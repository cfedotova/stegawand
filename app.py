from __future__ import annotations
import argparse
import base64
import io
import math
from typing import List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from flask import Flask, jsonify, render_template, request

from config import CFG
from distortions import DistortionConfig, apply_distortions
from infer import Stegawand
from message_codec import text_to_bits, bits_to_text
import residual_ops as R

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception:
    _HEIF_OK = False

app = Flask(__name__, template_folder="templates", static_folder="static")
STATE: dict = {"model": None, "ckpt": None}

def get_model() -> Stegawand:
    if STATE["model"] is None:
        STATE["model"] = Stegawand(ckpt_path=STATE["ckpt"])
        if not STATE["model"].loaded:
            print("[app] WARNING: no checkpoint loaded — decoded text will be random. "
                  "Train first with: python train.py")
    return STATE["model"]

def _b64_to_pil(data_url: str) -> Image.Image:
    mime = ""
    if data_url.startswith("data:") and "," in data_url:
        header, data_url = data_url.split(",", 1)
        if ";" in header:
            mime = header.split(":", 1)[1].split(";", 1)[0].lower()
    raw = base64.b64decode(data_url)
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as e:
        hint = ""
        heic_magic = raw[4:12] in (b"ftypheic", b"ftypheif", b"ftypmif1", b"ftypmsf1")
        if mime in ("image/heic", "image/heif") or heic_magic:
            hint = (" The file looks like HEIC/HEIF (Mac default). "
                    "Either convert it to PNG/JPG first, or install HEIC "
                    "support: `pip install pillow-heif` and restart the server.")
            if _HEIF_OK:
                hint = " HEIC support is loaded but this specific file could still not be read."
        raise ValueError(f"unsupported or corrupt image (mime={mime or 'unknown'}).{hint}") from e

def _tensor_to_b64(t: torch.Tensor) -> str:
    t = t.detach().clamp(0, 1).cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    pil = transforms.ToPILImage()(t)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _pil_to_tensor(img: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    tx = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return tx(img).unsqueeze(0).to(device)

def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).item()

def _bits_str(bit_tensor: torch.Tensor) -> str:
    return "".join(str(int(b)) for b in bit_tensor.flatten().tolist())

def _make_distortion_config(d: dict) -> DistortionConfig:
    return DistortionConfig(
        noise=bool(d.get("noise")),
        blur=bool(d.get("blur")),
        gray=bool(d.get("grayscale") or d.get("gray")),
        jpeg=bool(d.get("jpeg")),
        noise_sigma=float(d.get("noise_sigma", 0.02)),
        blur_sigma=float(d.get("blur_sigma", 1.0)),
        blur_kernel=int(d.get("blur_kernel", 5)),
        jpeg_quality=int(d.get("jpeg_quality", 75)),
    )

def _build_pipelines(cfg: dict) -> List[Tuple[str, Optional[DistortionConfig]]]:
    pipelines: List[Tuple[str, Optional[DistortionConfig]]] = [("clean", None)]
    if cfg.get("noise"):
        pipelines.append(("noise", DistortionConfig(
            noise=True, noise_sigma=float(cfg.get("noise_sigma", 0.02)))))
    if cfg.get("blur"):
        pipelines.append(("blur", DistortionConfig(
            blur=True, blur_sigma=float(cfg.get("blur_sigma", 1.0)),
            blur_kernel=int(cfg.get("blur_kernel", 5)))))
    if cfg.get("grayscale") or cfg.get("gray"):
        pipelines.append(("grayscale", DistortionConfig(gray=True)))
    if cfg.get("jpeg"):
        pipelines.append(("jpeg", DistortionConfig(
            jpeg=True, jpeg_quality=int(cfg.get("jpeg_quality", 75)))))
    return pipelines

def _common_ctx():
    return {
        "max_chars":     CFG.msg_chars,
        "msg_bits":      CFG.msg_bits,
        "image_size":    CFG.image_size,
        "model_loaded":  get_model().loaded,
    }

@app.get("/")
def page_about():
    return render_template("about.html", active="about", **_common_ctx())

@app.get("/encode")
def page_encode():
    return render_template("encode.html", active="encode", **_common_ctx())

@app.get("/decode")
def page_decode():
    return render_template("decode.html", active="decode", **_common_ctx())

@app.get("/stress")
def page_stress():
    return render_template("stress.html", active="stress", **_common_ctx())

@app.get("/theory")
def page_theory():
    return render_template("theory.html", active="theory", **_common_ctx())

@app.get("/demo")
def page_demo():
    return render_template("demo.html", active="demo", **_common_ctx())

@app.get("/api/sample-image")
def api_sample_image():
    import random as _r
    from pathlib import Path as _P
    candidates = list((_P(__file__).resolve().parent / "data" / "flowers-102" / "jpg").glob("image_*.jpg"))
    if not candidates:
        return jsonify({"error": "no sample images found in data/flowers-102/jpg/"}), 404
    if request.args.get("random") == "1":
        p = _r.choice(candidates)
    else:
        p = sorted(candidates)[min(50, len(candidates) - 1)]
    img = Image.open(p).convert("RGB")
    return jsonify({"image": _tensor_to_b64(_pil_to_tensor(img, CFG.image_size, torch.device("cpu"))),
                    "name": p.name})

@app.post("/api/encode")
def api_encode():
    data = request.get_json(force=True) or {}
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "no image provided"}), 400
    try:
        pil = _b64_to_pil(img_b64)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "empty message"}), 400
    msg_padded = msg[: CFG.msg_chars].ljust(CFG.msg_chars, " ")

    smooth = bool(data.get("smooth_residual", True))

    m = get_model()
    cover = _pil_to_tensor(pil, CFG.image_size, m.device)
    bits = text_to_bits(msg_padded, CFG.msg_chars, CFG.bits_per_char).unsqueeze(0).to(m.device)
    encoded_bits_str = _bits_str(bits[0].cpu())

    with torch.no_grad():
        stego, residual = m.encoder(cover, bits)
        if smooth:
            residual = R.gaussian_blur(residual, kernel_size=5, sigma=1.0)
            stego = (cover + residual).clamp(0.0, 1.0)

    psnr_db = _psnr(cover, stego)
    ghost = R.ghost_residual_view(cover[0], residual[0], amp=5.0)

    pipelines = _build_pipelines(data.get("distortions") or {})
    results = []
    with torch.no_grad():
        for name, dc in pipelines:
            distorted = stego if dc is None else apply_distortions(stego, dc)
            logits = m.decoder(distorted)
            pred = (torch.sigmoid(logits) > 0.5).float()
            decoded_text = bits_to_text(pred[0].cpu(), CFG.bits_per_char)
            bit_acc = (pred == bits).float().mean().item()
            results.append({
                "name":         name,
                "image":        _tensor_to_b64(distorted),
                "decoded":      decoded_text,
                "bit_acc":      round(bit_acc, 4),
                "decoded_bits": _bits_str(pred[0].cpu()),
            })

    return jsonify({
        "cover":         _tensor_to_b64(cover),
        "stego":         _tensor_to_b64(stego),
        "residual":      _tensor_to_b64(ghost),
        "psnr":          round(psnr_db, 2),
        "message_used":  msg_padded.rstrip(),
        "encoded_bits":  encoded_bits_str,
        "results":       results,
    })

@app.post("/api/decode")
def api_decode():
    data = request.get_json(force=True) or {}
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "no image provided"}), 400
    try:
        pil = _b64_to_pil(img_b64)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    m = get_model()
    inp = _pil_to_tensor(pil, CFG.image_size, m.device)

    dc = _make_distortion_config(data.get("distortions") or {})
    apply_dist = any([dc.noise, dc.blur, dc.gray, dc.jpeg])
    if apply_dist:
        with torch.no_grad():
            inp_processed = apply_distortions(inp, dc)
    else:
        inp_processed = inp

    with torch.no_grad():
        logits = m.decoder(inp_processed)
        pred = (torch.sigmoid(logits) > 0.5).float()
        decoded_text = bits_to_text(pred[0].cpu(), CFG.bits_per_char)
        bit_string = _bits_str(pred[0].cpu())

    return jsonify({
        "decoded":      decoded_text,
        "decoded_bits": bit_string,
        "input":        _tensor_to_b64(inp_processed),
    })

@app.post("/api/stress")
def api_stress():
    data = request.get_json(force=True) or {}
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "no image provided"}), 400
    try:
        pil = _b64_to_pil(img_b64)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "empty message"}), 400
    msg_padded = msg[: CFG.msg_chars].ljust(CFG.msg_chars, " ")

    intensity = float(data.get("intensity", 1.0))
    intensity = max(0.0, min(3.0, intensity))

    m = get_model()
    cover = _pil_to_tensor(pil, CFG.image_size, m.device)
    bits = text_to_bits(msg_padded, CFG.msg_chars, CFG.bits_per_char).unsqueeze(0).to(m.device)
    encoded_bits_str = _bits_str(bits[0].cpu())

    raw = data.get("distortions") or {}
    dc = DistortionConfig(
        noise=bool(raw.get("noise")),
        blur=bool(raw.get("blur")),
        gray=bool(raw.get("grayscale") or raw.get("gray")),
        jpeg=bool(raw.get("jpeg")),
        noise_sigma=float(raw.get("noise_sigma", 0.02)) * intensity,
        blur_sigma=max(0.2, float(raw.get("blur_sigma", 1.0)) * intensity),
        blur_kernel=int(raw.get("blur_kernel", 5)),
        jpeg_quality=max(15, int(float(raw.get("jpeg_quality", 75))
                                  - (intensity - 1.0) * 30)),
    )

    with torch.no_grad():
        stego, residual = m.encoder(cover, bits)
        residual = R.gaussian_blur(residual, kernel_size=5, sigma=1.0)
        stego = (cover + residual).clamp(0.0, 1.0)
        distorted = apply_distortions(stego, dc) if (dc.noise or dc.blur or dc.gray or dc.jpeg) else stego
        logits = m.decoder(distorted)
        pred = (torch.sigmoid(logits) > 0.5).float()
        decoded_text = bits_to_text(pred[0].cpu(), CFG.bits_per_char)
        bit_acc = (pred == bits).float().mean().item()
        decoded_bits = _bits_str(pred[0].cpu())

    psnr_db = _psnr(cover, stego)
    psnr_distorted = _psnr(cover, distorted)

    return jsonify({
        "cover":          _tensor_to_b64(cover),
        "stego":          _tensor_to_b64(stego),
        "distorted":      _tensor_to_b64(distorted),
        "psnr":           round(psnr_db, 2),
        "psnr_distorted": round(psnr_distorted, 2),
        "message_used":   msg_padded.rstrip(),
        "encoded_bits":   encoded_bits_str,
        "decoded_bits":   decoded_bits,
        "decoded":        decoded_text,
        "bit_acc":        round(bit_acc, 4),
        "intensity":      round(intensity, 2),
        "effective": {
            "noise_sigma":  round(dc.noise_sigma, 4),
            "blur_sigma":   round(dc.blur_sigma, 2),
            "jpeg_quality": dc.jpeg_quality,
        },
    })

@app.get("/api/codes")
def api_codes():
    n = int(request.args.get("n", 8))
    n = min(max(1, n), CFG.msg_bits)

    m = get_model()
    codes = m.encoder.bit_codes[:n].detach().cpu()
    codes64 = torch.nn.functional.interpolate(
        codes.unsqueeze(1), size=(64, 64), mode="nearest"
    ).squeeze(1)

    images = []
    for i, c in enumerate(codes64):
        gray = (c * 0.5 + 0.5).clamp(0, 1)
        rgb = gray.unsqueeze(0).expand(3, -1, -1)
        images.append({"i": i, "image": _tensor_to_b64(rgb)})
    return jsonify({"codes": images, "total_bits": CFG.msg_bits})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CFG.ckpt_path))
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    STATE["ckpt"] = args.ckpt
    get_model()
    print(f"[app] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
