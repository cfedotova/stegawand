from __future__ import annotations
import time
import torch
import torch.nn.functional as F

from config import CFG, DEVICE
from distortions import DistortionLayer, apply_distortions, DistortionConfig
from message_codec import random_bits, bit_accuracy, text_to_bits, bits_to_text
from models import build_models
import residual_ops as R

def _fake_covers(n: int, size: int) -> torch.Tensor:
    x = torch.linspace(0, 1, size).view(1, 1, 1, size).expand(n, 3, size, size).clone()
    y = torch.linspace(0, 1, size).view(1, 1, size, 1).expand(n, 3, size, size)
    img = 0.5 * x + 0.5 * y
    img = img + 0.05 * torch.randn_like(img)
    return img.clamp(0, 1)

def run() -> bool:
    print(f"[smoke] device = {DEVICE}, torch={torch.__version__}")
    ok = True

    text = "hi!"[: CFG.msg_chars]
    bits = text_to_bits(text, CFG.msg_chars)
    back = bits_to_text(bits)
    assert back.strip() == text.strip(), f"codec round-trip failed: {back!r}"
    print(f"[smoke] codec round-trip OK ({len(bits)} bits)")

    enc, dec = build_models(CFG)
    enc.to(DEVICE).eval(); dec.to(DEVICE).eval()
    nparams = sum(p.numel() for p in list(enc.parameters()) + list(dec.parameters()))
    print(f"[smoke] params total = {nparams/1e6:.2f}M (enc+dec)")

    distort = DistortionLayer(0.5, 0.5, 0.3, 0.4).to(DEVICE).train()
    enc.train(); dec.train()
    cover = _fake_covers(CFG.batch_size, CFG.image_size).to(DEVICE)
    msg = random_bits(CFG.batch_size, CFG.msg_bits, device=DEVICE)

    t0 = time.time()
    stego, residual = enc(cover, msg)
    noisy = distort(stego)
    logits = dec(noisy)
    loss = F.binary_cross_entropy_with_logits(logits, msg) + 1.6 * F.l1_loss(stego, cover)
    loss.backward()
    dt = (time.time() - t0) * 1000
    if torch.isnan(loss) or torch.isinf(loss):
        print("[smoke] FAIL: loss NaN/Inf"); ok = False
    else:
        print(f"[smoke] fwd+bwd OK loss={loss.item():.3f} ({dt:.0f} ms, batch={CFG.batch_size})")

    with torch.no_grad():
        amax = residual.abs().amax().item()
        coarse = F.avg_pool2d(residual, kernel_size=16).abs().mean().item()
    print(f"[smoke] residual |max|={amax:.4f} (cap≈{CFG.residual_amplitude}) "
          f"lowfreq_mean={coarse:.5f}")
    if amax > CFG.residual_amplitude * 1.1:
        print("[smoke] WARN: residual exceeded amplitude cap"); ok = False

    with torch.no_grad():
        enc.eval(); dec.eval()
        stego2, _ = enc(cover, msg)
        logits2 = dec(stego2)
        acc = bit_accuracy(logits2, msg)
    print(f"[smoke] untrained decode acc = {acc:.3f} (≈0.5 expected before training)")

    with torch.no_grad():
        for name, dc in [
            ("noise", DistortionConfig(noise=True)),
            ("blur",  DistortionConfig(blur=True)),
            ("gray",  DistortionConfig(gray=True)),
            ("jpeg",  DistortionConfig(jpeg=True)),
        ]:
            y = apply_distortions(cover[:1], dc)
            assert y.shape == cover[:1].shape
            assert torch.isfinite(y).all()
        print("[smoke] distortion pipeline OK")

    if DEVICE.type == "mps":
        try:
            mb = torch.mps.current_allocated_memory() / 1024 / 1024
            print(f"[smoke] MPS allocated: {mb:.1f} MB")
        except Exception:
            pass

    print("[smoke] DONE" + (" — all checks passed" if ok else " — issues above"))
    return ok

if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
