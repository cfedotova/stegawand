from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG, DEVICE, Config
from dataset import build_dataloaders
from distortions import DistortionLayer
from message_codec import random_bits, bit_accuracy
from models import build_models
import residual_ops as R

def _empty_mps_cache():
    if DEVICE.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def low_freq_l1(residual: torch.Tensor, pool: int = 16) -> torch.Tensor:
    coarse = F.avg_pool2d(residual, kernel_size=pool, stride=pool)
    return coarse.abs().mean()

def leakage_loss(residual: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
    r = residual.mean(dim=1, keepdim=True)
    c_low = F.avg_pool2d(cover.mean(dim=1, keepdim=True), kernel_size=8)
    c_low = F.interpolate(c_low, size=r.shape[-2:], mode="bilinear", align_corners=False)
    r_flat = (r - r.mean(dim=(2, 3), keepdim=True)).flatten(1)
    c_flat = (c_low - c_low.mean(dim=(2, 3), keepdim=True)).flatten(1)
    num = (r_flat * c_flat).sum(dim=1)
    den = r_flat.norm(dim=1).clamp_min(1e-6) * c_flat.norm(dim=1).clamp_min(1e-6)
    corr = num / den
    return corr.abs().mean()

def distortion_enabled(epoch: int, cfg: Config) -> bool:
    return epoch >= cfg.distortion_warmup_epochs

def image_weight(step: int, cfg: Config) -> float:
    if step < cfg.no_im_loss_steps:
        return 0.0
    elapsed = step - cfg.no_im_loss_steps
    if elapsed >= cfg.im_loss_ramp_steps:
        return cfg.w_image
    return cfg.w_image * elapsed / cfg.im_loss_ramp_steps

def train(cfg: Config = CFG, data_root: str = "data", resume: bool = True) -> Path:
    print(f"[train] device = {DEVICE}")
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder = build_models(cfg)
    encoder.to(DEVICE)
    decoder.to(DEVICE)

    distort = DistortionLayer(
        p_noise=cfg.p_noise, p_blur=cfg.p_blur, p_gray=cfg.p_gray, p_jpeg=cfg.p_jpeg,
        noise_sigma=cfg.noise_sigma, blur_kernel=cfg.blur_kernel,
        blur_sigma=cfg.blur_sigma, jpeg_quality=cfg.jpeg_quality,
    ).to(DEVICE)

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if resume and cfg.ckpt_path.exists():
        ck = torch.load(cfg.ckpt_path, map_location=DEVICE)
        try:
            encoder.load_state_dict(ck["encoder"])
            decoder.load_state_dict(ck["decoder"])
            opt.load_state_dict(ck["opt"])
            start_epoch = ck.get("epoch", 0)
            print(f"[train] resumed from epoch {start_epoch}")
        except (RuntimeError, KeyError) as e:
            print(f"[train] checkpoint at {cfg.ckpt_path} is incompatible with "
                  f"current architecture ({type(e).__name__}). Starting from scratch. "
                  f"Delete the file or pass --no-resume to silence this.")

    train_loader, val_loader = build_dataloaders(cfg, data_root=data_root)
    global_step = 0
    for epoch in range(start_epoch, cfg.epochs):
        use_distort = distortion_enabled(epoch, cfg)
        encoder.train(); decoder.train()

        distort.train(use_distort)
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}"
                   f" {'+distort' if use_distort else 'clean'}")
        for cover in bar:
            cover = cover.to(DEVICE, non_blocking=True)
            B = cover.size(0)
            msg = random_bits(B, cfg.msg_bits, device=DEVICE)

            stego, residual = encoder(cover, msg)
            noisy = distort(stego) if use_distort else stego
            logits = decoder(noisy)

            l_msg = bce(logits, msg)
            l_img = F.l1_loss(stego, cover)
            l_res = residual.abs().mean()
            l_lf = low_freq_l1(residual, pool=cfg.lowfreq_pool)
            l_leak = leakage_loss(residual, cover)

            w_img_now = image_weight(global_step, cfg)
            loss = (cfg.w_msg * l_msg
                    + w_img_now * l_img
                    + cfg.w_residual * l_res
                    + cfg.w_lowfreq * l_lf
                    + cfg.w_leakage * l_leak)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
            opt.step()

            if global_step % 20 == 0:
                acc = bit_accuracy(logits, msg)
                bar.set_postfix(loss=f"{loss.item():.3f}",
                                msg=f"{l_msg.item():.3f}",
                                img=f"{l_img.item():.4f}",
                                w_img=f"{w_img_now:.2f}",
                                acc=f"{acc:.3f}")
            global_step += 1
            if global_step % cfg.empty_cache_every == 0:
                _empty_mps_cache()

        encoder.eval(); decoder.eval(); distort.eval()
        with torch.no_grad():
            v_acc, v_l1, n = 0.0, 0.0, 0
            for cover in val_loader:
                cover = cover.to(DEVICE)
                B = cover.size(0)
                msg = random_bits(B, cfg.msg_bits, device=DEVICE)
                stego, residual = encoder(cover, msg)
                logits = decoder(stego)
                v_acc += bit_accuracy(logits, msg) * B
                v_l1 += F.l1_loss(stego, cover).item() * B
                n += B
            v_acc /= max(1, n); v_l1 /= max(1, n)
        dt = time.time() - t0
        print(f"[val] epoch {epoch + 1}: bit_acc={v_acc:.4f} stego_L1={v_l1:.5f} ({dt:.1f}s)")

        if (epoch + 1) % cfg.save_every == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch + 1,
                "config": vars(cfg),
            }, cfg.ckpt_path)
            print(f"[ckpt] saved -> {cfg.ckpt_path}")
        _empty_mps_cache()

    return cfg.ckpt_path

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    if args.epochs is not None:
        CFG.epochs = args.epochs
    if args.batch is not None:
        CFG.batch_size = args.batch
    train(CFG, data_root=args.data_root, resume=not args.no_resume)

if __name__ == "__main__":
    _cli()
