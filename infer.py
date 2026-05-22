from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch
from torchvision import transforms

from config import CFG, DEVICE, Config
from message_codec import text_to_bits, bits_to_text
from models import build_models
from distortions import apply_distortions, DistortionConfig
import residual_ops as R

def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tx = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return tx(img.convert("RGB")).unsqueeze(0)

def _to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1).cpu().squeeze(0)
    return transforms.ToPILImage()(t)

class Stegawand:
    def __init__(self, ckpt_path: Optional[Path] = None, cfg: Config = CFG):
        self.cfg = cfg
        self.device = DEVICE
        self.encoder, self.decoder = build_models(cfg)
        path = Path(ckpt_path) if ckpt_path else cfg.ckpt_path
        self.loaded = False
        if path.exists():
            ck = torch.load(path, map_location=self.device)
            self.encoder.load_state_dict(ck["encoder"])
            self.decoder.load_state_dict(ck["decoder"])
            self.loaded = True
        self.encoder.to(self.device).eval()
        self.decoder.to(self.device).eval()

    @torch.no_grad()
    def encode(self, cover: Image.Image, message: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
        c = _to_tensor(cover, self.cfg.image_size).to(self.device)
        bits = text_to_bits(message, self.cfg.msg_chars, self.cfg.bits_per_char).unsqueeze(0).to(self.device)
        stego, residual = self.encoder(c, bits)
        residual_view = R.amplify_for_view(residual, gain=10.0).expand(-1, 3, -1, -1)
        return _to_pil(c), _to_pil(stego), _to_pil(residual_view)

    @torch.no_grad()
    def decode(self, stego: Image.Image, dc: Optional[DistortionConfig] = None) -> Tuple[str, Image.Image]:
        s = _to_tensor(stego, self.cfg.image_size).to(self.device)
        if dc is not None:
            s = apply_distortions(s, dc)
        logits = self.decoder(s)
        bits = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()
        return bits_to_text(bits, self.cfg.bits_per_char), _to_pil(s)
