from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
import residual_ops as R

_CODES_SEED = 1337

def make_bit_codes(n_bits: int, h: int, w: int, seed: int = _CODES_SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    codes = torch.rand(n_bits, h, w, generator=g) > 0.5
    return codes.float() * 2.0 - 1.0

def encode_message_pattern(message: torch.Tensor, bit_codes: torch.Tensor) -> torch.Tensor:
    signed = message * 2.0 - 1.0
    pattern = torch.einsum("bn,nhw->bhw", signed, bit_codes)
    pattern = pattern.unsqueeze(1) / math.sqrt(bit_codes.shape[0])
    return pattern

class Encoder(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ch = cfg.enc_channels
        self.register_buffer(
            "bit_codes",
            make_bit_codes(cfg.msg_bits, cfg.image_size, cfg.image_size),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(3 + 1, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),    nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),    nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),    nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 3, kernel_size=1),
            nn.Tanh(),
        )

        last_conv = self.refine[-2]
        nn.init.normal_(last_conv.weight, std=0.10)
        nn.init.zeros_(last_conv.bias)

    def forward(self, cover: torch.Tensor, message: torch.Tensor):
        msg_pattern = encode_message_pattern(message, self.bit_codes)
        x = torch.cat([cover, msg_pattern], dim=1)
        raw = self.refine(x)
        bcap = R.brightness_capacity(cover, floor=self.cfg.brightness_mask_min)
        tcap = R.texture_capacity(cover, floor=self.cfg.texture_mask_min)
        residual = raw * bcap * tcap * self.cfg.residual_amplitude
        stego = (cover + residual).clamp(0.0, 1.0)
        return stego, residual

class Decoder(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ch = cfg.dec_channels
        self.register_buffer(
            "bit_codes",
            make_bit_codes(cfg.msg_bits, cfg.image_size, cfg.image_size),
        )
        self.estimator = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, kernel_size=1),
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        B, _, H, W = stego.shape
        pattern = self.estimator(stego).squeeze(1)
        logits = torch.einsum("bhw,nhw->bn", pattern, self.bit_codes)
        return logits / math.sqrt(H * W)

def build_models(cfg: Config):
    return Encoder(cfg), Decoder(cfg)
