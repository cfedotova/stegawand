from __future__ import annotations
from dataclasses import dataclass
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import residual_ops as R

@dataclass
class DistortionConfig:
    noise: bool = False
    blur: bool = False
    gray: bool = False
    jpeg: bool = False
    noise_sigma: float = 0.02
    blur_kernel: int = 5
    blur_sigma: float = 1.0
    jpeg_quality: int = 75

def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)

def to_grayscale(x: torch.Tensor) -> torch.Tensor:

    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    g = (x * w).sum(dim=1, keepdim=True)
    return g.expand_as(x)

def fake_jpeg(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    q = max(20, min(95, quality))
    factor = 0.55 + 0.45 * (q / 100.0)
    h, w = x.shape[-2:]
    nh, nw = max(8, int(h * factor)), max(8, int(w * factor))
    y = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

    step = (100 - q) / 100.0 * 0.015
    y = y + (torch.rand_like(y) - 0.5) * step
    return y.clamp(0.0, 1.0)

def apply_distortions(x: torch.Tensor, dc: DistortionConfig) -> torch.Tensor:
    if dc.blur:
        x = R.gaussian_blur(x, dc.blur_kernel, dc.blur_sigma)
    if dc.noise:
        x = add_gaussian_noise(x, dc.noise_sigma)
    if dc.gray:
        x = to_grayscale(x)
    if dc.jpeg:
        x = fake_jpeg(x, dc.jpeg_quality)
    return x.clamp(0.0, 1.0)

class DistortionLayer(nn.Module):

    def __init__(self, p_noise: float, p_blur: float, p_gray: float, p_jpeg: float,
                 noise_sigma: float = 0.02, blur_kernel: int = 5,
                 blur_sigma: float = 1.0, jpeg_quality: int = 75):
        super().__init__()
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_gray = p_gray
        self.p_jpeg = p_jpeg
        self.noise_sigma = noise_sigma
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.jpeg_quality = jpeg_quality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if random.random() < self.p_blur:
            x = R.gaussian_blur(x, self.blur_kernel, self.blur_sigma)
        if random.random() < self.p_noise:
            x = add_gaussian_noise(x, self.noise_sigma)
        if random.random() < self.p_gray:
            x = to_grayscale(x)
        if random.random() < self.p_jpeg:
            x = fake_jpeg(x, self.jpeg_quality)
        return x.clamp(0.0, 1.0)
