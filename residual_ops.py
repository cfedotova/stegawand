from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def gaussian_kernel1d(size: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    half = (size - 1) / 2.0
    x = torch.arange(size, device=device, dtype=dtype) - half
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()

def gaussian_kernel2d(size: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    k1 = gaussian_kernel1d(size, sigma, device=device, dtype=dtype)
    return torch.outer(k1, k1)

def gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    c = x.shape[1]
    k = gaussian_kernel2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    k = k.expand(c, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    return F.conv2d(x, k, padding=pad, groups=c)

def brightness_capacity(cover: torch.Tensor, floor: float = 0.25) -> torch.Tensor:
    brightness = cover.mean(dim=1, keepdim=True)

    cap = 1.0 - 4.0 * (brightness - 0.5).pow(2)
    return cap.clamp(floor, 1.0)

def _laplacian_energy(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    k = x.new_tensor([[0.0, 1.0, 0.0],
                      [1.0, -4.0, 1.0],
                      [0.0, 1.0, 0.0]])
    k = k.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
    e = F.conv2d(x, k, padding=1, groups=c).abs().sum(dim=1, keepdim=True)
    return e

def texture_capacity(cover: torch.Tensor, floor: float = 0.30,
                     smooth_size: int = 5, smooth_sigma: float = 1.2) -> torch.Tensor:
    e = _laplacian_energy(cover)
    e = gaussian_blur(e, smooth_size, smooth_sigma)

    e_max = e.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    e = e / e_max
    return e.clamp(floor, 1.0)

def smooth_residual(residual: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
    return gaussian_blur(residual, kernel_size, sigma)

def suppress_low_frequency(residual: torch.Tensor, pool: int = 16) -> torch.Tensor:
    coarse = F.avg_pool2d(residual, kernel_size=pool, stride=pool)
    coarse = F.interpolate(coarse, size=residual.shape[-2:], mode="bilinear", align_corners=False)
    return residual - coarse

def amplify_for_view(residual: torch.Tensor, gain: float = 10.0) -> torch.Tensor:
    g = residual.mean(dim=1, keepdim=True) * gain
    return (g * 0.5 + 0.5).clamp(0.0, 1.0)

def ghost_residual_view(cover: torch.Tensor, residual: torch.Tensor,
                        amp: float = 5.0, fade: float = 0.25) -> torch.Tensor:
    if cover.ndim == 4: cover = cover[0]
    if residual.ndim == 4: residual = residual[0]
    luma = (cover * cover.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0, keepdim=True)
    faded = (0.5 + (luma - 0.5) * fade).expand(3, -1, -1)
    return (faded + residual * amp).clamp(0.0, 1.0)
