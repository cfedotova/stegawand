from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@dataclass
class Config:
    msg_chars: int = 5
    bits_per_char: int = 8

    image_size: int = 256

    enc_channels: int = 32
    dec_channels: int = 48

    residual_amplitude: float = 0.05
    brightness_mask_min: float = 0.30
    texture_mask_min: float = 0.35

    batch_size: int = 6
    epochs: int = 8
    lr: float = 5e-4
    weight_decay: float = 0.0

    w_msg: float = 5.0
    w_image: float = 0.3

    distortion_warmup_epochs: int = 4

    no_im_loss_steps: int = 300
    im_loss_ramp_steps: int = 400

    p_noise: float = 0.40
    p_blur: float = 0.30
    p_gray: float = 0.10
    p_jpeg: float = 0.15
    noise_sigma: float = 0.02
    blur_kernel: int = 5
    blur_sigma: float = 1.0
    jpeg_quality: int = 75

    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False

    ckpt_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    ckpt_name: str = "stegawand.pt"
    save_every: int = 1

    empty_cache_every: int = 10

    @property
    def msg_bits(self) -> int:
        return self.msg_chars * self.bits_per_char

    @property
    def ckpt_path(self) -> Path:
        return self.ckpt_dir / self.ckpt_name

CFG = Config()
DEVICE = get_device()
