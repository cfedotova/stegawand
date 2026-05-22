from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102
from PIL import Image, UnidentifiedImageError

def _train_transform(size: int):
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def _eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

class CoverOnlyFlowers(Dataset):
    def __init__(self, root: str, split: str, size: int, download: bool = True):
        tx = _train_transform(size) if split == "train" else _eval_transform(size)
        self.base = Flowers102(root=root, split=split, download=download, transform=tx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img

class FolderImages(Dataset):
    def __init__(self, root: str, size: int, train: bool = True):
        self.paths = []
        for p in sorted(Path(root).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                self.paths.append(p)
        if not self.paths:
            raise RuntimeError(f"No images in {root}")
        self.tx = _train_transform(size) if train else _eval_transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except (UnidentifiedImageError, OSError):

            img = Image.open(self.paths[0]).convert("RGB")
        return self.tx(img)

def _try_flowers(root: str, split: str, size: int) -> Optional[Dataset]:
    try:
        return CoverOnlyFlowers(root, split, size, download=True)
    except Exception as e:
        print(f"[dataset] Flowers102 unavailable ({e}); will use fallback folder.")
        return None

def build_dataloaders(cfg, data_root: str = "data") -> tuple[DataLoader, DataLoader]:
    Path(data_root).mkdir(parents=True, exist_ok=True)
    custom = Path(data_root) / "custom"

    train_ds = _try_flowers(data_root, "train", cfg.image_size)
    if train_ds is None:
        if not custom.exists():
            raise RuntimeError(
                f"Flowers102 download failed and no fallback at {custom}. "
                f"Drop a few hundred .jpg/.png files into that folder."
            )
        train_ds = FolderImages(str(custom), cfg.image_size, train=True)
        val_ds = FolderImages(str(custom), cfg.image_size, train=False)
    else:
        val_ds = CoverOnlyFlowers(data_root, "val", cfg.image_size, download=True)

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        drop_last=True,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader
