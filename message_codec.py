from __future__ import annotations
import torch

def _ensure_length(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars]
    return text.ljust(max_chars, " ")

def text_to_bits(text: str, max_chars: int, bits_per_char: int = 8) -> torch.Tensor:
    text = _ensure_length(text, max_chars)
    bits = []
    for ch in text:
        v = ord(ch) & ((1 << bits_per_char) - 1)
        for i in reversed(range(bits_per_char)):
            bits.append((v >> i) & 1)
    return torch.tensor(bits, dtype=torch.float32)

def bits_to_text(bits: torch.Tensor, bits_per_char: int = 8) -> str:
    bits = (bits > 0.5).to(torch.int64).flatten().tolist()
    chars = []
    for i in range(0, len(bits), bits_per_char):
        chunk = bits[i : i + bits_per_char]
        if len(chunk) < bits_per_char:
            break
        v = 0
        for b in chunk:
            v = (v << 1) | int(b)

        chars.append(chr(v) if 32 <= v < 127 else "?")
    return "".join(chars).rstrip()

def random_bits(batch: int, n_bits: int, device=None) -> torch.Tensor:
    return torch.randint(0, 2, (batch, n_bits), device=device).float()

def bit_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred == target).float().mean().item()
