import torch
import math
from collections.abc import Callable, Iterable
import numpy as np
from typing import Tuple, IO, Any, BinaryIO
import numpy.typing as npt
import os
from pathlib import Path


def softmax(x: torch.Tensor, dim: int, temperature: float = 1.0) -> torch.Tensor:
    max_element, _ = torch.max(x, dim=dim, keepdim=True)
    scaled_x = (x - max_element) / temperature
    exp_x = torch.exp(scaled_x)
    norm_x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return norm_x

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = k.shape[-1]
    inv_d_k = 1 / math.sqrt(d_k)
    # [batch_size, seq_len_q, d_k] * [batch_size, d_k, seq_len_k] -> [batch_size, seq_len_q, seq_len_k]
    scores = q @ k.transpose(-2, -1) * inv_d_k 
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = softmax(scores, -1)
    output = probs @ v
    return output

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    # logits.shape = [batch_size, seq_len, vocab_size]
    m = torch.amax(logits, dim=-1, keepdim=True)
    nll = torch.log(torch.sum(torch.exp(logits-m), dim=-1, keepdim=True)) - logits + m
    targets = targets.unsqueeze(-1) # targets and nll need to have the same number of dimension
    nll = torch.gather(nll, dim=-1, index=targets).squeeze(-1)
    cross_entropy = torch.mean(nll)
    return cross_entropy

def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits=logits, targets=targets))

def learning_rate_schedule(t: int, alpha_min: float, alpha_max: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t * alpha_max / T_w
    if T_w <= t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t-T_w)*math.pi / (T_c - T_w))) * (alpha_max - alpha_min)
    if t > T_c:
        return alpha_min
    
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> torch.Tensor:
    l2 = torch.tensor(0.0)
    params = list(params)
    for p in params:
        grad = p.grad
        if grad is None:
            continue
        l2 += torch.sum(grad ** 2)
    if l2 > max_l2_norm**2:
        for p in params:
          if p.grad is not None:
            p.grad *= max_l2_norm / (torch.sqrt(l2) + eps)

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(x)
    start_indexes = np.expand_dims(np.random.randint(0, n - context_length, batch_size), axis=-1)
    length = np.expand_dims(np.arange(context_length), axis=0)
    indexes = start_indexes + length # [batch, 1] + [1, context] -> [batch, context]
    target_indexes = start_indexes + length + np.ones_like(length)
    input_np = x[indexes]
    target_np = x[target_indexes]

    inputs = torch.from_numpy(input_np).to(device, dtype=torch.long)
    targets = torch.from_numpy(target_np).to(device, dtype=torch.long)

    return inputs, targets

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

