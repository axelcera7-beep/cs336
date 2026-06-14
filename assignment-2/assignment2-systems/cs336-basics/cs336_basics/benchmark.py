import tomllib
from pathlib import Path
import torch
import time
import numpy as np 
import statistics

from cs336_basics.optimizer import AdamW
from cs336_basics.model import Transformer
from cs336_basics.utils import cross_entropy, gradient_clipping, learning_rate_schedule
import tomllib

def benchmarking_script(model_args, batch_size, context_length, vocab_size, device, w: int, n: int, lr_min: float = 3e-4, lr_max: float = 3e-4, warmup_steps: int = 300, num_steps: int =2500, max_l2_norm: float = 1.0):
    model = Transformer(**model_args)
    optimizer = AdamW(model.parameters())

    times_for = np.zeros(n)
    times_for_back = np.zeros(n)
    times_all = np.zeros(n)

    for w_iter in range(w):
        run_all(model, optimizer, vocab_size, batch_size, context_length, device, lr_min, lr_max, warmup_steps, num_steps, w_iter, max_l2_norm)
    
    for n_iter in range(n):
        times_for[n_iter] = run_forward_pass(model, optimizer, vocab_size, batch_size, context_length, device, lr_min, lr_max, warmup_steps, num_steps, n_iter)
        times_for_back[n_iter] = run_back_for_pass(model, optimizer, vocab_size, batch_size, context_length, device, lr_min, lr_max, warmup_steps, num_steps, n_iter)
        times_all[n_iter] = run_all(model, optimizer, vocab_size, batch_size, context_length, device, lr_min, lr_max, warmup_steps, num_steps, n_iter, max_l2_norm)
    print("times for forward mean",  times_for.mean(),"std", times_for.std(), "\n",
          "times for forward and backward mean",  times_for_back.mean(),"std", times_for_back.std(), "\n",
          "times for total",  times_all.mean(),"std", times_all.std(), "\n",    
          )

def _sync_gpu(device):
    if device == "cuda":
        torch.cuda.synchronize()

def run_forward_pass(model, optimizer, vocab_size, batch_size, context_length, device, lr_min, lr_max, warmup_steps, num_steps, step):
    lr = learning_rate_schedule(t=step, alpha_min=lr_min, alpha_max=lr_max, T_w=warmup_steps, T_c=num_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    _sync_gpu(device)
    time_start = time.perf_counter()
    optimizer.zero_grad()
    logits = model(inputs)    
    _sync_gpu(device)
    time_end = time.perf_counter()
    return time_end - time_start

def run_back_for_pass(model, optimizer, vocab_size, batch_size, context_length, device, lr_min, lr_max, warmup_steps, num_steps, step):
    lr = learning_rate_schedule(t=step, alpha_min=lr_min, alpha_max=lr_max, T_w=warmup_steps, T_c=num_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    _sync_gpu(device)
    time_start = time.perf_counter()
    optimizer.zero_grad()
    logits = model(inputs)  
    loss_train = cross_entropy(logits=logits, targets=targets)
    loss_train.backward()
    _sync_gpu(device)
    time_end = time.perf_counter()
    return time_end - time_start

def run_all(model, optimizer, vocab_size, batch_size, context_length, device, lr_min, lr_max, warmup_steps, num_steps, step, max_l2_norm):
    lr = learning_rate_schedule(t=step, alpha_min=lr_min, alpha_max=lr_max, T_w=warmup_steps, T_c=num_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets =torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    _sync_gpu(device)
    time_start = time.perf_counter()
    optimizer.zero_grad()
    logits = model(inputs)  
    loss_train = cross_entropy(logits=logits, targets=targets)
    loss_train.backward()
    gradient_clipping(params=model.parameters(), max_l2_norm=max_l2_norm)
    optimizer.step()
    _sync_gpu(device)
    time_end = time.perf_counter()
    return time_end - time_start

if __name__ == "__main__":
    
    CONFIG_DIR = Path(__file__).parent / "configs"

    MODEL_NAME    = "small"
    DATA_NAME     = "tinystories"
    TRAINING_NAME = "default"

    with open(CONFIG_DIR / "model" / f"{MODEL_NAME}.toml", "rb") as f:
        config_model = tomllib.load(f)

    with open(CONFIG_DIR / "data" / f"{DATA_NAME}.toml", "rb") as f:
        config_data = tomllib.load(f)

    with open(CONFIG_DIR / "training" / f"{TRAINING_NAME}.toml", "rb") as f:
        config_training = tomllib.load(f)

    model_args = {
        **config_model,                           
        "vocab_size": config_data["vocab_size"],  
        "device": config_training["device"],
    }

    batch_size = config_training["batch_size"]
    vocab_size = config_data["vocab_size"]
    device = config_training["device"]
    context_length = config_model["context_length"]
    warmup_iter = 10
    n_iter = 10

    benchmarking_script(model_args, batch_size, context_length, vocab_size, device, w=warmup_iter, n=n_iter)