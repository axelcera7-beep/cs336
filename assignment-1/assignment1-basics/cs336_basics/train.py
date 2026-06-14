import numpy as np
import torch
from cs336_basics.model import Transformer
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import data_loading, save_checkpoint, load_checkpoint, cross_entropy, gradient_clipping, learning_rate_schedule
import cProfile, pstats

checkpoint_dir = "/Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/artifacts/tinystories/checkpoints"

dataset = "tinystories"
device = "cpu"
num_steps = 2500

context_length = 256
num_layers = 4
num_heads = 16
d_model = 512
d_ff = 1344
batch_size = 256
lr_max = 3e-4
lr_min = 3e-5
warmup_steps = 300
max_l2_norm = 1.0
theta = 10000

steps_saves = 1000
steps_valid_loss = 10
steps_train_loss = 10

if dataset == "tinystories":
  train_data_path = "/Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/artifacts/tinystories/train.bin"
  valid_data_path = "/Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/artifacts/tinystories/valid.bin"
  vocab_size = 10000
if dataset == "owt":
  train_data_path = "/raid/home/students/cera_axe/assignment1-basics/artifacts/owt/train.bin"
  valid_data_path = "/raid/home/students/cera_axe/assignment1-basics/artifacts/owt/valid.bin"
  vocab_size = 32000

def training_together():

  model = Transformer(vocab_size=vocab_size, 
                        context_length=context_length, 
                        num_layers=num_layers, 
                        d_model=d_model,
                        num_heads=num_heads, 
                        d_ff = d_ff,
                        theta=theta,
                        device=device)
    
  data_train = np.memmap(train_data_path, dtype=np.uint16, mode="r")
  data_valid = np.memmap(valid_data_path, dtype=np.uint16, mode="r")
  num_params = sum([p.numel() for p in model.parameters()])
  print("on commence l'entrainement on a ", num_params, "paramètres")
  optimizer = AdamW(model.parameters())

  for step in range(num_steps):
      lr = learning_rate_schedule(t=step, alpha_min=lr_min, alpha_max=lr_max, T_w=warmup_steps, T_c=num_steps)
      for group in optimizer.param_groups:
          group["lr"] = lr

      inputs, targets = data_loading(x=data_train, batch_size=batch_size, context_length=context_length, device=device)
      optimizer.zero_grad()
      logits = model(inputs)
      loss_train = cross_entropy(logits=logits, targets=targets)
      loss_train.backward()
      gradient_clipping(params=model.parameters(), max_l2_norm=max_l2_norm)
      optimizer.step()

      if step % steps_valid_loss == 0:
        with torch.no_grad():
            val_losses = []
            for _ in range(20):  # 20 batches de validation
                inputs, targets = data_loading(x=data_valid, batch_size=batch_size, context_length=context_length, device=device)
                logits = model(inputs)
                val_losses.append(cross_entropy(logits=logits, targets=targets).item())
            loss_valid = np.mean(val_losses)
            print(f"step {step} valid loss {loss_valid:.4f}", flush=True)
        
      if step % steps_saves == 0:
          checkpoint_path = checkpoint_dir + f"/{dataset}_{step}.pt"
          save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=checkpoint_path)
          
      if step % steps_train_loss == 0:
          print(f"step {step} train loss {loss_train.item():.4f}", flush=True)

if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()
    training_together()
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats("cumulative")
    #stats.print_stats(100)
"""ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       10    0.000    0.000   33.742    3.374 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_tensor.py:576(backward)
       10    0.000    0.000   33.742    3.374 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/autograd/__init__.py:253(backward)
       10    0.000    0.000   33.741    3.374 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/autograd/graph.py:860(_engine_run_backward)
       10   33.741    3.374   33.741    3.374 {method 'run_backward' of 'torch._C._EngineBase' objects}
   640/20    0.002    0.000   28.832    1.442 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py:1775(_wrapped_call_impl)
   640/20    0.352    0.001   28.832    1.442 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py:1783(_call_impl)
       20    0.028    0.001   28.832    1.442 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:175(forward)
      300   21.800    0.073   21.800    0.073 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:24(forward)
       40    0.354    0.009   21.798    0.545 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:142(forward)
       40    0.500    0.013   11.833    0.296 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:55(forward)
       40    0.084    0.002    8.539    0.213 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:113(forward)
       20    2.112    0.106    3.980    0.199 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/utils.py:28(cross_entropy)
    23/16    0.000    0.000    3.465    0.217 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_ops.py:368(fallthrough)
       40    1.246    0.031    1.677    0.042 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/utils.py:17(scaled_dot_product_attention)
       80    1.150    0.014    1.600    0.020 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:81(forward)
       10    0.000    0.000    1.366    0.137 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/optim/optimizer.py:512(wrapper)
       10    1.202    0.120    1.365    0.136 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/optimizer.py:34(step)
       22    0.000    0.000    1.268    0.058 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_ops.py:311(py_impl)
       60    1.215    0.020    1.215    0.020 {built-in method torch.exp}
       40    0.466    0.012    1.097    0.027 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:62(silu)
      100    0.682    0.007    1.084    0.011 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:35(forward)
        3    0.000    0.000    0.837    0.279 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_ops.py:166(py_functionalize_impl)
      104    0.001    0.000    0.823    0.008 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/library.py:285(impl)
       40    0.630    0.016    0.630    0.016 {built-in method torch.sigmoid}
        1    0.000    0.000    0.630    0.630 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:154(__init__)
       16    0.000    0.000    0.629    0.039 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/nn/init.py:268(trunc_normal_)
       16    0.000    0.000    0.629    0.039 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/nn/init.py:86(_no_grad_trunc_normal_)
       16    0.504    0.032    0.504    0.032 {method 'erfinv_' of 'torch._C.TensorBase' objects}
       15    0.000    0.000    0.481    0.032 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:17(__init__)
      370    0.480    0.001    0.480    0.001 {built-in method torch.sum}
       80    0.448    0.006    0.448    0.006 {built-in method torch.stack}
        3    0.000    0.000    0.433    0.144 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_ops.py:283(__init__)
        1    0.000    0.000    0.403    0.403 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_higher_order_ops/triton_kernel_wrap.py:1077(__init__)
        1    0.000    0.000    0.401    0.401 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/__init__.py:1(<module>)
        1    0.000    0.000    0.386    0.386 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/_fully_shard/__init__.py:1(<module>)
        1    0.000    0.000    0.332    0.332 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/nn/modules/container.py:361(__init__)
        1    0.000    0.000    0.332    0.332 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/nn/modules/container.py:412(__iadd__)
        1    0.000    0.000    0.332    0.332 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/nn/modules/container.py:486(extend)
        3    0.000    0.000    0.332    0.111 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:165(<genexpr>)
        2    0.000    0.000    0.332    0.166 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:136(__init__)
       20    0.316    0.016    0.316    0.016 {built-in method torch.amax}
       27    0.001    0.000    0.312    0.012 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/library.py:129(define)
       40    0.111    0.003    0.309    0.008 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/utils.py:10(softmax)
      100    0.288    0.003    0.288    0.003 {built-in method torch.square}
       20    0.002    0.000    0.246    0.012 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/utils.py:61(data_loading)
        1    0.000    0.000    0.244    0.244 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_dynamo/exc.py:1(<module>)
        1    0.000    0.000    0.242    0.242 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_dynamo/utils.py:1(<module>)
       40    0.241    0.006    0.241    0.006 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/numpy/_core/memmap.py:359(__getitem__)
      920    0.002    0.000    0.231    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_tensor.py:40(wrapped)
      630    0.227    0.000    0.227    0.000 {method 'pow' of 'torch._C.TensorBase' objects}
        2    0.000    0.000    0.218    0.109 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:47(__init__)
        1    0.000    0.000    0.216    0.216 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/fx/experimental/symbolic_shapes.py:1(<module>)
        1    0.000    0.000    0.198    0.198 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/sympy/__init__.py:1(<module>)
       10    0.071    0.007    0.181    0.018 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/utils.py:48(gradient_clipping)
       40    0.168    0.004    0.168    0.004 {built-in method torch.reshape}
        1    0.000    0.000    0.148    0.148 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:8(__init__)
        1    0.000    0.000    0.139    0.139 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/_fully_shard/_fully_shard.py:1(<module>)
       20    0.124    0.006    0.124    0.006 {built-in method torch.gather}
       40    0.121    0.003    0.121    0.003 {method 'masked_fill' of 'torch._C.TensorBase' objects}
      801    0.002    0.000    0.116    0.000 <frozen importlib._bootstrap_external>:1090(get_code)
        2    0.000    0.000    0.114    0.057 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:101(__init__)
       16    0.114    0.007    0.114    0.007 {method 'uniform_' of 'torch._C.TensorBase' objects}
        1    0.000    0.000    0.110    0.110 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/sympy/polys/__init__.py:1(<module>)
        1    0.000    0.000    0.105    0.105 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_init.py:1(<module>)
        1    0.000    0.000    0.105    0.105 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_state.py:1(<module>)
      329    0.000    0.000    0.088    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/dataclasses.py:1294(wrap)
      329    0.006    0.000    0.087    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/dataclasses.py:929(_process_class)
2121/2070    0.015    0.000    0.085    0.000 {built-in method builtins.__build_class__}
      801    0.031    0.000    0.084    0.000 <frozen importlib._bootstrap_external>:779(_compile_bytecode)
        1    0.000    0.000    0.076    0.076 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/_flat_param.py:1(<module>)
       20    0.072    0.004    0.072    0.004 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/model.py:13(forward)
       40    0.070    0.002    0.070    0.002 {built-in method torch.max}
        1    0.000    0.000    0.069    0.069 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/_fsdp_extensions.py:1(<module>)
        1    0.000    0.000    0.069    0.069 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/fsdp/_shard_utils.py:1(<module>)
        1    0.000    0.000    0.068    0.068 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/tensor/__init__.py:1(<module>)
        1    0.000    0.000    0.067    0.067 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/tensor/_ops/__init__.py:1(<module>)
      329    0.000    0.000    0.059    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/dataclasses.py:1277(dataclass)
        1    0.000    0.000    0.059    0.059 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/tensor/_ops/_conv_ops.py:1(<module>)
       28    0.000    0.000    0.058    0.002 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/library.py:685(wrap)
      329    0.002    0.000    0.053    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/dataclasses.py:470(add_fns_to_class)
        1    0.000    0.000    0.053    0.053 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/tensor/_dtensor_spec.py:1(<module>)
      801    0.053    0.000    0.053    0.000 {built-in method marshal.loads}
        1    0.000    0.000    0.049    0.049 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/sympy/polys/polyfuncs.py:1(<module>)
        1    0.000    0.000    0.049    0.049 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/sympy/polys/specialpolys.py:1(<module>)
      9/5    0.000    0.000    0.048    0.010 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_library/custom_ops.py:338(inner)
      9/5    0.000    0.000    0.048    0.010 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_library/custom_ops.py:189(__init__)
      9/5    0.058    0.006    0.048    0.010 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_library/custom_ops.py:607(_register_to_dispatcher)
        1    0.000    0.000    0.047    0.047 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/distributed/tensor/placement_types.py:1(<module>)
        1    0.000    0.000    0.046    0.046 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/sympy/polys/rings.py:1(<module>)
        1    0.000    0.000    0.045    0.045 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/sympy/printing/__init__.py:1(<module>)
      8/4    0.000    0.000    0.044    0.011 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_library/custom_ops.py:148(inner)
       14    0.000    0.000    0.043    0.003 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/library.py:180(_register_fake)
       35    0.000    0.000    0.041    0.001 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_dynamo/decorators.py:708(wrapper)
       14    0.000    0.000    0.040    0.003 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_library/fake_impl.py:34(register)
       14    0.000    0.000    0.039    0.003 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_library/utils.py:36(get_source)
       14    0.000    0.000    0.039    0.003 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/inspect.py:1668(getframeinfo)
       13    0.000    0.000    0.039    0.003 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/_dynamo/eval_frame.py:1240(_fn)
       18    0.000    0.000    0.039    0.002 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/inspect.py:1054(findsource)
       10    0.038    0.004    0.039    0.004 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/torch/optim/optimizer.py:1027(zero_grad)
       32    0.004    0.000    0.038    0.001 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/inspect.py:1002(getmodule)
"""