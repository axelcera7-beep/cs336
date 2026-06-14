import torch
import torch.nn as nn
import math

from cs336_basics.utils import softmax, scaled_dot_product_attention

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None =None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float= 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x shape: (batch_size, sequence_length, d_model)
        squared_elements = torch.sum(torch.square(x), dim=-1, keepdim=True) / self.d_model + self.eps
        mean_squared_elements = torch.sqrt(squared_elements)
        
        x_normalized = x / mean_squared_elements * self.weight
        x_normalized = x_normalized.to(in_dtype)
        return x_normalized

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if not d_ff:
            d_ff = round(8 / 3 * d_model /64) * 64
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.silu(self.w1(x))
        x3 = self.w3(x)
        x_intermediate = x1 * x3
        x2 = self.w2(x_intermediate)
        return x2
    
    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        if not theta:
            theta = 10**5
        exponenent = (2*torch.arange(1, d_k // 2 + 1, device=device)-2) / d_k
        freq = (theta ** exponenent).unsqueeze(0)
        inv_freq = 1.0 / freq
        #[max_seq_len, 1] * [1, d/2]
        theta_i_k = torch.arange(max_seq_len, device=device).unsqueeze(1) * inv_freq
        cos_i_k = torch.cos(theta_i_k)
        sin_i_k = torch.sin(theta_i_k)
        self.register_buffer("cos_i_k", cos_i_k)
        self.register_buffer("sin_i_k", sin_i_k)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is not None:
            cos_i_k = self.cos_i_k[token_positions]
            sin_i_k = self.sin_i_k[token_positions]
        else:
            cos_i_k = self.cos_i_k[:seq_len]
            sin_i_k = self.sin_i_k[:seq_len]

        # x.shape = (batch_size, seq_len, d_k)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated_even = x_even * cos_i_k - x_odd * sin_i_k
        x_rotated_odd = x_even * sin_i_k + x_odd * cos_i_k
        stacked = torch.stack([x_rotated_even, x_rotated_odd], -1)
        x_rotated = stacked.flatten(-2)
        return x_rotated
    
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, context_length: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.context_length = context_length
        self.theta = theta
        self.device = device
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        if self.theta:
            rope = RoPE(theta=self.theta, d_k=self.head_dim, max_seq_len=self.context_length, device=self.device)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        mask = torch.triu(torch.ones(x.shape[-2],x.shape[-2], dtype=torch.bool, device=self.device), diagonal=1)
        attention_head = scaled_dot_product_attention(q=Q, k=K, v=V, mask=~mask)
        # x.shape = [batch_size,  num_heads, seq_len, head_dim]
        attention_head = attention_head.transpose(-2, -3) # [batch_size, seq_len, num_heads, head_dim]
        multi_head = torch.reshape(attention_head, (*attention_head.shape[:-2], self.d_model))
        output = self.output_proj(multi_head)
        return output
    
    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
        batch_size, seq_len, d_model = t.shape
        return t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads: int, d_ff: int, context_length: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadAttention(d_model=d_model, num_heads=num_heads, context_length=context_length, theta=theta, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
    def forward(self, x):
        residual = x
        norm_x = self.ln1(x)
        att_x = self.attn(norm_x)
        x1 = att_x + residual
        residual = x1
        norm_x1 = self.ln2(x1)
        ffn_x1 = self.ffn(norm_x1)

        return residual + ffn_x1
    
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, 
                 context_length: int, 
                 num_layers: int, 
                 d_model: int, 
                 num_heads: int,
                 d_ff: int,
                 theta: float | None = None, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(TransformerBlock(d_model=d_model,
                                                     num_heads=num_heads,
                                                     d_ff=d_ff,
                                                     theta=theta,
                                                     context_length=context_length,
                                                     device=device,
                                                     dtype=dtype) for i in range(num_layers))
        self.ln_final =  RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
    
    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)

        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x