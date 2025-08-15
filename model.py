import os
import sys
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from config import config


class RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None, dtype=torch.float32):
        super().__init__()
        dim = config.d_model // config.num_heads
        max_position_embeddings = config.max_seq_len
        base = config.rope_theta


        inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
        t = torch.arange(max_position_embeddings, device=device, dtype=dtype)
        freqs = torch.outer(t, inv)                               # [T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)                   # [T, D]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, dtype=None):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, config, device=None)
        super().__init__()
        self.dim = config.d_model
        self.n_heads = config.num_heads
        self.d = self.dim // n_heads
        self.causal = config.causal
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta

        self.rope = RotaryEmbedding(config, device)
        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.out = nn.Linear(dim, dim, bias = False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    
    def _to_heads(self, q, k, v):
        q, k, v = [item.view(B, T, self.h, self.d).transpose(1, 2) for item in (q, k, v)] # [B, H, T, D]
        return q, k, v


    def _apply_rope(self, x, cos, sin):
        """
        x: [B, H, T, D] with D even, using interleaved pairs (0,1), (2,3), ...
        cos/sin: [T, D/2]; will be broadcast across batch and heads.
        """
        # split even/odd channels
        x_even = x[..., ::2]  # [B,H,T,D/2]
        x_odd  = x[..., 1::2] # [B,H,T,D/2]

        # broadcast cos/sin to [1,1,T,D/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot  = x_odd  * cos + x_even * sin

        # re-interleave
        out = torch.empty_like(x)
        out[..., ::2] = x_even_rot
        out[..., 1::2] = x_odd_rot
        return out


    def forward(self, x, dtype=None):
        B, T, C = x.shape
        qkv = self.qkv(x) # [B, T, C] -> [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # [B, T, C] -> [B, H, T, D]
        q, k, v = self._to_heads(q, k, v)

        # rope the head embeds
        cos, sin = self.rope(T)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale # [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]
        if self.causal: # mask the upper triangular elements?
            mask = torch.ones(T, T, device=self.device, dtype=torch.bool).triu(1)
            att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1) # softmax across the keys dimension to get P. each query gets mapped to P * values
        att = self.attn_drop(att)
        o = torch.matmul(att, v) # [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]

        o = o.transpose(1, 2).contiguous().view(B, T, C) # back to [B, T, C]
        o = self.out(o)
        o = self.proj_drop(o)
        
        return o



class HRMBlock(nn.Module):
    """
    Attention + MLP + Residual + Norm
    - config.norm_order: 'pre' | 'post'
    - config.norm_kind : 'rms' | 'layer'
    - config.activation: 'swiglu' | 'gelu' | 'silu'
    """
    def __init__(self, config, attn: nn.Module):
        super().__init__()
        d, r = config.d_model, getattr(config, "mlp_ratio", 4.0)
        ff = getattr(config, "ffn_hidden", int(r * d))
        p = getattr(config, "dropout", 0.0)
        self.attn = attn
        self.resid_drop = nn.Dropout(p)
        self.norm_order = getattr(config, "norm_order", "pre")
        self.norm_kind = getattr(config, "norm_kind", "rms")
        # Norm params: use RMSNorm weights when requested; else LayerNorm modules
        if self.norm_kind == "rms":
            self.eps = getattr(config, "layer_norm_eps", 1e-6)
            self.n1 = nn.Parameter(torch.ones(d))
            self.n2 = nn.Parameter(torch.ones(d))
        else:
            self.n1 = nn.LayerNorm(d, eps=getattr(config, "layer_norm_eps", 1e-5))
            self.n2 = nn.LayerNorm(d, eps=getattr(config, "layer_norm_eps", 1e-5))
        # MLP: swiglu (default) or standard 2-layer
        self.act_kind = getattr(config, "activation", "swiglu")
        if self.act_kind == "swiglu":
            self.Wa = nn.Linear(d, ff, bias=getattr(config, "ffn_bias", True))
            self.Wb = nn.Linear(d, ff, bias=getattr(config, "ffn_bias", True))
            self.act = nn.SiLU()
            self.Wo = nn.Linear(ff, d, bias=getattr(config, "ffn_bias", True))
        else:
            self.fc1 = nn.Linear(d, ff, bias=getattr(config, "ffn_bias", True))
            self.fc2 = nn.Linear(ff, d, bias=getattr(config, "ffn_bias", True))
            self.act = nn.GELU() if self.act_kind == "gelu" else nn.SiLU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # --- norms
        if self.norm_kind == "rms":
            def rmsnorm(x, w):
                rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
                return w * (x / rms)
        # --- block
        if self.norm_order == "pre":
            x1 = rmsnorm(x, self.n1) if self.norm_kind == "rms" else self.n1(x)
            x = x + self.resid_drop(self.attn(x1))
            x2 = rmsnorm(x, self.n2) if self.norm_kind == "rms" else self.n2(x)
            if self.act_kind == "swiglu":
                y = self.Wo(self.drop(self.Wa(x2) * self.act(self.Wb(x2))))
            else:
                y = self.fc2(self.drop(self.act(self.fc1(x2))))
            x = x + self.resid_drop(y)
            return x
        else:  # post
            x = x + self.resid_drop(self.attn(x))
            x = (rmsnorm(x, self.n1) if self.norm_kind == "rms" else self.n1(x))
            if self.act_kind == "swiglu":
                y = self.Wo(self.drop(self.Wa(x) * self.act(self.Wb(x))))
            else:
                y = self.fc2(self.drop(self.act(self.fc1(x))))
            x = x + self.resid_drop(y)
            x = (rmsnorm(x, self.n2) if self.norm_kind == "rms" else self.n2(x))
            return x



class HRMRecurrentModule(nn.Module):
    def __init__(self, config, layers):
        super().__init__()

        self.num_puzzle_embs = config.num_puzzle_embs
        self.layers = nn.ModuleList(layers)


    def forward(self, hidden_states, additional_input=None):
        if additional_input is not None:
            hidden_states = hidden_states + additional_input
        
        for _layer in self.layers:
            hidden_states = _layers(hidden_states)
        
        return hidden_states

class HRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L_module = HRMRecurrentModule([HRMBlock(config, MultiHeadSelfAttentionWithRoPE(config)) for _i in range(config.T_L)])
        self.H_module = HRMRecurrentModule([HRMBlock(config, MultiHeadSelfAttentionWithRoPE(config)) for _i in range(config.H_L)])

    def forward(self, input_embeds):



def main():
    print('fello world.')


if __name__ == "__main__":
    main()
