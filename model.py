import os
import sys
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from common import trunc_normal_init_

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
        # emb = torch.cat((freqs, freqs), dim=-1)                   # [T, D]
        # rotate nahi karna hai. downstream fully vectorized _apply_rope me useful hoga but we used slicing instead.
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, seq_len: int, dtype=None):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.dim = config.d_model
        self.n_heads = config.num_heads
        self.d = self.dim // self.n_heads
        self.causal = config.causal
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta

        self.rope = RotaryEmbedding(config, device)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias = False)
        self.out = nn.Linear(self.dim, self.dim, bias = False)
        self.attn_drop = nn.Dropout(self.dropout)
        self.proj_drop = nn.Dropout(self.dropout)
    
    
    def _to_heads(self, q, k, v):
        B, T, C = q.shape
        q, k, v = [
            item.view(B, T, self.n_heads, self.d).transpose(1, 2)
            for item 
            in (q, k, v)
        ] # [B, H, T, D]
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
    
    
    def sdpa(self, q, k, v, causal=False):
        B, H, T, D = k.shape
        scale = 1.0 / math.sqrt(self.d)
        att = torch.matmul(
            q,                  # [B, H, T, D] ... 
            k.transpose(-2, -1) # ... @ [B, H, D, T] -> [B, H, T, T]
        ) * scale 
        if causal:
            mask = torch.ones(T, T, device=self.device, dtype=torch.bool).triu(1) 
            # [ [0, 1],  q1
            #   [0, 0] ] q2
            #   k1  k2
            # queries do not see future keys! 1s are filled with -inf
            att = att.masked_fill(mask, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1)) # [B, H, T, T]
        o = torch.matmul(att, v) # [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
        return o
        
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
        o = self.sdpa(q, k, v, self.causal)

        # BTC se BTHD view se gaye.
        # fir
        # BTHD se BTC view se wapas aaye.
        # LMAOOO
        
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


    def forward(self, x): # crazy forward pass, need to refactor whole class and make this forward pass more readable.
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
            hidden_states = _layer(hidden_states)
        
        return hidden_states


@dataclass
class InnerCarry:
    Z_L: torch.Tensor
    Z_H: torch.Tensor


@dataclass
class Carry:
    inner_carry: InnerCarry
    halted: torch.Tensor
    steps: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HRM_Inner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_puzzle_embs = config.num_puzzle_embs
        self.puzzle_embs = nn.Embedding(config.num_puzzle_embs, config.d_model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_tokens.weight = nn.Parameter(trunc_normal_init_(torch.empty(config.vocab_size, config.d_model), std=math.sqrt(config.d_model)))
        self.L_module = HRMRecurrentModule(config, [HRMBlock(config, MultiHeadSelfAttentionWithRoPE(config)) for _i in range(config.L_depth)])
        self.H_module = HRMRecurrentModule(config, [HRMBlock(config, MultiHeadSelfAttentionWithRoPE(config)) for _i in range(config.H_depth)])
        self.L_cycles = config.L_cycles
        self.H_cycles = config.H_cycles
        self.M_min, self.M_max = config.M_min, config.M_max
        self.register_buffer("Z_L_init", trunc_normal_init_(torch.empty(config.d_model, dtype=torch.float32), std=1))
        self.register_buffer("Z_H_init", trunc_normal_init_(torch.empty(config.d_model, dtype=torch.float32), std=1))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.q_head  = nn.Linear(config.d_model, 2, bias=True)
        for linearlayer in [self.lm_head, self.q_head]:
            linearlayer.weight = nn.Parameter(
                trunc_normal_init_(
                    tensor = torch.zeros_like(linearlayer.weight), 
                    std = 1.0 / math.sqrt(linearlayer.weight.shape[-1])
                )
            )
            if hasattr(linearlayer, "bias") and linearlayer.bias is not None:
                linearlayer.bias = nn.Parameter(torch.zeros_like(linearlayer.bias))
            
        # ek batch ke across halted ko track kiya, for multiple values of M
        # M= 1    2    3    4 (M_max)
        #    0 -> 0 -> 1 -> no forward pass (shuru me hi exit the forward())
        #    0 -> 0 -> 0 -> 1 -> no forward pass
        #    0 -> 0 -> 1 -> no forward pass
        #    0 -> 0 -> 0 -> 1 -> no forward pass
    

    def get_init_inner_carry(self, batch_size, max_seq_len) -> InnerCarry:
        return InnerCarry(
            self.Z_L_init.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_seq_len, 1).to(self.config.device), 
            self.Z_H_init.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_seq_len, 1).to(self.config.device),
        )
    
    
    def reset_inner_carry(self, reset_flag: torch.Tensor, inner_carry: InnerCarry) -> InnerCarry:
        """
        For sequences that are halted, reset the carry.
        """
        # batch has:
        # 1. inputs: [B, 81]
        # 2. labels: [B, 81]
        # 3. puzzle_identifiers: [B]
        # carry ka halted use karke torch.where se fresh batch reset karna hai.
        # remember that a carry has these things: inner_carry, halted, current_data.
        batch_size = inner_carry.Z_L.shape[0]
        
        init_inner_carry = self.get_init_inner_carry(batch_size, config.max_seq_len)
        # new_carry = Carry(
        #     inner_carry = InnerCarry(
        #         Z_L = torch.where(reset_flag.reshape(-1, 1, 1), init_inner_carry.Z_L, inner_carry.Z_L),
        #         Z_H = torch.where(reset_flag.reshape(-1, 1, 1), init_inner_carry.Z_H, inner_carry.Z_H)
        #     ),
        #     halted = torch.zeros(batch_size, dtype=torch.bool),
        ######## note that the line below will never work because it doesnt respect the shape of batch[k]
        #     current_data = {k: torch.where(reset_flag.reshape(-1, 1, 1), batch[k], inner_carry.current_data[k]) for k in batch.keys()}
        # )
        reset_flag = reset_flag.reshape(-1, 1, 1).to(torch.bool)
        new_inner_carry = InnerCarry(
            Z_L = torch.where(reset_flag, init_inner_carry.Z_L, inner_carry.Z_L),
            Z_H = torch.where(reset_flag, init_inner_carry.Z_H, inner_carry.Z_H)
        )
        
        return new_inner_carry


    def get_input_embeds(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # get the puzzle embeds from the puzzle_identifiers.
        puzzle_token_ids = batch["puzzle_identifiers"].unsqueeze(-1).to(self.config.device)
        # get the puzzle token embeds from the embed_tokens.
        puzzle_input_ids = batch["inputs"].to(self.config.device)
        # concatenate the puzzle token ids and the puzzle input ids.
        # get the emebds for the concatenated token seqence.
        input_puzzle_embeds = self.puzzle_embs(puzzle_token_ids)
        input_token_embeds = self.embed_tokens(puzzle_input_ids)
        
        input_embeds = torch.cat([input_puzzle_embeds, input_token_embeds], dim=puzzle_token_ids.ndim - 1)
        return input_embeds
    
    
    def forward(self, inner_carry: InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Input:
        - carry: InnerCarry
        Output:
        - new_inner_carry: InnerCarry
        - output_token_embeds: torch.Tensor of shape [B, T-1, D]
        - (q_halt_logits, q_continue_logits): Tuple[torch.Tensor of shape [B, D], torch.Tensor of shape [B, D]]
        """
        
        # Input encoding
        # for our case, this is just an nn.Embedding layer
        # first, get embeddings for the joint token sequence
        input_embeds = self.get_input_embeds(batch)
        
        # init Z_L and Z_H
        Z_H, Z_L = inner_carry.Z_H, inner_carry.Z_L
        
        # then, run one forward pass (apart from the last iter)
        with torch.no_grad():
            for _Hiter in range(self.H_cycles):
                for _Liter in range(self.L_cycles):
                    if not ((_Hiter + 1 == self.H_cycles) and (_Liter + 1 == self.L_cycles)):
                        Z_L = self.L_module(Z_L, Z_H + input_embeds)
                if not (_Hiter + 1 == self.H_cycles):
                    Z_H = self.H_module(Z_H, Z_L)
        
        Z_L = self.L_module(Z_L, Z_H + input_embeds)
        Z_H = self.H_module(Z_H, Z_L)
        
        # detach the gradients for the next forward pass.
        new_inner_carry = InnerCarry(
            Z_L.detach(),
            Z_H.detach()
        )
        
        # finally, run through the entire stack for inference.                
        output_token_embeds = self.lm_head(Z_H)[:, self.config.num_puzzle_embs:]
        # # q_head (for now its unused, disabled)
        # q_head. we are switching back to using it. hopefully it works. if god is willing.
        
        q_logits = self.q_head(Z_H[:, 0]).to(torch.float32)
        q_halt_logits, q_continue_logits = q_logits[..., 1], q_logits[..., 0]
        # q_halt_logits, q_continue_logits = None, None
        
        return new_inner_carry, output_token_embeds, (q_halt_logits, q_continue_logits)
        
        
class HRM_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner = HRM_Inner(config)
        # assert False, "Not Implemented yet"

    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> Carry:
        batch_size = batch["inputs"].shape[0]
        # ... for the whole batch now
        halted = torch.ones(batch_size, dtype=torch.bool).to(self.config.device)
        steps = torch.zeros(batch_size, dtype=torch.int32).to(self.config.device)
        # all halted, will be reset on the first forward pass.
        current_data = {k: v for k, v in batch.items()}
        inner_carry = self.inner.get_init_inner_carry(batch_size, config.max_seq_len)
        return Carry(inner_carry, halted, steps, current_data)
        
        
    def forward(self, batch: Dict[str, torch.Tensor], carry: Carry) -> Tuple[Carry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
        # we need to reset the carry for all halted sequences,
        # and then forward all the elements of the batch.
        
        # first, we initialize the carry by resetting the halted sequences.
        new_inner_carry = self.inner.reset_inner_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1, ) + (1, )*(batch[k].ndim - 1)), 
                batch[k], 
                v
            ) for k, v in carry.current_data.items()
        }
        
        # forward pass 
        # now with the resetted carry.
        new_inner_carry, output_token_embeds, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        
        outputs = {
            "output_token_embeds": output_token_embeds,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "g_continue": None # these will be computed in a moment. and only if the model is training.
        }
        
        # everything got a forward pass on it after resetting the halted segments.
        new_steps = new_steps + 1
        
        # check if any of the sequences are halted.
        halted = new_steps >= self.config.M_max
        is_last_step = halted
        halted = halted | (q_halt_logits > q_continue_logits)
        min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                          ) * torch.randint_like(new_steps, low=self.config.M_min, high=self.config.M_max + 1)
        halted = halted & (new_steps >= min_halt_steps)
        
        # might need to optimise  this routine to be carried over from the next forward pass.
        # but then this would change the order of the forward/backward pass interleaving.
        # this could have huge consequences or maybe minor consequences
        #     on the quality of the training. it could be positive, too.
        # would be good to read the literature on this like that one paper on DEEP SOMETHING
        # that was referenced inside HRM and then formulate a hypothesis on the best way to go about this.
        
        if self.training:
            with torch.no_grad():
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
            
            outputs["g_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))
        
        return Carry(new_inner_carry, halted, new_steps, new_current_data), outputs
        
                


def main():
    print('fello world.')


if __name__ == "__main__":
    main()
