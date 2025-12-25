"""
Memory slots module from the Project 3 notebook.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class MemorySlotModule(nn.Module):
    def __init__(self, M: int = 16, D: int = 256):
        super().__init__()
        self.M = M
        self.D = D
        self.mem = nn.Parameter(torch.randn(M, D) * 0.02)
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.write_proj = nn.Linear(D, D)
        self.gate_proj = nn.Linear(D, D)

    def init_memory(self, B: int) -> torch.Tensor:
        return self.mem.unsqueeze(0).expand(B, self.M, self.D).contiguous()

    def read(self, mem: torch.Tensor, q: torch.Tensor):
        Q = self.q_proj(q).unsqueeze(1)
        K = self.k_proj(mem)
        att = torch.softmax((Q * K).sum(-1) / math.sqrt(self.D), dim=-1)
        V = self.v_proj(mem)
        readout = (att.unsqueeze(-1) * V).sum(dim=1)
        return readout, att

    def write(self, mem: torch.Tensor, u: torch.Tensor):
        u_proj = self.write_proj(u).unsqueeze(1)
        gate = torch.sigmoid(self.gate_proj(mem) + u_proj)
        mem_new = mem * (1 - gate) + u_proj * gate
        return mem_new
