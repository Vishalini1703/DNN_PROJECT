"""
Memory-augmented next-text model from the Project 3 notebook.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .memory_module import MemorySlotModule
from .model_baseline import FrameTextEncoder


class MemoryNextTextModel(nn.Module):
    """
    Same as baseline, but maintains memory slots across timesteps.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, img_dim: int = 512, hidden: int = 256, M: int = 16):
        super().__init__()
        self.text_enc = FrameTextEncoder(vocab_size, d_model=d_model)
        self.fuse = nn.Linear(d_model + img_dim, hidden)
        self.mem_mod = MemorySlotModule(M=M, D=hidden)
        self.ctx_gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.dec_gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def context_hidden(self, ctx_input_ids, ctx_attn_mask, ctx_img_emb):
        B, K, _ = ctx_input_ids.shape
        tvec = self.text_enc(ctx_input_ids, ctx_attn_mask)
        z = torch.cat([tvec, ctx_img_emb], dim=-1)
        z = torch.tanh(self.fuse(z))
        mem = self.mem_mod.init_memory(B)
        z_mem = []
        for t_step in range(K):
            q = z[:, t_step]
            readout, _ = self.mem_mod.read(mem, q)
            mem = self.mem_mod.write(mem, q)
            z_mem.append(q + readout)
        z2 = torch.stack(z_mem, dim=1)
        _, hN = self.ctx_gru(z2)
        return hN

    def forward(self, ctx_input_ids, ctx_attn_mask, ctx_img_emb, tgt_input_ids):
        h0 = self.context_hidden(ctx_input_ids, ctx_attn_mask, ctx_img_emb)
        inp = tgt_input_ids[:, :-1]
        x = self.tok_emb(inp)
        out, _ = self.dec_gru(x, h0)
        logits = self.lm_head(out)
        return logits
