"""
Baseline next-text model from the Project 3 notebook.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrameTextEncoder(nn.Module):
    """
    Encodes context text tokens per frame into a single vector
    using token embedding + masked mean pooling.
    """

    def __init__(self, vocab_size: int, d_model: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        x = self.ln(x)
        m = attn_mask.unsqueeze(-1).float()
        x = x * m
        denom = m.sum(dim=2).clamp_min(1.0)
        pooled = x.sum(dim=2) / denom
        return pooled


class BaselineNextTextModel(nn.Module):
    """
    Fusion per timestep: [text_frame_vec || img_emb] -> GRU -> context summary -> LM head.
    Decoder uses teacher forcing with target tokens.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, img_dim: int = 512, hidden: int = 256):
        super().__init__()
        self.text_enc = FrameTextEncoder(vocab_size, d_model=d_model)
        self.fuse = nn.Linear(d_model + img_dim, hidden)
        self.ctx_gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.dec_gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def context_hidden(self, ctx_input_ids, ctx_attn_mask, ctx_img_emb):
        tvec = self.text_enc(ctx_input_ids, ctx_attn_mask)
        z = torch.cat([tvec, ctx_img_emb], dim=-1)
        z = torch.tanh(self.fuse(z))
        _, hN = self.ctx_gru(z)
        return hN

    def forward(self, ctx_input_ids, ctx_attn_mask, ctx_img_emb, tgt_input_ids):
        h0 = self.context_hidden(ctx_input_ids, ctx_attn_mask, ctx_img_emb)
        inp = tgt_input_ids[:, :-1]
        x = self.tok_emb(inp)
        out, _ = self.dec_gru(x, h0)
        logits = self.lm_head(out)
        return logits
