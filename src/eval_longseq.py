"""
Evaluation helpers for Project 3 (loss, perplexity, entity hallucination).
"""
from __future__ import annotations

import math
import re
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


re_entity_open = re.compile(r"<gdo\\s+([^>]+)>", re.IGNORECASE)


def extract_entity_ids_from_text(text: str):
    if text is None:
        return []
    return [m.group(1).strip() for m in re_entity_open.finditer(text)]


def lm_loss(logits, tgt_input_ids, pad_token_id: int):
    targets = tgt_input_ids[:, 1:].contiguous()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_token_id)


@torch.no_grad()
def greedy_decode(model, tokenizer, ctx_input_ids, ctx_attn_mask, ctx_img_emb, max_len: int = 80):
    model.eval()
    B = ctx_input_ids.size(0)
    start_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    if start_id is None:
        start_id = tokenizer.pad_token_id
    cur = torch.full((B, 1), start_id, device=ctx_input_ids.device, dtype=torch.long)

    h0 = model.context_hidden(ctx_input_ids, ctx_attn_mask, ctx_img_emb)
    h = h0
    out_tokens = []
    for _ in range(max_len):
        x = model.tok_emb(cur[:, -1:])
        o, h = model.dec_gru(x, h)
        logits = model.lm_head(o[:, -1])
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
        out_tokens.append(nxt)
    return torch.cat(out_tokens, dim=1)


@torch.no_grad()
def entity_hallucination_rate(decoded_texts, ctx_entity_sets_batch):
    rates = []
    for txt, ctx_sets in zip(decoded_texts, ctx_entity_sets_batch):
        pred_ents = set(extract_entity_ids_from_text(txt))
        ctx_union = set().union(*ctx_sets) if len(ctx_sets) else set()
        if len(pred_ents) == 0:
            rates.append(0.0)
        else:
            halluc = [e for e in pred_ents if e not in ctx_union]
            rates.append(len(halluc) / max(len(pred_ents), 1))
    return float(sum(rates) / max(len(rates), 1))


@torch.no_grad()
def eval_epoch(model, loader, tokenizer, embedder, add_embeddings_fn, max_batches: int | None = None):
    model.eval()
    tot = 0
    sum_loss = 0.0
    pbar = tqdm(loader, leave=False)
    for bi, batch in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break
        batch = add_embeddings_fn(batch)
        ctx_ids = batch["ctx_input_ids"].to(embedder.device)
        ctx_mask = batch["ctx_attn_mask"].to(embedder.device)
        ctx_img = batch["ctx_img_emb"]
        tgt_ids = batch["tgt_input_ids"].to(embedder.device)
        logits = model(ctx_ids, ctx_mask, ctx_img, tgt_ids)
        loss = lm_loss(logits, tgt_ids, tokenizer.pad_token_id)
        bs = ctx_ids.size(0)
        tot += bs
        sum_loss += loss.item() * bs
    avg_loss = sum_loss / max(tot, 1)
    ppl = math.exp(min(avg_loss, 10.0))
    return avg_loss, ppl


@torch.no_grad()
def eval_hallucination(model, loader, tokenizer, embedder, add_embeddings_fn, batches: int = 25):
    model.eval()
    decoded_texts = []
    ctx_sets_all = []
    for bi, batch in enumerate(tqdm(loader, desc="halluc eval", leave=False)):
        if bi >= batches:
            break
        batch = add_embeddings_fn(batch)
        ctx_ids = batch["ctx_input_ids"].to(embedder.device)
        ctx_mask = batch["ctx_attn_mask"].to(embedder.device)
        ctx_img = batch["ctx_img_emb"]
        out_ids = greedy_decode(model, tokenizer, ctx_ids, ctx_mask, ctx_img, max_len=80)
        txts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        decoded_texts.extend(txts)
        ctx_sets_all.extend(batch["ctx_entity_sets"])
    return entity_hallucination_rate(decoded_texts, ctx_sets_all)
