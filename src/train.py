"""
Training loop for Project 3 (baseline vs memory model).
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataloader import StoryTextNextDataset, build_index_table, collate_p3
from .encoders_image import ResNet18Embedder
from .encoders_text import load_tokenizer
from .model_baseline import BaselineNextTextModel
from .model_memory import MemoryNextTextModel


def build_splits(train_raw, test_raw, val_frac: float, seed: int, K: int):
    unique_ids = np.unique(np.array(train_raw["story_id"]))
    train_ids, val_ids = train_test_split(unique_ids, test_size=val_frac, random_state=seed, shuffle=True)
    storyid_to_idx = {sid: i for i, sid in enumerate(train_raw["story_id"])}
    train_story_indices = [storyid_to_idx[sid] for sid in train_ids]
    val_story_indices = [storyid_to_idx[sid] for sid in val_ids]

    train_index, _ = build_index_table(train_raw, train_story_indices, K=K, require_markers=False)
    val_index, _ = build_index_table(train_raw, val_story_indices, K=K, require_markers=False)
    test_story_indices = list(range(len(test_raw)))
    test_index, _ = build_index_table(test_raw, test_story_indices, K=K, require_markers=False)
    return train_index, val_index, test_index


def add_image_embeddings_to_batch(batch, embedder: ResNet18Embedder):
    B = len(batch["ctx_images"])
    K = len(batch["ctx_images"][0])
    flat_ctx = []
    for i in range(B):
        flat_ctx.extend(batch["ctx_images"][i])
    ctx_e = embedder.embed_pil_list(flat_ctx)
    ctx_e = ctx_e.view(B, K, -1)
    tgt_e = embedder.embed_pil_list(batch["target_images"])
    batch["ctx_img_emb"] = ctx_e
    batch["tgt_img_emb"] = tgt_e
    return batch


def lm_loss(logits, tgt_input_ids, pad_token_id: int):
    targets = tgt_input_ids[:, 1:].contiguous()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_token_id)


def extract_entity_ids_from_text(text: str):
    import re

    re_entity_open = re.compile(r"<gdo\\s+([^>]+)>", re.IGNORECASE)
    if text is None:
        return []
    return [m.group(1).strip() for m in re_entity_open.finditer(text)]


@torch.no_grad()
def greedy_decode(model, tokenizer, ctx_input_ids, ctx_attn_mask, ctx_img_emb, max_len: int = 80):
    model.eval()
    B = ctx_input_ids.size(0)
    start_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    if start_id is None:
        start_id = tokenizer.pad_token_id
    cur = torch.full((B, 1), start_id, device=ctx_input_ids.device, dtype=torch.long)

    h0 = model.context_hidden(ctx_input_ids, ctx_attn_mask, ctx_img_emb)
    dec_gru = model.dec_gru
    tok_emb = model.tok_emb
    lm_head = model.lm_head

    h = h0
    out_tokens: List[torch.Tensor] = []
    for _ in range(max_len):
        x = tok_emb(cur[:, -1:])
        o, h = dec_gru(x, h)
        logits = lm_head(o[:, -1])
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
        out_tokens.append(nxt)

    out_ids = torch.cat(out_tokens, dim=1)
    return out_ids


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


def run_epoch(model, loader, tokenizer, embedder, optimizer, train: bool, max_batches: int | None):
    model.train(train)
    tot = 0
    sum_loss = 0.0
    pbar = tqdm(loader, leave=False)
    for bi, batch in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break
        batch = add_image_embeddings_to_batch(batch, embedder)
        ctx_ids = batch["ctx_input_ids"].to(embedder.device)
        ctx_mask = batch["ctx_attn_mask"].to(embedder.device)
        ctx_img = batch["ctx_img_emb"]
        tgt_ids = batch["tgt_input_ids"].to(embedder.device)

        logits = model(ctx_ids, ctx_mask, ctx_img, tgt_ids)
        loss = lm_loss(logits, tgt_ids, tokenizer.pad_token_id)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = ctx_ids.size(0)
        tot += bs
        sum_loss += loss.item() * bs
        pbar.set_postfix({"loss": sum_loss / tot, "ppl": math.exp(min(sum_loss / tot, 10.0))})

    avg_loss = sum_loss / max(tot, 1)
    ppl = math.exp(min(avg_loss, 10.0))
    return avg_loss, ppl


@torch.no_grad()
def eval_hallucination(model, loader, tokenizer, embedder, batches: int = 25):
    model.eval()
    decoded_texts = []
    ctx_sets_all = []
    for bi, batch in enumerate(tqdm(loader, desc="halluc eval", leave=False)):
        if bi >= batches:
            break
        batch = add_image_embeddings_to_batch(batch, embedder)
        ctx_ids = batch["ctx_input_ids"].to(embedder.device)
        ctx_mask = batch["ctx_attn_mask"].to(embedder.device)
        ctx_img = batch["ctx_img_emb"]
        out_ids = greedy_decode(model, tokenizer, ctx_ids, ctx_mask, ctx_img, max_len=80)
        txts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        decoded_texts.extend(txts)
        ctx_sets_all.extend(batch["ctx_entity_sets"])
    return entity_hallucination_rate(decoded_texts, ctx_sets_all)


def main():
    parser = argparse.ArgumentParser(description="Train baseline or memory model for Project 3.")
    parser.add_argument("--cache_dir", type=str, default=r"E:\\_cache\\storyreasoning\\datasets")
    parser.add_argument("--tokenizer_cache_dir", type=str, default=r"E:\\_cache\\storyreasoning\\transformers")
    parser.add_argument("--tokenizer_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--torch_cache_dir", type=str, default=r"E:\\_cache\\storyreasoning\\torch")
    parser.add_argument("--k_steps", type=int, default=6)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--ctx_max_len", type=int, default=128)
    parser.add_argument("--tgt_max_len", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--model", choices=["baseline", "memory"], default="memory")
    parser.add_argument("--mem_slots", type=int, default=16)
    parser.add_argument("--val_cap", type=int, default=120)
    parser.add_argument("--halluc_batches", type=int, default=25)
    parser.add_argument("--save_path", type=str, default=r"E:\\_cache\\storyreasoning\\checkpoints\\p3_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("daniel3303/StoryReasoning", cache_dir=args.cache_dir)
    train_raw = ds["train"]
    test_raw = ds["test"]

    train_index, val_index, _ = build_splits(train_raw, test_raw, args.val_frac, 42, args.k_steps)
    train_ds = StoryTextNextDataset(train_raw, train_index, K=args.k_steps)
    val_ds = StoryTextNextDataset(train_raw, val_index, K=args.k_steps)

    tok = load_tokenizer(args.tokenizer_name, Path(args.tokenizer_cache_dir))
    collate = lambda b: collate_p3(b, tok, args.k_steps, args.ctx_max_len, args.tgt_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    embedder = ResNet18Embedder(device=device, cache_dir=Path(args.torch_cache_dir))

    if args.model == "baseline":
        model = BaselineNextTextModel(tok.vocab_size, d_model=256, img_dim=512, hidden=256).to(device)
    else:
        model = MemoryNextTextModel(tok.vocab_size, d_model=256, img_dim=512, hidden=256, M=args.mem_slots).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, tok, embedder, optimizer, train=True, max_batches=None)
        va_loss, va_ppl = run_epoch(model, val_loader, tok, embedder, optimizer, train=False, max_batches=args.val_cap)
        hall = eval_hallucination(model, val_loader, tok, embedder, batches=args.halluc_batches)
        print(f"\\nEpoch {ep} ({args.model})")
        print(f"  train loss={tr_loss:.4f} ppl={tr_ppl:.2f}")
        print(f"  val   loss={va_loss:.4f} ppl={va_ppl:.2f}")
        print(f"  val entity-hallucination-rate={hall:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "val_loss": va_loss,
                    "val_ppl": va_ppl,
                    "halluc_rate": hall,
                    "K": args.k_steps,
                    "CTX_MAX_LEN": args.ctx_max_len,
                    "TGT_MAX_LEN": args.tgt_max_len,
                    "model": args.model,
                },
                args.save_path,
            )
            print("  saved", args.save_path)


if __name__ == "__main__":
    main()
