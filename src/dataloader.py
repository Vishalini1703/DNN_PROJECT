"""
Dataset and collation utilities adapted from the Project 3 notebook.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

re_frame = re.compile(r"<gdi\\s+image\\d+>\\s*(.*?)(?=(?:<gdi\\s+image\\d+>)|$)", re.DOTALL | re.IGNORECASE)
re_entity_open = re.compile(r"<gdo\\s+([^>]+)>", re.IGNORECASE)


def extract_frame_texts(story_text: str) -> List[str]:
    if story_text is None:
        return []
    chunks = [m.group(1).strip() for m in re_frame.finditer(story_text)]
    return [c for c in chunks if len(c) > 0]


def extract_entity_ids(text: str) -> List[str]:
    if text is None:
        return []
    return [m.group(1).strip() for m in re_entity_open.finditer(text)]


def build_index_table(
    split,
    story_row_indices: Sequence[int],
    K: int = 6,
    require_markers: bool = False,
) -> Tuple[List[Tuple[int, int, int]], int]:
    idx_rows: List[Tuple[int, int, int]] = []
    kept = 0
    for row_idx in story_row_indices:
        ex = split[int(row_idx)]
        fc = int(ex["frame_count"])
        story = ex.get("story", "")
        if require_markers:
            if story is None or "<gdi image" not in story.lower():
                continue
        if fc < K + 1:
            continue
        for t in range(K, fc):
            idx_rows.append((int(row_idx), int(t), int(fc)))
        kept += 1
    return idx_rows, kept


@dataclass
class StoryTextSample:
    story_id: str
    row_idx: int
    t_index: int
    frame_count: int
    ctx_images: List[Any]
    target_image: Any
    ctx_texts: List[str]
    target_text: str
    ctx_entity_sets: List[set]
    target_entity_set: set


class StoryTextNextDataset(Dataset):
    def __init__(self, base_split, index_rows: Sequence[Tuple[int, int, int]], K: int = 6):
        self.base = base_split
        self.index = list(index_rows)
        self.K = K

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row_idx, t, _ = self.index[i]
        ex = self.base[int(row_idx)]
        imgs = ex["images"]
        story = ex.get("story", "")
        frame_texts = extract_frame_texts(story)

        if len(frame_texts) == 0:
            frame_texts = [story] * int(ex["frame_count"])

        maxT = min(len(imgs), len(frame_texts), int(ex["frame_count"]))
        if maxT <= self.K:
            raise IndexError("Not enough frames after alignment.")
        t = min(int(t), maxT - 1)

        ctx_imgs = imgs[t - self.K : t]
        tgt_img = imgs[t]
        ctx_txts = frame_texts[t - self.K : t]
        tgt_txt = frame_texts[t]

        ctx_ents = [set(extract_entity_ids(s)) for s in ctx_txts]
        tgt_ents = set(extract_entity_ids(tgt_txt))

        return {
            "story_id": ex["story_id"],
            "row_idx": int(row_idx),
            "t_index": int(t),
            "frame_count": int(ex["frame_count"]),
            "ctx_images": ctx_imgs,
            "target_image": tgt_img,
            "ctx_texts": ctx_txts,
            "target_text": tgt_txt,
            "ctx_entity_sets": ctx_ents,
            "target_entity_set": tgt_ents,
        }


def collate_p3(
    batch: List[Dict[str, Any]],
    tokenizer,
    K: int,
    ctx_max_len: int,
    tgt_max_len: int,
) -> Dict[str, Any]:
    ctx_texts = [b["ctx_texts"] for b in batch]
    tgt_texts = [b["target_text"] for b in batch]
    flat_ctx = [ctx_texts[i][j] for i in range(len(batch)) for j in range(K)]

    ctx_enc = tokenizer(
        flat_ctx,
        padding=True,
        truncation=True,
        max_length=ctx_max_len,
        return_tensors="pt",
    )
    B = len(batch)
    Lc = ctx_enc["input_ids"].shape[1]
    ctx_input_ids = ctx_enc["input_ids"].view(B, K, Lc)
    ctx_attn_mask = ctx_enc["attention_mask"].view(B, K, Lc)

    tgt_enc = tokenizer(
        tgt_texts,
        padding=True,
        truncation=True,
        max_length=tgt_max_len,
        return_tensors="pt",
    )
    tgt_input_ids = tgt_enc["input_ids"]
    tgt_attn_mask = tgt_enc["attention_mask"]

    ctx_images = [b["ctx_images"] for b in batch]
    target_images = [b["target_image"] for b in batch]

    return {
        "story_id": [b["story_id"] for b in batch],
        "row_idx": torch.tensor([b["row_idx"] for b in batch], dtype=torch.long),
        "t_index": torch.tensor([b["t_index"] for b in batch], dtype=torch.long),
        "frame_count": torch.tensor([b["frame_count"] for b in batch], dtype=torch.long),
        "ctx_input_ids": ctx_input_ids,
        "ctx_attn_mask": ctx_attn_mask,
        "tgt_input_ids": tgt_input_ids,
        "tgt_attn_mask": tgt_attn_mask,
        "ctx_images": ctx_images,
        "target_images": target_images,
        "ctx_entity_sets": [b["ctx_entity_sets"] for b in batch],
        "target_entity_set": [b["target_entity_set"] for b in batch],
    }
