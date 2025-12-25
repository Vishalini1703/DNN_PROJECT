"""
ResNet18 image embedding utilities (frozen backbone).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torchvision import models, transforms


class ResNet18Embedder:
    def __init__(self, device: torch.device, cache_dir: Path | None = None):
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(cache_dir)
            torch.hub.set_dir(str(cache_dir))

        self.device = device
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def embed_pil_list(self, pil_list: List) -> torch.Tensor:
        batch = torch.stack([self.transform(im.convert("RGB")) for im in pil_list], dim=0).to(self.device)
        return self.model(batch)
