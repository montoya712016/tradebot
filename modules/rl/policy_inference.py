# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from .train_rl import QNet


@dataclass
class LoadedPolicy:
    model: QNet
    device: torch.device


def load_policy(checkpoint_path: str | Path, *, device: str = "cuda") -> LoadedPolicy:
    p = Path(checkpoint_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint nao encontrado: {p}")
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    try:
        ckpt = torch.load(p, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(p, map_location=dev)
    model = QNet(int(ckpt["state_dim"]), int(ckpt["n_actions"]), int(ckpt["hidden_dim"])).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return LoadedPolicy(model=model, device=dev)


def choose_action(policy: LoadedPolicy, state: np.ndarray, *, valid_actions: list[int] | None = None) -> int:
    with torch.no_grad():
        x = torch.from_numpy(state.astype(np.float32, copy=False)).to(policy.device).unsqueeze(0)
        q = policy.model(x)
        if valid_actions:
            idx = torch.as_tensor(valid_actions, dtype=torch.long, device=policy.device)
            qv = q.index_select(1, idx)
            j = int(torch.argmax(qv, dim=1).item())
            return int(valid_actions[j])
        return int(torch.argmax(q, dim=1).item())
