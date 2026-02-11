# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch

from .train_rl import QNet


@dataclass
class LoadedPolicy:
    model: QNet
    device: torch.device
    state_buf: torch.Tensor | None = None
    action_index_cache: dict[tuple[int, ...], torch.Tensor] = field(default_factory=dict)


def load_policy(checkpoint_path: str | Path, *, device: str = "cuda") -> LoadedPolicy:
    p = Path(checkpoint_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint nao encontrado: {p}")
    dev_req = str(device).strip().lower()
    if dev_req == "auto" and torch.cuda.is_available():
        dev = torch.device("cuda")
    elif dev_req.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    try:
        ckpt = torch.load(p, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(p, map_location=dev)
    state_dim = int(ckpt["state_dim"])
    n_actions = int(ckpt["n_actions"])
    hidden_dim = int(ckpt["hidden_dim"])
    # Em modo auto, manter CUDA quando disponivel para evitar trocas inesperadas de device
    # entre treino/backtest e facilitar consistencia operacional.
    model = QNet(state_dim, n_actions, hidden_dim).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return LoadedPolicy(model=model, device=dev)


def choose_action(policy: LoadedPolicy, state: np.ndarray, *, valid_actions: list[int] | None = None) -> int:
    s_np = state.astype(np.float32, copy=False)
    d = int(s_np.shape[0])
    if policy.state_buf is None or int(policy.state_buf.shape[1]) != d:
        policy.state_buf = torch.empty((1, d), dtype=torch.float32, device=policy.device)
    with torch.inference_mode():
        policy.state_buf[0].copy_(torch.from_numpy(s_np), non_blocking=True)
        q = policy.model(policy.state_buf)
        if valid_actions:
            key = tuple(int(a) for a in valid_actions)
            idx = policy.action_index_cache.get(key)
            if idx is None:
                idx = torch.as_tensor(key, dtype=torch.long, device=policy.device)
                policy.action_index_cache[key] = idx
            qv = q.index_select(1, idx)
            j = int(torch.argmax(qv, dim=1).item())
            return int(valid_actions[j])
        return int(torch.argmax(q, dim=1).item())
