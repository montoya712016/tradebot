# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import os
import time
from typing import Callable, Deque, Generic, Iterable, Iterator, TypeVar

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class AdaptiveParallelPolicy:
    max_ram_pct: float = 85.0
    min_free_mb: float = 1024.0
    per_worker_mem_mb: float = 512.0
    min_workers: int = 1
    poll_interval_s: float = 0.5
    log_every_s: float = 10.0


def _system_mem() -> tuple[float, float]:
    if psutil is None:
        return 0.0, float("inf")
    vm = psutil.virtual_memory()
    used_pct = float(vm.percent)
    avail_mb = float(vm.available) / (1024.0 * 1024.0)
    return used_pct, avail_mb


def _recommended_workers(
    current: int,
    max_workers: int,
    policy: AdaptiveParallelPolicy,
) -> int:
    if max_workers <= 1:
        return 1
    if psutil is None:
        return max(1, min(max_workers, current if current > 0 else max_workers))

    used_pct, avail_mb = _system_mem()
    hard_block = (used_pct >= float(policy.max_ram_pct)) or (avail_mb <= float(policy.min_free_mb))
    if hard_block:
        return max(0, int(policy.min_workers))

    per_w = max(1.0, float(policy.per_worker_mem_mb))
    extra_mb = max(0.0, avail_mb - float(policy.min_free_mb))
    budget = int(extra_mb // per_w)
    desired = max(int(policy.min_workers), budget)
    desired = min(int(max_workers), desired)
    # evita oscilar demais; aumenta/reduz no máximo 1 por ciclo
    if desired > current:
        return min(current + 1, desired)
    if desired < current:
        return max(current - 1, desired)
    return desired


def run_adaptive_thread_map(
    items: Iterable[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int,
    policy: AdaptiveParallelPolicy,
    task_name: str = "adaptive",
) -> Iterator[tuple[T, Future[R]]]:
    """
    Executa map em threads com concorrência adaptativa baseada em RAM.
    - Nunca aborta por RAM alta; apenas reduz submissão e espera.
    - Retorna resultados no formato (item, future) conforme conclusão.
    """
    q: Deque[T] = deque(items)
    if not q:
        return

    max_workers = max(1, int(max_workers))
    min_workers = max(0, int(policy.min_workers))
    if min_workers > max_workers:
        min_workers = max_workers

    in_flight: dict[Future[R], T] = {}
    running_cap = min(max_workers, max(1, min_workers))
    last_log = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while q or in_flight:
            running_cap = _recommended_workers(len(in_flight), max_workers, policy)
            while q and len(in_flight) < running_cap:
                item = q.popleft()
                fut = ex.submit(worker_fn, item)
                in_flight[fut] = item

            if not in_flight:
                now = time.perf_counter()
                if now - last_log >= float(policy.log_every_s):
                    used_pct, avail_mb = _system_mem()
                    print(
                        f"[adaptive:{task_name}] throttle: ram={used_pct:.1f}% avail={avail_mb:.0f}MB cap={running_cap}",
                        flush=True,
                    )
                    last_log = now
                time.sleep(max(0.1, float(policy.poll_interval_s)))
                continue

            done, _ = wait(
                list(in_flight.keys()),
                timeout=max(0.05, float(policy.poll_interval_s)),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue

            for fut in done:
                item = in_flight.pop(fut)
                yield item, fut

