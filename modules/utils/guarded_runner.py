from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Iterable, Iterator, TypeVar

from .adaptive_parallel import AdaptiveParallelPolicy, run_adaptive_thread_map
from .thermal_guard import ThermalGuard


T = TypeVar("T")
R = TypeVar("R")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class GuardedParallelDefaults:
    max_ram_pct: float = 78.0
    min_free_mb: float = 3072.0
    per_worker_mem_mb: float = 768.0
    critical_ram_pct: float = 92.0
    critical_min_free_mb: float = 1024.0
    abort_on_critical_ram: bool = True
    min_workers: int = 1
    poll_interval_s: float = 0.3
    log_every_s: float = 20.0
    throttle_sleep_s: float = 3.0


class GuardedRunner:
    """
    Wrapper central para rotinas longas:
    - thermal guard padronizado
    - policy de RAM/concurrency adaptativa padronizada
    - classificador de erros de guard
    """

    def __init__(
        self,
        *,
        log_prefix: str,
        thermal_guard: ThermalGuard | None = None,
        env_prefix: str | None = None,
        defaults: GuardedParallelDefaults | None = None,
    ) -> None:
        self.log_prefix = str(log_prefix)
        self.thermal_guard = thermal_guard or ThermalGuard.from_env()
        self.env_prefix = (str(env_prefix or "").strip().upper() or None)
        self.defaults = defaults or GuardedParallelDefaults()

    def log(self, msg: str) -> None:
        print(f"{self.log_prefix} {msg}", flush=True)

    def thermal_wait(self, where: str) -> None:
        self.thermal_guard.wait_until_safe(
            where=where,
            force_sample=False,
            logger=lambda m: self.log(m),
        )

    @staticmethod
    def is_guard_error(exc: BaseException | str | None) -> bool:
        s = str(exc or "").lower()
        return (
            ("ram guard" in s)
            or ("ram critica" in s)
            or ("[thermal-guard]" in s)
            or ("critical" in s and "thermal" in s)
            or ("timeout quente" in s)
        )

    def _ek(self, suffix: str) -> str | None:
        if not self.env_prefix:
            return None
        return f"{self.env_prefix}_{suffix}"

    def _getf(self, suffix: str, default: float) -> float:
        key = self._ek(suffix)
        return _env_float(key, default) if key else float(default)

    def _getb(self, suffix: str, default: bool) -> bool:
        key = self._ek(suffix)
        return _env_bool(key, default) if key else bool(default)

    def make_policy(self, **overrides: float | int | bool) -> AdaptiveParallelPolicy:
        d = self.defaults
        policy = AdaptiveParallelPolicy(
            max_ram_pct=self._getf("RAM_PCT", d.max_ram_pct),
            min_free_mb=self._getf("MIN_FREE_MB", d.min_free_mb),
            per_worker_mem_mb=self._getf("PER_WORKER_MB", d.per_worker_mem_mb),
            critical_ram_pct=self._getf("CRITICAL_RAM_PCT", d.critical_ram_pct),
            critical_min_free_mb=self._getf("CRITICAL_MIN_FREE_MB", d.critical_min_free_mb),
            abort_on_critical_ram=self._getb("ABORT_ON_CRITICAL_RAM", d.abort_on_critical_ram),
            min_workers=int(d.min_workers),
            poll_interval_s=float(d.poll_interval_s),
            log_every_s=float(d.log_every_s),
            throttle_sleep_s=self._getf("THROTTLE_SLEEP_S", d.throttle_sleep_s),
        )
        for k, v in overrides.items():
            if hasattr(policy, k):
                setattr(policy, k, v)  # dataclass mutable
        return policy

    def adaptive_map(
        self,
        items: Iterable[T],
        worker_fn: Callable[[T], R],
        *,
        max_workers: int,
        policy: AdaptiveParallelPolicy | None = None,
        task_name: str,
    ) -> Iterator[tuple[T, object]]:
        pol = policy or self.make_policy()
        return run_adaptive_thread_map(
            items,
            worker_fn,
            max_workers=int(max_workers),
            policy=pol,
            task_name=str(task_name),
        )

