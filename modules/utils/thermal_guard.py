# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import time
from typing import Callable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def _is_finite(v: float | None) -> bool:
    if v is None:
        return False
    try:
        if np is not None:
            return bool(np.isfinite(v))
        return not (v != v or v in (float("inf"), float("-inf")))
    except Exception:
        return False


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


@dataclass
class ThermalSample:
    cpu_c: float | None
    gpu_c: float | None
    checked_at: float

    @property
    def observed(self) -> tuple[float, ...]:
        vals: list[float] = []
        if _is_finite(self.cpu_c):
            vals.append(float(self.cpu_c))
        if _is_finite(self.gpu_c):
            vals.append(float(self.gpu_c))
        return tuple(vals)

    @property
    def has_any(self) -> bool:
        return len(self.observed) > 0

    @property
    def peak_c(self) -> float | None:
        if not self.has_any:
            return None
        return float(max(self.observed))


@dataclass
class ThermalGuardConfig:
    enabled: bool = True
    max_temp_c: float = 80.0
    resume_temp_c: float = 70.0
    check_every_s: float = 10.0
    cooldown_s: float = 15.0
    log_prefix: str = "[thermal-guard]"


class ThermalGuard:
    def __init__(self, cfg: ThermalGuardConfig):
        self.cfg = cfg
        self._last_check_ts: float = 0.0
        self._last_sample = ThermalSample(cpu_c=None, gpu_c=None, checked_at=0.0)
        self._warned_no_sensors = False

    @classmethod
    def from_env(cls) -> "ThermalGuard":
        enabled = _env_bool("SNIPER_THERMAL_GUARD", True)
        max_temp = _env_float("SNIPER_THERMAL_MAX_TEMP_C", 80.0)
        resume_temp = _env_float("SNIPER_THERMAL_RESUME_BELOW_C", 70.0)
        if resume_temp >= max_temp:
            resume_temp = max_temp - 5.0
        cfg = ThermalGuardConfig(
            enabled=enabled,
            max_temp_c=max_temp,
            resume_temp_c=resume_temp,
            check_every_s=max(1.0, _env_float("SNIPER_THERMAL_CHECK_EVERY_S", 10.0)),
            cooldown_s=max(1.0, _env_float("SNIPER_THERMAL_COOLDOWN_S", 15.0)),
        )
        return cls(cfg)

    def _read_cpu_temp_c_psutil(self) -> float | None:
        if psutil is None or not hasattr(psutil, "sensors_temperatures"):
            return None
        try:
            temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
        except Exception:
            return None
        values: list[float] = []
        for entries in (temps or {}).values():
            for ent in entries or ():
                cur = getattr(ent, "current", None)
                if cur is None:
                    continue
                try:
                    fv = float(cur)
                except Exception:
                    continue
                if _is_finite(fv):
                    values.append(fv)
        if not values:
            return None
        return float(max(values))

    def _read_cpu_temp_c_windows(self) -> float | None:
        if os.name != "nt":
            return None
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature | "
            "Select-Object -ExpandProperty CurrentTemperature",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, timeout=3.0)
        except Exception:
            return None
        vals: list[float] = []
        for line in out.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                c = (float(s) / 10.0) - 273.15
            except Exception:
                continue
            if _is_finite(c):
                vals.append(c)
        if not vals:
            return None
        return float(max(vals))

    def _read_gpu_temp_c_nvidia_smi(self) -> float | None:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        cmd = [exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        try:
            out = subprocess.check_output(cmd, text=True, timeout=3.0)
        except Exception:
            return None
        vals: list[float] = []
        for line in out.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                v = float(s)
            except Exception:
                continue
            if _is_finite(v):
                vals.append(v)
        if not vals:
            return None
        return float(max(vals))

    def sample(self, *, force: bool = False) -> ThermalSample:
        now = time.time()
        if (not force) and ((now - self._last_check_ts) < float(self.cfg.check_every_s)):
            return self._last_sample
        cpu_c = self._read_cpu_temp_c_psutil()
        if cpu_c is None:
            cpu_c = self._read_cpu_temp_c_windows()
        gpu_c = self._read_gpu_temp_c_nvidia_smi()
        self._last_check_ts = now
        self._last_sample = ThermalSample(cpu_c=cpu_c, gpu_c=gpu_c, checked_at=now)
        return self._last_sample

    def should_wait(self, *, force_sample: bool = False) -> tuple[bool, ThermalSample]:
        sample = self.sample(force=force_sample)
        peak = sample.peak_c
        if peak is None:
            return False, sample
        return bool(peak >= float(self.cfg.max_temp_c)), sample

    def wait_until_safe(
        self,
        *,
        where: str,
        force_sample: bool = False,
        logger: Callable[[str], None] | None = None,
    ) -> ThermalSample:
        if not bool(self.cfg.enabled):
            return self.sample(force=force_sample)
        wait_now, sample = self.should_wait(force_sample=force_sample)
        if not sample.has_any:
            if (not self._warned_no_sensors) and logger is not None:
                logger(f"{self.cfg.log_prefix} ativo sem sensores CPU/GPU; seguindo sem throttle")
                self._warned_no_sensors = True
            return sample
        if not wait_now:
            return sample

        if logger is not None:
            logger(
                f"{self.cfg.log_prefix} quente em {where}: cpu={sample.cpu_c}C gpu={sample.gpu_c}C "
                f"(limite={self.cfg.max_temp_c:.1f}C). aguardando..."
            )
        while True:
            time.sleep(float(self.cfg.cooldown_s))
            sample = self.sample(force=True)
            peak = sample.peak_c
            if peak is None:
                if logger is not None:
                    logger(f"{self.cfg.log_prefix} sensores indisponiveis; retomando")
                return sample
            if peak < float(self.cfg.resume_temp_c):
                if logger is not None:
                    logger(
                        f"{self.cfg.log_prefix} liberado em {where}: cpu={sample.cpu_c}C gpu={sample.gpu_c}C "
                        f"(alvo<{self.cfg.resume_temp_c:.1f}C)"
                    )
                return sample
            if logger is not None:
                logger(
                    f"{self.cfg.log_prefix} ainda quente: cpu={sample.cpu_c}C gpu={sample.gpu_c}C "
                    f"(alvo<{self.cfg.resume_temp_c:.1f}C)"
                )

