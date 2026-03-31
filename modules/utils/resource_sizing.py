from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
from typing import Any

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


@dataclass(frozen=True)
class HostResources:
    cpu_count: int
    total_mb: float
    available_mb: float
    used_pct: float


@dataclass(frozen=True)
class WorkloadProfile:
    min_workers: int
    max_workers: int
    cpu_fraction: float
    reserve_cores: int
    per_worker_mb: float
    reserve_mb: float


@dataclass(frozen=True)
class WorkerDecision:
    kind: str
    workers: int
    cpu_cap: int
    memory_cap: int
    requested: int
    host: HostResources
    profile_per_worker_mb: float
    observed_per_worker_mb: float

    @property
    def summary(self) -> str:
        return (
            f"{self.kind}: workers={self.workers} "
            f"(cpu_cap={self.cpu_cap}, mem_cap={self.memory_cap}, "
            f"avail={self.host.available_mb:.0f}MB, cpu={self.host.cpu_count}, "
            f"per_worker_mb={self.profile_per_worker_mb:.0f}, "
            f"observed={self.observed_per_worker_mb:.0f})"
        )


_WORKLOADS: dict[str, WorkloadProfile] = {
    "ohlc_1m": WorkloadProfile(1, 12, 0.50, 2, 768.0, 2048.0),
    "ohlc_5m": WorkloadProfile(1, 12, 0.50, 2, 512.0, 2048.0),
    "feature_cache": WorkloadProfile(1, 10, 0.35, 2, 3072.0, 4096.0),
    "feature_cache_global": WorkloadProfile(1, 8, 0.25, 2, 4096.0, 6144.0),
    "dataset": WorkloadProfile(1, 8, 0.30, 2, 4096.0, 6144.0),
    "labels_refresh": WorkloadProfile(1, 10, 0.35, 2, 2048.0, 4096.0),
    "explore": WorkloadProfile(1, 10, 0.35, 2, 3072.0, 4096.0),
}

_TELEMETRY_FILE = "parallel_runtime_telemetry.json"
_MAX_SAMPLES_PER_WORKLOAD = 32


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "tradebot":
            return p
    return here.parents[2]


def telemetry_file_path() -> Path:
    return _repo_root() / "modules" / "utils" / _TELEMETRY_FILE


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def host_resources() -> HostResources:
    cpu_count = max(1, int(os.cpu_count() or 1))
    if psutil is None:
        return HostResources(
            cpu_count=cpu_count,
            total_mb=0.0,
            available_mb=float("inf"),
            used_pct=0.0,
        )
    vm = psutil.virtual_memory()
    return HostResources(
        cpu_count=cpu_count,
        total_mb=float(vm.total) / (1024.0 * 1024.0),
        available_mb=float(vm.available) / (1024.0 * 1024.0),
        used_pct=float(vm.percent),
    )


def host_key(host: HostResources | None = None) -> str:
    h = host or host_resources()
    total_rounded = int(round(float(h.total_mb)))
    return f"cpu{int(h.cpu_count)}_mem{total_rounded}"


def _coerce_workers(value: object) -> int:
    try:
        return max(0, int(str(value).strip()))
    except Exception:
        return 0


def _default_telemetry() -> dict[str, Any]:
    return {"version": 1, "updated_utc": "", "hosts": {}}


def load_runtime_telemetry() -> dict[str, Any]:
    path = telemetry_file_path()
    if not path.exists():
        return _default_telemetry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default_telemetry()
        data.setdefault("version", 1)
        data.setdefault("updated_utc", "")
        data.setdefault("hosts", {})
        return data
    except Exception:
        return _default_telemetry()


def save_runtime_telemetry(data: dict[str, Any]) -> None:
    path = telemetry_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data or {})
    payload["version"] = 1
    payload["updated_utc"] = _utc_now_iso()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _percentile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    q = min(1.0, max(0.0, float(q)))
    idx = int(round((len(vals) - 1) * q))
    return float(vals[idx])


def get_workload_telemetry(kind: str, host: HostResources | None = None) -> dict[str, Any]:
    h = host or host_resources()
    data = load_runtime_telemetry()
    hk = host_key(h)
    host_entry = dict((data.get("hosts") or {}).get(hk) or {})
    workloads = dict(host_entry.get("workloads") or {})
    return dict(workloads.get(str(kind).strip().lower()) or {})


def record_workload_observation(
    kind: str,
    *,
    workers: int,
    duration_s: float,
    peak_process_mb: float = 0.0,
    peak_used_pct: float = 0.0,
    min_available_mb: float = 0.0,
    units: int = 0,
    unit_name: str = "",
    success: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    h = host_resources()
    data = load_runtime_telemetry()
    hk = host_key(h)
    hosts = dict(data.get("hosts") or {})
    host_entry = dict(hosts.get(hk) or {})
    host_entry["cpu_count"] = int(h.cpu_count)
    host_entry["total_mb"] = float(h.total_mb)
    host_entry["updated_utc"] = _utc_now_iso()

    workloads = dict(host_entry.get("workloads") or {})
    wk = str(kind).strip().lower() or "explore"
    entry = dict(workloads.get(wk) or {})
    samples = list(entry.get("samples") or [])

    workers_i = max(1, int(workers))
    duration_f = max(0.0, float(duration_s))
    peak_process_f = max(0.0, float(peak_process_mb))
    per_worker_mb = peak_process_f / float(workers_i) if workers_i > 0 else 0.0
    sec_per_worker = duration_f / float(workers_i) if workers_i > 0 else 0.0
    sec_per_unit = (duration_f / float(units)) if int(units) > 0 else 0.0

    sample = {
        "utc": _utc_now_iso(),
        "success": bool(success),
        "workers": int(workers_i),
        "duration_s": float(duration_f),
        "peak_process_mb": float(peak_process_f),
        "estimated_per_worker_mb": float(per_worker_mb),
        "peak_used_pct": float(peak_used_pct),
        "min_available_mb": float(min_available_mb),
        "units": int(units),
        "unit_name": str(unit_name or ""),
        "seconds_per_worker": float(sec_per_worker),
        "seconds_per_unit": float(sec_per_unit),
        "metadata": dict(metadata or {}),
    }
    samples.append(sample)
    samples = samples[-_MAX_SAMPLES_PER_WORKLOAD:]

    success_samples = [s for s in samples if bool(s.get("success", False))]
    per_worker_values = [float(s.get("estimated_per_worker_mb") or 0.0) for s in success_samples]
    sec_worker_values = [float(s.get("seconds_per_worker") or 0.0) for s in success_samples]
    sec_unit_values = [float(s.get("seconds_per_unit") or 0.0) for s in success_samples if float(s.get("seconds_per_unit") or 0.0) > 0.0]
    worker_values = [int(s.get("workers") or 0) for s in success_samples]

    entry.update(
        {
            "runs": int(len(samples)),
            "success_runs": int(len(success_samples)),
            "last_workers": int(workers_i),
            "last_duration_s": float(duration_f),
            "last_peak_process_mb": float(peak_process_f),
            "last_estimated_per_worker_mb": float(per_worker_mb),
            "last_peak_used_pct": float(peak_used_pct),
            "last_min_available_mb": float(min_available_mb),
            "p90_estimated_per_worker_mb": float(_percentile(per_worker_values, 0.90)),
            "p95_estimated_per_worker_mb": float(_percentile(per_worker_values, 0.95)),
            "max_estimated_per_worker_mb": float(max(per_worker_values) if per_worker_values else 0.0),
            "p90_seconds_per_worker": float(_percentile(sec_worker_values, 0.90)),
            "p90_seconds_per_unit": float(_percentile(sec_unit_values, 0.90)),
            "max_successful_workers": int(max(worker_values) if worker_values else 0),
            "samples": samples,
        }
    )
    workloads[wk] = entry
    host_entry["workloads"] = workloads
    hosts[hk] = host_entry
    data["hosts"] = hosts
    save_runtime_telemetry(data)


def _effective_per_worker_mb(kind: str, profile: WorkloadProfile, host: HostResources) -> float:
    telemetry = get_workload_telemetry(kind, host)
    observed = 0.0
    if int(telemetry.get("success_runs", 0)) >= 2:
        observed = max(
            float(telemetry.get("p90_estimated_per_worker_mb", 0.0) or 0.0),
            float(telemetry.get("max_estimated_per_worker_mb", 0.0) or 0.0) * 0.85,
        )
    if observed <= 0.0:
        return float(profile.per_worker_mb)
    return max(float(profile.per_worker_mb), float(observed) * 1.15)


def recommend_workers(kind: str, requested: int = 0) -> WorkerDecision:
    wk = str(kind).strip().lower() or "explore"
    profile = _WORKLOADS.get(wk, _WORKLOADS["explore"])
    host = host_resources()
    requested_i = _coerce_workers(requested)
    effective_per_worker_mb = _effective_per_worker_mb(wk, profile, host)
    telemetry = get_workload_telemetry(wk, host)
    observed_per_worker_mb = float(telemetry.get("p90_estimated_per_worker_mb", 0.0) or 0.0)

    cpu_free = max(1, int(host.cpu_count) - int(profile.reserve_cores))
    cpu_cap = int(math.ceil(float(cpu_free) * float(profile.cpu_fraction)))
    cpu_cap = max(int(profile.min_workers), min(int(profile.max_workers), cpu_cap))

    if math.isfinite(float(host.available_mb)):
        budget_mb = max(0.0, float(host.available_mb) - float(profile.reserve_mb))
        memory_cap = int(budget_mb // max(1.0, float(effective_per_worker_mb)))
    else:
        memory_cap = int(profile.max_workers)
    memory_cap = max(int(profile.min_workers), min(int(profile.max_workers), memory_cap))

    workers = min(cpu_cap, memory_cap)
    workers = max(int(profile.min_workers), min(int(profile.max_workers), workers))
    if requested_i > 0:
        workers = max(int(profile.min_workers), min(int(requested_i), workers))

    return WorkerDecision(
        kind=wk,
        workers=int(workers),
        cpu_cap=int(cpu_cap),
        memory_cap=int(memory_cap),
        requested=int(requested_i),
        host=host,
        profile_per_worker_mb=float(effective_per_worker_mb),
        observed_per_worker_mb=float(observed_per_worker_mb),
    )


def apply_env_worker_default(env_name: str, kind: str, default: int = 0) -> int:
    raw = str(os.getenv(env_name, "") or "").strip()
    if raw:
        return _coerce_workers(raw)
    decision = recommend_workers(kind, requested=int(default))
    os.environ[env_name] = str(int(decision.workers))
    return int(decision.workers)


__all__ = [
    "HostResources",
    "WorkloadProfile",
    "WorkerDecision",
    "host_resources",
    "host_key",
    "telemetry_file_path",
    "load_runtime_telemetry",
    "save_runtime_telemetry",
    "get_workload_telemetry",
    "record_workload_observation",
    "recommend_workers",
    "apply_env_worker_default",
]
