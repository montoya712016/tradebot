from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_REASONING_EFFORTS = ["low", "medium", "high", "xhigh"]
SUPPORTED_ACCESS_MODES = ["danger-full-access", "workspace-write", "read-only"]
CURATED_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-mini",
]


class _AppServerClient:
    def __init__(self, *, codex_exe: str, cwd: Path, env: dict[str, str]) -> None:
        self.codex_exe = codex_exe
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.initialized = False
        self.proc = subprocess.Popen(
            [codex_exe, "app-server", "--listen", "stdio://"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(cwd),
            env=env,
        )
        self.stdout_queue: queue.Queue[str] = queue.Queue()
        self.stderr_lines: list[str] = []
        self.req_id = 0
        threading.Thread(target=self._stdout_reader, daemon=True).start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()

    def _stdout_reader(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self.stdout_queue.put(line)

    def _stderr_reader(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            text = str(line or "").rstrip("\n")
            if text:
                self.stderr_lines.append(text)
            if len(self.stderr_lines) > 200:
                self.stderr_lines = self.stderr_lines[-200:]

    def send(self, method: str, params: dict[str, Any]) -> str:
        self.req_id += 1
        req_id = str(self.req_id)
        payload = {"id": req_id, "method": method, "params": params}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return req_id

    def wait_for(self, req_id: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        notifications: list[dict[str, Any]] = []
        seen: list[dict[str, Any]] = []
        while time.time() < deadline:
            try:
                raw = self.stdout_queue.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None:
                    return {
                        "process_exit": self.proc.returncode,
                        "notifications": notifications,
                        "seen": seen,
                        "stderr_tail": self.stderr_lines[-20:],
                    }
                continue
            parsed = _safe_json_loads(raw)
            if parsed is None:
                seen.append({"raw": raw})
                continue
            seen.append(parsed)
            if "method" in parsed and "id" not in parsed:
                notifications.append(parsed)
            if req_id is not None and parsed.get("id") == req_id:
                return {
                    "message": parsed,
                    "notifications": notifications,
                    "seen": seen,
                    "stderr_tail": self.stderr_lines[-20:],
                }
        return {
            "timeout": True,
            "notifications": notifications,
            "seen": seen,
            "stderr_tail": self.stderr_lines[-20:],
        }

    def close(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            return
        try:
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent), suffix=".tmp") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _tail_text(path: Path, max_chars: int = 24000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _normalize_prompt(prompt: str) -> str:
    lines = [str(line).rstrip() for line in str(prompt or "").replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _existing_file(path_str: str) -> str:
    raw = str(path_str or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    return str(resolved) if resolved.exists() and resolved.is_file() else ""


def _truncate_text(text: str, max_chars: int = 180) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _new_conversation_id() -> str:
    return f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


LOG_TS_PREFIX = re.compile(r"^\[ts=(?P<ts>[^\]]+)\]\s?(?P<body>.*)$")
RUNNING_STALE_AFTER_SECONDS = 150


def _split_timestamped_log_line(text: str) -> tuple[str, str]:
    raw = str(text or "").rstrip("\n")
    match = LOG_TS_PREFIX.match(raw)
    if not match:
        return "", raw
    return str(match.group("ts") or "").strip(), str(match.group("body") or "")


def _parse_iso_dt(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _latest_log_timestamp(raw_log: str) -> str:
    latest = ""
    for line in str(raw_log or "").splitlines():
        event_ts, _ = _split_timestamped_log_line(line)
        if event_ts:
            latest = event_ts
    return latest


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return False
    output = str(proc.stdout or "")
    if "No tasks are running" in output:
        return False
    return str(pid) in output


def _codex_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def _load_codex_config() -> dict[str, Any]:
    path = _codex_config_path()
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_model() -> str:
    cfg = _load_codex_config()
    raw = str(cfg.get("model", "") or "").strip()
    return raw or "gpt-5.4"


def _default_reasoning_effort() -> str:
    cfg = _load_codex_config()
    raw = str(cfg.get("model_reasoning_effort", "") or "").strip().lower()
    if raw in SUPPORTED_REASONING_EFFORTS:
        return raw
    return "medium"


def _default_access_mode() -> str:
    raw = str(os.getenv("ASTRA_CODEX_REMOTE_ACCESS", "danger-full-access") or "").strip().lower()
    if raw in SUPPORTED_ACCESS_MODES:
        return raw
    return "danger-full-access"


def _find_codex_executable() -> str:
    env_override = _existing_file(os.getenv("ASTRA_CODEX_CLI", ""))
    if env_override:
        return env_override

    direct = shutil.which("codex") or shutil.which("codex.exe")
    if direct:
        return str(Path(direct).resolve())

    user_home = Path.home()
    base_dirs = [
        user_home / ".vscode" / "extensions",
        user_home / ".cursor" / "extensions",
    ]
    suffixes = [
        Path("bin") / "windows-x86_64" / "codex.exe",
        Path("bin") / "codex.exe",
    ]
    for base in base_dirs:
        if not base.exists():
            continue
        for ext_dir in sorted(base.glob("openai.chatgpt-*"), reverse=True):
            for suffix in suffixes:
                candidate = ext_dir / suffix
                if candidate.exists() and candidate.is_file():
                    return str(candidate.resolve())
    return ""


def _find_ripgrep_executable(codex_exe: str = "") -> str:
    env_override = _existing_file(os.getenv("ASTRA_REMOTE_RG", ""))
    if env_override:
        return env_override

    direct = shutil.which("rg") or shutil.which("rg.exe")
    if direct:
        return str(Path(direct).resolve())

    codex_path = Path(str(codex_exe or "").strip())
    if codex_path.exists():
        sibling = codex_path.with_name("rg.exe")
        if sibling.exists() and sibling.is_file():
            return str(sibling.resolve())

    user_home = Path.home()
    base_dirs = [
        user_home / ".vscode" / "extensions",
        user_home / ".cursor" / "extensions",
    ]
    suffixes = [
        Path("bin") / "windows-x86_64" / "rg.exe",
        Path("bin") / "rg.exe",
    ]
    for base in base_dirs:
        if not base.exists():
            continue
        for ext_dir in sorted(base.glob("openai.chatgpt-*"), reverse=True):
            for suffix in suffixes:
                candidate = ext_dir / suffix
                if candidate.exists() and candidate.is_file():
                    return str(candidate.resolve())
    return ""


def _find_git_executable() -> str:
    env_override = _existing_file(os.getenv("ASTRA_REMOTE_GIT", ""))
    if env_override:
        return env_override
    direct = shutil.which("git") or shutil.which("git.exe")
    if direct:
        return str(Path(direct).resolve())
    common_paths = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd" / "git.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "git.exe",
    ]
    for candidate in common_paths:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return ""


def _prepend_path_entries(base_path: str, entries: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in [*entries, *(str(base_path or "").split(os.pathsep))]:
        item = str(raw or "").strip()
        if not item:
            continue
        norm = os.path.normcase(os.path.normpath(item))
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(item)
    return os.pathsep.join(ordered)


def _supported_models() -> list[str]:
    default_model = _default_model()
    out: list[str] = []
    for value in [default_model, *CURATED_MODELS]:
        item = str(value or "").strip()
        if not item or item in out:
            continue
        out.append(item)
    return out


def _git_status_map(repo_root: Path) -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain=v1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return {}
    out: dict[str, str] = {}
    for line in str(proc.stdout or "").splitlines():
        raw = str(line or "").rstrip()
        if len(raw) < 4:
            continue
        status = raw[:2]
        path = raw[3:].strip()
        if path:
            out[path] = status
    return out


def _build_command_summary(command: str) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    marker = " -Command "
    if marker in text and "powershell.exe" in text.lower():
        text = text.split(marker, 1)[1].strip()
        if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
            text = text[1:-1]
    return _truncate_text(text, max_chars=220)


def _filtered_activity_text(item: dict[str, Any]) -> str:
    text = str(item.get("text", "") or "").strip()
    return text


def _response_error_text(snapshot: dict[str, Any]) -> str:
    message = snapshot.get("message")
    if not isinstance(message, dict):
        if snapshot.get("timeout"):
            return "Timeout aguardando resposta do Codex app-server."
        if "process_exit" in snapshot:
            return f"Codex app-server encerrou antes da resposta (exit={snapshot.get('process_exit')})."
        return "Resposta invalida do Codex app-server."
    error = message.get("error")
    if isinstance(error, dict):
        return str(error.get("message", "") or error)
    return ""


def _extract_thread_id(snapshot: dict[str, Any]) -> str:
    message = snapshot.get("message")
    if not isinstance(message, dict):
        return ""
    result = message.get("result")
    if not isinstance(result, dict):
        return ""
    thread = result.get("thread")
    if isinstance(thread, dict):
        thread_id = str(thread.get("id", "") or "").strip()
        if thread_id:
            return thread_id
    return str(result.get("threadId", "") or "").strip()


def _extract_turn_usage(notification: dict[str, Any]) -> dict[str, int]:
    params = notification.get("params")
    if not isinstance(params, dict):
        return {}
    turn = params.get("turn")
    raw_usage = turn.get("usage") if isinstance(turn, dict) else params.get("usage")
    if not isinstance(raw_usage, dict):
        return {}
    return {
        "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
        "cached_input_tokens": int(raw_usage.get("cached_input_tokens", 0) or 0),
        "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
    }


def _normalize_app_server_item(raw_item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw_item, dict):
        return None
    item_type = str(raw_item.get("type", "") or "").strip()
    normalized_type = item_type.replace("-", "_")
    if normalized_type.lower() in {"agentmessage", "agent_message"}:
        return None
    if "command" in normalized_type.lower():
        return {
            "type": "command_execution",
            "command": str(
                raw_item.get("command", "")
                or raw_item.get("rawCommand", "")
                or raw_item.get("title", "")
                or raw_item.get("text", "")
                or ""
            ),
            "exit_code": int(raw_item.get("exitCode", raw_item.get("exit_code", 0)) or 0),
        }
    if normalized_type.lower() in {"mcptoolcall", "mcp_tool_call"}:
        name = str(raw_item.get("toolName", "") or raw_item.get("serverName", "") or "MCP tool")
        return {"type": "command_execution", "command": name, "exit_code": int(raw_item.get("exitCode", 0) or 0)}
    return None


def _access_mode_to_turn_sandbox_policy(access_mode: str) -> dict[str, Any]:
    mapping = {
        "danger-full-access": {"type": "dangerFullAccess"},
        "workspace-write": {"type": "workspaceWrite"},
        "read-only": {"type": "readOnly"},
    }
    return dict(mapping.get(str(access_mode or "").strip().lower(), {"type": "dangerFullAccess"}))


def _build_transcript(job: dict[str, Any], raw_log: str) -> dict[str, Any]:
    agent_messages: list[str] = []
    activity: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    trace_lines: list[str] = []
    session_id = str(job.get("session_id", "") or "").strip()

    for line in str(raw_log or "").splitlines():
        event_ts, body = _split_timestamped_log_line(line)
        event = _safe_json_loads(body)
        if event is None:
            stripped = str(body or "").strip()
            if stripped.startswith("[remote]"):
                trace_lines.append(stripped)
            continue
        etype = str(event.get("type", "") or "")
        if etype == "thread.started":
            maybe_thread_id = str(event.get("thread_id", "") or "").strip()
            if maybe_thread_id:
                session_id = maybe_thread_id
            continue
        if etype == "turn.started":
            activity.append({"kind": "system", "status": "running", "text": "Sessao iniciada.", "timestamp": event_ts})
            continue
        if etype == "turn.completed":
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                usage = {
                    "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
                    "cached_input_tokens": int(raw_usage.get("cached_input_tokens", 0) or 0),
                    "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
                }
            activity.append({"kind": "system", "status": "completed", "text": "Turno concluido.", "timestamp": event_ts})
            continue
        if etype in {"item.started", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "")
            if item_type == "agent_message":
                text = _filtered_activity_text(item)
                if text:
                    agent_messages.append(text)
                continue
            if item_type == "command_execution":
                command_summary = _build_command_summary(str(item.get("command", "") or ""))
                status = "running" if etype == "item.started" else ("completed" if int(item.get("exit_code", 0) or 0) == 0 else "failed")
                prefix = "Executando" if etype == "item.started" else "Comando concluido"
                if int(item.get("exit_code", 0) or 0) != 0 and etype == "item.completed":
                    prefix = f"Comando falhou ({int(item.get('exit_code', 0) or 0)})"
                activity.append(
                    {
                        "kind": "tool",
                        "status": status,
                        "text": f"{prefix}: {command_summary}" if command_summary else prefix,
                        "timestamp": event_ts,
                    }
                )
                continue

    final_response = ""
    progress_messages: list[str] = []
    status = str(job.get("status", "") or "")
    if agent_messages:
        if status == "completed":
            final_response = agent_messages[-1]
            progress_messages = agent_messages[:-1]
        elif status == "failed":
            final_response = agent_messages[-1]
            progress_messages = agent_messages[:-1]
        else:
            progress_messages = agent_messages

    for msg in progress_messages:
        activity.insert(0, {"kind": "assistant", "status": "info", "text": msg, "timestamp": ""})

    status_line = ""
    if status == "running":
        if activity:
            status_line = str(activity[-1].get("text", "") or "")
        elif progress_messages:
            status_line = progress_messages[-1]
        else:
            status_line = "Codex esta trabalhando."
    elif status == "completed":
        status_line = "Resposta pronta."
    elif status == "failed":
        status_line = str(job.get("error", "") or "Job falhou.")
    elif status == "queued":
        status_line = "Job aguardando execucao."
    elif status == "cancelling":
        status_line = "Cancelamento solicitado."

    filtered_trace = "\n".join(trace_lines[-20:]).strip()
    return {
        "session_id": session_id,
        "final_response": final_response,
        "progress_messages": progress_messages,
        "activity": activity[-24:],
        "usage": usage,
        "status_line": status_line,
        "filtered_trace": filtered_trace,
    }


class RemoteControlManager:
    def __init__(self, *, repo_root: Path, storage_dir: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.storage_dir = Path(storage_dir).resolve()
        self.jobs_path = self.storage_dir / "jobs.json"
        self.logs_dir = self.storage_dir / "logs"
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = self._load_jobs()
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._app_server: _AppServerClient | None = None

    def _reconcile_job_state(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            current = self._jobs.get(str(job_id))
            job = dict(current) if isinstance(current, dict) else None
            proc = self._procs.get(str(job_id))
        if not job:
            return None
        if str(job.get("status", "") or "") != "running":
            return job

        if proc is not None:
            if proc.poll() is None:
                return job
            with self._lock:
                self._procs.pop(str(job_id), None)

        pid = int(job.get("pid", 0) or 0)
        if pid and _pid_exists(pid):
            return job

        raw_log = _tail_text(self._log_path_for_job(job), max_chars=240000)
        latest_ts = _latest_log_timestamp(raw_log)
        reference_dt = _parse_iso_dt(latest_ts) or _parse_iso_dt(str(job.get("started_at", "") or "")) or _parse_iso_dt(str(job.get("created_at", "") or ""))
        if reference_dt is None:
            return self._update_job(
                str(job_id),
                status="failed",
                finished_at=_utc_now_iso(),
                error="Job ficou sem processo vivo e sem timestamps de progresso.",
            )

        age_seconds = (datetime.now(timezone.utc) - reference_dt).total_seconds()
        if age_seconds <= RUNNING_STALE_AFTER_SECONDS:
            return job

        return self._update_job(
            str(job_id),
            status="failed",
            finished_at=_utc_now_iso(),
            error=f"Execucao interrompida ou travada: sem processo vivo e sem atividade nova ha {int(age_seconds)}s.",
            exit_code=-1,
        )

    def _reconcile_all_running_jobs(self) -> None:
        with self._lock:
            running_ids = [job_id for job_id, job in self._jobs.items() if str(job.get("status", "") or "") == "running"]
        for job_id in running_ids:
            self._reconcile_job_state(job_id)

    def _load_jobs(self) -> dict[str, dict[str, Any]]:
        if not self.jobs_path.exists():
            return {}
        try:
            data = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for job_id, payload in data.items():
            if isinstance(payload, dict):
                out[str(job_id)] = payload
        return out

    def _save_jobs(self) -> None:
        with self._lock:
            _write_json_atomic(self.jobs_path, self._jobs)

    def _command_preview(self, cmd: list[str]) -> str:
        return subprocess.list2cmdline([str(part) for part in cmd])

    def _resolved_conversation_id(self, job: dict[str, Any], transcript: dict[str, Any] | None = None) -> str:
        conversation_id = str(job.get("conversation_id", "") or "").strip()
        if conversation_id:
            return conversation_id
        session_id = str(job.get("session_id", "") or "").strip()
        if session_id:
            return session_id
        if isinstance(transcript, dict):
            maybe_session = str(transcript.get("session_id", "") or "").strip()
            if maybe_session:
                return maybe_session
        return str(job.get("id", "") or "").strip()

    def _jobs_with_transcripts(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        with self._lock:
            rows = [dict(job) for job in self._jobs.values()]
        rows.sort(key=lambda item: str(item.get("created_at", "")))
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for job in rows:
            transcript = self.transcript(job.get("id", ""))
            fresh_job = self.get_job(job.get("id", "")) or job
            out.append((fresh_job, transcript))
        return out

    def _latest_session_id_for_conversation(self, conversation_id: str) -> str:
        target = str(conversation_id or "").strip()
        if not target:
            return ""
        latest: tuple[str, str] | None = None
        for job, transcript in self._jobs_with_transcripts():
            resolved = self._resolved_conversation_id(job, transcript)
            if resolved != target:
                continue
            if str(job.get("backend", "") or "codex-exec") != "codex-app-server":
                continue
            session_id = str(job.get("session_id", "") or transcript.get("session_id", "") or "").strip()
            if not session_id:
                continue
            created_at = str(job.get("created_at", "") or "")
            latest = (created_at, session_id)
        return latest[1] if latest else ""

    def _update_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            job = dict(self._jobs[str(job_id)])
            job.update(changes)
            self._jobs[str(job_id)] = job
            self._save_jobs()
            return dict(job)

    def codex_executable(self) -> str:
        return _find_codex_executable()

    def _build_subprocess_env(self, codex_exe: str) -> dict[str, str]:
        env = dict(os.environ)
        rg_exe = _find_ripgrep_executable(codex_exe)
        git_exe = _find_git_executable()

        extra_dirs = [
            str(Path(codex_exe).resolve().parent) if codex_exe else "",
            str(Path(rg_exe).resolve().parent) if rg_exe else "",
            str(Path(git_exe).resolve().parent) if git_exe else "",
        ]
        env["PATH"] = _prepend_path_entries(env.get("PATH", ""), extra_dirs)
        if rg_exe:
            env.setdefault("ASTRA_REMOTE_RG", rg_exe)
        if git_exe:
            env.setdefault("ASTRA_REMOTE_GIT", git_exe)
        return env

    def _get_app_server(self, codex_exe: str) -> _AppServerClient:
        with self._lock:
            current = self._app_server
            if current is not None and current.proc.poll() is None:
                return current
            if current is not None:
                try:
                    current.close()
                except Exception:
                    pass
            current = _AppServerClient(codex_exe=codex_exe, cwd=self.repo_root, env=self._build_subprocess_env(codex_exe))
            self._app_server = current
            return current

    def capabilities(self) -> dict[str, Any]:
        self._reconcile_all_running_jobs()
        codex_path = self.codex_executable()
        rg_path = _find_ripgrep_executable(codex_path)
        git_path = _find_git_executable()
        return {
            "codex_available": bool(codex_path),
            "codex_path": codex_path or "",
            "rg_available": bool(rg_path),
            "rg_path": rg_path or "",
            "git_available": bool(git_path),
            "git_path": git_path or "",
            "repo_root": str(self.repo_root),
            "storage_dir": str(self.storage_dir),
            "running_jobs": len([job for job in self._jobs.values() if str(job.get("status")) == "running"]),
            "defaults": {
                "model": _default_model(),
                "reasoning_effort": _default_reasoning_effort(),
                "access_mode": _default_access_mode(),
            },
            "supported_models": [{"id": value, "label": value} for value in _supported_models()],
            "supported_reasoning_efforts": [{"id": value, "label": value} for value in SUPPORTED_REASONING_EFFORTS],
            "supported_access_modes": [
                {"id": "danger-full-access", "label": "Full access"},
                {"id": "workspace-write", "label": "Workspace write"},
                {"id": "read-only", "label": "Read only"},
            ],
        }

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = [dict(job) for job in self._jobs.values()]
        rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        out: list[dict[str, Any]] = []
        for job in rows[: max(1, int(limit))]:
            transcript = self.transcript(job.get("id", ""))
            fresh_job = self.get_job(job.get("id", "")) or job
            job = dict(fresh_job)
            job["conversation_id"] = self._resolved_conversation_id(job, transcript)
            job["response_preview"] = _truncate_text(str(transcript.get("final_response", "") or ""), 180)
            job["status_line"] = str(transcript.get("status_line", "") or "")
            out.append(job)
        return out

    def list_conversations(self, *, limit: int = 20) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for job, transcript in self._jobs_with_transcripts():
            conversation_id = self._resolved_conversation_id(job, transcript)
            created_at = str(job.get("created_at", "") or "")
            latest_key = str(job.get("finished_at") or job.get("started_at") or created_at or "")
            entry = grouped.get(conversation_id)
            if entry is None:
                grouped[conversation_id] = {
                    "id": conversation_id,
                    "created_at": created_at,
                    "updated_at": latest_key,
                    "created_by": str(job.get("created_by", "") or ""),
                    "turn_count": 1,
                    "first_prompt": str(job.get("prompt", "") or ""),
                    "latest_job": job,
                    "latest_transcript": transcript,
                }
                continue
            entry["turn_count"] = int(entry.get("turn_count", 0) or 0) + 1
            if created_at < str(entry.get("created_at", "") or ""):
                entry["created_at"] = created_at
                entry["first_prompt"] = str(job.get("prompt", "") or "")
            if latest_key >= str(entry.get("updated_at", "") or ""):
                entry["updated_at"] = latest_key
                entry["latest_job"] = job
                entry["latest_transcript"] = transcript

        conversations: list[dict[str, Any]] = []
        for conversation in grouped.values():
            latest_job = dict(conversation.get("latest_job", {}) or {})
            latest_transcript = dict(conversation.get("latest_transcript", {}) or {})
            conversations.append(
                {
                    "id": str(conversation.get("id", "") or ""),
                    "created_at": str(conversation.get("created_at", "") or ""),
                    "updated_at": str(conversation.get("updated_at", "") or ""),
                    "created_by": str(conversation.get("created_by", "") or ""),
                    "turn_count": int(conversation.get("turn_count", 0) or 0),
                    "prompt": str(conversation.get("first_prompt", "") or ""),
                    "latest_prompt": str(latest_job.get("prompt", "") or ""),
                    "status": str(latest_job.get("status", "") or "queued"),
                    "model": str(latest_job.get("model", "") or ""),
                    "reasoning_effort": str(latest_job.get("reasoning_effort", "") or ""),
                    "access_mode": str(latest_job.get("access_mode", "") or ""),
                    "latest_job_id": str(latest_job.get("id", "") or ""),
                    "response_preview": _truncate_text(
                        str(
                            latest_transcript.get("final_response", "")
                            or latest_job.get("final_response", "")
                            or latest_transcript.get("status_line", "")
                            or ""
                        ),
                        180,
                    ),
                    "status_line": str(latest_transcript.get("status_line", "") or ""),
                }
            )
        conversations.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return conversations[: max(1, int(limit))]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        target = str(conversation_id or "").strip()
        if not target:
            return None
        turns: list[dict[str, Any]] = []
        for job, transcript in self._jobs_with_transcripts():
            resolved = self._resolved_conversation_id(job, transcript)
            if resolved != target:
                continue
            turns.append({"job": job, "transcript": transcript})
        if not turns:
            return None
        turns.sort(key=lambda item: str(item["job"].get("created_at", "") or ""))
        latest_turn = turns[-1]
        first_turn = turns[0]
        return {
            "conversation": {
                "id": target,
                "created_at": str(first_turn["job"].get("created_at", "") or ""),
                "updated_at": str(
                    latest_turn["job"].get("finished_at")
                    or latest_turn["job"].get("started_at")
                    or latest_turn["job"].get("created_at")
                    or ""
                ),
                "created_by": str(first_turn["job"].get("created_by", "") or ""),
                "turn_count": len(turns),
                "prompt": str(first_turn["job"].get("prompt", "") or ""),
                "latest_prompt": str(latest_turn["job"].get("prompt", "") or ""),
                "status": str(latest_turn["job"].get("status", "") or "queued"),
                "model": str(latest_turn["job"].get("model", "") or ""),
                "reasoning_effort": str(latest_turn["job"].get("reasoning_effort", "") or ""),
                "access_mode": str(latest_turn["job"].get("access_mode", "") or ""),
                "latest_job_id": str(latest_turn["job"].get("id", "") or ""),
            },
            "turns": turns,
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        reconciled = self._reconcile_job_state(str(job_id))
        if reconciled is not None:
            return dict(reconciled)
        with self._lock:
            job = self._jobs.get(str(job_id))
            return dict(job) if isinstance(job, dict) else None

    def _log_path_for_job(self, job: dict[str, Any]) -> Path:
        log_path = Path(str(job.get("log_path", "") or ""))
        if log_path.is_absolute():
            return log_path
        return self.storage_dir / "logs" / log_path

    def transcript(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {
                "session_id": "",
                "final_response": "",
                "progress_messages": [],
                "activity": [],
                "usage": {},
                "status_line": "",
                "filtered_trace": "",
            }
        raw_log = _tail_text(self._log_path_for_job(job), max_chars=400000)
        transcript = _build_transcript(job, raw_log)
        session_id = str(transcript.get("session_id", "") or "").strip()
        updates: dict[str, Any] = {}
        if session_id and not str(job.get("session_id", "") or "").strip():
            updates["session_id"] = session_id
        if session_id and not str(job.get("conversation_id", "") or "").strip():
            updates["conversation_id"] = session_id
        if updates:
            self._update_job(str(job_id), **updates)
        return transcript

    def filtered_log(self, job_id: str) -> str:
        transcript = self.transcript(job_id)
        lines: list[str] = []
        for item in transcript.get("activity", []):
            if isinstance(item, dict):
                status = str(item.get("status", "") or "").upper()
                text = str(item.get("text", "") or "").strip()
                if text:
                    lines.append(f"[{status}] {text}")
        final_response = str(transcript.get("final_response", "") or "").strip()
        if final_response:
            lines.append("")
            lines.append("[FINAL]")
            lines.append(final_response)
        trace = str(transcript.get("filtered_trace", "") or "").strip()
        if trace:
            lines.append("")
            lines.append("[TRACE]")
            lines.append(trace)
        return "\n".join(lines).strip()

    def create_codex_job(
        self,
        *,
        prompt: str,
        created_by: str,
        model: str,
        reasoning_effort: str,
        access_mode: str,
        conversation_id: str = "",
    ) -> dict[str, Any]:
        prompt = _normalize_prompt(prompt)
        model = str(model or "").strip() or _default_model()
        reasoning_effort = str(reasoning_effort or "").strip().lower() or _default_reasoning_effort()
        access_mode = str(access_mode or "").strip().lower() or _default_access_mode()
        requested_conversation_id = str(conversation_id or "").strip()

        if not prompt:
            raise ValueError("Prompt vazio.")
        if len(prompt) > 12000:
            raise ValueError("Prompt muito longo para o remote.")
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise ValueError("Reasoning effort invalido.")
        if access_mode not in SUPPORTED_ACCESS_MODES:
            raise ValueError("Modo de acesso invalido.")
        if model not in _supported_models():
            raise ValueError("Modelo invalido para este remote.")

        with self._lock:
            self._reconcile_all_running_jobs()
            for existing in self._jobs.values():
                if str(existing.get("status")) == "running":
                    raise RuntimeError("Ja existe um job em execucao.")

        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        local_conversation_id = requested_conversation_id or _new_conversation_id()
        resume_session_id = self._latest_session_id_for_conversation(local_conversation_id) if requested_conversation_id else ""
        log_path = self.logs_dir / f"{job_id}.log"
        dirty_before = _git_status_map(self.repo_root)

        job = {
            "id": job_id,
            "backend": "codex-app-server",
            "conversation_id": local_conversation_id,
            "session_id": resume_session_id,
            "resume_session_id": resume_session_id,
            "kind": "codex",
            "status": "queued",
            "created_at": _utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "created_by": str(created_by or "").strip() or "unknown",
            "prompt": prompt,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "access_mode": access_mode,
            "command": "",
            "cwd": str(self.repo_root),
            "log_path": str(log_path),
            "exit_code": None,
            "changed_files": [],
            "diff_stat": "",
            "error": "",
            "worktree_dirty_before": bool(dirty_before),
            "worktree_dirty_before_count": len(dirty_before),
        }
        with self._lock:
            self._jobs[job_id] = job
            self._save_jobs()
        thread = threading.Thread(target=self._run_codex_job, args=(job_id,), daemon=True)
        thread.start()
        return job

    def _build_codex_command(self, _job: dict[str, Any], codex_exe: str) -> list[str]:
        return [codex_exe, "app-server", "--listen", "stdio://"]

    def _run_codex_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        log_path = self._log_path_for_job(job)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        codex_exe = self.codex_executable()
        if not codex_exe:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now_iso(),
                error="Executavel do Codex nao encontrado neste ambiente.",
            )
            with log_path.open("a", encoding="utf-8", errors="replace") as log_fh:
                log_fh.write(f"[remote][error] codex executable not found for job={job_id}\n")
            return

        cmd = self._build_codex_command(job, codex_exe)
        command_preview = self._command_preview(cmd)
        self._update_job(
            job_id,
            status="running",
            started_at=_utc_now_iso(),
            command=command_preview,
        )

        dirty_before = _git_status_map(self.repo_root)

        with log_path.open("a", encoding="utf-8", errors="replace") as log_fh:
            env = self._build_subprocess_env(codex_exe)
            rg_exe = str(env.get("ASTRA_REMOTE_RG", "") or "")
            git_exe = str(env.get("ASTRA_REMOTE_GIT", "") or "")
            log_fh.write(f"[remote] job={job_id} started_at={_utc_now_iso()}\n")
            log_fh.write(f"[remote] cwd={self.repo_root}\n")
            log_fh.write(f"[remote] command={command_preview}\n\n")
            log_fh.write(f"[remote] codex_exe={codex_exe}\n")
            log_fh.write(f"[remote] rg_exe={rg_exe or 'unavailable'}\n")
            log_fh.write(f"[remote] git_exe={git_exe or 'unavailable'}\n")
            log_fh.write(f"[remote] path_head={os.pathsep.join(str(env.get('PATH', '')).split(os.pathsep)[:6])}\n\n")
            log_fh.flush()
            try:
                client = self._get_app_server(codex_exe)
            except Exception as exc:
                log_fh.write(f"[remote][error] failed to start codex app-server: {type(exc).__name__}: {exc}\n")
                log_fh.flush()
                self._update_job(
                    job_id,
                    status="failed",
                    finished_at=_utc_now_iso(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            stderr_start = len(client.stderr_lines)
            with self._lock:
                self._procs[job_id] = client.proc
            self._update_job(job_id, pid=int(client.proc.pid or 0))

            def emit(event: dict[str, Any]) -> None:
                log_fh.write(f"[ts={_utc_now_iso()}] {json.dumps(event, ensure_ascii=False)}\n")
                log_fh.flush()

            def emit_trace(message: str) -> None:
                log_fh.write(f"[ts={_utc_now_iso()}] [remote] {message}\n")
                log_fh.flush()

            def process_notifications(
                notifications: list[dict[str, Any]],
                *,
                agent_chunks: list[str],
                state: dict[str, Any],
            ) -> None:
                for note in notifications:
                    method = str(note.get("method", "") or "")
                    params = note.get("params")
                    if not isinstance(params, dict):
                        continue
                    if method == "thread/started":
                        thread_id = str(params.get("threadId", "") or "").strip()
                        if not thread_id:
                            thread = params.get("thread")
                            if isinstance(thread, dict):
                                thread_id = str(thread.get("id", "") or "").strip()
                        if thread_id:
                            state["thread_id"] = thread_id
                            emit({"type": "thread.started", "thread_id": thread_id})
                        continue
                    if method == "turn/started":
                        emit({"type": "turn.started"})
                        continue
                    if method == "item/started":
                        item = _normalize_app_server_item(params.get("item") or {})
                        if item is not None:
                            emit({"type": "item.started", "item": item})
                        continue
                    if method == "item/completed":
                        item = _normalize_app_server_item(params.get("item") or {})
                        if item is not None:
                            emit({"type": "item.completed", "item": item})
                        continue
                    if method == "item/agentMessage/delta":
                        delta = str(params.get("delta", "") or "")
                        if delta:
                            agent_chunks.append(delta)
                        continue
                    if method == "turn/completed":
                        usage = _extract_turn_usage(note)
                        if usage:
                            state["usage"] = usage
                        state["completed"] = True
                        emit({"type": "turn.completed", "usage": usage})
                        continue
                    if method == "error":
                        emit_trace(f"app-server notification error: {json.dumps(params, ensure_ascii=False)}")
                        continue
                    if method in {"mcpServer/startupStatus/updated", "thread/status/changed", "account/updated"}:
                        continue
                    if method in {"item/commandExecution/outputDelta", "command/exec/outputDelta", "item/reasoning/summaryTextDelta", "item/reasoning/textDelta"}:
                        continue
                    emit_trace(f"notify {method}")

            state: dict[str, Any] = {"thread_id": str(job.get("resume_session_id", "") or "").strip(), "completed": False, "usage": {}}
            agent_chunks: list[str] = []
            exit_code = 1
            error_text = ""

            try:
                if not client.initialized:
                    init_id = client.send(
                        "initialize",
                        {
                            "clientInfo": {"name": "astra-remote", "title": "Astra Remote", "version": "1.0"},
                            "capabilities": {"experimentalApi": True},
                        },
                    )
                    init_snapshot = client.wait_for(init_id, timeout=15)
                    process_notifications(init_snapshot.get("notifications", []), agent_chunks=agent_chunks, state=state)
                    init_error = _response_error_text(init_snapshot)
                    if init_error:
                        raise RuntimeError(init_error)
                    client.initialized = True

                thread_id = str(job.get("resume_session_id", "") or "").strip()
                if not thread_id:
                    start_id = client.send(
                        "thread/start",
                        {
                            "cwd": str(self.repo_root),
                            "approvalPolicy": "never",
                            "sandbox": str(job.get("access_mode", "") or _default_access_mode()),
                            "model": str(job.get("model", "") or _default_model()),
                            "modelProvider": "openai",
                            "experimentalRawEvents": False,
                            "persistExtendedHistory": True,
                            "ephemeral": False,
                        },
                    )
                    start_snapshot = client.wait_for(start_id, timeout=20)
                    process_notifications(start_snapshot.get("notifications", []), agent_chunks=agent_chunks, state=state)
                    start_error = _response_error_text(start_snapshot)
                    if start_error:
                        raise RuntimeError(start_error)
                    thread_id = _extract_thread_id(start_snapshot)
                    if not thread_id:
                        raise RuntimeError("thread/start nao retornou thread id.")
                    state["thread_id"] = thread_id
                    self._update_job(job_id, session_id=thread_id)

                turn_params = {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": str(job.get("prompt", "") or ""), "text_elements": []}],
                    "model": str(job.get("model", "") or _default_model()),
                    "effort": str(job.get("reasoning_effort", "") or _default_reasoning_effort()),
                    "approvalPolicy": "never",
                    "sandboxPolicy": _access_mode_to_turn_sandbox_policy(str(job.get("access_mode", "") or _default_access_mode())),
                }
                turn_id = client.send("turn/start", turn_params)
                turn_snapshot = client.wait_for(turn_id, timeout=20)
                process_notifications(turn_snapshot.get("notifications", []), agent_chunks=agent_chunks, state=state)
                turn_error = _response_error_text(turn_snapshot)
                if turn_error:
                    if str(job.get("resume_session_id", "") or "").strip():
                        emit_trace(f"resume thread failed; criando nova thread: {turn_error}")
                        start_id = client.send(
                            "thread/start",
                            {
                                "cwd": str(self.repo_root),
                                "approvalPolicy": "never",
                                "sandbox": str(job.get("access_mode", "") or _default_access_mode()),
                                "model": str(job.get("model", "") or _default_model()),
                                "modelProvider": "openai",
                                "experimentalRawEvents": False,
                                "persistExtendedHistory": True,
                                "ephemeral": False,
                            },
                        )
                        start_snapshot = client.wait_for(start_id, timeout=20)
                        process_notifications(start_snapshot.get("notifications", []), agent_chunks=agent_chunks, state=state)
                        start_error = _response_error_text(start_snapshot)
                        if start_error:
                            raise RuntimeError(start_error)
                        thread_id = _extract_thread_id(start_snapshot)
                        if not thread_id:
                            raise RuntimeError("thread/start nao retornou thread id.")
                        state["thread_id"] = thread_id
                        self._update_job(job_id, session_id=thread_id)
                        turn_id = client.send(
                            "turn/start",
                            {
                                **turn_params,
                                "threadId": thread_id,
                            },
                        )
                        turn_snapshot = client.wait_for(turn_id, timeout=20)
                        process_notifications(turn_snapshot.get("notifications", []), agent_chunks=agent_chunks, state=state)
                        retry_error = _response_error_text(turn_snapshot)
                        if retry_error:
                            raise RuntimeError(retry_error)
                    else:
                        raise RuntimeError(turn_error)

                deadline = time.time() + 1800
                while time.time() < deadline:
                    snapshot = client.wait_for(None, timeout=5)
                    process_notifications(snapshot.get("notifications", []), agent_chunks=agent_chunks, state=state)
                    if state.get("completed"):
                        break
                    if snapshot.get("process_exit") is not None:
                        break

                if state.get("completed"):
                    exit_code = 0
                else:
                    error_text = "Codex app-server nao concluiu o turno dentro do tempo esperado."
                    emit_trace(error_text)

            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                emit_trace(f"error: {error_text}")
            finally:
                final_text = "".join(agent_chunks).strip()
                if final_text:
                    emit({"type": "item.completed", "item": {"type": "agent_message", "text": final_text}})
                for stderr_line in client.stderr_lines[stderr_start:][-12:]:
                    emit_trace(f"stderr: {stderr_line}")
                exit_code = 0 if exit_code == 0 else 1
                log_fh.write(f"\n[remote] job={job_id} exit_code={exit_code}\n")
                log_fh.flush()

        with self._lock:
            self._procs.pop(job_id, None)

        dirty_after = _git_status_map(self.repo_root)
        changed_files = sorted([path for path, status in dirty_after.items() if dirty_before.get(path) != status])
        if not changed_files:
            changed_files = sorted(set(dirty_after.keys()) - set(dirty_before.keys()))

        diff_stat = ""
        if not dirty_before:
            try:
                proc = subprocess.run(
                    ["git", "-C", str(self.repo_root), "diff", "--stat"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                diff_stat = str(proc.stdout or "").strip()
            except Exception:
                diff_stat = ""
        elif changed_files:
            diff_stat = "Repositorio ja tinha mudancas pendentes antes deste job; diff isolado nao esta disponivel."
        else:
            diff_stat = "Nenhuma mudanca nova detectada neste job."

        final_status = "completed" if int(exit_code) == 0 else "failed"
        if int(exit_code) != 0 and not error_text:
            error_text = f"Codex app-server terminou com exit code {exit_code}."

        self._update_job(
            job_id,
            status=final_status,
            finished_at=_utc_now_iso(),
            exit_code=int(exit_code),
            pid=0,
            changed_files=changed_files,
            diff_stat=diff_stat,
            error=error_text,
            session_id=str(state.get("thread_id", "") or job.get("session_id", "") or "").strip() if 'state' in locals() else str(job.get("session_id", "") or ""),
        )
        transcript = self.transcript(job_id)
        self._update_job(
            job_id,
            final_response=str(transcript.get("final_response", "") or "").strip(),
            session_id=str(transcript.get("session_id", "") or job.get("session_id", "") or "").strip(),
        )

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            proc = self._procs.get(str(job_id))
        if proc is None:
            raise RuntimeError("Job nao esta em execucao.")
        try:
            proc.terminate()
            with self._lock:
                if self._app_server is not None and self._app_server.proc.pid == proc.pid:
                    self._app_server = None
        except Exception as exc:
            raise RuntimeError(f"Falha ao cancelar job: {type(exc).__name__}: {exc}") from exc
        self._update_job(
            job_id,
            status="cancelling",
            error="Cancelamento solicitado pelo usuario.",
        )
        return self.get_job(job_id) or {}
