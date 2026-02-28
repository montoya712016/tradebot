# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import sys
import time
from typing import Iterable, Iterator, TextIO, TypeVar


T = TypeVar("T")


def progress(
    it: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    prefix: str = "prog",
    fallback: str = "eta",
    min_interval_s: float = 1.0,
) -> Iterable[T]:
    """
    Barra de progresso reutilizavel (tqdm opcional).

    Parameters
    ----------
    it:
        Iteravel de entrada.
    total:
        Total esperado (opcional).
    desc:
        Descricao exibida na barra/log.
    prefix:
        Prefixo usado no fallback sem tqdm (ex.: 'wf', 'corr').
    fallback:
        'eta' => imprime progresso periodico com ETA
        'print' => imprime uma linha simples no inicio
        'none' => sem fallback, retorna o iteravel original
    min_interval_s:
        Intervalo minimo entre logs no fallback 'eta'.
    """
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, total=total, desc=desc)
    except Exception:
        mode = str(fallback).strip().lower()
        if mode == "none":
            return it
        if mode == "print":
            if desc:
                print(f"[{prefix}] {desc} ...", flush=True)
            return it
        return _progress_eta_fallback(
            it,
            total=total,
            desc=desc,
            prefix=prefix,
            min_interval_s=float(min_interval_s),
        )


def _progress_eta_fallback(
    it: Iterable[T],
    *,
    total: int | None,
    desc: str,
    prefix: str,
    min_interval_s: float,
) -> Iterator[T]:
    t0 = time.perf_counter()
    last = 0.0
    n = int(total) if total is not None else None

    for i, x in enumerate(it, start=1):
        now = time.perf_counter()
        if now - last >= max(0.1, float(min_interval_s)):
            last = now
            if n:
                pct = 100.0 * i / max(1, n)
                avg = (now - t0) / max(1, i)
                eta = avg * max(0, n - i)
                print(
                    f"[{prefix}] {desc}: {i}/{n} ({pct:5.1f}%) ETA {eta/60.0:5.1f}m",
                    flush=True,
                )
            else:
                print(f"[{prefix}] {desc}: {i}", flush=True)
        yield x


def fmt_eta(seconds: float, *, pad_seconds: bool = False) -> str:
    try:
        sec_f = float(seconds)
    except Exception:
        sec_f = 0.0
    if (not math.isfinite(sec_f)) or sec_f <= 0.0:
        return "00s" if bool(pad_seconds) else "0s"
    sec = int(round(sec_f))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    if pad_seconds:
        return f"{s:02d}s"
    return f"{s}s"


def ascii_bar(done: int, total: int, *, width: int = 26, fill: str = "#", empty: str = "-") -> str:
    total_i = max(1, int(total))
    done_i = int(max(0, min(int(done), total_i)))
    width_i = max(1, int(width))
    n = int(round(width_i * (done_i / total_i)))
    return (str(fill) * n) + (str(empty) * (width_i - n))


class LineProgressPrinter:
    """
    Progress printer de linha unica (TTY) com fallback para logs periodicos em nao-TTY.

    Uso tipico:
        p = LineProgressPrinter(prefix="cache", total=100, stream=sys.stderr)
        p.update(17, current="BTCUSDT")
        p.close()
    """

    def __init__(
        self,
        *,
        prefix: str = "prog",
        total: int | None = None,
        width: int = 26,
        stream: TextIO | None = None,
        min_interval_s: float = 1.0,
        show_eta: bool = True,
        pad_seconds: bool = False,
    ) -> None:
        self.prefix = str(prefix)
        self.total = int(total) if total is not None else None
        self.width = int(max(1, int(width)))
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval_s = float(max(0.1, float(min_interval_s)))
        self.show_eta = bool(show_eta)
        self.pad_seconds = bool(pad_seconds)

        self._t0 = time.perf_counter()
        self._last_print_ts = 0.0
        self._last_len = 0
        self._used_tty = False

    def reset(self) -> None:
        self._t0 = time.perf_counter()
        self._last_print_ts = 0.0
        self._last_len = 0
        self._used_tty = False

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def _format_done_total(self, done: int) -> str:
        if self.total is None:
            return str(int(done))
        w = max(1, len(str(int(max(0, self.total)))))
        return f"{int(done):>{w}}/{int(self.total):<{w}}"

    def _calc_pct(self, done: int) -> float | None:
        if self.total is None:
            return None
        return 100.0 * float(int(done)) / max(1.0, float(self.total))

    def _calc_eta(self, done: int) -> float | None:
        if (self.total is None) or (int(done) <= 0):
            return 0.0 if self.total is not None else None
        elapsed = max(0.0, time.perf_counter() - self._t0)
        avg = elapsed / max(1, int(done))
        return avg * max(0, int(self.total) - int(done))

    def render(
        self,
        done: int,
        *,
        current: str = "",
        extra: str = "",
        pct: float | None = None,
        eta_seconds: float | None = None,
    ) -> str:
        done_i = int(max(0, int(done)))
        parts: list[str] = [f"[{self.prefix}]"]
        if self.total is not None:
            parts.append(f"[{ascii_bar(done_i, self.total, width=self.width)}]")
        parts.append(self._format_done_total(done_i))

        pct_v = self._calc_pct(done_i) if pct is None else float(pct)
        if pct_v is not None:
            parts.append(f"{pct_v:5.1f}%")

        if self.show_eta:
            eta_v = self._calc_eta(done_i) if eta_seconds is None else float(eta_seconds)
            if eta_v is not None:
                parts.append(f"ETA {fmt_eta(eta_v, pad_seconds=self.pad_seconds)}")

        line = " ".join(parts)
        cur = str(current or "").strip()
        ext = str(extra or "").strip()
        if cur:
            line += f" | {cur}"
        if ext:
            line += f" | {ext}"
        return line

    def update(
        self,
        done: int,
        *,
        current: str = "",
        extra: str = "",
        pct: float | None = None,
        eta_seconds: float | None = None,
        force: bool = False,
    ) -> None:
        line = self.render(
            int(done),
            current=current,
            extra=extra,
            pct=pct,
            eta_seconds=eta_seconds,
        )

        if self._is_tty():
            self._used_tty = True
            self.stream.write("\r" + line + (" " * max(0, self._last_len - len(line))))
            self.stream.flush()
            self._last_len = max(self._last_len, len(line))
            return

        now = time.perf_counter()
        if (not force) and (now - self._last_print_ts < self.min_interval_s):
            return
        self._last_print_ts = now
        print(line, flush=True)
        self._last_len = max(self._last_len, len(line))

    def close(self) -> None:
        if self._used_tty:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except Exception:
                pass


__all__ = ["progress", "fmt_eta", "ascii_bar", "LineProgressPrinter"]
