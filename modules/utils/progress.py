# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import time
import math


class ProgressPrinter:
    """
    Simple progress bar with ETA. Uses in-place updates on TTY; otherwise prints periodically.
    """

    def __init__(
        self,
        *,
        prefix: str,
        total: int,
        width: int = 26,
        stream=None,
        print_every_s: float = 5.0,
        force_inplace: bool = False,
    ) -> None:
        self.prefix = str(prefix)
        self.total = max(1, int(total))
        self.width = max(5, int(width))
        self.stream = stream or sys.stderr
        self.print_every_s = float(max(0.1, print_every_s))
        self.force_inplace = bool(force_inplace)
        self._t0 = time.perf_counter()
        self._last_len = 0
        self._last_print_ts = 0.0

    def _bar(self, done: int) -> str:
        done = int(max(0, min(done, self.total)))
        n = int(round(self.width * (done / float(self.total))))
        return ("#" * n) + ("-" * (self.width - n))

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        if (not math.isfinite(seconds)) or seconds <= 0:
            return "0s"
        sec = int(round(seconds))
        h = sec // 3600
        m = (sec % 3600) // 60
        s2 = sec % 60
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s2:02d}s"
        return f"{s2}s"

    def update(self, done: int, *, suffix: str | None = None) -> None:
        done = int(max(0, min(done, self.total)))
        pct = 100.0 * done / float(self.total)
        elapsed = time.perf_counter() - self._t0
        rate = elapsed / max(1, done)
        eta = rate * max(0, self.total - done)
        line = f"{self.prefix} [{self._bar(done)}] {done:>3}/{self.total:<3} {pct:5.1f}% ETA {self._fmt_eta(eta)}"
        if suffix:
            line = f"{line} | {suffix}"
        if self.force_inplace or self.stream.isatty():
            self.stream.write("\r" + line + (" " * max(0, self._last_len - len(line))))
            self.stream.flush()
        else:
            now = time.perf_counter()
            if now - self._last_print_ts >= self.print_every_s:
                print(line, flush=True)
                self._last_print_ts = now
        self._last_len = max(self._last_len, len(line))

    def close(self) -> None:
        if self.force_inplace or self.stream.isatty():
            self.stream.write("\n")
            self.stream.flush()
