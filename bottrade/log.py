"""Console logging setup — compact, aligned, one line per event."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.agent = record.name.replace("bottrade.", "").ljust(10)[:10]
        return super().format(record)


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    root = logging.getLogger("bottrade")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.propagate = False

    fmt = _Formatter("%(asctime)s %(levelname)-5s %(agent)s %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(_Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
