"""Keep the dashboard's own modules current between reruns (decision 0013).

Streamlit reruns app.py in the same Python process, and on Streamlit Community Cloud a push
updates the files without restarting it. app.py itself is read again, but modules it imported
earlier (common.*, hosting.live) stay as they were in memory, so new code calling a new
function in them fails until someone reboots the app. refresh() runs at the top of app.py:
when any project module's file has changed since this process last loaded it (or this process
has not checked it yet), it reloads all of them, settings first, so none keeps stale values
imported from another.

This module is imported once per process and never reloaded, so it remembers what it saw.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PACKAGES = ("common", "hosting", "dashboard")
SKIP = {__name__, "dashboard.app"}    # this module, and the app (Streamlit reads it again each run)
FIRST = ["common.settings", "common.study", "common.contracts"]   # what the others import from
_SEEN: dict[str, float] = {}          # module name -> file modification time when last (re)loaded
_BUSY = False                         # a reload runs module code, which must not start another


def _mtime(module) -> float | None:
    path = getattr(module, "__file__", None)
    return Path(path).stat().st_mtime if path and Path(path).exists() else None


def refresh(packages: tuple[str, ...] = PACKAGES) -> list[str]:
    """Reload every loaded project module if any of them changed on disk. Returns the names reloaded."""
    global _BUSY
    if _BUSY:
        return []
    names = [n for n in list(sys.modules) if n.split(".")[0] in packages and n not in SKIP
             and _mtime(sys.modules[n]) is not None]
    if all(_SEEN.get(n) == _mtime(sys.modules[n]) for n in names):
        return []
    order = [n for n in FIRST if n in names] + sorted(n for n in names if n not in FIRST)
    _BUSY = True
    try:
        for name in order:
            importlib.reload(sys.modules[name])
            _SEEN[name] = _mtime(sys.modules[name])
    finally:
        _BUSY = False
    return order
