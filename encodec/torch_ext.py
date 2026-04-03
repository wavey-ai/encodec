from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load

_LOCK = threading.Lock()
_MODULE = None
_LOAD_ERROR: Optional[Exception] = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _env_bool("ENCODEC_TORCH_EXT", False)


def load_extension():
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR

    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        if _LOAD_ERROR is not None:
            raise _LOAD_ERROR

        repo_root = Path(__file__).resolve().parents[1]
        source = repo_root / "native" / "encodec_torch_ext" / "encodec_torch_ext.cpp"
        build_dir = repo_root / "native" / "encodec_torch_ext" / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"

        try:
            _MODULE = load(
                name="encodec_torch_ext",
                sources=[str(source)],
                build_directory=str(build_dir),
                extra_cflags=["-O3", "-std=c++17"],
                verbose=_env_bool("ENCODEC_TORCH_EXT_VERBOSE", False),
            )
            return _MODULE
        except Exception as exc:  # pragma: no cover - build failures are environment-specific.
            _LOAD_ERROR = exc
            raise
