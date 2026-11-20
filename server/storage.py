"""Atomic JSON storage helpers for settings, jobs, and run histories."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from server.config import (
    JOBS_FILE,
    RUNS_FILE,
    SETTINGS_FILE,
    VOICE_LIBRARY_FILE,
    VOICE_RUNS_FILE,
    ensure_dirs,
)


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_settings() -> dict[str, Any]:
    return read_json(
        SETTINGS_FILE,
        {"colab_fallback_enabled": False},
    )


def save_settings(settings: dict[str, Any]) -> None:
    write_json_atomic(SETTINGS_FILE, settings)


def load_jobs() -> list[dict[str, Any]]:
    return read_json(JOBS_FILE, [])


def save_jobs(jobs: list[dict[str, Any]]) -> None:
    write_json_atomic(JOBS_FILE, jobs)


def load_voice_runs() -> list[dict[str, Any]]:
    return read_json(VOICE_RUNS_FILE, [])


def save_voice_runs(runs: list[dict[str, Any]]) -> None:
    write_json_atomic(VOICE_RUNS_FILE, runs)


def load_voice_library() -> list[dict[str, Any]]:
    return read_json(VOICE_LIBRARY_FILE, [])


def save_voice_library(items: list[dict[str, Any]]) -> None:
    write_json_atomic(VOICE_LIBRARY_FILE, items)


def load_video_runs() -> list[dict[str, Any]]:
    return read_json(RUNS_FILE, [])


def save_video_runs(runs: list[dict[str, Any]]) -> None:
    write_json_atomic(RUNS_FILE, runs)
