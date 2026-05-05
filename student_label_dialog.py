
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_WORKER = Path(__file__).resolve().parent / "student_label_dialog_worker.py"


def _env_for_worker_subprocess() -> dict[str, str]:
    env = os.environ.copy()
    v = env.get("PYTHONUTF8")
    if v is not None and str(v) not in ("0", "1"):
        env.pop("PYTHONUTF8", None)
    return env


def prompt_compact_label(
    *,
    initial: str,
    vocab_json_path: Path,
    title: str = "Telemetry label override",
) -> Optional[str]:
    if not _WORKER.is_file():
        print(f"[student] Missing worker script: {_WORKER}", flush=True)
        return None
    if not vocab_json_path.is_file():
        print(f"[student] Missing vocab file: {vocab_json_path}", flush=True)
        return None

    payload = json.dumps(
        {
            "vocab_path": str(vocab_json_path.resolve()),
            "initial": initial,
            "title": title,
        },
        ensure_ascii=False,
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(_WORKER)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
            env=_env_for_worker_subprocess(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[student] Label dialog subprocess failed: {e}", flush=True)
        return None

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if err:
            print(err, flush=True)
        return None

    raw_out = (proc.stdout or "").strip()
    if not raw_out:
        return None
    last_line = raw_out.splitlines()[-1]
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        print(f"[student] Bad dialog JSON: {raw_out!r}", flush=True)
        return None

    if not data.get("ok"):
        return None
    label = data.get("label")
    return str(label) if label is not None else None
