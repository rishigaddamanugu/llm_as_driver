
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

from distillation_collect import (
    append_offset_training_rows,
    append_raw_rollout_blocks,
    build_offset_training_dataset,
    save_rollout_npz,
)
from expert_model import BundledExpert
from telemetry_labels import auto_label_telemetry, compact_telemetry_label_from_tags


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _flush_every_from_env() -> int:
    raw = os.environ.get("METADRIVE_DISTILLATION_FLUSH_EVERY", "500").strip()
    try:
        n = int(raw)
    except ValueError:
        return 500
    return max(0, n)


def _part_index_from_filename(stem: str, filename: str) -> int | None:
    m = re.match(rf"{re.escape(stem)}_part(\d+)\.npz\Z", filename)
    return int(m.group(1)) if m else None


def _next_part_index(parent: Path, stem: str) -> int:
    mx = -1
    if not parent.is_dir():
        return 0
    for child in parent.iterdir():
        idx = _part_index_from_filename(stem, child.name)
        if idx is not None:
            mx = max(mx, idx)
    return mx + 1


def _chunk_timestep_end(path: Path) -> int | None:
    try:
        with np.load(path) as z:
            st = int(z["global_timestep_start"])
            n = int(z["num_rows"])
            return st + n
    except (OSError, ValueError, KeyError):
        return None


def _primary_rollout_length(primary_npz: Path) -> int | None:
    if not primary_npz.exists():
        return None
    try:
        with np.load(primary_npz) as z:
            return int(z["expert_observations"].shape[0])
    except (OSError, ValueError, KeyError):
        return None


def _next_global_timestep_from_npz(parent: Path, stem: str, primary_npz: Path) -> int:
    end = 0
    if parent.is_dir():
        for child in parent.glob(f"{stem}_part*.npz"):
            e = _chunk_timestep_end(child)
            if e is not None:
                end = max(end, e)
    n_primary = _primary_rollout_length(primary_npz)
    if n_primary is not None:
        end = max(end, n_primary)
    return end


def _max_index_in_text_tail(path: Path, label: str) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 262_144))
        tail = f.read().decode("utf-8", errors="replace")
    pat = re.compile(rf"BEGIN {re.escape(label)} (\d+)")
    matches = [int(m) for m in pat.findall(tail)]
    return max(matches) if matches else None


def _next_global_row_from_offset_text(offset_path: Path) -> int:
    m = _max_index_in_text_tail(offset_path, "training_row_index")
    return (m + 1) if m is not None else 0


def _next_global_timestep_from_raw_text(raw_path: Path) -> int:
    m = _max_index_in_text_tail(raw_path, "timestep_index")
    return (m + 1) if m is not None else 0


def _load_bridge_from_last_npz(
    parent: Path, stem: str, primary_npz: Path
) -> tuple[np.ndarray | None, np.ndarray | None]:
    best_end = -1
    best_path: Path | None = None
    if parent.is_dir():
        for child in parent.glob(f"{stem}_part*.npz"):
            e = _chunk_timestep_end(child)
            if e is not None and e > best_end:
                best_end = e
                best_path = child
    n_primary = _primary_rollout_length(primary_npz)
    if n_primary is not None and n_primary > 0:
        if n_primary > best_end:
            best_path = primary_npz
    if best_path is None:
        return None, None
    try:
        with np.load(best_path) as z:
            obs = z["expert_observations"]
            act = z["expert_actions"]
            if obs.shape[0] == 0:
                return None, None
            return (
                np.asarray(obs[-1], dtype=np.float32).reshape(-1),
                np.asarray(act[-1], dtype=np.float32).reshape(-1),
            )
    except (OSError, ValueError, KeyError):
        return None, None


class KeyboardDistillationRecorder:

    def __init__(
        self,
        *,
        output_npz: Path,
        deterministic_expert: bool,
        flush_every: int,
    ) -> None:
        self._output_npz = Path(output_npz)
        self._deterministic_expert = deterministic_expert
        self._flush_every = flush_every

        stem = self._output_npz.stem
        parent = self._output_npz.parent
        self._raw_path = parent / f"{stem}_raw.txt"
        self._offset_path = parent / f"{stem}_offset_training.txt"

        self._expert = BundledExpert()
        self._obs_rows: list[np.ndarray] = []
        self._action_rows: list[np.ndarray] = []
        self._telemetry_labels: list[str] = []

        self._part: int = 0
        self._global_t: int = 0
        self._global_row: int = 0
        self._bridge_obs: np.ndarray | None = None
        self._bridge_action: np.ndarray | None = None

        self._resume_append_state()

    def _resume_append_state(self) -> None:
        stem = self._output_npz.stem
        parent = self._output_npz.parent
        self._part = _next_part_index(parent, stem)
        t_npz = _next_global_timestep_from_npz(parent, stem, self._output_npz)
        t_raw = _next_global_timestep_from_raw_text(self._raw_path)
        self._global_t = max(t_npz, t_raw)
        self._global_row = _next_global_row_from_offset_text(self._offset_path)
        self._bridge_obs, self._bridge_action = _load_bridge_from_last_npz(
            parent, stem, self._output_npz
        )
        print(
            f"Distillation resume: next chunk index _part={self._part:06d}, "
            f"next_global_timestep={self._global_t}, next_training_row={self._global_row}, "
            f"bridge_loaded={'yes' if self._bridge_obs is not None else 'no'}.\n",
            flush=True,
        )

    @classmethod
    def from_env_defaults(cls) -> KeyboardDistillationRecorder:
        out = os.environ.get("METADRIVE_COLLECT_OUT", "data/distillation/drive_main_session.npz")
        return cls(
            output_npz=Path(out),
            deterministic_expert=_env_true("METADRIVE_EXPERT_DETERMINISTIC"),
            flush_every=_flush_every_from_env(),
        )

    @property
    def output_npz(self) -> Path:
        return self._output_npz

    @property
    def deterministic_expert(self) -> bool:
        return self._deterministic_expert

    @property
    def flush_every(self) -> int:
        return self._flush_every

    def record_before_step(self, env: MetaDriveEnv) -> None:
        ego = env.agent
        tags = auto_label_telemetry(ego, None)
        label = compact_telemetry_label_from_tags(tags)
        action, obs = self._expert.predict(
            ego,
            deterministic=self._deterministic_expert,
            need_obs=True,
        )
        self._obs_rows.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        self._action_rows.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self._telemetry_labels.append(label)

        if self._flush_every > 0 and len(self._obs_rows) >= self._flush_every:
            self._flush()

    def _allocate_unique_npz_path(self) -> Path:
        parent = self._output_npz.parent
        stem = self._output_npz.stem
        primary = self._output_npz
        if not primary.exists():
            return primary
        k = self._part
        while True:
            p = parent / f"{stem}_part{k:06d}.npz"
            if not p.exists():
                return p
            k += 1

    def _next_unused_part_path(self) -> Path:
        parent = self._output_npz.parent
        stem = self._output_npz.stem
        k = self._part
        while True:
            p = parent / f"{stem}_part{k:06d}.npz"
            if not p.exists():
                self._part = k + 1
                return p
            k += 1

    def _append_chunk_text_and_globals(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        telem: np.ndarray,
        chunk_start: int,
    ) -> None:
        n = int(obs.shape[0])
        append_raw_rollout_blocks(
            self._raw_path,
            expert_observations=obs,
            expert_actions=act,
            telemetry_labels_T_at_same_timestep_as_observation=telem,
            global_timestep_index_start=chunk_start,
        )
        self._global_t += n

        if self._bridge_obs is not None and self._bridge_action is not None:
            append_offset_training_rows(
                self._offset_path,
                observation_input_time_t=self._bridge_obs.reshape(1, -1),
                telemetry_label_T_observation_time_t_plus_one=np.array([telem[0]], dtype=object),
                expert_action_target_time_t=self._bridge_action.reshape(1, -1),
                global_training_row_index_start=self._global_row,
                time_index_t_start=chunk_start - 1,
            )
            self._global_row += 1

        if n >= 2:
            off = build_offset_training_dataset(obs, act, telem)
            append_offset_training_rows(
                self._offset_path,
                observation_input_time_t=off["observation_input_time_t"],
                telemetry_label_T_observation_time_t_plus_one=off[
                    "telemetry_label_T_observation_time_t_plus_one"
                ],
                expert_action_target_time_t=off["expert_action_target_time_t"],
                global_training_row_index_start=self._global_row,
                time_index_t_start=chunk_start,
            )
            self._global_row += n - 1

        self._bridge_obs = np.asarray(obs[-1], dtype=np.float32).reshape(-1)
        self._bridge_action = np.asarray(act[-1], dtype=np.float32).reshape(-1)

    def _flush(self) -> None:
        n = len(self._obs_rows)
        if n == 0:
            return

        chunk_start = self._global_t
        obs = np.stack(self._obs_rows, axis=0)
        act = np.stack(self._action_rows, axis=0)
        telem = np.asarray(self._telemetry_labels, dtype=object)

        part_path = self._next_unused_part_path()
        part_idx = self._part - 1
        np.savez_compressed(
            part_path,
            expert_observations=obs,
            expert_actions=act,
            telemetry_labels_T_at_same_timestep_as_observation=telem,
            global_timestep_start=np.int64(chunk_start),
            num_rows=np.int64(n),
            chunk_part_index=np.int64(part_idx),
        )

        self._append_chunk_text_and_globals(obs, act, telem, chunk_start)

        self._obs_rows.clear()
        self._action_rows.clear()
        self._telemetry_labels.clear()

        print(
            f"Distillation flush: wrote {n} rows (chunk {part_path.name}; "
            f"next global timestep {self._global_t}).",
            flush=True,
        )

    def save(self) -> None:
        if self._flush_every == 0:
            if not self._obs_rows:
                return
            obs = np.stack(self._obs_rows, axis=0)
            act = np.stack(self._action_rows, axis=0)
            telem = np.asarray(self._telemetry_labels, dtype=object)
            chunk_start = self._global_t
            out_path = self._allocate_unique_npz_path()
            save_rollout_npz(
                out_path,
                expert_observations=obs,
                expert_actions=act,
                telemetry_labels_T_at_same_timestep_as_observation=telem,
                include_offset_training_arrays=True,
                write_human_readable=False,
            )
            self._append_chunk_text_and_globals(obs, act, telem, chunk_start)
            print(f"Saved distillation rollout: {out_path}", flush=True)
            return

        if self._obs_rows:
            self._flush()


@contextmanager
def keyboard_distillation_session(
    enabled: bool,
) -> Generator[KeyboardDistillationRecorder | None, None, None]:
    if not enabled:
        yield None
        return
    recorder = KeyboardDistillationRecorder.from_env_defaults()
    out_dir = recorder.output_npz.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fe = recorder.flush_every
    if fe > 0:
        print(
            f"Distillation output directory:\n  {out_dir}\n"
            f"  First automatic chunk + text append after {fe} steps "
            f"(then every {fe} steps). Remainder on exit (Ctrl+C). "
            f"Deterministic expert: {recorder.deterministic_expert}. "
            "Override interval: METADRIVE_DISTILLATION_FLUSH_EVERY=N.\n",
            flush=True,
        )
    else:
        print(
            f"Distillation output directory:\n  {out_dir}\n"
            f"  Single .npz + text at exit only ({recorder.output_npz.name}); "
            "METADRIVE_DISTILLATION_FLUSH_EVERY>0 enables periodic chunks. "
            f"Deterministic expert: {recorder.deterministic_expert}.\n",
            flush=True,
        )
    try:
        yield recorder
    finally:
        recorder.save()
