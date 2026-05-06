#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from telemetry_factorization import build_vocab, encode_with_vocab, factors_from_compact_label


def _sorted_chunk_paths(glob_pattern: str) -> list[Path]:
    return sorted(Path().glob(glob_pattern))


def _load_parallel_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        observations = np.asarray(data["expert_observations"], dtype=np.float32)
        actions = np.asarray(data["expert_actions"], dtype=np.float32)
        telemetry = np.asarray(
            data["telemetry_labels_T_at_same_timestep_as_observation"],
            dtype=object,
        )
    return observations, actions, telemetry


def _concat_parallel_arrays(
    paths: Iterable[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    telemetry_rows: list[np.ndarray] = []
    for path in paths:
        obs, actions, telemetry = _load_parallel_arrays(path)
        obs_rows.append(obs)
        action_rows.append(actions)
        telemetry_rows.append(telemetry)
    if not obs_rows:
        raise FileNotFoundError("No rollout files found.")
    return (
        np.concatenate(obs_rows, axis=0),
        np.concatenate(action_rows, axis=0),
        np.concatenate(telemetry_rows, axis=0),
    )


def _factorize_telemetry_labels(telemetry_labels: np.ndarray) -> dict[str, object]:
    telemetry_as_strings = [str(x) for x in telemetry_labels.reshape(-1).tolist()]
    factors = [factors_from_compact_label(s) for s in telemetry_as_strings]

    speed_vocab, speed_id_to_token = build_vocab([f.speed for f in factors])
    long_vocab, long_id_to_token = build_vocab([f.longitudinal for f in factors])
    steer_vocab, steer_id_to_token = build_vocab([f.steering for f in factors])
    route_vocab, route_id_to_token = build_vocab([f.route_bin for f in factors])

    return {
        "telemetry_speed_id": np.asarray(
            [encode_with_vocab(f.speed, speed_vocab) for f in factors], dtype=np.int32
        ),
        "telemetry_longitudinal_id": np.asarray(
            [encode_with_vocab(f.longitudinal, long_vocab) for f in factors], dtype=np.int32
        ),
        "telemetry_steering_id": np.asarray(
            [encode_with_vocab(f.steering, steer_vocab) for f in factors], dtype=np.int32
        ),
        "telemetry_route_bin_id": np.asarray(
            [encode_with_vocab(f.route_bin, route_vocab) for f in factors], dtype=np.int32
        ),
        "telemetry_overspeed": np.asarray([f.overspeed for f in factors], dtype=np.float32),
        "telemetry_off_lane": np.asarray([f.off_lane for f in factors], dtype=np.float32),
        "speed_token_to_id": speed_vocab,
        "speed_id_to_token": speed_id_to_token,
        "longitudinal_token_to_id": long_vocab,
        "longitudinal_id_to_token": long_id_to_token,
        "steering_token_to_id": steer_vocab,
        "steering_id_to_token": steer_id_to_token,
        "route_bin_token_to_id": route_vocab,
        "route_bin_id_to_token": route_id_to_token,
    }


def _build_aligned_training_dataset(
    expert_observations: np.ndarray,
    expert_actions: np.ndarray,
    telemetry_labels_T_at_same_timestep_as_observation: np.ndarray,
) -> dict[str, np.ndarray]:
    n = int(expert_observations.shape[0])
    if n < 1:
        raise ValueError("Need at least 1 timestep.")
    if expert_actions.shape[0] != n:
        raise ValueError("expert_observations and expert_actions length mismatch.")
    if telemetry_labels_T_at_same_timestep_as_observation.shape[0] != n:
        raise ValueError("expert_observations and telemetry_labels length mismatch.")
    return {
        "observation_input_time_t": expert_observations.copy(),
        "telemetry_label_same_timestep_as_observation": telemetry_labels_T_at_same_timestep_as_observation.copy(),
        "expert_action_target_time_t": expert_actions.copy(),
    }


def _build_offset_training_dataset(
    expert_observations: np.ndarray,
    expert_actions: np.ndarray,
    telemetry_labels_T_at_same_timestep_as_observation: np.ndarray,
) -> dict[str, np.ndarray]:
    n = int(expert_observations.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 timesteps for offset dataset, got {n}.")
    if expert_actions.shape[0] != n:
        raise ValueError("expert_observations and expert_actions length mismatch.")
    if telemetry_labels_T_at_same_timestep_as_observation.shape[0] != n:
        raise ValueError("expert_observations and telemetry_labels length mismatch.")
    return {
        "observation_input_time_t": expert_observations[:-1].copy(),
        "telemetry_label_T_observation_time_t_plus_one": telemetry_labels_T_at_same_timestep_as_observation[
            1:
        ].copy(),
        "expert_action_target_time_t": expert_actions[:-1].copy(),
    }


def main() -> None:
    input_glob = os.environ.get(
        "METADRIVE_TRAIN_INPUT_GLOB",
        "data/distillation/drive_main_session_part*.npz",
    )
    single_fallback = Path(
        os.environ.get(
            "METADRIVE_TRAIN_INPUT_SINGLE",
            "data/distillation/drive_main_session.npz",
        )
    )
    out_npz = Path(os.environ.get("METADRIVE_TRAIN_OUT_NPZ", "data/distillation/train_ready.npz"))
    out_vocab = Path(
        os.environ.get(
            "METADRIVE_TRAIN_OUT_VOCAB_JSON",
            "data/distillation/telemetry_vocab.json",
        )
    )

    chunk_paths = _sorted_chunk_paths(input_glob)
    if chunk_paths:
        source_paths = chunk_paths
        print(f"Using {len(source_paths)} chunk files from glob: {input_glob}", flush=True)
    elif single_fallback.exists():
        source_paths = [single_fallback]
        print(f"No chunk files found; using single file: {single_fallback}", flush=True)
    else:
        raise FileNotFoundError(
            "No training input files found. Expected chunk files matching "
            f"{input_glob} or fallback file {single_fallback}."
        )

    expert_observations, expert_actions, telemetry_labels = _concat_parallel_arrays(source_paths)
    align = os.environ.get(
        "METADRIVE_TRAIN_TELEMETRY_ALIGN",
        "obs_t_telem_t_plus_1",
    ).strip().lower()
    if align in ("same", "aligned", "same_timestep"):
        offset = _build_aligned_training_dataset(
            expert_observations,
            expert_actions,
            telemetry_labels,
        )
        telemetry_selected = offset["telemetry_label_same_timestep_as_observation"]
        schema_txt = (
            "input: o_t + T_t (same index); target: a_t"
        )
    else:
        offset = _build_offset_training_dataset(
            expert_observations,
            expert_actions,
            telemetry_labels,
        )
        telemetry_selected = offset["telemetry_label_T_observation_time_t_plus_one"]
        schema_txt = "input: o_t + T_{t+1}; target: a_t"

    fac = _factorize_telemetry_labels(telemetry_selected)

    pack: dict[str, object] = {
        "expert_observations": expert_observations,
        "expert_actions": expert_actions,
        "telemetry_labels_T_at_same_timestep_as_observation": telemetry_labels,
        "observation_input_time_t": offset["observation_input_time_t"],
        "telemetry_label_selected": telemetry_selected,
        "telemetry_speed_id": fac["telemetry_speed_id"],
        "telemetry_longitudinal_id": fac["telemetry_longitudinal_id"],
        "telemetry_steering_id": fac["telemetry_steering_id"],
        "telemetry_route_bin_id": fac["telemetry_route_bin_id"],
        "telemetry_overspeed": fac["telemetry_overspeed"],
        "telemetry_off_lane": fac["telemetry_off_lane"],
        "expert_action_target_time_t": offset["expert_action_target_time_t"],
        "telemetry_speed_vocab": np.asarray(fac["speed_id_to_token"], dtype=object),
        "telemetry_longitudinal_vocab": np.asarray(fac["longitudinal_id_to_token"], dtype=object),
        "telemetry_steering_vocab": np.asarray(fac["steering_id_to_token"], dtype=object),
        "telemetry_route_bin_vocab": np.asarray(fac["route_bin_id_to_token"], dtype=object),
        "dataset_schema_description": np.array(schema_txt, dtype=object),
        "telemetry_align_mode": np.array(align, dtype=object),
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **pack)

    out_vocab.write_text(
        json.dumps(
            {
                "speed_id_to_token": fac["speed_id_to_token"],
                "speed_token_to_id": fac["speed_token_to_id"],
                "longitudinal_id_to_token": fac["longitudinal_id_to_token"],
                "longitudinal_token_to_id": fac["longitudinal_token_to_id"],
                "steering_id_to_token": fac["steering_id_to_token"],
                "steering_token_to_id": fac["steering_token_to_id"],
                "route_bin_id_to_token": fac["route_bin_id_to_token"],
                "route_bin_token_to_id": fac["route_bin_token_to_id"],
                "schema": "factorized_telemetry_v1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    num_parallel_rows = int(expert_observations.shape[0])
    num_train_rows = int(offset["observation_input_time_t"].shape[0])
    print(f"Wrote train dataset: {out_npz}", flush=True)
    print(f"Wrote telemetry vocab: {out_vocab}", flush=True)
    print(
        f"Rows -> parallel: {num_parallel_rows}, training_rows: {num_train_rows}, "
        f"speed_vocab={len(fac['speed_id_to_token'])} "
        f"long_vocab={len(fac['longitudinal_id_to_token'])} "
        f"steer_vocab={len(fac['steering_id_to_token'])} "
        f"route_vocab={len(fac['route_bin_id_to_token'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
