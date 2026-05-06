#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import launch_metadrive_demo as render_launch
from metadrive_config import build_metadrive_config
from expert_model import BundledExpert
from metadrive.envs.metadrive_env import MetaDriveEnv
from telemetry_labels import auto_label_telemetry, compact_telemetry_label_from_tags


def _environment_flag_true(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _comma_separated_observation_vector(observation_vector: np.ndarray) -> str:
    return ",".join(f"{float(value):.8g}" for value in np.asarray(observation_vector).reshape(-1))


def write_rollout_human_readable_text_files(
    numpy_archive_path: Path,
    *,
    expert_observations: np.ndarray,
    expert_actions: np.ndarray,
    telemetry_labels_T_at_same_timestep_as_observation: np.ndarray,
) -> Tuple[Path, Path]:
    output_directory = numpy_archive_path.parent
    stem = numpy_archive_path.stem
    raw_text_path = output_directory / f"{stem}_raw.txt"
    offset_text_path = output_directory / f"{stem}_offset_training.txt"

    number_of_timesteps = int(expert_observations.shape[0])

    with raw_text_path.open("w", encoding="utf-8") as raw_file:
        raw_file.write(
            "# RAW ROLLOUT (parallel arrays, one timestep per block).\n"
            "# Index i logs expert_observations[i], expert_actions[i], and the label string at step i.\n"
            "# Training uses (o_t, label[t+1], a_t): label index t+1 is the oracle T(obs_{t+1}).\n\n"
        )
        for time_index in range(number_of_timesteps):
            steer = float(expert_actions[time_index, 0])
            throttle_brake = float(expert_actions[time_index, 1])
            label_string = str(telemetry_labels_T_at_same_timestep_as_observation[time_index])
            raw_file.write("=" * 80 + "\n")
            raw_file.write(f"BEGIN timestep_index {time_index}\n")
            raw_file.write(f"expert_actions_steer_component: {steer}\n")
            raw_file.write(f"expert_actions_throttle_brake_component: {throttle_brake}\n")
            raw_file.write(
                "telemetry_labels_T_at_same_timestep_as_observation:\n"
                f"  {label_string}\n"
            )
            raw_file.write(
                f"expert_observations_vector_length: {expert_observations.shape[1]}\n"
                "expert_observations_comma_separated:\n"
            )
            raw_file.write(_comma_separated_observation_vector(expert_observations[time_index]))
            raw_file.write("\n\n")

    with offset_text_path.open("w", encoding="utf-8") as offset_file:
        offset_file.write(
            "# OFFSET TRAINING ROWS (one block per training row).\n"
            "# Row r: observation_input_time_t[r] == expert_observations[r],\n"
            "#   telemetry_label_T_observation_time_t_plus_one[r] == T(observation_{r+1}),\n"
            "#   expert_action_target_time_t[r] == expert_actions[r].\n\n"
        )
        if number_of_timesteps < 2:
            offset_file.write(
                "# No offset rows: need at least two timesteps (got "
                f"{number_of_timesteps}).\n"
            )
        else:
            offset_arrays = build_offset_training_dataset(
                expert_observations,
                expert_actions,
                telemetry_labels_T_at_same_timestep_as_observation,
            )
            observation_input_time_t = offset_arrays["observation_input_time_t"]
            telemetry_label_T_observation_time_t_plus_one = offset_arrays[
                "telemetry_label_T_observation_time_t_plus_one"
            ]
            expert_action_target_time_t = offset_arrays["expert_action_target_time_t"]
            number_of_training_rows = int(observation_input_time_t.shape[0])

            for training_row_index in range(number_of_training_rows):
                time_index_t = training_row_index
                steer = float(expert_action_target_time_t[training_row_index, 0])
                throttle_brake = float(expert_action_target_time_t[training_row_index, 1])
                label_next = str(telemetry_label_T_observation_time_t_plus_one[training_row_index])
                offset_file.write("=" * 80 + "\n")
                offset_file.write(f"BEGIN training_row_index {training_row_index}\n")
                offset_file.write(f"time_index_t: {time_index_t}\n")
                offset_file.write(
                    "telemetry_label_T_observation_time_t_plus_one:\n"
                    f"  {label_next}\n"
                )
                offset_file.write(f"expert_action_target_time_t_steer_component: {steer}\n")
                offset_file.write(
                    f"expert_action_target_time_t_throttle_brake_component: {throttle_brake}\n"
                )
                offset_file.write(
                    f"observation_input_time_t_vector_length: {observation_input_time_t.shape[1]}\n"
                    "observation_input_time_t_comma_separated:\n"
                )
                offset_file.write(
                    _comma_separated_observation_vector(observation_input_time_t[training_row_index])
                )
                offset_file.write("\n\n")

    return raw_text_path, offset_text_path


def _raw_header() -> str:
    return (
        "# RAW ROLLOUT (parallel arrays, one timestep per block).\n"
        "# Index i logs expert_observations[i], expert_actions[i], and the label string at step i.\n"
        "# Training pairs (o_t, label[t+1], a_t) — oracle for the next observation.\n\n"
    )


def _offset_header() -> str:
    return (
        "# OFFSET TRAINING ROWS (one block per training row).\n"
        "# Row r: observation_input_time_t[r] == expert_observations[r],\n"
        "#   telemetry_label_T_observation_time_t_plus_one[r] == T(observation_{r+1}),\n"
        "#   expert_action_target_time_t[r] == expert_actions[r].\n\n"
    )


def append_raw_rollout_blocks(
    raw_text_path: Path,
    *,
    expert_observations: np.ndarray,
    expert_actions: np.ndarray,
    telemetry_labels_T_at_same_timestep_as_observation: np.ndarray,
    global_timestep_index_start: int,
) -> None:
    raw_text_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(expert_observations.shape[0])
    mode = "a" if raw_text_path.exists() and raw_text_path.stat().st_size > 0 else "w"
    with raw_text_path.open(mode, encoding="utf-8") as raw_file:
        if mode == "w":
            raw_file.write(_raw_header())
        for j in range(n):
            time_index = global_timestep_index_start + j
            steer = float(expert_actions[j, 0])
            throttle_brake = float(expert_actions[j, 1])
            label_string = str(telemetry_labels_T_at_same_timestep_as_observation[j])
            raw_file.write("=" * 80 + "\n")
            raw_file.write(f"BEGIN timestep_index {time_index}\n")
            raw_file.write(f"expert_actions_steer_component: {steer}\n")
            raw_file.write(f"expert_actions_throttle_brake_component: {throttle_brake}\n")
            raw_file.write(
                "telemetry_labels_T_at_same_timestep_as_observation:\n"
                f"  {label_string}\n"
            )
            raw_file.write(
                f"expert_observations_vector_length: {expert_observations.shape[1]}\n"
                "expert_observations_comma_separated:\n"
            )
            raw_file.write(_comma_separated_observation_vector(expert_observations[j]))
            raw_file.write("\n\n")


def append_offset_training_rows(
    offset_text_path: Path,
    *,
    observation_input_time_t: np.ndarray,
    telemetry_label_T_observation_time_t_plus_one: np.ndarray,
    expert_action_target_time_t: np.ndarray,
    global_training_row_index_start: int,
    time_index_t_start: int,
) -> None:
    offset_text_path.parent.mkdir(parents=True, exist_ok=True)
    k = int(observation_input_time_t.shape[0])
    if k == 0:
        return
    mode = "a" if offset_text_path.exists() and offset_text_path.stat().st_size > 0 else "w"
    with offset_text_path.open(mode, encoding="utf-8") as offset_file:
        if mode == "w":
            offset_file.write(_offset_header())
        for j in range(k):
            training_row_index = global_training_row_index_start + j
            time_index_t = time_index_t_start + j
            steer = float(expert_action_target_time_t[j, 0])
            throttle_brake = float(expert_action_target_time_t[j, 1])
            label_next = str(telemetry_label_T_observation_time_t_plus_one[j])
            offset_file.write("=" * 80 + "\n")
            offset_file.write(f"BEGIN training_row_index {training_row_index}\n")
            offset_file.write(f"time_index_t: {time_index_t}\n")
            offset_file.write(
                "telemetry_label_T_observation_time_t_plus_one:\n"
                f"  {label_next}\n"
            )
            offset_file.write(f"expert_action_target_time_t_steer_component: {steer}\n")
            offset_file.write(
                f"expert_action_target_time_t_throttle_brake_component: {throttle_brake}\n"
            )
            offset_file.write(
                f"observation_input_time_t_vector_length: {observation_input_time_t.shape[1]}\n"
                "observation_input_time_t_comma_separated:\n"
            )
            offset_file.write(
                _comma_separated_observation_vector(observation_input_time_t[j])
            )
            offset_file.write("\n\n")


def collect_parallel_arrays(
    environment: MetaDriveEnv,
    bundled_expert: BundledExpert,
    *,
    max_steps: int,
    deterministic_expert: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert_observations_list: List[np.ndarray] = []
    expert_actions_list: List[np.ndarray] = []
    telemetry_labels_T_at_same_timestep_as_observation_list: List[str] = []

    environment.reset()
    for _step_index in range(max_steps):
        ego_vehicle = environment.agent
        telemetry_tags = auto_label_telemetry(ego_vehicle, None)
        telemetry_label_string = compact_telemetry_label_from_tags(telemetry_tags)

        expert_action_vector, expert_observation_vector = bundled_expert.predict(
            ego_vehicle,
            deterministic=deterministic_expert,
            need_obs=True,
        )
        expert_observations_list.append(
            np.asarray(expert_observation_vector, dtype=np.float32).reshape(-1)
        )
        expert_actions_list.append(np.asarray(expert_action_vector, dtype=np.float32).reshape(-1))
        telemetry_labels_T_at_same_timestep_as_observation_list.append(telemetry_label_string)

        step_output = environment.step(expert_action_vector)
        if len(step_output) == 5:
            _observation, _reward, terminated, truncated, step_info = step_output
            episode_done = terminated or truncated
        else:
            _observation, _reward, episode_done, step_info = step_output

        if episode_done and step_info.get("arrive_dest"):
            environment.reset(environment.current_seed + 1)

    expert_observations = np.stack(expert_observations_list, axis=0)
    expert_actions = np.stack(expert_actions_list, axis=0)
    telemetry_labels_T_at_same_timestep_as_observation = np.asarray(
        telemetry_labels_T_at_same_timestep_as_observation_list,
        dtype=object,
    )
    return expert_observations, expert_actions, telemetry_labels_T_at_same_timestep_as_observation


def build_offset_training_dataset(
    expert_observations: np.ndarray,
    expert_actions: np.ndarray,
    telemetry_labels_T_at_same_timestep_as_observation: np.ndarray,
) -> Dict[str, np.ndarray]:
    number_of_timesteps = expert_observations.shape[0]
    if number_of_timesteps < 2:
        raise ValueError("Need at least 2 timesteps for offset dataset.")
    if expert_actions.shape[0] != number_of_timesteps:
        raise ValueError("expert_observations and expert_actions length mismatch.")
    if telemetry_labels_T_at_same_timestep_as_observation.shape[0] != number_of_timesteps:
        raise ValueError("expert_observations and telemetry_labels length mismatch.")

    return {
        "observation_input_time_t": expert_observations[:-1].copy(),
        "telemetry_label_T_observation_time_t_plus_one": telemetry_labels_T_at_same_timestep_as_observation[
            1:
        ].copy(),
        "expert_action_target_time_t": expert_actions[:-1].copy(),
    }


def save_rollout_npz(
    output_path: str | Path,
    *,
    expert_observations: np.ndarray,
    expert_actions: np.ndarray,
    telemetry_labels_T_at_same_timestep_as_observation: np.ndarray,
    include_offset_training_arrays: bool = True,
    write_human_readable: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numpy_payload: Dict[str, Any] = {
        "expert_observations": expert_observations,
        "expert_actions": expert_actions,
        "telemetry_labels_T_at_same_timestep_as_observation": telemetry_labels_T_at_same_timestep_as_observation,
        "dataset_schema_description": np.array(
            "parallel log index i: observations[i], actions[i], label string at step i. "
            "Supervision (offset rows): observation_input_time_t[t]=o_t, "
            "telemetry_label_T_observation_time_t_plus_one[t]=T(obs_{t+1}), "
            "expert_action_target_time_t[t]=a_t",
            dtype=object,
        ),
    }
    if include_offset_training_arrays and expert_observations.shape[0] >= 2:
        offset_arrays = build_offset_training_dataset(
            expert_observations,
            expert_actions,
            telemetry_labels_T_at_same_timestep_as_observation,
        )
        numpy_payload["observation_input_time_t"] = offset_arrays["observation_input_time_t"]
        numpy_payload["telemetry_label_T_observation_time_t_plus_one"] = offset_arrays[
            "telemetry_label_T_observation_time_t_plus_one"
        ]
        numpy_payload["expert_action_target_time_t"] = offset_arrays["expert_action_target_time_t"]

    np.savez_compressed(output_path, **numpy_payload)

    if write_human_readable:
        raw_text_path, offset_text_path = write_rollout_human_readable_text_files(
            output_path,
            expert_observations=expert_observations,
            expert_actions=expert_actions,
            telemetry_labels_T_at_same_timestep_as_observation=telemetry_labels_T_at_same_timestep_as_observation,
        )
        print(f"Wrote human-readable raw rollout: {raw_text_path}", flush=True)
        print(f"Wrote human-readable offset training: {offset_text_path}", flush=True)


def main() -> None:
    maximum_collection_steps = int(os.environ.get("METADRIVE_COLLECT_STEPS", "5000"))
    output_file_path = Path(
        os.environ.get("METADRIVE_COLLECT_OUT", "data/distillation/rollout.npz")
    )
    use_onscreen_rendering = not _environment_flag_true("METADRIVE_COLLECT_HEADLESS")
    use_deterministic_expert = _environment_flag_true("METADRIVE_EXPERT_DETERMINISTIC")

    if sys.platform == "darwin" and os.environ.get("METADRIVE_DEBUG_GL"):
        from panda3d.core import loadPrcFileData

        loadPrcFileData("", "notify-level-display debug")

    render_launch.apply_metadrive_render_patches()

    environment_config = build_metadrive_config(
        use_render=use_onscreen_rendering,
        control_mode="programmatic",
    )
    environment = MetaDriveEnv(config=environment_config)
    bundled_expert = BundledExpert()
    try:
        print(
            f"Collecting {maximum_collection_steps} steps -> {output_file_path} "
            f"(rendering={'on' if use_onscreen_rendering else 'off'})",
            flush=True,
        )
        (
            expert_observations,
            expert_actions,
            telemetry_labels_T_at_same_timestep_as_observation,
        ) = collect_parallel_arrays(
            environment,
            bundled_expert,
            max_steps=maximum_collection_steps,
            deterministic_expert=use_deterministic_expert,
        )
        save_rollout_npz(
            output_file_path,
            expert_observations=expert_observations,
            expert_actions=expert_actions,
            telemetry_labels_T_at_same_timestep_as_observation=telemetry_labels_T_at_same_timestep_as_observation,
            include_offset_training_arrays=True,
        )
        offset_row_count = expert_observations.shape[0] - 1
        print(
            f"Saved expert_observations {expert_observations.shape}, "
            f"expert_actions {expert_actions.shape}, "
            f"telemetry_labels {telemetry_labels_T_at_same_timestep_as_observation.shape}; "
            f"offset training rows: {offset_row_count}.",
            flush=True,
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
