
from __future__ import annotations

from typing import Any

from metadrive_config import MetaDriveEnvConfig, build_metadrive_config
from drive_sessions import (
    KeyboardSession,
    ProgrammaticSession,
    RandomSession,
    run_keyboard_session,
    run_programmatic_session,
    run_random_session,
)

__all__ = [
    "MetaDriveEnvConfig",
    "KeyboardSession",
    "ProgrammaticSession",
    "RandomSession",
    "build_metadrive_config",
    "ppo_expert_vehicle_config",
    "run_keyboard_session",
    "run_programmatic_session",
    "run_random_session",
]


def ppo_expert_vehicle_config(*, show_lidar_overlay: bool = True) -> dict[str, Any]:
    return MetaDriveEnvConfig.expert_vehicle_config(show_lidar_overlay=show_lidar_overlay)
