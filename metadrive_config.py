
from __future__ import annotations

import sys
from typing import Any, Literal

ControlMode = Literal["keyboard", "random", "programmatic"]


class MetaDriveEnvConfig:

    @staticmethod
    def expert_vehicle_config(*, show_lidar_overlay: bool = True) -> dict[str, Any]:
        return dict(
            enable_reverse=True,
            lidar=dict(
                num_lasers=240,
                distance=50,
                num_others=4,
                gaussian_noise=0.0,
                dropout_prob=0.0,
                add_others_navi=False,
            ),
            side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
            lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
            show_lidar=show_lidar_overlay,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
        )

    @classmethod
    def build(
        cls,
        *,
        use_render: bool,
        control_mode: ControlMode,
        mac_balanced_preset: bool = False,
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "use_render": use_render,
            "multi_thread_render": False,
            "traffic_density": 0.1,
            "num_scenarios": 10000,
            "random_agent_model": False,
            "controller": "keyboard",
            "map": 4,
            "start_seed": 10,
        }
        if sys.platform == "darwin":
            cfg["window_size"] = (960, 720)
        if mac_balanced_preset and use_render:
            cfg["force_render_fps"] = 60

        if control_mode == "keyboard":
            cfg["manual_control"] = True
            cfg["vehicle_config"] = cls.expert_vehicle_config(show_lidar_overlay=True)
        elif control_mode == "programmatic":
            cfg["manual_control"] = False
            cfg["vehicle_config"] = cls.expert_vehicle_config(show_lidar_overlay=True)
        else:
            cfg["manual_control"] = False

        return cfg


def build_metadrive_config(
    *,
    use_render: bool,
    control_mode: str,
    mac_balanced_preset: bool = False,
) -> dict[str, Any]:
    if control_mode not in ("keyboard", "random", "programmatic"):
        control_mode = "keyboard"
    return MetaDriveEnvConfig.build(
        use_render=use_render,
        control_mode=control_mode,
        mac_balanced_preset=mac_balanced_preset,
    )
