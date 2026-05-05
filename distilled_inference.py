
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch

from expert_model import BundledExpert
from telemetry_factorization import encode_with_vocab, factors_from_compact_label, factors_from_tags
from telemetry_labels import auto_label_telemetry, compact_telemetry_label_from_tags
from train_distill import DistillMLP

if TYPE_CHECKING:
    from student_label_control import StudentLabelState


class StrictDistilledPolicy:

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        vocab_json_path: Path,
        device: torch.device,
        label_state: Optional["StudentLabelState"] = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._vocab_json_path = vocab_json_path
        self._device = device
        self._label_state = label_state
        self._expert = BundledExpert()

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing distilled checkpoint: {self._checkpoint_path}. Run `python train_distill.py`."
            )
        if not self._vocab_json_path.exists():
            raise FileNotFoundError(
                f"Missing telemetry vocab json: {self._vocab_json_path}. "
                "Run `python prepare_training_data.py`."
            )

        vocab_data = json.loads(self._vocab_json_path.read_text(encoding="utf-8"))
        required_vocab_keys = (
            "speed_token_to_id",
            "longitudinal_token_to_id",
            "steering_token_to_id",
            "route_bin_token_to_id",
        )
        for k in required_vocab_keys:
            if k not in vocab_data or not isinstance(vocab_data[k], dict):
                raise ValueError(f"Invalid vocab json format (missing `{k}`): {self._vocab_json_path}")
        self._speed_to_id = {str(k): int(v) for k, v in vocab_data["speed_token_to_id"].items()}
        self._long_to_id = {
            str(k): int(v) for k, v in vocab_data["longitudinal_token_to_id"].items()
        }
        self._steer_to_id = {str(k): int(v) for k, v in vocab_data["steering_token_to_id"].items()}
        self._route_to_id = {str(k): int(v) for k, v in vocab_data["route_bin_token_to_id"].items()}

        checkpoint = torch.load(self._checkpoint_path, map_location=self._device)
        required_keys = (
            "model_state_dict",
            "obs_dim",
            "speed_vocab_size",
            "longitudinal_vocab_size",
            "steering_vocab_size",
            "route_bin_vocab_size",
            "telemetry_embed_dim",
            "hidden_dim",
        )
        for k in required_keys:
            if k not in checkpoint:
                raise KeyError(f"Checkpoint missing key `{k}`: {self._checkpoint_path}")

        num_hidden_blocks = int(checkpoint.get("num_hidden_blocks", 1))
        dropout = float(checkpoint.get("dropout", 0.0))

        model = DistillMLP(
            obs_dim=int(checkpoint["obs_dim"]),
            speed_vocab_size=int(checkpoint["speed_vocab_size"]),
            longitudinal_vocab_size=int(checkpoint["longitudinal_vocab_size"]),
            steering_vocab_size=int(checkpoint["steering_vocab_size"]),
            route_bin_vocab_size=int(checkpoint["route_bin_vocab_size"]),
            telemetry_embed_dim=int(checkpoint["telemetry_embed_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_hidden_blocks=num_hidden_blocks,
            dropout=dropout,
        ).to(self._device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        self._model = model

        checks = (
            (len(self._speed_to_id), int(checkpoint["speed_vocab_size"]), "speed"),
            (len(self._long_to_id), int(checkpoint["longitudinal_vocab_size"]), "longitudinal"),
            (len(self._steer_to_id), int(checkpoint["steering_vocab_size"]), "steering"),
            (len(self._route_to_id), int(checkpoint["route_bin_vocab_size"]), "route_bin"),
        )
        for got, expected, name in checks:
            if got != expected:
                raise ValueError(
                    f"Telemetry vocab size mismatch for {name}: vocab_json={got} checkpoint={expected}"
                )

    @property
    def label_to_id(self) -> dict[str, int]:
        return {}

    @property
    def vocab_json_path(self) -> Path:
        return self._vocab_json_path

    def _encode_factors(self, factors) -> tuple[torch.Tensor, ...]:
        speed_t = torch.tensor(
            [encode_with_vocab(factors.speed, self._speed_to_id)],
            dtype=torch.long,
            device=self._device,
        )
        long_t = torch.tensor(
            [encode_with_vocab(factors.longitudinal, self._long_to_id)],
            dtype=torch.long,
            device=self._device,
        )
        steer_t = torch.tensor(
            [encode_with_vocab(factors.steering, self._steer_to_id)],
            dtype=torch.long,
            device=self._device,
        )
        route_t = torch.tensor(
            [encode_with_vocab(factors.route_bin, self._route_to_id)],
            dtype=torch.long,
            device=self._device,
        )
        overspeed_t = torch.tensor([float(factors.overspeed)], dtype=torch.float32, device=self._device)
        off_lane_t = torch.tensor([float(factors.off_lane)], dtype=torch.float32, device=self._device)
        return speed_t, long_t, steer_t, route_t, overspeed_t, off_lane_t

    @classmethod
    def from_config(
        cls,
        *,
        checkpoint_path: str,
        vocab_json_path: str,
        device_name: str = "",
        label_state: Optional["StudentLabelState"] = None,
    ) -> StrictDistilledPolicy:
        checkpoint = Path(checkpoint_path)
        vocab_json = Path(vocab_json_path)
        if not device_name:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif (
                sys.platform == "darwin"
                and getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_name)
        return cls(
            checkpoint_path=checkpoint,
            vocab_json_path=vocab_json,
            device=device,
            label_state=label_state,
        )

    def action_for_env(self, env: Any) -> np.ndarray:
        ego = env.agent
        telemetry_tags = auto_label_telemetry(ego, None)
        auto_label = compact_telemetry_label_from_tags(telemetry_tags)
        if self._label_state is not None:
            self._label_state.last_auto_label = auto_label
            label = (
                self._label_state.override_label
                if self._label_state.override_label is not None
                else auto_label
            )
        else:
            label = auto_label

        factors = (
            factors_from_compact_label(label)
            if (self._label_state is not None and self._label_state.override_label is not None)
            else factors_from_tags(telemetry_tags)
        )
        if self._label_state is not None:
            self._label_state.last_error = None

        _expert_action, expert_obs = self._expert.predict(
            ego,
            deterministic=True,
            need_obs=True,
        )
        obs_np = np.asarray(expert_obs, dtype=np.float32).reshape(1, -1)
        obs_t = torch.from_numpy(obs_np).to(self._device)
        speed_t, long_t, steer_t, route_t, overspeed_t, off_lane_t = self._encode_factors(factors)

        with torch.no_grad():
            pred = self._model(
                obs_t, speed_t, long_t, steer_t, route_t, overspeed_t, off_lane_t
            ).squeeze(0).cpu().numpy()
        return np.asarray(pred, dtype=np.float32).reshape(-1)
