
from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np

ExpertReturn = Union[Tuple[np.ndarray, np.ndarray], np.ndarray]


class BundledExpert:

    def __init__(self) -> None:
        self.last_action: np.ndarray | None = None
        self.last_observation: np.ndarray | None = None

    def predict(
        self,
        vehicle: Any,
        *,
        deterministic: bool = False,
        need_obs: bool = True,
    ) -> ExpertReturn:
        from metadrive.examples import expert

        raw = expert(vehicle, deterministic=deterministic, need_obs=need_obs)

        if need_obs:
            action_arr, obs_arr = raw
            action_out = np.asarray(action_arr, dtype=np.float32).reshape(-1).copy()
            obs_out = np.asarray(obs_arr, dtype=np.float32).reshape(-1).copy()
            self.last_action = action_out
            self.last_observation = obs_out
            return action_out, obs_out

        action_out = np.asarray(raw, dtype=np.float32).reshape(-1).copy()
        self.last_action = action_out
        self.last_observation = None
        return action_out
