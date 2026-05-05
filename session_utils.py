
from __future__ import annotations

from typing import Any


class MetaDriveTransitions:

    @staticmethod
    def unpack_reset(reset_out: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(reset_out, tuple):
            info = reset_out[1] if len(reset_out) > 1 else {}
            return reset_out[0], info if isinstance(info, dict) else {}
        return reset_out, {}

    @staticmethod
    def unpack_step(step_out: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            return obs, reward, bool(terminated or truncated), info
        obs, reward, done, info = step_out
        return obs, reward, bool(done), info
