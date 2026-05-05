
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID_DEFAULT = "google/flan-t5-small"
MODEL_DIR_DEFAULT = Path("data/local_models/flan-t5-small")


@dataclass(frozen=True)
class TelemetryFormatResult:
    speed: str
    longitudinal: str
    steering: str
    overspeed: bool
    off_lane: bool
    route_bin: str

    def to_compact_label(self) -> str:
        parts = [
            f"speed_natural_language:{self.speed}",
            f"longitudinal_natural_language:{self.longitudinal}",
            f"steering_natural_language:{self.steering}",
        ]
        if self.overspeed:
            parts.append("overspeed")
        if self.off_lane:
            parts.append("off_lane")
        parts.append(f"route_completion:{self.route_bin}")
        return "|".join(parts)


def download_local_formatter_model(
    *,
    model_id: str = MODEL_ID_DEFAULT,
    local_dir: Path = MODEL_DIR_DEFAULT,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    return local_dir


class TelemetryLabelLLM:
    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR_DEFAULT,
        max_new_tokens: int = 128,
    ) -> None:
        self._model_dir = model_dir
        self._max_new_tokens = max_new_tokens
        self._ensure_local_model()
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    def _ensure_local_model(self) -> None:
        if self._model_dir.is_dir() and (self._model_dir / "config.json").is_file():
            return
        try:
            download_local_formatter_model(local_dir=self._model_dir)
        except Exception as e:
            raise RuntimeError(
                f"Could not download local formatter model to {self._model_dir}: {e}"
            ) from e

    @staticmethod
    def _prompt(user_text: str) -> str:
        return (
            "Convert this driving command to JSON.\n"
            "Keys: speed, longitudinal, steering, overspeed, off_lane, route_bin.\n"
            "Use underscore tokens and route_bin b00..b19.\n"
            "Return JSON only.\n"
            f"User command: {user_text}\n"
        )

    @staticmethod
    def _strict_retry_prompt(user_text: str) -> str:
        return (
            "Return one JSON object only.\n"
            '{"speed":"...","longitudinal":"...","steering":"...",'
            '"overspeed":false,"off_lane":false,"route_bin":"b08"}\n'
            f"User command: {user_text}\n"
        )

    @staticmethod
    def _normalize_route_bin(v: Any) -> str:
        raw = str(v or "").strip()
        if re.fullmatch(r"b\d{2}", raw):
            i = int(raw[1:])
            return f"b{max(0, min(19, i)):02d}"
        return "b08"

    @staticmethod
    def _as_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "on"}

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m is None:
            raise ValueError(f"Model output does not contain JSON: {text!r}")
        return json.loads(m.group(0))

    @staticmethod
    def _fallback_from_text(user_text: str) -> dict[str, Any]:
        t = user_text.lower()
        speed = "at_a_modest_speed"
        if any(x in t for x in ("stop", "stopped", "very slow", "crawl")):
            speed = "nearly_stopped"
        elif any(x in t for x in ("slow", "slower")):
            speed = "moving_slowly"
        elif any(x in t for x in ("fast", "faster", "quick", "highway")):
            speed = "moving_quickly"
        elif "cruise" in t:
            speed = "cruising"

        longitudinal = "pressing_the_gas_slightly"
        if any(x in t for x in ("brake", "slow down")):
            longitudinal = "slowing_down_a_bit"
        elif any(x in t for x in ("hard throttle", "floor it", "accelerate hard")):
            longitudinal = "accelerating_hard"
        elif any(x in t for x in ("accelerate", "speed up", "more gas")):
            longitudinal = "accelerating_gently"

        steering = "driving_nearly_straight"
        if "left" in t:
            steering = "turning_left_gently"
        elif "right" in t:
            steering = "turning_right_gently"

        route_bin = "b08"
        if any(x in t for x in ("start", "beginning")):
            route_bin = "b00"
        elif any(x in t for x in ("end", "finish")):
            route_bin = "b19"
        elif any(x in t for x in ("middle", "mid")):
            route_bin = "b10"

        overspeed = any(x in t for x in ("overspeed", "speeding", "over speed limit"))
        off_lane = any(x in t for x in ("off lane", "offlane", "out of lane"))
        return {
            "speed": speed,
            "longitudinal": longitudinal,
            "steering": steering,
            "overspeed": overspeed,
            "off_lane": off_lane,
            "route_bin": route_bin,
        }

    def _generate_text(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def format(self, user_text: str) -> TelemetryFormatResult:
        raw = self._generate_text(self._prompt(user_text))
        try:
            payload = self._extract_json(raw)
        except Exception:
            raw_retry = self._generate_text(self._strict_retry_prompt(user_text))
            try:
                payload = self._extract_json(raw_retry)
            except Exception:
                payload = self._fallback_from_text(user_text)
        return TelemetryFormatResult(
            speed=str(payload.get("speed", "at_a_modest_speed")).strip() or "at_a_modest_speed",
            longitudinal=str(payload.get("longitudinal", "pressing_the_gas_slightly")).strip()
            or "pressing_the_gas_slightly",
            steering=str(payload.get("steering", "driving_nearly_straight")).strip()
            or "driving_nearly_straight",
            overspeed=self._as_bool(payload.get("overspeed", False)),
            off_lane=self._as_bool(payload.get("off_lane", False)),
            route_bin=self._normalize_route_bin(payload.get("route_bin", "b08")),
        )
