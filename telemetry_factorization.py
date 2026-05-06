
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from telemetry_labels import route_completion_bucket_suffix

_UNK = "__UNK__"


@dataclass(frozen=True)
class TelemetryFactors:
    speed: str
    longitudinal: str
    steering: str
    overspeed: int
    off_lane: int
    route_bin: str


def neutral_inference_telemetry_factors() -> TelemetryFactors:
    """Unknown human oracle for o_{t+1}: all categorical slots map to ``__UNK__``."""
    return TelemetryFactors(
        speed=_UNK,
        longitudinal=_UNK,
        steering=_UNK,
        overspeed=0,
        off_lane=0,
        route_bin=_UNK,
    )


def _norm_phrase(v: str) -> str:
    return str(v or "").strip() or _UNK


def _norm_route_bin(v: str) -> str:
    raw = str(v or "").strip()
    if raw == _UNK:
        return _UNK
    if raw.startswith("b") and raw[1:].isdigit():
        return f"b{int(raw[1:]):02d}"
    try:
        return route_completion_bucket_suffix(float(raw))
    except ValueError:
        return "b00"


def factors_from_compact_label(compact: str) -> TelemetryFactors:
    speed = _UNK
    longitudinal = _UNK
    steering = _UNK
    overspeed = 0
    off_lane = 0
    route_bin = "b00"
    for part in str(compact).split("|"):
        if part.startswith("speed_natural_language:"):
            speed = _norm_phrase(part.split(":", 1)[1])
        elif part.startswith("longitudinal_natural_language:"):
            longitudinal = _norm_phrase(part.split(":", 1)[1])
        elif part.startswith("steering_natural_language:"):
            steering = _norm_phrase(part.split(":", 1)[1])
        elif part == "overspeed":
            overspeed = 1
        elif part == "off_lane":
            off_lane = 1
        elif part.startswith("route_completion:"):
            route_bin = _norm_route_bin(part.split(":", 1)[1])
    return TelemetryFactors(
        speed=speed,
        longitudinal=longitudinal,
        steering=steering,
        overspeed=overspeed,
        off_lane=off_lane,
        route_bin=route_bin,
    )


def factors_from_tags(tags: Mapping[str, object]) -> TelemetryFactors:
    return TelemetryFactors(
        speed=_norm_phrase(str(tags.get("speed_natural_language", "?")).replace(" ", "_")),
        longitudinal=_norm_phrase(
            str(tags.get("longitudinal_natural_language", "?")).replace(" ", "_")
        ),
        steering=_norm_phrase(str(tags.get("steering_natural_language", "?")).replace(" ", "_")),
        overspeed=1 if bool(tags.get("overspeed")) else 0,
        off_lane=0 if bool(tags.get("on_lane", True)) else 1,
        route_bin=route_completion_bucket_suffix(float(tags.get("route_completion", 0.0))),
    )


def build_vocab(values: list[str]) -> tuple[dict[str, int], list[str]]:
    uniq = sorted(set(values))
    if _UNK not in uniq:
        uniq = [_UNK] + uniq
    return {v: i for i, v in enumerate(uniq)}, uniq


def encode_with_vocab(value: str, vocab: Mapping[str, int]) -> int:
    unk_id = int(vocab.get(_UNK, 0))
    return int(vocab.get(value, unk_id))
