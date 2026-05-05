
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from metadrive.constants import TerminationState


THROTTLE_FORWARD_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.65, 1.01, "accelerating hard"),
    (0.30, 0.65, "accelerating moderately"),
    (0.10, 0.30, "accelerating gently"),
    (0.03, 0.10, "pressing the gas slightly"),
    (0.0, 0.03, "coasting on the throttle (near zero forward input)"),
)

THROTTLE_REVERSE_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (-1.01, -0.55, "strong braking or strong reverse input"),
    (-0.55, -0.25, "slowing down a lot"),
    (-0.25, -0.10, "slowing down noticeably"),
    (-0.10, -0.03, "slowing down a bit"),
    (-0.03, 0.0, "light braking or slight reverse input"),
)

STEERING_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.35, 1.01, "turning right sharply"),
    (0.12, 0.35, "turning right gently"),
    (-0.12, 0.12, "driving nearly straight"),
    (-0.35, -0.12, "turning left gently"),
    (-1.01, -0.35, "turning left sharply"),
)

SPEED_RATIO_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 0.08, "nearly stopped"),
    (0.08, 0.22, "moving slowly"),
    (0.22, 0.45, "at a modest speed"),
    (0.45, 0.70, "cruising"),
    (0.70, 0.92, "moving quickly"),
    (0.92, 10.0, "near the vehicle's top speed"),
)


def _band_label_for_value(value: float, bands: Tuple[Tuple[float, float, str], ...]) -> str:
    for low, high, label in bands:
        if low <= value < high:
            return label
    return bands[-1][2]


def _longitudinal_natural_language(throttle_brake: float, speed_kmh: float, enable_reverse: bool) -> str:
    throttle_brake_value = float(throttle_brake)
    if throttle_brake_value >= 0:
        return _band_label_for_value(throttle_brake_value, THROTTLE_FORWARD_BANDS)
    base = _band_label_for_value(throttle_brake_value, THROTTLE_REVERSE_BANDS)
    if enable_reverse and speed_kmh < 1.5 and throttle_brake_value < -0.08:
        return base + " (likely reversing at very low speed)"
    if enable_reverse and throttle_brake_value < -0.08 and speed_kmh > 3.0:
        return base + " (negative longitudinal input while moving forward — often braking)"
    return base


def _steering_natural_language(steering_norm: float) -> str:
    return _band_label_for_value(float(steering_norm), STEERING_BANDS)


def _speed_natural_language(speed_kmh: float, max_speed_kmh: float) -> str:
    if max_speed_kmh <= 1e-3:
        return "speed unknown (max speed unset)"
    speed_ratio = max(0.0, min(1.5, speed_kmh / max_speed_kmh))
    return _band_label_for_value(speed_ratio, SPEED_RATIO_BANDS)


def auto_label_telemetry(vehicle: Any, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info = info or {}
    enable_reverse = bool(getattr(vehicle, "config", {}).get("enable_reverse", False))
    throttle_brake = float(getattr(vehicle, "throttle_brake", 0.0))
    steering = float(getattr(vehicle, "steering", 0.0))
    speed_kmh = float(getattr(vehicle, "speed_km_h", 0.0))
    max_speed_kmh = float(getattr(vehicle, "max_speed_km_h", 80.0))

    telemetry_tags: Dict[str, Any] = {
        "speed_kmh": round(speed_kmh, 2),
        "max_speed_kmh": max_speed_kmh,
        "throttle_brake": round(throttle_brake, 3),
        "steering_norm": round(steering, 3),
        "overspeed": bool(getattr(vehicle, "overspeed", False)),
        "on_lane": bool(getattr(vehicle, "on_lane", True)),
        "longitudinal_natural_language": _longitudinal_natural_language(
            throttle_brake, speed_kmh, enable_reverse
        ),
        "steering_natural_language": _steering_natural_language(steering),
        "speed_natural_language": _speed_natural_language(speed_kmh, max_speed_kmh),
    }

    lane = getattr(vehicle, "lane", None)
    if lane is not None and getattr(lane, "speed_limit", None) is not None:
        telemetry_tags["lane_speed_limit_kmh"] = float(lane.speed_limit)

    navigation = getattr(vehicle, "navigation", None)
    if navigation is not None:
        try:
            telemetry_tags["route_completion"] = round(
                float(getattr(navigation, "route_completion", 0.0)), 4
            )
        except Exception:
            pass

    for termination_key in (
        TerminationState.CRASH_VEHICLE,
        TerminationState.CRASH_OBJECT,
        TerminationState.CRASH_BUILDING,
        TerminationState.CRASH_SIDEWALK,
        TerminationState.CRASH_HUMAN,
        TerminationState.OUT_OF_ROAD,
        TerminationState.SUCCESS,
        TerminationState.MAX_STEP,
    ):
        if termination_key in info:
            telemetry_tags[f"info_{termination_key}"] = bool(info[termination_key])

    if "step_reward" in info:
        telemetry_tags["step_reward"] = round(float(info["step_reward"]), 4)

    return telemetry_tags


_ROUTE_COMPLETION_LABEL_BINS: int = 20


def route_completion_bucket_suffix(route_completion: float, *, num_bins: int = _ROUTE_COMPLETION_LABEL_BINS) -> str:
    x = max(0.0, min(1.0, float(route_completion)))
    idx = int(x * num_bins)
    if idx >= num_bins:
        idx = num_bins - 1
    return f"b{idx:02d}"


def _replace_route_completion_suffix(compact: str, new_suffix: str) -> str:
    if "route_completion:" not in compact:
        return compact
    parts = compact.split("|")
    out: list[str] = []
    for p in parts:
        if p.startswith("route_completion:"):
            out.append(f"route_completion:{new_suffix}")
        else:
            out.append(p)
    return "|".join(out)


def strip_route_completion_segment(compact: str) -> str:
    parts = [p for p in compact.split("|") if not p.startswith("route_completion:")]
    return "|".join(parts)


def _route_completion_segment_from_compact(compact: str) -> Optional[str]:
    for p in compact.split("|"):
        if p.startswith("route_completion:"):
            return p
    return None


def route_completion_value_from_segment(segment: str) -> Optional[float]:
    if not segment.startswith("route_completion:"):
        return None
    raw = segment[len("route_completion:") :]
    if len(raw) >= 2 and raw[0] == "b" and raw[1:].isdigit():
        idx = int(raw[1:])
        idx = max(0, min(_ROUTE_COMPLETION_LABEL_BINS - 1, idx))
        return (idx + 0.5) / _ROUTE_COMPLETION_LABEL_BINS
    try:
        return float(raw)
    except ValueError:
        return None


def nearest_vocab_label_for_strip_route_match(
    stripped_compact: str,
    label_to_id: Mapping[str, int],
    route_completion_observed: float,
) -> Optional[str]:
    best_key: Optional[str] = None
    best_dist = float("inf")
    obs_rc = max(0.0, min(1.0, float(route_completion_observed)))
    for k in label_to_id:
        if strip_route_completion_segment(k) != stripped_compact:
            continue
        seg = _route_completion_segment_from_compact(k)
        if seg is None:
            continue
        rv = route_completion_value_from_segment(seg)
        if rv is None:
            continue
        d = abs(rv - obs_rc)
        if d < best_dist:
            best_dist = d
            best_key = k
        elif d == best_dist and (best_key is None or k < best_key):
            best_key = k
    return best_key


def compact_label_vocab_lookup_variants(telemetry_tags: Dict[str, Any]) -> List[str]:
    primary = compact_telemetry_label_from_tags(telemetry_tags)
    seen: list[str] = []

    def add(s: str) -> None:
        if s not in seen:
            seen.append(s)

    add(primary)
    if "route_completion" not in telemetry_tags:
        return seen
    rc = float(telemetry_tags["route_completion"])
    for suf in (
        route_completion_bucket_suffix(rc),
        f"{rc:.3f}",
        f"{rc:.2f}",
        f"{rc:.4f}",
    ):
        add(_replace_route_completion_suffix(primary, suf))
    add(strip_route_completion_segment(primary))
    return seen


def compact_telemetry_label_from_tags(telemetry_tags: Dict[str, Any]) -> str:
    speed_phrase = str(telemetry_tags.get("speed_natural_language", "?")).replace(" ", "_")
    longitudinal_phrase = str(telemetry_tags.get("longitudinal_natural_language", "?")).replace(" ", "_")
    steering_phrase = str(telemetry_tags.get("steering_natural_language", "?")).replace(" ", "_")
    label_segments = [
        f"speed_natural_language:{speed_phrase}",
        f"longitudinal_natural_language:{longitudinal_phrase}",
        f"steering_natural_language:{steering_phrase}",
    ]
    if telemetry_tags.get("overspeed"):
        label_segments.append("overspeed")
    if not telemetry_tags.get("on_lane", True):
        label_segments.append("off_lane")
    if "route_completion" in telemetry_tags:
        rc_bin = route_completion_bucket_suffix(float(telemetry_tags["route_completion"]))
        label_segments.append(f"route_completion:{rc_bin}")
    return "|".join(label_segments)


def format_telemetry_natural_language(vehicle: Any, info: Optional[Dict[str, Any]] = None) -> str:
    telemetry_tags = auto_label_telemetry(vehicle, info)
    sentences: List[str] = [
        f"Speed is about {telemetry_tags['speed_kmh']} km/h ({telemetry_tags['speed_natural_language']}).",
        f"Longitudinal command reads as: {telemetry_tags['longitudinal_natural_language']}.",
        f"Steering reads as: {telemetry_tags['steering_natural_language']}.",
    ]
    if telemetry_tags.get("overspeed"):
        sentences.append("The car is over the current lane speed limit.")
    if not telemetry_tags.get("on_lane", True):
        sentences.append("Lane keeping: vehicle reports it is not fully on the lane.")
    if "route_completion" in telemetry_tags:
        sentences.append(
            f"Route progress is roughly {telemetry_tags['route_completion'] * 100:.1f}% "
            "along the navigation route."
        )
    if "lane_speed_limit_kmh" in telemetry_tags:
        sentences.append(
            f"This lane's speed limit is about {telemetry_tags['lane_speed_limit_kmh']:.0f} km/h."
        )

    if info:
        if info.get(TerminationState.CRASH_VEHICLE):
            sentences.append("Contact: crash with another vehicle reported this step.")
        if info.get(TerminationState.OUT_OF_ROAD):
            sentences.append("Out-of-road condition is flagged this step.")
        if info.get(TerminationState.SUCCESS):
            sentences.append("Arrival at destination is flagged this step.")

    return " ".join(sentences)


def telemetry_natural_language_lines(vehicle: Any, info: Optional[Dict[str, Any]] = None) -> List[str]:
    telemetry_tags = auto_label_telemetry(vehicle, info)
    lines = [
        f"[telemetry] speed={telemetry_tags['speed_kmh']} km/h — {telemetry_tags['speed_natural_language']}",
        f"[telemetry] throttle_brake={telemetry_tags['throttle_brake']:.3f} — "
        f"{telemetry_tags['longitudinal_natural_language']}",
        f"[telemetry] steering={telemetry_tags['steering_norm']:.3f} — "
        f"{telemetry_tags['steering_natural_language']}",
    ]
    if "route_completion" in telemetry_tags:
        lines.append(f"[telemetry] route_completion={telemetry_tags['route_completion']:.4f}")
    return lines
