
from __future__ import annotations

from typing import Mapping, Optional

_ROUTE_BINS: int = 20


def strip_route_completion_segment(compact: str) -> str:
    parts = [p for p in compact.split("|") if not p.startswith("route_completion:")]
    return "|".join(parts)


def _route_segment_from_compact(compact: str) -> Optional[str]:
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
        idx = max(0, min(_ROUTE_BINS - 1, idx))
        return (idx + 0.5) / _ROUTE_BINS
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
        seg = _route_segment_from_compact(k)
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


def resolve_training_key(
    candidate: str,
    label_to_id: Mapping[str, int],
) -> tuple[Optional[str], str]:
    cand = candidate.strip()
    if not cand:
        return None, "none"
    if cand in label_to_id:
        return cand, "exact"
    seg = _route_segment_from_compact(cand)
    obs = route_completion_value_from_segment(seg) if seg else 0.05
    stripped = strip_route_completion_segment(cand)
    nearest = nearest_vocab_label_for_strip_route_match(stripped, label_to_id, obs)
    if nearest is not None:
        return nearest, "nearest_route"
    return None, "none"
