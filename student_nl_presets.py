
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NlPreset:

    sentence: str
    speed: str
    longitudinal: str
    steering: str
    overspeed: bool = False
    off_lane: bool = False
    route_bin: str = "b08"


NL_PRESETS: list[NlPreset] = [
    NlPreset(
        "Easy cruise — modest speed, light throttle, straight",
        "at_a_modest_speed",
        "pressing_the_gas_slightly",
        "driving_nearly_straight",
        route_bin="b10",
    ),
    NlPreset(
        "Slow roll — nearly stopped, tiny gas, straight",
        "nearly_stopped",
        "pressing_the_gas_slightly",
        "driving_nearly_straight",
        route_bin="b02",
    ),
    NlPreset(
        "City crawl — moving slowly, gentle accel, straight",
        "moving_slowly",
        "accelerating_gently",
        "driving_nearly_straight",
        route_bin="b06",
    ),
    NlPreset(
        "Highway pace — cruising, moderate throttle, straight",
        "cruising",
        "accelerating_moderately",
        "driving_nearly_straight",
        route_bin="b14",
    ),
    NlPreset(
        "Hammer down — quick, hard throttle, straight",
        "moving_quickly",
        "accelerating_hard",
        "driving_nearly_straight",
        route_bin="b12",
    ),
    NlPreset(
        "Easy arc — modest speed, light gas, gentle left",
        "at_a_modest_speed",
        "pressing_the_gas_slightly",
        "turning_left_gently",
        route_bin="b09",
    ),
    NlPreset(
        "Wide turn — slow, light gas, gentle right",
        "moving_slowly",
        "pressing_the_gas_slightly",
        "turning_right_gently",
        route_bin="b07",
    ),
    NlPreset(
        "Brake check — modest speed, braking a bit, straight",
        "at_a_modest_speed",
        "slowing_down_a_bit",
        "driving_nearly_straight",
        route_bin="b05",
    ),
    NlPreset(
        "Heavy brake — slow, strong braking, straight",
        "moving_slowly",
        "slowing_down_a_lot",
        "driving_nearly_straight",
        route_bin="b04",
    ),
    NlPreset(
        "Settle in — modest speed, gentle accel, straight",
        "at_a_modest_speed",
        "accelerating_gently",
        "driving_nearly_straight",
        route_bin="b11",
    ),
    NlPreset(
        "Start of lap — nearly stopped, light gas, straight, very beginning of route",
        "nearly_stopped",
        "pressing_the_gas_slightly",
        "driving_nearly_straight",
        route_bin="b00",
    ),
    NlPreset(
        "Near finish — modest speed, light gas, straight, end of route",
        "at_a_modest_speed",
        "pressing_the_gas_slightly",
        "driving_nearly_straight",
        route_bin="b18",
    ),
    NlPreset(
        "Speeding context — overspeed, cruising, hard throttle, straight",
        "cruising",
        "accelerating_hard",
        "driving_nearly_straight",
        overspeed=True,
        route_bin="b13",
    ),
    NlPreset(
        "Off lane recovery — slow, light gas, straight, off lane",
        "moving_slowly",
        "pressing_the_gas_slightly",
        "driving_nearly_straight",
        off_lane=True,
        route_bin="b06",
    ),
]
