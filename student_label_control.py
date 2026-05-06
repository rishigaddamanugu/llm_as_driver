
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from direct.showbase.DirectObject import DirectObject

from student_label_dialog import prompt_compact_label
from telemetry_factorization import factors_from_compact_label


@dataclass
class StudentLabelState:
    override_label: str | None = None
    last_auto_label: str = ""
    last_error: str | None = None


def _truncate(s: str, n: int = 72) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def space_key_observed_down(engine: Any) -> bool:
    try:
        from direct.showbase.InputStateGlobal import inputState

        for name in ("space", " "):
            try:
                if inputState.isSet(name):
                    return True
            except Exception:
                pass
    except Exception:
        pass

    try:
        from panda3d.core import KeyboardButton

        mw = getattr(engine, "mouseWatcherNode", None)
        if mw is not None:
            if mw.isButtonDown(KeyboardButton.space()):
                return True
            btn_space_char = getattr(KeyboardButton, "asciiKey", None) or getattr(
                KeyboardButton, "ascii_key", None
            )
            if callable(btn_space_char):
                try:
                    if mw.isButtonDown(btn_space_char(" ")):
                        return True
                except Exception:
                    try:
                        if mw.isButtonDown(btn_space_char(ord(" "))):
                            return True
                    except Exception:
                        pass
    except Exception:
        pass

    return False


def pygame_wants_label_dialog() -> bool:
    try:
        import pygame
    except ImportError:
        return False
    try:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        return bool(keys[pygame.K_SPACE] or keys[pygame.K_l])
    except Exception:
        return False


def any_label_hotkey_down(engine: Any | None) -> bool:
    if pygame_wants_label_dialog():
        return True
    if engine is None:
        return False
    return space_key_observed_down(engine)


def open_label_dialog_for_state(state: StudentLabelState, vocab_json_path: Path) -> None:
    seed = state.override_label or state.last_auto_label
    result = prompt_compact_label(initial=seed, vocab_json_path=vocab_json_path)
    if result is None:
        print("[student] label dialog cancelled or failed (see terminal)", flush=True)
        return
    state.override_label = result
    state.last_error = None
    print(f"[student] override = typed label ({len(result)} chars)", flush=True)


def make_student_label_before_step(
    state: StudentLabelState,
    vocab_json_path: Path,
    *,
    file_trigger_path: Path | None = None,
) -> Callable[[Any], None]:
    prev = False
    logged = False

    def before_step(env: Any) -> None:
        nonlocal prev, logged
        if not logged:
            print(
                "[student] Press Space or L in the game window to edit the telemetry label.",
                flush=True,
            )
            logged = True

        if file_trigger_path is not None and file_trigger_path.is_file():
            try:
                file_trigger_path.unlink()
            except OSError:
                pass
            else:
                print("[student] file trigger → opening label dialog", flush=True)
                open_label_dialog_for_state(state, vocab_json_path)
                return

        eng = getattr(env, "engine", None)
        down = any_label_hotkey_down(eng)
        if down and not prev:
            open_label_dialog_for_state(state, vocab_json_path)
        prev = down

    return before_step


def student_mode_hud_text(*, state: StudentLabelState, file_trigger_hint: str = "") -> dict[str, str]:
    if state.override_label is None:
        lines = {
            "Mode": "no user telemetry",
            "model": "UNK factors until you set a label",
        }
    else:
        f = factors_from_compact_label(state.override_label)
        lines = {
            "Mode": "user telemetry",
            "speed": f.speed,
            "longitudinal": f.longitudinal,
            "steering": f.steering,
            "route_bin": f.route_bin,
            "overspeed": "yes" if f.overspeed else "no",
            "off_lane": "yes" if f.off_lane else "no",
            "compact": _truncate(state.override_label, 96),
        }
    if state.override_label is not None and file_trigger_hint:
        lines["Backup"] = f"touch {file_trigger_hint}"
    if state.last_error:
        lines["Note"] = _truncate(state.last_error, 48)
    return lines


class StudentLabelHotkeys(DirectObject):
    def __init__(
        self,
        engine,
        state: StudentLabelState,
        presets: list[str],
        *,
        vocab_json_path: Path,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._state = state
        self._vocab_json_path = vocab_json_path
        self._presets = list(presets)[:9]
        self._bind_keys()

    def _open_label_dialog(self) -> None:
        open_label_dialog_for_state(self._state, self._vocab_json_path)

    def _bind_keys(self) -> None:
        accept = getattr(self._engine, "accept", self.accept)

        accept("[", self._clear)
        accept(";", self._pin)
        accept("semicolon", self._pin)
        accept("l", self._open_label_dialog)
        accept("L", self._open_label_dialog)
        for i in range(1, 10):
            accept(str(i), self._preset, [i - 1])

    def _clear(self) -> None:
        self._state.override_label = None
        self._state.last_error = None
        print("[student] user telemetry cleared (model uses neutral UNK factors)", flush=True)

    def _pin(self) -> None:
        auto = self._state.last_auto_label
        if not auto:
            self._state.last_error = "No auto label yet"
            return
        self._state.override_label = auto
        self._state.last_error = None
        print("[student] pinned live label as override", flush=True)

    def _preset(self, index: int) -> None:
        if index >= len(self._presets):
            self._state.last_error = f"No preset slot {index + 1}"
            return
        label = self._presets[index]
        self._state.override_label = label
        self._state.last_error = None
        print(f"[student] preset {index + 1} applied", flush=True)


def attach_student_label_hotkeys(
    engine,
    state: StudentLabelState,
    presets: list[str],
    *,
    vocab_json_path: Path,
) -> StudentLabelHotkeys:
    return StudentLabelHotkeys(
        engine, state, presets, vocab_json_path=vocab_json_path
    )


def print_student_mode_key_help() -> None:
    print(
        "\n--- Student mode: user telemetry ---\n"
        "  [     Clear user telemetry (model uses neutral factors)\n"
        "  ;     Pin **live scene** label from auto (then becomes user telemetry)\n"
        "  1-9   Apply configured preset strings (see APP_CONFIG student_label_presets)\n"
        "  Space / L  Label dialog (click the game window first, then press)\n",
        flush=True,
    )
