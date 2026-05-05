
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from panda3d.core import loadPrcFileData

import launch_metadrive_demo as render_launch
from drive_sessions import (
    KeyboardSession,
    ProgrammaticSession,
    RandomSession,
)
from distilled_inference import StrictDistilledPolicy
from distillation_recorder import keyboard_distillation_session
from expert_model import BundledExpert
from metadrive_config import MetaDriveEnvConfig
from metadrive.envs.metadrive_env import MetaDriveEnv
from student_label_control import (
    StudentLabelState,
    attach_student_label_hotkeys,
    make_student_label_before_step,
    print_student_mode_key_help,
    student_mode_hud_text,
)

APP_CONFIG = {
    "mode": "student",
    "manual_start_with_expert": False,
    "fallback_headless": False,
    "mac_balanced": False,
    "record_distillation_in_expert_mode": True,
    "distilled_checkpoint": "data/distillation/checkpoints/distill_mlp.pt",
    "distilled_vocab_json": "data/distillation/telemetry_vocab.json",
    "distilled_device": "",
    "student_label_presets": [],
    "student_label_file_trigger": "student_label_open.flag",
}

def print_macos_tips() -> None:
    print(
        "\nIf the window still fails on macOS, try:\n"
        "  1) Run from Terminal.app (not SSH / remote-only).\n"
        "  2) `pip uninstall opencv-python -y && pip install opencv-python-headless` "
        "(fixes duplicate libSDL2 from pygame+opencv).\n"
        "  2b) launch_metadrive_demo patches simplePBR on mac (MSAA=0). If you still see "
        "'NoneType' / set_shader, try software GL (step 3).\n"
        "  3) Software GL (slow): "
        "`export MESA_GL_VERSION_OVERRIDE=3.3` or "
        "`export __GL_ALLOW_SOFTWARE_RENDERING=1` then re-run.\n"
        "  4) Last resort: run MetaDrive in an Ubuntu VM / Linux box with GPU.\n"
        "  5) Pink / weird colors? Toggle `METADRIVE_FRAMEBUFFER_SRGB` between 0 and 1. "
        "Avoid `METADRIVE_MAC_GL=compat` unless you need the 3.2 fallback (can shift colors).\n",
        flush=True,
    )


class DriveApp:

    def __init__(self) -> None:
        self._mode = str(APP_CONFIG["mode"]).strip().lower()
        self._manual_start_with_expert = bool(APP_CONFIG["manual_start_with_expert"])
        self._fallback_headless = bool(APP_CONFIG["fallback_headless"])
        self._mac_balanced = sys.platform == "darwin" and bool(APP_CONFIG["mac_balanced"])
        self._record_distillation_in_expert_mode = bool(APP_CONFIG["record_distillation_in_expert_mode"])
        self._distilled_checkpoint = str(APP_CONFIG["distilled_checkpoint"])
        self._distilled_vocab_json = str(APP_CONFIG["distilled_vocab_json"])
        self._distilled_device = str(APP_CONFIG["distilled_device"])

    def run(self) -> None:
        if sys.platform == "darwin" and os.environ.get("METADRIVE_DEBUG_GL"):
            loadPrcFileData("", "notify-level-display debug")

        render_launch.apply_metadrive_render_patches()

        modes: list[tuple[bool, str]] = [(True, "onscreen")]
        if self._fallback_headless:
            modes.append((False, "headless"))

        last_err: Exception | None = None
        for use_render, label in modes:
            env: MetaDriveEnv | None = None
            try:
                print(f"Starting MetaDrive ({label}, mode={self._mode})...", flush=True)
                if self._mode in ("manual", "expert", "student") and not use_render:
                    print("This mode needs an onscreen window; skipping headless.", flush=True)
                    continue

                assert self._mode in ("manual", "random", "expert", "student")
                env = MetaDriveEnv(
                    config=MetaDriveEnvConfig.build(
                        use_render=use_render,
                        control_mode=(
                            "keyboard"
                            if self._mode == "manual"
                            else ("random" if self._mode == "random" else "programmatic")
                        ),
                        mac_balanced_preset=self._mac_balanced,
                    ),
                )
                self._run_session(env)
                print("Done.", flush=True)
                return
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if use_render and self._fallback_headless and (
                    "could not open window" in msg or "unable to open" in msg
                ):
                    print("Onscreen failed; trying headless...\n", flush=True)
                else:
                    print(f"\nMetaDrive error: {e}\n", file=sys.stderr, flush=True)
                    if sys.platform == "darwin" and use_render:
                        print_macos_tips()
                    raise
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

        if last_err is not None:
            raise last_err

    def _run_session(self, env: MetaDriveEnv) -> None:
        if self._mode == "manual":
            KeyboardSession(
                env,
                start_with_expert=self._manual_start_with_expert,
                record_distillation=False,
            ).run()
        elif self._mode == "random":
            RandomSession(env, 10_000).run()
        elif self._mode == "expert":
            expert = BundledExpert()
            if self._record_distillation_in_expert_mode:
                with keyboard_distillation_session(True) as recorder:
                    ProgrammaticSession(
                        env,
                        lambda e: expert.predict(e.agent)[0],
                        before_step_fn=(
                            (lambda e: recorder.record_before_step(e))
                            if recorder is not None
                            else None
                        ),
                    ).run()
            else:
                ProgrammaticSession(
                    env,
                    lambda e: expert.predict(e.agent)[0],
                ).run()
        elif self._mode == "student":
            label_state = StudentLabelState()
            presets = list(APP_CONFIG.get("student_label_presets") or [])
            distilled_policy = StrictDistilledPolicy.from_config(
                checkpoint_path=self._distilled_checkpoint,
                vocab_json_path=self._distilled_vocab_json,
                device_name=self._distilled_device,
                label_state=label_state,
            )
            _hotkey_listener_holder: list[Any] = []

            def _student_on_start(e: MetaDriveEnv) -> None:
                eng = getattr(e, "engine", None)
                if eng is None:
                    print("Student mode: no engine — label hotkeys unavailable.", flush=True)
                    return
                _hotkey_listener_holder.append(
                    attach_student_label_hotkeys(
                        eng,
                        label_state,
                        presets,
                        vocab_json_path=distilled_policy.vocab_json_path,
                    )
                )

            print(
                "Running distilled inference policy (strict mode; errors stop execution).",
                flush=True,
            )
            print_student_mode_key_help()
            _tr_raw = str(APP_CONFIG.get("student_label_file_trigger") or "").strip()
            _tr_path: Path | None
            if _tr_raw:
                _tp = Path(_tr_raw).expanduser()
                _tr_path = _tp if _tp.is_absolute() else (Path.cwd() / _tp)
            else:
                _tr_path = None
            ProgrammaticSession(
                env,
                distilled_policy.action_for_env,
                on_session_start=_student_on_start,
                before_step_fn=make_student_label_before_step(
                    label_state,
                    distilled_policy.vocab_json_path,
                    file_trigger_path=_tr_path,
                ),
                hud_text_provider=lambda: student_mode_hud_text(
                    state=label_state,
                    file_trigger_hint=str(_tr_path) if _tr_path else "",
                ),
            ).run()
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")
