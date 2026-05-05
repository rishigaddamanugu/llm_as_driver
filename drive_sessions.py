
from __future__ import annotations

import sys
import time
from typing import Any, Callable

from metadrive.constants import HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv

import key_mapping
from distillation_recorder import keyboard_distillation_session
from session_utils import MetaDriveTransitions


class _TopCenterHud:

    def __init__(self) -> None:
        self._label = None
        try:
            from direct.gui.OnscreenText import OnscreenText
            from direct.showbase.ShowBaseGlobal import base
            from panda3d.core import TextNode

            self._label = OnscreenText(
                text="",
                parent=base.a2dTopCenter,
                pos=(0.0, -0.08),
                align=TextNode.ACenter,
                fg=(1, 1, 1, 1),
                bg=(0, 0, 0, 0.65),
                scale=0.048,
                mayChange=True,
            )
        except Exception:
            self._label = None

    def update(self, lines: dict[str, str]) -> bool:
        if self._label is None:
            return False
        text = "\n".join(f"{k}: {v}" for k, v in lines.items())
        self._label.setText(text)
        return True

    def destroy(self) -> None:
        if self._label is not None:
            try:
                self._label.destroy()
            except Exception:
                pass
            self._label = None


class KeyboardSession:

    NEUTRAL_ACTION: list[float] = [0.0, 0.0]

    def __init__(
        self,
        env: MetaDriveEnv,
        *,
        start_with_expert: bool = False,
        record_distillation: bool = True,
    ) -> None:
        self._env = env
        self._start_with_expert = start_with_expert
        self._record_distillation = record_distillation

    def run(self) -> None:
        key_mapping.print_control_context(mac=sys.platform == "darwin")
        print(HELP_MESSAGE)
        key_mapping.print_keyboard_focus_reminder()
        key_mapping.print_expert_toggle_hint()

        self._env.reset()
        self._env.agent.expert_takeover = bool(self._start_with_expert)
        hud = _TopCenterHud()
        if self._start_with_expert:
            print(
                "Starting with expert_takeover=True (bundled PPO). "
                "Watch the car - or press t to turn off.\n",
                flush=True,
            )
        try:
            with keyboard_distillation_session(self._record_distillation) as recorder:
                while True:
                    expert_takeover_on = bool(getattr(self._env.agent, "expert_takeover", False))
                    if recorder is not None and expert_takeover_on:
                        recorder.record_before_step(self._env)

                    step_out = self._env.step(self.NEUTRAL_ACTION)
                    _obs, _reward, done, info = MetaDriveTransitions.unpack_step(step_out)

                    lines = key_mapping.render_overlay_labels(expert_takeover=expert_takeover_on)
                    if not hud.update(lines):
                        self._env.render(text=lines)
                    else:
                        self._env.render()
                    if done:
                        takeover = getattr(self._env.agent, "expert_takeover", False)
                        if info.get("arrive_dest"):
                            self._env.reset(self._env.current_seed + 1)
                        else:
                            self._env.reset()
                        self._env.agent.expert_takeover = takeover
        except KeyboardInterrupt:
            print("\nStopped (KeyboardInterrupt).", flush=True)
        finally:
            hud.destroy()


class RandomSession:

    def __init__(self, env: MetaDriveEnv, max_steps: int) -> None:
        self._env = env
        self._max_steps = max_steps

    def run(self) -> None:
        reset_out = self._env.reset()
        _obs, _info = MetaDriveTransitions.unpack_reset(reset_out)

        for i in range(self._max_steps):
            action = self._env.action_space.sample()
            step_out = self._env.step(action)
            _obs, _reward, done, _info = MetaDriveTransitions.unpack_step(step_out)
            if done:
                reset_out = self._env.reset()
                _obs, _info = MetaDriveTransitions.unpack_reset(reset_out)
            if (i + 1) % 500 == 0:
                print(f"...step {i + 1}", flush=True)


class ProgrammaticSession:

    def __init__(
        self,
        env: MetaDriveEnv,
        policy_fn: Callable[[MetaDriveEnv], Any],
        *,
        use_render_text: bool = True,
        before_step_fn: Callable[[MetaDriveEnv], None] | None = None,
        on_session_start: Callable[[MetaDriveEnv], None] | None = None,
        hud_text_provider: Callable[[], dict[str, str]] | None = None,
        freeze_sim_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._env = env
        self._policy_fn = policy_fn
        self._use_render_text = use_render_text
        self._before_step_fn = before_step_fn
        self._on_session_start = on_session_start
        self._hud_text_provider = hud_text_provider
        self._freeze_sim_fn = freeze_sim_fn

    @staticmethod
    def _default_hud_lines() -> dict[str, str]:
        return {
            "Driving": "Programmatic policy",
            "Goal": "Follow road / checkpoints - avoid crashes",
            "Launcher": "expert_model.predict -> step",
        }

    def _render_with_hud(self, hud: _TopCenterHud, lines: dict[str, str]) -> None:
        if not hud.update(lines):
            self._env.render(text=lines)
            return
        self._env.render()

    def run(self) -> None:
        self._env.reset()
        hud = _TopCenterHud()
        if self._on_session_start is not None:
            self._on_session_start(self._env)
        try:
            while True:
                if self._freeze_sim_fn is not None and self._freeze_sim_fn():
                    if self._use_render_text:
                        lines = (
                            self._hud_text_provider()
                            if self._hud_text_provider is not None
                            else self._default_hud_lines()
                        )
                        self._render_with_hud(hud, lines)
                    time.sleep(1.0 / 60.0)
                    continue
                if self._before_step_fn is not None:
                    self._before_step_fn(self._env)
                action = self._policy_fn(self._env)
                step_out = self._env.step(action)
                _obs, _reward, done, info = MetaDriveTransitions.unpack_step(step_out)
                if self._use_render_text:
                    lines = (
                        self._hud_text_provider()
                        if self._hud_text_provider is not None
                        else self._default_hud_lines()
                    )
                    self._render_with_hud(hud, lines)
                if done:
                    if info.get("arrive_dest"):
                        self._env.reset(self._env.current_seed + 1)
                    else:
                        self._env.reset()
        except KeyboardInterrupt:
            print("\nStopped (KeyboardInterrupt).", flush=True)
        finally:
            hud.destroy()


def run_keyboard_session(
    env: MetaDriveEnv,
    *,
    start_with_expert: bool = False,
    record_distillation: bool = True,
) -> None:
    KeyboardSession(
        env,
        start_with_expert=start_with_expert,
        record_distillation=record_distillation,
    ).run()


def run_random_session(env: MetaDriveEnv, max_steps: int) -> None:
    RandomSession(env, max_steps).run()


def run_programmatic_session(
    env: MetaDriveEnv,
    policy_fn: Callable[[MetaDriveEnv], Any],
    *,
    use_render_text: bool = True,
    before_step_fn: Callable[[MetaDriveEnv], None] | None = None,
    on_session_start: Callable[[MetaDriveEnv], None] | None = None,
    hud_text_provider: Callable[[], dict[str, str]] | None = None,
    freeze_sim_fn: Callable[[], bool] | None = None,
) -> None:
    ProgrammaticSession(
        env,
        policy_fn,
        use_render_text=use_render_text,
        before_step_fn=before_step_fn,
        on_session_start=on_session_start,
        hud_text_provider=hud_text_provider,
        freeze_sim_fn=freeze_sim_fn,
    ).run()
