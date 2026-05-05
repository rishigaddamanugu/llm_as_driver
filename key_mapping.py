
from __future__ import annotations

EXPERT_TOGGLE_KEY = "t"

VIEW_THIRDPERSON = "q"
VIEW_TOPDOWN = "b"


def print_control_context(*, mac: bool) -> None:
    print(
        "\n"
        "--------------------------------------------------------------------\n"
        " WHAT THIS IS\n"
        "--------------------------------------------------------------------\n"
        " MetaDrive is a driving simulator: procedural roads, traffic, checkpoints,\n"
        " speed limits, crashes, and episode resets. You are controlling one car.\n"
        " The 3D view + HUD look busy - that is normal (speed, mini-map, sensors).\n"
        "\n"
        " WHY COLORS / SHADING FEEL WEIRD (especially on Mac)\n"
        "--------------------------------------------------------------------\n"
        + (
            " What you see online (papers, GitHub, YouTube) is almost always Linux or\n"
            " Windows with full OpenGL: HDR tonemap, MSAA, no crash workarounds.\n"
            " macOS only exposes an older OpenGL compatibility layer; Apple deprecated\n"
            " OpenGL years ago. MetaDrive then uses simplified buffers / tonemap paths\n"
            " here so it runs at all - so the image can look \"wrong\" vs reference\n"
            " footage even when your install is fine. Not your fault.\n"
            " Graphics env (optional): METADRIVE_FRAMEBUFFER_SRGB=0|1; METADRIVE_MAC_GL=compat if window fails\n"
            if mac
            else " Reference videos use Linux/Windows + full GL.\n"
        )
        + "\n"
        " TWO PLACES TO WATCH\n"
        "--------------------------------------------------------------------\n"
        " - Game window = driving, camera, **t** (expert). Must have focus.\n"
        " - This Terminal = log lines (e.g. when **t** toggles autopilot).\n"
        "\n"
        " HELP TEXT BELOW IS GENERIC METADRIVE - OVERRIDES FROM THIS LAUNCHER\n"
        "--------------------------------------------------------------------\n"
        " - **t** is NOT listed there: it toggles MetaDrive's bundled PPO \"expert\"\n"
        "   (ML policy shipped inside the pip package - not your LLM project).\n"
        "   Install **torch** in your env to use the Torch expert (see requirements-metadrive-macos.txt);\n"
        "   otherwise MetaDrive uses the numpy expert and may print NumPy matmul warnings.\n"
        " - **S**: stock help says \"braking\" only; this launcher enables **reverse**\n"
        "   so **S** can back up the car, not only brake.\n"
        " - Chase camera: **mouse** orbits the view (stock MetaDrive).\n"
        " - **Q** / **B**: third-person vs top-down (stock MetaDrive).\n"
        " - If the world looks like **flat gray** with **no lane paint**: you may have hit **2**\n"
        "   (wireframe) or **3** (textures off). Tap **2** then **3** once in the game window.\n"
        "\n",
        flush=True,
    )


def print_keyboard_focus_reminder() -> None:
    print(
        "\n>>> **Critical:** Click inside the **MetaDrive game window** so it has keyboard focus.\n"
        "    Keys go to the window, **not** this Terminal - otherwise WASD / mouse / t do nothing.\n"
        ">>> **W A S D** - drive; **S** can reverse (enabled in this launcher).\n"
        ">>> **Mouse** - orbit chase camera (MetaDrive default).\n"
        ">>> **lowercase t** - toggle bundled **PPO expert** autopilot.\n"
        ">>> **Esc** - quit (handled inside MetaDrive).\n",
        flush=True,
    )


def print_expert_toggle_hint() -> None:
    print(
        "When you press **t**, MetaDrive prints `The expert takeover is set to: True/False` in **this** Terminal - "
        "if you never see that line, the game window does not have keyboard focus.\n",
        flush=True,
    )


def render_overlay_labels(*, expert_takeover: bool) -> dict[str, str]:
    return {
        "Driving": "Expert (t toggles)" if expert_takeover else "You (WASD)",
        "Goal": "Follow road / checkpoints - avoid crashes",
        "Launcher": "Mac compat",
    }
