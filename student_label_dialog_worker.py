
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from telemetry_label_llm import MODEL_DIR_DEFAULT, TelemetryLabelLLM
from student_label_vocab_resolve import resolve_training_key


def _load_vocab_data(vocab_path: Path) -> dict[str, object]:
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("vocab json must be an object")
    return data


def _analyze_vocab(vocab_data: dict[str, object]) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    sp: set[str] = set()
    lg: set[str] = set()
    st: set[str] = set()
    label_to_id_raw = vocab_data.get("label_to_id")
    label_to_id: dict[str, int] = {}
    if isinstance(label_to_id_raw, dict):
        label_to_id = {str(k): int(v) for k, v in label_to_id_raw.items()}
        for key in label_to_id:
            for part in key.split("|"):
                if part.startswith("speed_natural_language:"):
                    sp.add(part.split(":", 1)[1])
                elif part.startswith("longitudinal_natural_language:"):
                    lg.add(part.split(":", 1)[1])
                elif part.startswith("steering_natural_language:"):
                    st.add(part.split(":", 1)[1])
    else:
        for k, target in (
            ("speed_id_to_token", sp),
            ("longitudinal_id_to_token", lg),
            ("steering_id_to_token", st),
        ):
            vals = vocab_data.get(k)
            if isinstance(vals, list):
                for item in vals:
                    tok = str(item).strip()
                    if tok and not tok.startswith("__"):
                        target.add(tok)
    return sorted(sp), sorted(lg), sorted(st), label_to_id


def _parse_initial_compact(initial: str) -> dict[str, object]:
    d: dict[str, object] = {
        "speed": "",
        "long": "",
        "steer": "",
        "overspeed": False,
        "off_lane": False,
        "route": "b05",
    }
    if not initial.strip():
        return d
    for part in initial.split("|"):
        if part.startswith("speed_natural_language:"):
            d["speed"] = part.split(":", 1)[1]
        elif part.startswith("longitudinal_natural_language:"):
            d["long"] = part.split(":", 1)[1]
        elif part.startswith("steering_natural_language:"):
            d["steer"] = part.split(":", 1)[1]
        elif part == "overspeed":
            d["overspeed"] = True
        elif part == "off_lane":
            d["off_lane"] = True
        elif part.startswith("route_completion:"):
            suf = part.split(":", 1)[1]
            m = re.match(r"^b(\d{2})$", suf)
            if m:
                d["route"] = f"b{int(m.group(1)):02d}"
            else:
                try:
                    x = max(0.0, min(1.0, float(suf)))
                    idx = min(19, max(0, int(x * 20)))
                    d["route"] = f"b{idx:02d}"
                except ValueError:
                    pass
    return d


def _route_bin_labels() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i in range(20):
        suf = f"b{i:02d}"
        lo = i * 5
        hi = (i + 1) * 5
        out.append((f"Where on route: ~{lo}%–{hi}% along map ({suf})", suf))
    return out


def _bring_to_front(root, dlg_title: str) -> None:
    try:
        root.title(dlg_title)
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.update_idletasks()
    except Exception:
        pass

    def _again() -> None:
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.focus_force()
        except Exception:
            pass

    try:
        root.after(80, _again)
        root.after(250, lambda: root.attributes("-topmost", False))
    except Exception:
        pass


def _run_dialog(
    *,
    vocab_data: dict[str, object],
    initial: str,
    title: str,
    vocab_path: Path,
) -> str | None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    out: list[str | None] = [None]

    root.resizable(True, True)
    root.minsize(820, 540)

    speeds, longs, steers, label_to_id = _analyze_vocab(vocab_data)
    ini = _parse_initial_compact(initial)

    def ensure_in_list(options: list[str], val: str) -> list[str]:
        if val and val not in options:
            return [val] + options
        return options

    speeds = ensure_in_list(speeds, str(ini.get("speed", "")))
    longs = ensure_in_list(longs, str(ini.get("long", "")))
    steers = ensure_in_list(steers, str(ini.get("steer", "")))
    for name, arr in (("speeds", speeds), ("longs", longs), ("steers", steers)):
        if not arr:
            arr.append("")

    body = ttk.Frame(root, padding=10)
    body.pack(fill=tk.BOTH, expand=True)

    vocab_hint = (
        f"{len(label_to_id)} compact labels known"
        if label_to_id
        else "Factorized telemetry vocab loaded"
    )
    accept_hint = (
        "OK resolves nearest route-bin match when needed."
        if label_to_id
        else "OK uses the compact label directly."
    )
    header = (
        f"Vocabulary: {vocab_path}\n"
        f"{vocab_hint} — enter a command and let the local LLM format it. {accept_hint}"
    )
    ttk.Label(body, text=header, wraplength=880, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

    pf = ttk.Frame(body, padding=6)
    pf.pack(fill=tk.BOTH, expand=True)

    speed_var = tk.StringVar(value=str(ini.get("speed") or (speeds[0] if speeds else "")))
    long_var = tk.StringVar(value=str(ini.get("long") or (longs[0] if longs else "")))
    steer_var = tk.StringVar(value=str(ini.get("steer") or (steers[0] if steers else "")))
    os_var = tk.BooleanVar(value=bool(ini.get("overspeed")))
    ol_var = tk.BooleanVar(value=bool(ini.get("off_lane")))

    route_pairs = _route_bin_labels()
    route_disp_to_suf = {d: s for d, s in route_pairs}
    route_displays = [d for d, _ in route_pairs]
    want_route = str(ini.get("route", "b05"))
    default_rd = next((d for d, s in route_pairs if s == want_route), route_pairs[5][0])
    route_var = tk.StringVar(value=default_rd)

    preview_var = tk.StringVar(value="")
    err_var = tk.StringVar(value="")

    def build_compact() -> str:
        sp = speed_var.get().strip()
        lg = long_var.get().strip()
        st = steer_var.get().strip()
        parts = [
            f"speed_natural_language:{sp}",
            f"longitudinal_natural_language:{lg}",
            f"steering_natural_language:{st}",
        ]
        if os_var.get():
            parts.append("overspeed")
        if ol_var.get():
            parts.append("off_lane")
        rd = route_var.get()
        suf = route_disp_to_suf.get(rd, "b05")
        parts.append(f"route_completion:{suf}")
        return "|".join(parts)

    def update_preview(*_a: object) -> None:
        preview_var.set(build_compact())

    llm_cmd_var = tk.StringVar(value="")
    llm_status_var = tk.StringVar(value="")
    llm_formatted_var = tk.StringVar(value="")
    llm_holder: dict[str, object] = {"model": None}

    row = 0
    ttk.Label(pf, text="Natural-language command", font=("TkDefaultFont", 10, "bold")).grid(
        row=row, column=0, sticky=tk.W, pady=(0, 2)
    )
    ttk.Label(
        pf,
        text="Type your command and click Generate.",
        foreground="#555",
    ).grid(row=row + 1, column=0, columnspan=2, sticky=tk.W)
    llm_cmd = ttk.Entry(pf, textvariable=llm_cmd_var, width=62)
    llm_cmd.grid(row=row + 2, column=0, sticky=tk.EW, pady=(0, 6))
    ttk.Button(pf, text="Generate", command=lambda: apply_llm()).grid(
        row=row + 2, column=1, sticky=tk.E
    )
    ttk.Label(pf, textvariable=llm_status_var, foreground="#555").grid(
        row=row + 3, column=0, columnspan=2, sticky=tk.W, pady=(0, 10)
    )
    ttk.Label(pf, text="Formatted output", font=("TkDefaultFont", 10, "bold")).grid(
        row=row + 4, column=0, sticky=tk.W, pady=(0, 2)
    )
    ttk.Label(
        pf,
        textvariable=llm_formatted_var,
        justify=tk.LEFT,
        wraplength=860,
        foreground="#222",
    ).grid(row=row + 5, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

    def apply_llm() -> None:
        cmd = llm_cmd_var.get().strip()
        if not cmd:
            llm_status_var.set("Enter a command first.")
            llm_formatted_var.set("")
            return
        try:
            model = llm_holder.get("model")
            if model is None:
                model = TelemetryLabelLLM(model_dir=MODEL_DIR_DEFAULT)
                llm_holder["model"] = model
            result = model.format(cmd)
            speed_var.set(result.speed)
            long_var.set(result.longitudinal)
            steer_var.set(result.steering)
            os_var.set(result.overspeed)
            ol_var.set(result.off_lane)
            rd = next((d for d, s in route_pairs if s == result.route_bin), route_pairs[5][0])
            route_var.set(rd)
            update_preview()
            err_var.set("")
            llm_status_var.set("Command formatted.")
            llm_formatted_var.set(
                "speed={speed} | longitudinal={longitudinal} | steering={steering} | "
                "overspeed={overspeed} | off_lane={off_lane} | route={route}\n{compact}".format(
                    speed=result.speed,
                    longitudinal=result.longitudinal,
                    steering=result.steering,
                    overspeed="true" if result.overspeed else "false",
                    off_lane="true" if result.off_lane else "false",
                    route=result.route_bin,
                    compact=result.to_compact_label(),
                )
            )
        except Exception as e:
            llm_status_var.set(f"LLM error: {e}")
            llm_formatted_var.set("")
    llm_cmd.bind("<Return>", lambda _e: apply_llm())
    pf.columnconfigure(0, weight=1)

    def finish(value: str | None) -> None:
        out[0] = value
        root.quit()

    def try_resolve_and_finish(cand: str) -> bool:
        if not label_to_id:

            finish(cand)
            return True
        key, mode = resolve_training_key(cand, label_to_id)
        if key is None:
            err_var.set(
                "That combination is not in the training vocabulary (including nearest route-bin match). "
                "Try another preset or adjust speed / gas / steer / route, or pick an exact key from Browse."
            )
            return False
        if mode == "nearest_route":
            print(
                f"[student-worker] Resolved via nearest route bin: {cand!r} -> {key!r}",
                file=sys.stderr,
                flush=True,
            )
        finish(key)
        return True

    def on_ok() -> None:
        err_var.set("")
        cand = build_compact()
        try_resolve_and_finish(cand)

    def on_cancel() -> None:
        finish(None)

    ttk.Label(body, textvariable=err_var, foreground="#b00020", wraplength=880).pack(
        anchor=tk.W, pady=(4, 6)
    )

    bf_btn = ttk.Frame(body)
    bf_btn.pack(fill=tk.X)
    ttk.Button(bf_btn, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(bf_btn, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)

    root.protocol("WM_DELETE_WINDOW", on_cancel)

    root.update_idletasks()
    w = max(840, root.winfo_reqwidth())
    h = max(560, root.winfo_reqheight())
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 6)
    root.geometry(f"{int(min(sw - 40, w))}x{int(min(sh - 60, h))}+{x}+{y}")

    _bring_to_front(root, title)
    llm_cmd.focus_set()

    print(
        "[student-worker] Label window opened. If hidden: Cmd+Tab or Dock → Python/Tk.",
        file=sys.stderr,
        flush=True,
    )

    root.mainloop()

    try:
        root.destroy()
    except Exception:
        pass
    return out[0]


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": f"stdin: {e}"}), flush=True)
        return 2

    vocab_path = Path(payload["vocab_path"])
    initial = str(payload.get("initial", ""))
    title = str(payload.get("title", "Telemetry label override"))

    try:
        vocab_data = _load_vocab_data(vocab_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        return 3

    try:
        label = _run_dialog(
            vocab_data=vocab_data,
            initial=initial,
            title=title,
            vocab_path=vocab_path.resolve(),
        )
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        return 4

    if label is None:
        print(json.dumps({"ok": False}), flush=True)
        return 0
    print(json.dumps({"ok": True, "label": label}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
