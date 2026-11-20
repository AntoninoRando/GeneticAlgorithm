#!/usr/bin/env python3
"""Interactive menu for the GP scheduler project.

    ./menu.py            arrow keys and Enter (or the number of an entry)
    ./menu.py --plain    numbered menus only (works in any terminal, and on Windows)
    ./menu.py --dry-run  show the commands the menu would run, run nothing

The menu does nothing you could not type yourself: every entry builds one
ordinary command (``python3 main.py ...``, ``scripts/build.sh``, ...), shows
it, and runs it.  The command is always printed first, so the menu is also a
way to learn the commands, and anything you did through it can be repeated
without it.

What it covers
--------------
* Train a formula: pick the data, a starting configuration from
  ``runConfigurations/`` and any setting of ``main.py``; save the result as a
  new named configuration.
* Data: generate realistic data (``data_processing/realistic_trace``), split
  the original log into worlds, write the demo data, show what is there.
* Build the C++ module, run the tests, run the thesis experiments, open the
  documentation, plot a snapshot.

How the file is organised
-------------------------
1. ``SETTINGS``: one row per setting of ``main.py`` (key, flag, type, help).
   ``main.py`` cannot be imported here, because it needs the compiled module
   and the menu must work before the first build.  ``tests/test_menu.py``
   checks that this table and ``main.py`` agree.
2. Command builders: pure functions that turn choices into a command line.
   No input, no output, so they are easy to test.
3. ``Project``: what is on disk (worlds, orbit files, the compiled module).
4. The two user interfaces, ``KeyUI`` (arrow keys) and ``PlainUI`` (numbers).
5. The screens, one function each, and ``main``.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
ORIGINAL_WORLDS = "data/worlds"
REALISTIC_WORLDS = "data/realistic/worlds"
DEMO_WORLDS = "data/demo/worlds"


# ═════════════════════════════════════════════════════════════════════════════
# 1. The settings of main.py
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Setting:
    key: str                 # key of GAConfig in main.py and of the JSON files
    label: str               # what the menu shows
    kind: str                # "int", "float", "bool" or "str"
    default: object          # must equal main.default_config()[key]
    flag: str                # command-line flag; for bool the positive form
    help: str
    group: str
    off_flag: str = ""       # bool only: the negative form (default --no-<flag>)
    minimum: Optional[float] = None
    maximum: Optional[float] = None


SETTINGS: List[Setting] = [
    # ── the search ──────────────────────────────────────────────────────────
    Setting("generations", "Generations", "int", 200, "--generations",
            "How many generations to evolve.", "Search", minimum=1),
    Setting("population_size", "Population size", "int", 100, "--population",
            "Formulas per generation.", "Search", minimum=2),
    Setting("max_depth", "Maximum tree depth", "int", 7, "--max-depth",
            "Largest depth of a formula. The thesis uses 4.", "Search", minimum=1),
    Setting("seed", "Random seed", "int", -1, "--seed",
            "0 or more makes the run reproducible; -1 is different every time.", "Search",
            minimum=-1),
    Setting("convergence_threshold", "Generations before a new world", "int", 10,
            "--convergence-threshold",
            "Generations without improvement before moving to the next world.", "Search",
            minimum=1),
    Setting("worlds_per_eval", "Worlds per evaluation", "int", 1, "--worlds-per-eval",
            "Score every formula on this many worlds at once (3 in the robust setup).",
            "Search", minimum=1),
    Setting("csv_row_limit", "Jobs loaded per world", "int", 100, "--csv-row-limit",
            "0 loads every job of the world.", "Search"),
    # ── operators ───────────────────────────────────────────────────────────
    Setting("hoist_rate", "Hoist rate", "float", 0.05, "--hoist-rate",
            "Chance that an offspring is replaced by one of its subtrees (fights bloat).",
            "Operators", minimum=0.0, maximum=1.0),
    Setting("tournament_size", "Tournament size", "int", 5, "--tournament-size",
            "Candidates per selection tournament: higher means stronger selection.",
            "Operators", minimum=1),
    Setting("initial_max_depth", "Depth of the first trees", "int", -1, "--initial-max-depth",
            "-1 uses the maximum tree depth.", "Operators", minimum=-1),
    Setting("internal_crossover_bias", "Crossover bias to operators", "float", 0.0,
            "--internal-crossover-bias",
            "Chance that a crossover point is an operator, not a leaf (Koza: 0.9).",
            "Operators", minimum=0.0, maximum=1.0),
    Setting("clone_retries", "Clone retries", "int", 0, "--clone-retries",
            "Re-mutate offspring that are copies of a parent (3 in the robust setup).",
            "Operators", minimum=0),
    Setting("depth_aware_mutation", "Depth-aware mutation", "bool", False,
            "--depth-aware-mutation", "Mutation never builds a tree above the depth limit.",
            "Operators"),
    Setting("depth_aware_crossover", "Depth-aware crossover", "bool", False,
            "--depth-aware-crossover", "Crossover never builds a tree above the depth limit.",
            "Operators"),
    Setting("terminal_weighting", "Adaptive terminal weighting", "bool", False,
            "--terminal-weighting", "Mutation prefers the inputs the best formulas use.",
            "Operators"),
    Setting("prune_constant_terminals", "Skip constant inputs", "bool", False,
            "--prune-constant-terminals", "Never draw inputs that are the same in every world.",
            "Operators"),
    # ── decoder and fitness ─────────────────────────────────────────────────
    Setting("fixed_bounds", "Fixed-bound fitness", "bool", False, "--fixed-bounds",
            "Penalise unscheduled jobs; pushes towards scheduling more of them.", "Fitness"),
    Setting("track_busy_until", "Live 'busy until' input", "bool", False, "--track-busy-until",
            "Formulas can see when each satellite becomes free.", "Fitness"),
    Setting("earliest_free_tie_break", "Earliest-free tie-break", "bool", False,
            "--earliest-free-tie-break",
            "Equal scores go to the satellite that is free first.", "Fitness"),
    # ── output ──────────────────────────────────────────────────────────────
    Setting("print_every", "Print progress every", "int", 50, "--print-every",
            "Generations between progress reports; 0 prints none.", "Output", minimum=0),
    Setting("collect_metrics", "Write metrics (CSV, plot)", "bool", True, "--enable-metrics",
            "Per-generation metrics in the metrics folder.", "Output",
            off_flag="--disable-metrics"),
    Setting("metrics_dir", "Metrics folder", "str", "metrics", "--metrics-dir",
            "Where the metrics files go.", "Output"),
    Setting("metrics_prefix", "Metrics file prefix", "str", "", "--metrics-prefix",
            "Empty uses a timestamp.", "Output"),
    Setting("metrics_flush_every", "Refresh metrics every", "int", 1, "--metrics-flush-every",
            "Generations between two writes of the metrics dump.", "Output", minimum=1),
    Setting("inspect_snapshots", "Print world summaries", "bool", False, "--inspect-snapshots",
            "Readable summary of each world when it is loaded.", "Output"),
    Setting("save_snapshots", "Save worlds as JSON", "bool", False, "--save-snapshots",
            "Write each world to the snapshots folder.", "Output"),
    Setting("snapshots_dir", "Snapshots folder", "str", "snapshots", "--snapshots-dir",
            "Where the snapshot files go.", "Output"),
    Setting("snapshots_prefix", "Snapshot file prefix", "str", "", "--snapshots-prefix",
            "Empty uses a timestamp.", "Output"),
    # ── data (set through the "Data" row of the training screen) ────────────
    Setting("demo", "Demo data", "bool", False, "--demo",
            "Toy data: its numbers mean nothing.", "Data"),
    Setting("worlds_dir", "Worlds folder", "str", ORIGINAL_WORLDS, "--worlds-dir",
            "Folder of world_*.csv files to train on.", "Data"),
]
SETTING = {s.key: s for s in SETTINGS}
GROUPS = ["Search", "Operators", "Fitness", "Output"]      # "Data" has its own row


def default_values() -> Dict[str, object]:
    return {s.key: s.default for s in SETTINGS}


def setting_tokens(setting: Setting, value) -> List[str]:
    """Command-line tokens that set ``setting`` to ``value``."""
    if setting.kind == "bool":
        if value:
            return [setting.flag]
        return [setting.off_flag or "--no-" + setting.flag[2:]]
    return [setting.flag, str(value)]


def parse_value(setting: Setting, text: str):
    """Turn typed text into a value of the setting's type; ValueError if invalid."""
    text = text.strip()
    if setting.kind == "str":
        return text
    if setting.kind == "bool":
        if text.lower() in ("y", "yes", "true", "on", "1"):
            return True
        if text.lower() in ("n", "no", "false", "off", "0"):
            return False
        raise ValueError("answer yes or no")
    try:
        value = int(text) if setting.kind == "int" else float(text)
    except ValueError:
        raise ValueError("a whole number is needed" if setting.kind == "int"
                         else "a number is needed") from None
    if setting.minimum is not None and value < setting.minimum:
        raise ValueError(f"must be at least {setting.minimum:g}")
    if setting.maximum is not None and value > setting.maximum:
        raise ValueError(f"must be at most {setting.maximum:g}")
    return value


def show_value(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if value == "":
        return "(empty)"
    return str(value)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Command builders (pure functions)
# ═════════════════════════════════════════════════════════════════════════════
PY = "python3"          # shown in the menu; replaced by sys.executable when run


def train_command(base: Optional[str], base_values: Dict[str, object],
                  values: Dict[str, object]) -> List[str]:
    """``main.py`` command for ``values``, given the configuration it starts from.

    ``base`` is the name of a file in runConfigurations/ (or None), and
    ``base_values`` what that file plus the built-in defaults give.  Only the
    settings that differ from it become flags, so the command stays short and
    shows exactly what you changed.
    """
    command = [PY, "main.py", "--no-start-prompt"]
    if base:
        command += ["-c", base]
    for setting in SETTINGS:
        if values[setting.key] != base_values[setting.key]:
            command += setting_tokens(setting, values[setting.key])
    return command


# Defaults of `python3 -m data_processing.realistic_trace` (checked by the tests).
GENERATOR_DEFAULTS = {
    "runs": 5, "seed": 2026, "rate": 10.0, "energy_budget": 100000.0, "policy": "OrbitAware",
    "energy_cpu_hz": "calibrate", "output_dir": "data/realistic", "log_batch": False,
}


def generator_command(source: str, orbit_file: str, options: Dict[str, object]) -> List[str]:
    """Command of the realistic trace generator.

    ``source`` is "file" (use ``orbit_file``), "download" or "walker".
    """
    command = [PY, "-m", "data_processing.realistic_trace"]
    if source == "download":
        command.append("--download")
    elif source == "walker":
        command.append("--walker")
    elif orbit_file != "data/orbits/starlink.csv":
        command += ["--orbits", orbit_file]
    flags = {"runs": "--runs", "seed": "--seed", "rate": "--rate",
             "energy_budget": "--energy-budget", "policy": "--policy",
             "energy_cpu_hz": "--energy-cpu-hz", "output_dir": "--output-dir"}
    for key, flag in flags.items():
        if options[key] != GENERATOR_DEFAULTS[key]:
            value = options[key]
            command += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
    if options.get("log_batch"):
        command.append("--log-batch")
    return command


def tests_command(test_file: Optional[str]) -> List[str]:
    if test_file is None:
        return [PY, "-m", "unittest", "discover", "-s", "tests", "-v"]
    return [PY, "-m", "unittest", "tests." + Path(test_file).stem, "-v"]


def results_folder(script: str, worlds_dir: str) -> str:
    """Where an experiment writes.  Results on other worlds than the original
    ones never go to the folders the thesis reads."""
    base = {"experiments": "thesis/experiments/results",
            "ablation": "thesis/experiments/results_ablation"}[script]
    if worlds_dir == ORIGINAL_WORLDS:
        return base
    tag = "realistic" if worlds_dir == REALISTIC_WORLDS else \
        re.sub(r"[^A-Za-z0-9]+", "_", worlds_dir).strip("_")
    return f"{base}_{tag}"


def baselines_command(worlds_dir: str) -> List[str]:
    command = [PY, "thesis/experiments/run_experiments.py", "--baselines-only"]
    if worlds_dir != ORIGINAL_WORLDS:
        command += ["--worlds-dir", worlds_dir]
    return command


def experiments_command(worlds_dir: str, seeds: int, out: str) -> List[str]:
    command = [PY, "thesis/experiments/run_experiments.py", "--out", out, "--seeds", str(seeds)]
    if worlds_dir != ORIGINAL_WORLDS:
        command += ["--worlds-dir", worlds_dir]
    return command


def ablation_command(worlds_dir: str, configs: Sequence[str], seeds: int, seed_start: int,
                     procs: Optional[int], out: str) -> List[str]:
    command = [PY, "thesis/experiments/run_ablation.py", "--configs", ",".join(configs),
               "--seeds", str(seeds), "--out", out]
    if seed_start:
        command += ["--seed-start", str(seed_start)]
    if procs:
        command += ["--procs", str(procs)]
    if worlds_dir != ORIGINAL_WORLDS:
        command += ["--worlds-dir", worlds_dir]
    return command


def display(command: Sequence[str]) -> str:
    return shlex.join(command)


# ═════════════════════════════════════════════════════════════════════════════
# 3. What is on disk
# ═════════════════════════════════════════════════════════════════════════════
class Project:
    """Reads the state of the repository.  Nothing here changes a file except
    :meth:`save_run_config`."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self._module: Optional[dict] = None

    # ── data ────────────────────────────────────────────────────────────────
    def count_worlds(self, worlds_dir: str) -> int:
        folder = self.root / worlds_dir
        manifest = folder / "manifest.json"
        if manifest.is_file():
            try:
                return int(json.loads(manifest.read_text(encoding="utf-8"))["num_worlds"])
            except (ValueError, KeyError, OSError):
                pass
        return len(list(folder.glob("world_*.csv"))) if folder.is_dir() else 0

    def has_original_log(self) -> bool:
        return (self.root / "data/example.csv").is_file()

    def orbit_files(self) -> List[str]:
        folder = self.root / "data/orbits"
        if not folder.is_dir():
            return []
        return sorted(str(p.relative_to(self.root)).replace(os.sep, "/")
                      for p in folder.iterdir() if p.suffix.lower() in (".csv", ".tle", ".txt"))

    def count_satellites(self, orbit_file: str) -> int:
        try:
            with (self.root / orbit_file).open(encoding="utf-8-sig") as handle:
                lines = [line for line in handle if line.strip()]
        except OSError:
            return 0
        if lines and "OBJECT_NAME" in lines[0]:
            return len(lines) - 1
        return sum(1 for line in lines if line.startswith("1 "))

    def realistic_report(self) -> Optional[dict]:
        try:
            return json.loads((self.root / "data/realistic/report.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None

    def snapshot_files(self) -> List[str]:
        folder = self.root / "snapshots"
        return sorted(str(p.relative_to(self.root)).replace(os.sep, "/")
                      for p in folder.glob("*.json")) if folder.is_dir() else []

    # ── run configurations ──────────────────────────────────────────────────
    def run_configs(self) -> Dict[str, dict]:
        configs = {}
        for path in sorted((self.root / "runConfigurations").glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                configs[path.stem] = data
        return configs

    def base_values(self, name: Optional[str]) -> Dict[str, object]:
        """Built-in defaults, overridden by the known keys of the named file:
        the same layering as main.resolve_config."""
        values = default_values()
        if name:
            for key, value in self.run_configs().get(name, {}).items():
                if key in SETTING:
                    values[key] = value
        return values

    def save_run_config(self, name: str, values: Dict[str, object]) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("use only letters, digits, - and _")
        path = self.root / "runConfigurations" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = {s.key: values[s.key] for s in SETTINGS}
        path.write_text(json.dumps(ordered, indent=4) + "\n", encoding="utf-8")
        return path

    # ── code ────────────────────────────────────────────────────────────────
    def test_files(self) -> List[str]:
        return sorted(p.name for p in (self.root / "tests").glob("test_*.py"))

    def ablation_configs(self) -> List[str]:
        """Names of the configurations of run_ablation.py, read from its source
        (importing it would need the compiled module)."""
        try:
            tree = ast.parse((self.root / "thesis/experiments/run_ablation.py").read_text("utf-8"))
        except (OSError, SyntaxError):
            return []
        for node in tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and any(getattr(t, "id", "") == "CONFIGS" for t in node.targets)):
                return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
        return []

    def module(self, refresh: bool = False) -> dict:
        """Is the compiled `scheduler` module there, and which build is it?"""
        if self._module is None or refresh:
            info = {"built": False, "build_type": "", "optimized": False, "error": ""}
            try:
                out = subprocess.run(
                    [sys.executable, "-W", "ignore", "-c",
                     "import scheduler; print(getattr(scheduler, 'build_type', '?'), "
                     "getattr(scheduler, 'optimized', True))"],
                    cwd=self.root, capture_output=True, text=True, timeout=60)
                if out.returncode == 0 and out.stdout.split():
                    kind, optimized = out.stdout.split()[:2]
                    info.update(built=True, build_type=kind, optimized=optimized == "True")
                else:
                    info["error"] = (out.stderr.strip().splitlines() or ["not built"])[-1]
            except (OSError, subprocess.TimeoutExpired) as exc:
                info["error"] = str(exc)
            self._module = info
        return self._module

    def has_sgp4(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("sgp4") is not None

    def data_choices(self) -> List[Tuple[str, str, str, str]]:
        """(id, label, hint, reason it cannot be used or "") for each data source."""
        n_orig, n_real = self.count_worlds(ORIGINAL_WORLDS), self.count_worlds(REALISTIC_WORLDS)
        original_ok = n_orig > 0 or self.has_original_log()
        return [
            ("original", "Original log", f"{n_orig} worlds in {ORIGINAL_WORLDS}" if n_orig
             else ("data/example.csv, not split yet" if original_ok else "not on this machine"),
             "" if original_ok else "The original log is not here. Use generated data, or put "
                                    "the log at data/example.csv."),
            ("realistic", "Generated realistic data",
             f"{n_real} worlds in {REALISTIC_WORLDS}" if n_real else "not generated yet",
             "" if n_real else "Generate it first: Data > Generate realistic data."),
            ("demo", "Demo data (toy)", "numbers mean nothing; rewritten at every run", ""),
        ]

    def best_data(self) -> str:
        for ident, _, _, reason in self.data_choices():
            if not reason:
                return ident
        return "demo"


def apply_data(values: Dict[str, object], ident: str, folder: str = "") -> None:
    """Set the two data settings from a data choice."""
    values["demo"] = ident == "demo"
    values["worlds_dir"] = {"original": ORIGINAL_WORLDS, "realistic": REALISTIC_WORLDS,
                            "demo": ORIGINAL_WORLDS}.get(ident, folder or ORIGINAL_WORLDS)


def data_of(values: Dict[str, object]) -> str:
    if values["demo"]:
        return "demo"
    return {ORIGINAL_WORLDS: "original", REALISTIC_WORLDS: "realistic"}.get(
        str(values["worlds_dir"]), "folder")


# ═════════════════════════════════════════════════════════════════════════════
# 4. User interfaces
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Item:
    label: str
    hint: str = ""
    disabled: str = ""       # if set: why the entry cannot be chosen
    heading: bool = False    # a section title, not selectable


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text): return self._wrap("1", text)
    def dim(self, text): return self._wrap("2", text)
    def accent(self, text): return self._wrap("36", text)
    def good(self, text): return self._wrap("32", text)
    def warn(self, text): return self._wrap("33", text)
    def bad(self, text): return self._wrap("31", text)
    def reverse(self, text): return self._wrap("7", text)


class PlainUI:
    """Numbered menus read with input().  Works everywhere and is what the
    tests drive; ``KeyUI`` adds arrow keys on top of it."""

    keys_help = "number + Enter to choose · b back · q quit"

    def __init__(self, read: Callable[[str], str] = input,
                 write: Callable[[str], None] = print, color: Optional[bool] = None):
        self.read, self.write = read, write
        if color is None:
            color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        self.style = Style(bool(color))

    # ── building blocks ─────────────────────────────────────────────────────
    def _numbering(self, items: Sequence[Item]) -> List[Optional[int]]:
        numbers, n = [], 0
        for item in items:
            if item.heading:
                numbers.append(None)
            else:
                n += 1
                numbers.append(n)
        return numbers

    def _parts(self, title: str, header: str, items: Sequence[Item], footer: str,
               current: int = -1) -> Tuple[List[str], List[str], List[str], int]:
        """The screen as (top, entries, bottom, row of the current entry)."""
        s = self.style
        top = ["", s.bold(title)]
        top += [s.dim(line) for line in header.splitlines()] if header else []
        top.append("")
        body: List[str] = []
        at = 0
        width = max((len(i.label) for i in items if not i.heading), default=0)
        for index, (item, number) in enumerate(zip(items, self._numbering(items))):
            if item.heading:
                body += ["", s.dim("  " + item.label)] if item.label else [""]
                continue
            text = f"{number:>3}  {item.label.ljust(width)}"
            hint = item.hint + (" (unavailable)" if item.disabled else "")
            if index == current:
                at = len(body)
                body.append(s.accent(">" + text[1:]) + ("   " + s.dim(hint) if hint else ""))
            elif item.disabled:
                body.append(s.dim(text + ("   " + hint if hint else "")))
            else:
                body.append(text + ("   " + s.dim(hint) if hint else ""))
        bottom = [""] + ([s.dim(line) for line in footer.splitlines()] if footer else [])
        return top, body, bottom, at

    def select(self, title: str, items: Sequence[Item], header: str = "", footer: str = "",
               index: int = 0) -> Optional[int]:
        """Show a menu; return the index of the chosen entry, or None for back."""
        numbers = self._numbering(items)
        while True:
            top, body, bottom, _ = self._parts(title, header, items, footer)
            for line in top + body + bottom:
                self.write(line)
            answer = self.read(self.style.dim(f"[{self.keys_help}] ") + "> ").strip().lower()
            if answer in ("b", "back", ""):
                return None
            if answer in ("q", "quit", "exit"):
                raise SystemExit(0)
            if answer.isdigit() and int(answer) in numbers:
                chosen = numbers.index(int(answer))
                if items[chosen].disabled:
                    self.write(self.style.warn("  Not available: " + items[chosen].disabled))
                    continue
                return chosen
            self.write(self.style.warn("  Type the number of an entry."))

    def multiselect(self, title: str, items: Sequence[Item], chosen: Sequence[int],
                    header: str = "") -> Optional[List[int]]:
        """Tick any number of entries; None for back."""
        picked = set(chosen)
        cursor = len(items)                          # start on "Done"
        while True:
            shown = [Item(("[x] " if i in picked else "[ ] ") + item.label, item.hint)
                     for i, item in enumerate(items)]
            shown.append(Item(f"Done ({len(picked)} selected)"))
            answer = self.select(title, shown, header,
                                 "Choose an entry to tick or untick it, then Done.",
                                 index=cursor)
            if answer is None:
                return None
            if answer == len(items):
                return sorted(picked)
            picked.symmetric_difference_update({answer})
            cursor = answer

    def ask(self, prompt: str, default: str = "", check: Optional[Callable] = None,
            hint: str = ""):
        """Ask for a line of text.  ``check`` converts it and raises ValueError
        with a readable message if it is not acceptable.  Empty keeps the default."""
        while True:
            if hint:
                self.write(self.style.dim("  " + hint))
            suffix = f" [{default}]" if default != "" else ""
            text = self.read(f"  {prompt}{suffix}: ").strip()
            if text == "":
                text = default
            try:
                return check(text) if check else text
            except ValueError as exc:
                self.write(self.style.warn(f"  {exc}"))

    def confirm(self, question: str, default: bool = True) -> bool:
        answer = self.read(f"  {question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        if answer == "":
            return default
        return answer in ("y", "yes")

    def message(self, text: str) -> None:
        self.write(text)

    def pause(self) -> None:
        self.read(self.style.dim("  Press Enter to go back to the menu. "))


class KeyUI(PlainUI):
    """Menus driven by single keys: arrows or j/k to move, Enter to choose, a
    digit to choose directly, q / Esc / Left to go back.  Needs a Unix terminal."""

    keys_help = "↑↓ move · Enter choose · 1-9 jump · q or ← back"

    def __init__(self):
        super().__init__()
        import termios
        import tty
        self._termios, self._tty = termios, tty
        self._fd = sys.stdin.fileno()

    @staticmethod
    def available() -> bool:
        try:
            import termios  # noqa: F401
            import tty      # noqa: F401
        except ImportError:
            return False
        return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

    def _key(self) -> str:
        """Read one key press and name it."""
        import select
        data = os.read(self._fd, 1)
        if data == b"\x1b":                           # Esc alone, or the start of an arrow key
            while select.select([self._fd], [], [], 0.03)[0]:
                data += os.read(self._fd, 1)
                # Sequences are ESC [ ... letter, ESC O letter or ESC [ digits ~
                if len(data) >= 3 and (data[-1:].isalpha() or data[-1:] == b"~"):
                    break
        names = {b"\x1b[A": "up", b"\x1b[B": "down", b"\x1b[C": "right", b"\x1b[D": "left",
                 b"\x1bOA": "up", b"\x1bOB": "down", b"\x1bOC": "right", b"\x1bOD": "left",
                 b"\x1b[H": "home", b"\x1b[F": "end", b"\x1b[1~": "home", b"\x1b[4~": "end",
                 b"\x1b[5~": "pageup", b"\x1b[6~": "pagedown",
                 b"\r": "enter", b"\n": "enter", b"\x1b": "escape", b"\x7f": "left",
                 b"\x03": "interrupt", b"\x04": "quit", b" ": "enter"}
        if data in names:
            return names[data]
        return data.decode("utf-8", "ignore")

    def select(self, title: str, items: Sequence[Item], header: str = "", footer: str = "",
               index: int = 0) -> Optional[int]:
        # Single-key mode for as long as the menu is on screen.  TCSANOW, not
        # the default TCSAFLUSH: keys typed ahead must not be thrown away.
        old = self._termios.tcgetattr(self._fd)
        self._tty.setcbreak(self._fd, self._termios.TCSANOW)
        try:
            return self._choose(title, items, header, footer, index)
        finally:
            self._termios.tcsetattr(self._fd, self._termios.TCSANOW, old)

    def _choose(self, title: str, items: Sequence[Item], header: str, footer: str,
                index: int) -> Optional[int]:
        selectable = [i for i, item in enumerate(items) if not item.heading]
        if not selectable:
            return None
        numbers = self._numbering(items)
        current = index if index in selectable else selectable[0]
        note = ""
        while True:
            top, body, bottom, at = self._parts(title, header, items, footer, current)
            rows = shutil.get_terminal_size((100, 30)).lines
            room = max(5, rows - len(top) - len(bottom) - 3)
            if len(body) > room:                     # scroll so that the cursor stays visible
                start = min(max(0, at - room // 2), len(body) - room)
                body = body[start:start + room]
            sys.stdout.write("\033[2J\033[H" + "\n".join(top + body + bottom) + "\n")
            sys.stdout.write(self.style.warn("  " + note) + "\n" if note else "\n")
            sys.stdout.write(self.style.dim("  " + self.keys_help) + "\n")
            sys.stdout.flush()
            note = ""
            key = self._key()
            where = selectable.index(current)
            if key in ("up", "k"):
                current = selectable[(where - 1) % len(selectable)]
            elif key in ("down", "j", "\t"):
                current = selectable[(where + 1) % len(selectable)]
            elif key in ("home", "pageup"):
                current = selectable[0]
            elif key in ("end", "pagedown"):
                current = selectable[-1]
            elif key in ("left", "escape", "q", "b"):
                return None
            elif key == "interrupt":
                raise KeyboardInterrupt
            elif key == "quit":
                raise SystemExit(0)
            elif key in ("enter", "right") or (key.isdigit() and int(key) in numbers):
                if key.isdigit():
                    current = numbers.index(int(key))
                if items[current].disabled:
                    note = "Not available: " + items[current].disabled
                else:
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.flush()
                    return current


# ═════════════════════════════════════════════════════════════════════════════
# 5. Running commands
# ═════════════════════════════════════════════════════════════════════════════
class Runner:
    """Shows a command and runs it in the repository root."""

    def __init__(self, ui: PlainUI, project: Project, dry_run: bool = False):
        self.ui, self.project, self.dry_run = ui, project, dry_run
        self.history: List[List[str]] = []

    def run(self, command: Sequence[str], pause: bool = True) -> int:
        command = list(command)
        self.history.append(command)
        s = self.ui.style
        self.ui.message("")
        self.ui.message(s.bold("$ ") + s.accent(display(command)))
        if self.dry_run:
            self.ui.message(s.dim("  (dry run: not executed)"))
            code = 0
        else:
            real = [sys.executable if token == PY else token for token in command]
            sys.stdout.flush()                       # our lines first, then the command's
            started = time.time()
            try:
                code = subprocess.call(real, cwd=self.project.root)
            except KeyboardInterrupt:
                code = 130                           # the child got the same Ctrl-C
            except OSError as exc:
                self.ui.message(s.bad(f"  Could not start: {exc}"))
                code = 127
            took = time.time() - started
            verdict = s.good("finished") if code == 0 else (
                s.warn("stopped (Ctrl-C)") if code == 130 else s.bad(f"failed, exit code {code}"))
            self.ui.message(f"\n  {verdict} after {took:.1f} s")
        if pause:
            self.ui.pause()
        return code


# ═════════════════════════════════════════════════════════════════════════════
# 6. Screens
# ═════════════════════════════════════════════════════════════════════════════
def edit_setting(ui: PlainUI, setting: Setting, value):
    """Let the user change one setting; returns the new value."""
    if setting.kind == "bool":
        return not value
    limits = []
    if setting.minimum is not None:
        limits.append(f"at least {setting.minimum:g}")
    if setting.maximum is not None:
        limits.append(f"at most {setting.maximum:g}")
    hint = setting.help + (f" ({', '.join(limits)})" if limits else "")
    if setting.kind == "str":
        hint += " Type - for empty."
        text = ui.ask(setting.label, str(value), hint=hint)
        return "" if text == "-" else text
    return ui.ask(setting.label, str(value), lambda text: parse_value(setting, text), hint)


def choose_worlds(ui: PlainUI, project: Project, title: str, current: str = "",
                  allow_demo: bool = False) -> Optional[Tuple[str, str]]:
    """Pick a data source.  Returns (id, worlds folder) or None."""
    choices = [c for c in project.data_choices() if allow_demo or c[0] != "demo"]
    items = [Item(label, hint, reason) for _, label, hint, reason in choices]
    items.append(Item("Another folder...", "any folder with world_*.csv files"))
    start = next((i for i, c in enumerate(choices) if c[0] == current), 0)
    answer = ui.select(title, items, index=start)
    if answer is None:
        return None
    if answer == len(choices):
        folder = ui.ask("Folder", REALISTIC_WORLDS)
        if project.count_worlds(folder) == 0:
            ui.message(ui.style.warn(f"  No worlds found in {folder}."))
            ui.pause()
            return None
        return "folder", folder
    ident = choices[answer][0]
    return ident, {"original": ORIGINAL_WORLDS, "realistic": REALISTIC_WORLDS}.get(ident, "")


def describe_config(data: dict) -> str:
    values = default_values()
    values.update({k: v for k, v in data.items() if k in SETTING})
    text = (f"{values['generations']} generations, population {values['population_size']}, "
            f"depth {values['max_depth']}")
    extras = [SETTING[k].label.lower() for k in ("fixed_bounds", "clone_retries",
                                                 "worlds_per_eval", "track_busy_until")
              if values[k] != SETTING[k].default]
    return text + ("; " + ", ".join(extras) if extras else "")


def data_label(project: Project, values: Dict[str, object]) -> str:
    ident = data_of(values)
    if ident == "demo":
        return "demo data (toy)"
    count = project.count_worlds(str(values["worlds_dir"]))
    name = {"original": "original log", "realistic": "generated realistic data"}.get(
        ident, str(values["worlds_dir"]))
    return f"{name}, {count} worlds" if count else f"{name}: NOT FOUND"


def screen_train(ui: PlainUI, project: Project, runner: Runner) -> None:
    module = project.module()
    if not module["built"] and not runner.dry_run:
        ui.message(ui.style.warn("\n  The C++ module is not built yet, so nothing can be trained.\n"
                                 "  Build it first: main menu > Build the C++ module."))
        ui.pause()
        return
    base: Optional[str] = None
    values = project.base_values(base)
    apply_data(values, project.best_data())
    cursor = 0
    while True:
        base_values = project.base_values(base)
        command = train_command(base, base_values, values)
        items = [
            Item("Run now"),
            Item("Starting configuration", base or "built-in defaults"),
            Item("Data", data_label(project, values)),
        ]
        rows: List[Optional[Setting]] = [None, None, None]
        for group in GROUPS:
            items.append(Item(group, heading=True))
            rows.append(None)
            for setting in (s for s in SETTINGS if s.group == group):
                changed = values[setting.key] != base_values[setting.key]
                items.append(Item(setting.label,
                                  show_value(values[setting.key]) + ("  *" if changed else "")))
                rows.append(setting)
        items += [Item("", heading=True), Item("Save these settings as a configuration..."),
                  Item("Back to the main menu")]
        rows += [None, None, None]
        save_at, back_at = len(items) - 2, len(items) - 1
        footer = "Command:  " + display(command) + "\n* = changed from the starting configuration"
        if not module["optimized"]:
            footer += "\nWARNING: the module is a Debug build, many times slower. Rebuild it."
        answer = ui.select("Train a scheduling formula", items,
                           "Change what you like, then choose Run now.", footer, cursor)
        if answer is None or answer == back_at:
            return
        cursor = answer
        if answer == 0:
            if data_label(project, values).endswith("NOT FOUND"):
                ui.message(ui.style.warn("\n  The chosen data folder has no worlds. "
                                         "Pick other data, or generate it in the Data menu."))
                ui.pause()
                continue
            runner.run(command)
        elif answer == 1:
            configs = project.run_configs()
            names = list(configs)
            options = [Item("Built-in defaults", describe_config({}))]
            options += [Item(name, describe_config(configs[name])) for name in names]
            picked = ui.select("Starting configuration", options,
                               "Files in runConfigurations/. Your changes are added on top.",
                               index=(names.index(base) + 1) if base in names else 0)
            if picked is not None:
                keep = data_of(values), str(values["worlds_dir"])
                base = names[picked - 1] if picked else None
                values = project.base_values(base)
                chosen = configs.get(base, {}) if base else {}
                if "demo" not in chosen and "worlds_dir" not in chosen:
                    apply_data(values, *keep)        # the file says nothing about data
        elif answer == 2:
            picked = choose_worlds(ui, project, "Which data?", data_of(values), allow_demo=True)
            if picked:
                apply_data(values, *picked)
        elif answer == save_at:
            name = ui.ask("Name of the configuration (file in runConfigurations/)", "my-run")
            try:
                target = project.root / "runConfigurations" / f"{name}.json"
                if target.exists() and not ui.confirm(f"{name}.json exists. Replace it?", False):
                    continue
                path = project.save_run_config(name, values)
            except ValueError as exc:
                ui.message(ui.style.warn(f"  Not saved: {exc}"))
            else:
                base = name
                ui.message(ui.style.good(f"  Saved {path.relative_to(project.root)}. ")
                           + f"Run it any time with:  ./run {name} --no-start-prompt")
            ui.pause()
        elif rows[answer] is not None:
            setting = rows[answer]
            values[setting.key] = edit_setting(ui, setting, values[setting.key])


def screen_generate(ui: PlainUI, project: Project, runner: Runner) -> None:
    if not project.has_sgp4():
        ui.message(ui.style.warn("\n  The generator needs the 'sgp4' package."))
        if ui.confirm("Install it now with pip?", True):
            if runner.run([PY, "-m", "pip", "install", "sgp4"]) != 0:
                return
        else:
            return
    files = project.orbit_files()
    source, orbit_file = ("file", files[-1]) if files else ("download", "data/orbits/starlink.csv")
    for candidate in files:                          # prefer a complete download
        if candidate.endswith("/starlink.csv"):
            orbit_file = candidate
    options = dict(GENERATOR_DEFAULTS)
    cursor = 0
    while True:
        if source == "file":
            source_text = f"{orbit_file} ({project.count_satellites(orbit_file)} satellites)"
        else:
            source_text = {"download": "download from CelesTrak now",
                           "walker": "idealised constellation (NOT Starlink)"}[source]
        command = generator_command(source, orbit_file, options)
        worlds = options["runs"] * 30
        items = [
            Item("Generate now", f"about {worlds} worlds; takes well under a minute"),
            Item("Orbits", source_text),
            Item("Independent runs", f"{options['runs']}  (30 worlds each)"),
            Item("Random seed", str(options["seed"])),
            Item("Requests per second", f"{options['rate']:g}  (the thesis uses 2 to 10)"),
            Item("Battery per satellite [J]", f"{options['energy_budget']:g}  "
                                              "(the thesis uses 80000 to 120000)"),
            Item("Scheduler of the simulated system", str(options["policy"])),
            Item("Energy CPU frequency", str(options["energy_cpu_hz"])),
            Item("Output folder", str(options["output_dir"])),
            Item("Back"),
        ]
        footer = "Command:  " + display(command)
        if options["output_dir"] == "data/realistic" and project.count_worlds(REALISTIC_WORLDS):
            footer += "\nThis replaces the generated data that is there now."
        answer = ui.select("Generate realistic data", items,
                           "A re-run of the published simulation model on real Starlink orbits.\n"
                           "It is generated data: say so with every result.", footer, cursor)
        if answer is None or answer == len(items) - 1:
            return
        cursor = answer
        if answer == 0:
            if runner.run(command) == 0 and source == "download":
                source, orbit_file = "file", "data/orbits/starlink.csv"
        elif answer == 1:
            files = project.orbit_files()
            choices = [Item(f, f"{project.count_satellites(f)} satellites") for f in files]
            choices += [Item("Download the current Starlink orbits",
                             "from celestrak.org, about 1.7 MB; one download every 2 hours"),
                        Item("Idealised constellation",
                             "works offline; NOT Starlink, never report its numbers")]
            picked = ui.select("Which orbits?", choices,
                               "Keep an orbit file once you have it: a new download gives "
                               "different data.")
            if picked is not None:
                if picked < len(files):
                    source, orbit_file = "file", files[picked]
                else:
                    source = "download" if picked == len(files) else "walker"
        elif answer == 2:
            options["runs"] = ui.ask("Independent runs", str(options["runs"]),
                                     lambda t: _number(t, int, 1), "Each run is 300 simulated "
                                     "seconds and gives 30 worlds.")
        elif answer == 3:
            options["seed"] = ui.ask("Random seed", str(options["seed"]),
                                     lambda t: _number(t, int, 0))
        elif answer == 4:
            options["rate"] = ui.ask("Requests per second", f"{options['rate']:g}",
                                     lambda t: _number(t, float, 0.1))
        elif answer == 5:
            options["energy_budget"] = ui.ask("Battery per satellite in joules",
                                              f"{options['energy_budget']:g}",
                                              lambda t: _number(t, float, 1.0))
        elif answer == 6:
            picked = ui.select("Scheduler of the simulated system", [
                Item("OrbitAware", "also requires the task to finish before the satellite sets"),
                Item("DTS", "deadline and energy only")])
            if picked is not None:
                options["policy"] = ("OrbitAware", "DTS")[picked]
        elif answer == 7:
            picked = ui.select("Energy CPU frequency", [
                Item("calibrate", "fit it to the two published figures (recommended)"),
                Item("draw", "one random value per run, like the original simulator"),
                Item("a number...", "in hertz, for example 4.5e9")])
            if picked is not None:
                options["energy_cpu_hz"] = ("calibrate", "draw")[picked] if picked < 2 else \
                    f"{ui.ask('Frequency in hertz', '4.5e9', lambda t: _number(t, float, 1e6)):g}"
        elif answer == 8:
            options["output_dir"] = ui.ask("Output folder", str(options["output_dir"]))


def _number(text: str, kind, minimum):
    try:
        value = kind(text)
    except ValueError:
        raise ValueError("a number is needed") from None
    if value < minimum:
        raise ValueError(f"must be at least {minimum:g}")
    return value


def show_data_status(ui: PlainUI, project: Project) -> None:
    s = ui.style
    lines = ["", s.bold("Data on this machine"), ""]
    for ident, label, hint, reason in project.data_choices():
        mark = s.good("yes") if not reason or ident == "demo" else s.warn("no ")
        lines.append(f"  {mark}  {label}: {hint}")
    files = project.orbit_files()
    lines.append("")
    lines.append("  Orbit files: " + (", ".join(
        f"{f} ({project.count_satellites(f)} satellites)" for f in files) or "none"))
    report = project.realistic_report()
    if report:
        mean = report.get("mean_over_runs", {})
        lines += ["", s.bold("Generated data (data/realistic/report.json)"), "",
                  f"  {report.get('runs')} runs, {report.get('worlds')} worlds, "
                  f"{report.get('rows')} rows; seed {report.get('seed')}",
                  f"  orbits: {report.get('orbits', {}).get('satellites')} satellites, epoch "
                  f"{report.get('orbits', {}).get('median_epoch_utc')}",
                  f"  success rate {mean.get('success_rate', 0):.3f} (published 0.757), "
                  f"rejections due to energy {mean.get('energy_share_of_rejections', 0):.3f} "
                  "(published 0.61)"]
    for line in lines:
        ui.message(line)
    ui.pause()


def screen_data(ui: PlainUI, project: Project, runner: Runner) -> None:
    while True:
        log = project.has_original_log()
        items = [
            Item("Generate realistic data", "from real Starlink orbits; for work without the "
                                            "original log"),
            Item("Show what data is here", "worlds, orbit files, the last generation report"),
            Item("Split the original log into worlds", "data/example.csv -> data/worlds",
                 "" if log else "data/example.csv is not on this machine."),
            Item("Write the demo data", "toy data in data/demo, for trying the program"),
            Item("Back"),
        ]
        answer = ui.select("Data", items)
        if answer is None or answer == 4:
            return
        if answer == 0:
            screen_generate(ui, project, runner)
        elif answer == 1:
            show_data_status(ui, project)
        elif answer == 2:
            window = ui.ask("Length of a world, in the time unit of the log", "10",
                            lambda t: _number(t, float, 0.001))
            command = [PY, "-m", "data_processing.csv_preprocessing.world_splitter"]
            if window != 10:
                command += ["--window-minutes", f"{window:g}"]
            runner.run(command)
        elif answer == 3:
            runner.run([PY, "-m", "data_processing.demo_data"])


def screen_build(ui: PlainUI, project: Project, runner: Runner) -> None:
    while True:
        module = project.module()
        state = (f"built: {module['build_type']}" + ("" if module["optimized"]
                                                    else " (UNOPTIMISED, slow)")
                 if module["built"] else "not built yet")
        need = "" if shutil.which("cmake") else "cmake is not installed on this machine."
        items = [
            Item("Build (optimised)", "the build for every run and experiment", need),
            Item("Rebuild from scratch", "delete build/ first; use after odd build errors", need),
            Item("Debug build", "unoptimised, for gdb only: many times slower", need),
            Item("Check which build Python loads"),
            Item("Back"),
        ]
        answer = ui.select("Build the C++ module", items, "Now: " + state + "\nRebuild after "
                           "every change to bindings.cpp or include/scheduler.")
        if answer is None or answer == 4:
            return
        if answer == 0:
            runner.run(["bash", "scripts/build.sh"])
        elif answer == 1:
            runner.run(["bash", "scripts/build-py-new.sh"])
        elif answer == 2:
            if ui.confirm("A debug build is far too slow for training. Build it anyway?", False):
                runner.run(["bash", "scripts/build-debug.sh"])
        elif answer == 3:
            runner.run([PY, "-c", "import scheduler; print(scheduler.__file__); "
                                  "print('build type:', scheduler.build_type, "
                                  "'| optimised:', scheduler.optimized)"])
        project.module(refresh=True)


def screen_tests(ui: PlainUI, project: Project, runner: Runner) -> None:
    while True:
        files = project.test_files()
        items = [Item("All tests", "about 10 seconds")]
        items += [Item(name) for name in files] + [Item("Back")]
        answer = ui.select("Run the tests", items,
                           "Tests that need the original data are skipped when it is missing.")
        if answer is None or answer == len(items) - 1:
            return
        runner.run(tests_command(None if answer == 0 else files[answer - 1]))


def screen_experiments(ui: PlainUI, project: Project, runner: Runner) -> None:
    while True:
        items = [
            Item("Hand-written baselines and the energy ceiling", "a second; writes nothing"),
            Item("Compare GP options (ablation)", "several seeds per option; minutes to hours"),
            Item("Full thesis study", "main comparison and parameter sweeps; hours"),
            Item("Make the LaTeX tables and plots", "from a results folder"),
            Item("Back"),
        ]
        answer = ui.select("Thesis experiments", items,
                           "Worlds 0 to 19 train, the others test. Results on generated data\n"
                           "go to their own folder and must never be mixed with the others.")
        if answer is None or answer == 4:
            return
        if answer == 3:
            folder = ui.ask("Results folder", "thesis/experiments/results")
            runner.run([PY, "thesis/experiments/make_report.py", folder])
            continue
        picked = choose_worlds(ui, project, "On which worlds?", project.best_data())
        if picked is None:
            continue
        worlds = picked[1]
        if project.count_worlds(worlds) <= 20:
            ui.message(ui.style.warn(f"\n  {worlds} has {project.count_worlds(worlds)} worlds; "
                                     "more than 20 are needed (20 train, the rest test)."))
            ui.pause()
            continue
        if answer == 0:
            runner.run(baselines_command(worlds))
        elif answer == 1:
            names = project.ablation_configs()
            start = [names.index(n) for n in ("baseline", "fixed") if n in names]
            chosen = ui.multiselect("Which options?", [Item(n) for n in names], start,
                                    "'baseline' is the default GP; keep it to get p-values.")
            if not chosen:
                continue
            seeds = ui.ask("Seeds per option", "10", lambda t: _number(t, int, 1),
                           "10 or more for a statistical test.")
            first = ui.ask("First seed", "0", lambda t: _number(t, int, 0),
                           "Use a fresh range to confirm an earlier result.")
            out = ui.ask("Results folder", results_folder("ablation", worlds))
            command = ablation_command(worlds, [names[i] for i in chosen], seeds, first, None, out)
            runs = len(chosen) * seeds
            if ui.confirm(f"{runs} training runs, roughly {max(1, runs // 6)} to "
                          f"{max(2, runs // 3)} minutes on this machine. Start?", True):
                runner.run(command)
        elif answer == 2:
            seeds = ui.ask("Seeds", "10", lambda t: _number(t, int, 1))
            out = ui.ask("Results folder", results_folder("experiments", worlds))
            if ui.confirm("This runs hundreds of trainings and can take hours. Start?", False):
                runner.run(experiments_command(worlds, seeds, out))


def screen_docs(ui: PlainUI, project: Project, runner: Runner) -> None:
    while True:
        items = [
            Item("Open the onboarding guide in a browser", "served at http://localhost:8000; "
                                                           "Ctrl-C stops it"),
            Item("Where are the documents?", "paths of the guide, readme and thesis"),
            Item("Build the API documentation (Sphinx)", "scripts/make-doc.sh"),
            Item("Back"),
        ]
        answer = ui.select("Documentation", items)
        if answer is None or answer == 3:
            return
        if answer == 0:
            ui.message("\n  Open http://localhost:8000 in your browser (in VS Code the port is "
                       "forwarded automatically).\n  Press Ctrl-C here to stop the server.")
            runner.run([PY, "-m", "http.server", "8000", "--directory", "docs/onboarding"])
        elif answer == 1:
            for line in ("", "  Onboarding guide   docs/onboarding/index.html (start here)",
                         "  Realistic data     docs/onboarding/realistic-data.html",
                         "  Readme             readme.md",
                         "  Data pipeline      data_processing/DATA_PIPELINE.md",
                         "  Thesis             thesis/main.tex, thesis/literature_notes.md",
                         "  Every command      ./menu.py --dry-run shows them without running"):
                ui.message(line)
            ui.pause()
        elif answer == 2:
            runner.run(["bash", "scripts/make-doc.sh"])


def screen_tools(ui: PlainUI, project: Project, runner: Runner) -> None:
    while True:
        snapshots = project.snapshot_files()
        items = [
            Item("Plot the timeline of a saved world", f"{len(snapshots)} file(s) in snapshots/",
                 "" if snapshots else "No snapshots yet: train with 'Save worlds as JSON' on."),
            Item("Debug the C++ code with gdb", "builds a debug module, then starts gdb",
                 "" if shutil.which("gdb") else "gdb is not installed on this machine."),
            Item("Back"),
        ]
        answer = ui.select("Tools", items)
        if answer is None or answer == 2:
            return
        if answer == 0:
            picked = ui.select("Which snapshot?", [Item(name) for name in snapshots])
            if picked is not None:
                runner.run([PY, "snapshot_timeline_plot.py", snapshots[picked]])
        elif answer == 1:
            if ui.confirm("This replaces the optimised module by a debug build. Continue?", False):
                runner.run(["bash", "scripts/debug_cpp.sh"])
                project.module(refresh=True)


def status_header(project: Project) -> str:
    module = project.module()
    build = (module["build_type"] + ("" if module["optimized"] else " (slow!)")
             if module["built"] else "NOT BUILT")
    n_orig, n_real = project.count_worlds(ORIGINAL_WORLDS), project.count_worlds(REALISTIC_WORLDS)
    return (f"C++ module: {build}   |   original data: "
            f"{str(n_orig) + ' worlds' if n_orig else 'missing'}   |   generated data: "
            f"{str(n_real) + ' worlds' if n_real else 'none'}")


def screen_main(ui: PlainUI, project: Project, runner: Runner) -> None:
    screens = [
        ("Train a scheduling formula", "choose data and settings, then run main.py",
         screen_train),
        ("Data", "generate realistic data, split the log, see what is here", screen_data),
        ("Build the C++ module", "needed once, and after every C++ change", screen_build),
        ("Run the tests", "all of them or one file", screen_tests),
        ("Thesis experiments", "baselines, comparisons with several seeds, reports",
         screen_experiments),
        ("Documentation", "open the onboarding guide", screen_docs),
        ("Tools", "plot a saved world, debug with gdb", screen_tools),
    ]
    cursor = 0
    while True:
        items = [Item(label, hint) for label, hint, _ in screens] + [Item("Quit")]
        header = status_header(project)
        if runner.dry_run:
            header += "\nDRY RUN: commands are shown, not executed."
        answer = ui.select("GP scheduler", items, header, index=cursor)
        if answer is None or answer == len(screens):
            return
        cursor = answer
        screens[answer][2](ui, project, runner)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive menu for building, running and testing the GP scheduler. "
                    "Every entry shows the command it runs.")
    parser.add_argument("--plain", action="store_true",
                        help="numbered menus instead of arrow keys")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the commands instead of running them")
    args = parser.parse_args(argv)

    ui = KeyUI() if (not args.plain and KeyUI.available()) else PlainUI()
    project = Project()
    runner = Runner(ui, project, dry_run=args.dry_run)
    try:
        screen_main(ui, project, runner)
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
