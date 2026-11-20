"""Tests for the interactive menu (menu.py).

Run from the repository root, after building the `scheduler` extension:

    python3 -m unittest discover -s tests -v

The menu only builds command lines and runs them, so the tests check three
things: that its table of settings agrees with main.py (every flag it can
produce is understood, with the meaning the menu shows), that the command
builders give the commands documented in the guide, and that a scripted
session through the menus ends in the expected command.  Sessions run in
"dry run" mode: nothing is executed.  None of the tests needs data/.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.getcwd())

import main  # noqa: E402
import menu  # noqa: E402


def resolve(argv, config_dir="runConfigurations"):
    with contextlib.redirect_stdout(io.StringIO()):
        return main.resolve_config(list(argv), config_dir=config_dir)[0]


def other_value(setting):
    """A valid value different from the default."""
    if setting.kind == "bool":
        return not setting.default
    if setting.kind == "int":
        return setting.default + 3
    if setting.kind == "float":
        return 0.25 if setting.default != 0.25 else 0.5
    return "some/where" if "dir" in setting.key else "tag"


class SettingsTable(unittest.TestCase):
    def test_same_keys_and_defaults_as_main(self):
        self.assertEqual({s.key for s in menu.SETTINGS}, set(main.GAConfig.__annotations__))
        self.assertEqual(menu.default_values(), dict(main.default_config()))
        self.assertEqual(len(menu.SETTINGS), len(menu.SETTING))          # no key twice

    def test_every_setting_reaches_main_through_its_flag(self):
        for setting in menu.SETTINGS:
            value = other_value(setting)
            config = resolve(menu.setting_tokens(setting, value))
            self.assertEqual(config[setting.key], value, setting.key)
            self.assertEqual(type(config[setting.key]), type(setting.default), setting.key)

    def test_a_switch_can_be_turned_off_again(self):
        # A configuration file turns everything on; the menu's "off" tokens must win.
        switches = [s for s in menu.SETTINGS if s.kind == "bool"]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "all-on.json").write_text(json.dumps({s.key: True for s in switches}))
            for setting in switches:
                config = resolve(["-c", "all-on"] + menu.setting_tokens(setting, False), folder)
                self.assertIs(config[setting.key], False, setting.key)
                self.assertIs(resolve(["-c", "all-on"], folder)[setting.key], True)

    def test_typed_values_are_checked(self):
        generations = menu.SETTING["generations"]
        self.assertEqual(menu.parse_value(generations, " 40 "), 40)
        for bad in ("0", "-3", "many", "2.5"):
            with self.assertRaises(ValueError):
                menu.parse_value(generations, bad)
        hoist = menu.SETTING["hoist_rate"]
        self.assertEqual(menu.parse_value(hoist, "0.2"), 0.2)
        with self.assertRaises(ValueError):
            menu.parse_value(hoist, "1.5")
        self.assertEqual(menu.parse_value(menu.SETTING["seed"], "-1"), -1)
        self.assertEqual(menu.show_value(True), "on")
        self.assertEqual(menu.show_value(""), "(empty)")

    def test_every_group_is_shown(self):
        shown = {s.group for s in menu.SETTINGS} - {"Data"}
        self.assertEqual(shown, set(menu.GROUPS))


class Commands(unittest.TestCase):
    def setUp(self):
        self.project = menu.Project()

    def test_training_command_lists_only_what_changed(self):
        base = self.project.base_values(None)
        self.assertEqual(menu.train_command(None, base, dict(base)),
                         ["python3", "main.py", "--no-start-prompt"])
        values = dict(base, generations=50, fixed_bounds=True)
        menu.apply_data(values, "realistic")
        command = menu.train_command(None, base, values)
        self.assertEqual(command, ["python3", "main.py", "--no-start-prompt",
                                   "--generations", "50", "--fixed-bounds",
                                   "--worlds-dir", "data/realistic/worlds"])
        # main.py reads the command as the menu means it.
        config = resolve(command[2:])
        self.assertEqual({k: config[k] for k in values}, values)

    def test_training_command_on_top_of_a_configuration_file(self):
        base = self.project.base_values("robust")
        self.assertEqual(base["clone_retries"], 3)                      # from robust.json
        self.assertEqual(base["tournament_size"], 5)                    # built-in default
        values = dict(base, clone_retries=0, seed=7)
        command = menu.train_command("robust", base, values)
        self.assertEqual(command, ["python3", "main.py", "--no-start-prompt", "-c", "robust",
                                   "--seed", "7", "--clone-retries", "0"])
        config = resolve(command[2:])
        self.assertEqual({k: config[k] for k in values}, values)

    def test_data_choice_overrides_a_file_that_selects_other_data(self):
        base = self.project.base_values("demo")                         # demo.json: demo on
        self.assertTrue(base["demo"])
        values = dict(base)
        menu.apply_data(values, "realistic")
        command = menu.train_command("demo", base, values)
        self.assertIn("--no-demo", command)
        config = resolve(command[2:])
        self.assertFalse(config["demo"])
        self.assertEqual(config["worlds_dir"], "data/realistic/worlds")
        self.assertEqual(menu.data_of(values), "realistic")
        menu.apply_data(values, "demo")
        self.assertEqual(menu.data_of(values), "demo")
        menu.apply_data(values, "folder", "somewhere/worlds")
        self.assertEqual(menu.data_of(values), "folder")

    def test_generator_command(self):
        defaults = dict(menu.GENERATOR_DEFAULTS)
        module = ["python3", "-m", "data_processing.realistic_trace"]
        self.assertEqual(menu.generator_command("file", "data/orbits/starlink.csv", defaults),
                         module)
        self.assertEqual(menu.generator_command("download", "", defaults), module + ["--download"])
        options = dict(defaults, runs=2, rate=4.0, energy_budget=80000.0, policy="DTS",
                       energy_cpu_hz="draw", seed=9, output_dir="data/other")
        self.assertEqual(
            menu.generator_command("file", "data/orbits/starlink_partial_2026-11-19.csv", options),
            module + ["--orbits", "data/orbits/starlink_partial_2026-11-19.csv", "--runs", "2",
                      "--seed", "9", "--rate", "4", "--energy-budget", "80000", "--policy", "DTS",
                      "--energy-cpu-hz", "draw", "--output-dir", "data/other"])
        self.assertEqual(menu.generator_command("walker", "", dict(defaults, log_batch=True)),
                         module + ["--walker", "--log-batch"])

    def test_generator_defaults_match_the_generator(self):
        try:
            from data_processing.realistic_trace import generate
        except (ImportError, SystemExit):
            self.skipTest("needs the sgp4 package")
        parsed = generate._build_parser().parse_args([])
        self.assertEqual(menu.GENERATOR_DEFAULTS, {
            "runs": parsed.runs, "seed": parsed.seed, "rate": parsed.rate,
            "energy_budget": parsed.energy_budget, "policy": parsed.policy,
            "energy_cpu_hz": parsed.energy_cpu_hz, "output_dir": parsed.output_dir,
            "log_batch": parsed.log_batch})
        self.assertEqual(parsed.orbits, "data/orbits/starlink.csv")
        # The options the menu can produce are accepted by the generator's parser.
        options = dict(menu.GENERATOR_DEFAULTS, runs=2, rate=4.0, energy_cpu_hz="4.5e+09")
        parsed = generate._build_parser().parse_args(
            menu.generator_command("walker", "", options)[3:])
        self.assertEqual((parsed.walker, parsed.runs, parsed.rate), (True, 2, 4.0))

    def test_experiment_commands_keep_generated_results_apart(self):
        self.assertEqual(menu.results_folder("ablation", "data/worlds"),
                         "thesis/experiments/results_ablation")
        self.assertEqual(menu.results_folder("ablation", "data/realistic/worlds"),
                         "thesis/experiments/results_ablation_realistic")
        self.assertEqual(menu.results_folder("experiments", "my data/worlds"),
                         "thesis/experiments/results_my_data_worlds")
        self.assertEqual(menu.baselines_command("data/worlds"),
                         ["python3", "thesis/experiments/run_experiments.py", "--baselines-only"])
        self.assertEqual(menu.baselines_command("data/realistic/worlds")[-2:],
                         ["--worlds-dir", "data/realistic/worlds"])
        command = menu.ablation_command("data/realistic/worlds", ["baseline", "fixed"], 10, 20,
                                        None, "out")
        self.assertEqual(command, ["python3", "thesis/experiments/run_ablation.py", "--configs",
                                   "baseline,fixed", "--seeds", "10", "--out", "out",
                                   "--seed-start", "20", "--worlds-dir", "data/realistic/worlds"])
        self.assertEqual(menu.experiments_command("data/worlds", 5, "out"),
                         ["python3", "thesis/experiments/run_experiments.py", "--out", "out",
                          "--seeds", "5"])

    def test_ablation_names_are_read_from_the_script(self):
        sys.path.insert(0, "thesis/experiments")
        self.addCleanup(sys.path.remove, "thesis/experiments")
        import run_ablation
        self.assertEqual(self.project.ablation_configs(), list(run_ablation.CONFIGS))

    def test_test_commands(self):
        self.assertEqual(menu.tests_command(None),
                         ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"])
        self.assertEqual(menu.tests_command("test_menu.py"),
                         ["python3", "-m", "unittest", "tests.test_menu", "-v"])
        self.assertIn("test_menu.py", self.project.test_files())


class ProjectOnDisk(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.project = menu.Project(self.root)

    def test_worlds_are_counted_from_the_manifest_or_the_files(self):
        self.assertEqual(self.project.count_worlds("data/worlds"), 0)
        folder = self.root / "data/worlds"
        folder.mkdir(parents=True)
        for k in range(3):
            (folder / f"world_{k:03d}.csv").write_text("x\n")
        self.assertEqual(self.project.count_worlds("data/worlds"), 3)
        (folder / "manifest.json").write_text(json.dumps({"num_worlds": 30}))
        self.assertEqual(self.project.count_worlds("data/worlds"), 30)

    def test_data_choices_follow_what_exists(self):
        choices = {c[0]: c for c in self.project.data_choices()}
        self.assertTrue(choices["original"][3] and choices["realistic"][3])   # both unavailable
        self.assertEqual(choices["demo"][3], "")
        self.assertEqual(self.project.best_data(), "demo")
        folder = self.root / "data/realistic/worlds"
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"num_worlds": 150}))
        self.assertEqual(self.project.best_data(), "realistic")
        self.assertIn("150 worlds", {c[0]: c for c in self.project.data_choices()}["realistic"][2])
        (self.root / "data/example.csv").write_text("a\n")
        self.assertEqual(self.project.best_data(), "original")

    def test_orbit_files(self):
        self.assertEqual(self.project.orbit_files(), [])
        folder = self.root / "data/orbits"
        folder.mkdir(parents=True)
        (folder / "starlink.csv").write_text("OBJECT_NAME,X\nA,1\nB,2\n")
        (folder / "iss.tle").write_text("ISS\n1 25544U\n2 25544\n")
        (folder / "notes.md").write_text("ignored")
        self.assertEqual(self.project.orbit_files(), ["data/orbits/iss.tle",
                                                      "data/orbits/starlink.csv"])
        self.assertEqual(self.project.count_satellites("data/orbits/starlink.csv"), 2)
        self.assertEqual(self.project.count_satellites("data/orbits/iss.tle"), 1)
        self.assertEqual(self.project.count_satellites("data/orbits/none.csv"), 0)

    def test_saved_configuration_is_read_back_by_main(self):
        values = dict(menu.default_values(), generations=40, fixed_bounds=True, seed=3)
        menu.apply_data(values, "realistic")
        path = self.project.save_run_config("my-run", values)
        self.assertEqual(path, self.root / "runConfigurations/my-run.json")
        self.assertEqual(dict(resolve(["-c", "my-run"], str(path.parent))), values)
        self.assertEqual(self.project.base_values("my-run"), values)
        self.assertEqual(list(self.project.run_configs()), ["my-run"])
        # Started from the saved file, an unchanged run needs no further flag.
        self.assertEqual(menu.train_command("my-run", values, dict(values)),
                         ["python3", "main.py", "--no-start-prompt", "-c", "my-run"])
        for bad in ("", "../x", "a b", "run.json"):
            with self.assertRaises(ValueError):
                self.project.save_run_config(bad, values)


class Script:
    """Answers for a PlainUI session, and everything it printed."""

    def __init__(self, answers):
        self.answers, self.lines = list(answers), []

    def read(self, prompt=""):
        self.lines.append(prompt)
        if not self.answers:
            raise EOFError
        return str(self.answers.pop(0))

    def write(self, text=""):
        self.lines.append(str(text))

    @property
    def text(self):
        return "\n".join(self.lines)


def number_of(setting_key):
    """Number of a setting's row on the training screen (after Run, Config, Data)."""
    rows = [s for group in menu.GROUPS for s in menu.SETTINGS if s.group == group]
    return 4 + rows.index(menu.SETTING[setting_key])


class PlainSessions(unittest.TestCase):
    """Whole sessions through the numbered menus, in dry-run mode."""

    def session(self, answers, project=None):
        script = Script(answers)
        ui = menu.PlainUI(script.read, script.write, color=False)
        project = project or menu.Project()
        runner = menu.Runner(ui, project, dry_run=True)
        try:
            menu.screen_main(ui, project, runner)
        except (EOFError, SystemExit):
            pass
        return runner.history, script.text

    def test_train_with_two_changes(self):
        history, text = self.session([
            1,                                              # Train
            number_of("generations"), 25,
            number_of("fixed_bounds"),                      # a switch: choosing it toggles it
            1, "",                                          # Run now, Enter to come back
            "b", "q"])
        self.assertEqual(len(history), 1)
        command = history[0]
        self.assertEqual(command[:3], ["python3", "main.py", "--no-start-prompt"])
        self.assertIn("--fixed-bounds", command)
        self.assertEqual(command[command.index("--generations") + 1], "25")
        self.assertIn("$ " + menu.display(command), text)
        self.assertIn("dry run", text)
        self.assertIn("* = changed from the starting configuration", text)

    def test_wrong_input_is_explained_not_accepted(self):
        history, text = self.session([
            1, number_of("generations"), "lots", 0, 30, 1, "", "b", "q"])
        self.assertIn("a whole number is needed", text)
        self.assertIn("must be at least 1", text)
        self.assertEqual(history[0][history[0].index("--generations") + 1], "30")

    def test_starting_configuration_and_save(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runConfigurations").mkdir()
            (root / "runConfigurations/quick.json").write_text(
                json.dumps({"generations": 12, "population_size": 30}))
            project = menu.Project(root)
            save_number = 4 + len([s for s in menu.SETTINGS if s.group in menu.GROUPS])
            history, text = self.session([
                1,
                2, 2,                                       # starting configuration: quick
                number_of("seed"), 5,
                save_number, "seeded", "",                  # save as "seeded", Enter
                1, "", "b", "q"], project)
            saved = json.loads((root / "runConfigurations/seeded.json").read_text())
            self.assertEqual((saved["generations"], saved["population_size"], saved["seed"]),
                             (12, 30, 5))
            # After saving, the run starts from the new file and needs no extra flag.
            self.assertEqual(history[-1][:5],
                             ["python3", "main.py", "--no-start-prompt", "-c", "seeded"])
            self.assertNotIn("--seed", history[-1])
            self.assertIn("./run seeded --no-start-prompt", text)

    def test_unavailable_entries_say_why(self):
        with tempfile.TemporaryDirectory() as folder:
            history, text = self.session([2, 3, "b", "q"], menu.Project(Path(folder)))
        self.assertEqual(history, [])
        self.assertIn("Not available: data/example.csv is not on this machine.", text)

    def test_tests_and_baselines(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tests").mkdir()
            (root / "tests/test_alpha.py").write_text("")
            worlds = root / "data/realistic/worlds"
            worlds.mkdir(parents=True)
            (worlds / "manifest.json").write_text(json.dumps({"num_worlds": 150}))
            history, _ = self.session([
                4, 2, "", "b",                              # Tests > test_alpha.py
                5, 1, 2, "", "b",                           # Experiments > baselines > realistic
                "q"], menu.Project(root))
        self.assertEqual(history, [
            ["python3", "-m", "unittest", "tests.test_alpha", "-v"],
            ["python3", "thesis/experiments/run_experiments.py", "--baselines-only",
             "--worlds-dir", "data/realistic/worlds"]])

    def test_ablation_needs_enough_worlds_and_a_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "thesis/experiments").mkdir(parents=True)
            (root / "thesis/experiments/run_ablation.py").write_text(
                'CONFIGS = {"baseline": dict(), "fixed": dict(fixedBounds=True), "x": dict()}\n')
            worlds = root / "data/realistic/worlds"
            worlds.mkdir(parents=True)
            (worlds / "manifest.json").write_text(json.dumps({"num_worlds": 150}))
            history, text = self.session([
                5, 2, 2,                                    # Experiments > ablation > realistic
                3, 4,                                       # tick "x", then Done
                4, 0, "",                                   # 4 seeds, first seed 0, default folder
                "y", "", "b", "q"], menu.Project(root))
        self.assertEqual(history, [[
            "python3", "thesis/experiments/run_ablation.py", "--configs", "baseline,fixed,x",
            "--seeds", "4", "--out", "thesis/experiments/results_ablation_realistic",
            "--worlds-dir", "data/realistic/worlds"]])
        self.assertIn("12 training runs", text)

    def test_generate_with_the_idealised_constellation(self):
        with tempfile.TemporaryDirectory() as folder:
            project = menu.Project(Path(folder))
            project.has_sgp4 = lambda: True
            history, text = self.session([
                2, 1,                                       # Data > Generate
                2, 2,                                       # Orbits > idealised (no files here)
                3, 2,                                       # runs: 2
                1, "", "b", "b", "q"], project)
        self.assertEqual(history, [["python3", "-m", "data_processing.realistic_trace",
                                    "--walker", "--runs", "2"]])
        self.assertIn("about 60 worlds", text)
        self.assertIn("NOT Starlink", text)


@unittest.skipIf(os.name == "nt", "the arrow-key interface needs a Unix terminal")
class ArrowKeys(unittest.TestCase):
    """menu.py in a real pseudo-terminal, driven by key presses."""

    def setUp(self):
        import pty
        import subprocess
        master, slave = pty.openpty()
        env = dict(os.environ, TERM="xterm", LINES="45", COLUMNS="140")
        env.pop("NO_COLOR", None)
        self.proc = subprocess.Popen([sys.executable, "menu.py", "--dry-run"], stdin=slave,
                                     stdout=slave, stderr=slave, env=env, close_fds=True)
        os.close(slave)
        self.master, self.seen = master, ""
        self.addCleanup(os.close, master)
        self.addCleanup(self.proc.kill)

    def wait_for(self, text, timeout=30.0):
        import select
        import time
        deadline = time.time() + timeout
        while text not in self.seen:
            left = deadline - time.time()
            if left <= 0 or not select.select([self.master], [], [], left)[0]:
                self.fail(f"did not see {text!r}; last output:\n{self.seen[-1500:]}")
            try:
                self.seen += os.read(self.master, 65536).decode("utf-8", "replace")
            except OSError:
                break
        self.assertIn(text, self.seen)

    def press(self, keys):
        self.seen = ""
        os.write(self.master, keys)

    def test_navigate_with_keys(self):
        self.wait_for("↑↓ move")
        self.assertIn("GP scheduler", self.seen)
        self.assertIn("DRY RUN", self.seen)
        self.press(b"\x1b[B")                               # Down: cursor on "Data"
        self.wait_for("↑↓ move")
        self.assertRegex(self.seen, r"> 2  Data")
        self.press(b"\x1b[A\r")                             # Up, Enter: the training screen
        self.wait_for("Change what you like")               # header of the training screen
        self.wait_for("↑↓ move")
        self.press(b"jjj\r")                                # three down: Generations
        self.wait_for("Generations [")
        self.press(b"33\n")
        self.wait_for("--generations 33")                   # shown in the command preview
        self.wait_for("↑↓ move")
        self.press(b"1")                                    # digit: Run now
        self.wait_for("dry run")
        self.assertIn("main.py --no-start-prompt", self.seen)
        self.assertIn("--generations 33", self.seen)
        self.press(b"\n")                                   # back to the training screen
        self.wait_for("↑↓ move")
        self.press(b"\x1b")                                 # Esc: main menu
        self.wait_for("GP scheduler")
        self.wait_for("↑↓ move")
        self.press(b"q")                                    # quit
        self.assertEqual(self.proc.wait(timeout=15), 0)


if __name__ == "__main__":
    unittest.main()
