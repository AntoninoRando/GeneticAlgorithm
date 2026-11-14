"""Tests for the configuration layering of main.py.

Run from the repository root, after building the `scheduler` extension
(main.py imports it):

    python3 -m unittest discover -s tests -v

The rule under test: built-in defaults, then the JSON file named by -c, then
the flags TYPED on the command line.  A typed flag wins whatever its value.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.getcwd())

import main  # noqa: E402


class ConfigLayering(unittest.TestCase):
    def setUp(self):
        # A private configuration directory, so the tests do not depend on the
        # contents of runConfigurations/.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_dir = self._tmp.name
        self.write("lab", {
            "generations": 5000,
            "max_depth": 4,
            "clone_retries": 3,
            "terminal_weighting": True,
            "fixed_bounds": True,
            "collect_metrics": False,
            "metrics_prefix": "from_file",
            "someFutureKey": {"ignored": True},
        })

    def write(self, name, data):
        with open(os.path.join(self.config_dir, f"{name}.json"), "w") as f:
            json.dump(data, f)

    def resolve(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return main.resolve_config(list(argv), config_dir=self.config_dir)

    # ── defaults ────────────────────────────────────────────────────────────
    def test_no_arguments_gives_the_built_in_defaults(self):
        config, start_prompt = self.resolve()
        self.assertEqual(config, main.default_config())
        self.assertTrue(start_prompt)

    def test_every_option_maps_to_a_config_key(self):
        # dest == GAConfig key for every option, except the two that are not
        # settings of the run.  Guards against adding a flag that is parsed
        # and then silently dropped.
        keys = set(main.GAConfig.__annotations__)
        dests = {a.dest for a in main.parser._actions} - {"help", "config", "no_start_prompt"}
        self.assertEqual(dests - keys, set())

    def test_defaults_cover_every_config_key(self):
        self.assertEqual(set(main.default_config()), set(main.GAConfig.__annotations__))

    # ── file over defaults ──────────────────────────────────────────────────
    def test_file_overrides_defaults_and_ignores_unknown_keys(self):
        config, _ = self.resolve("-c", "lab")
        self.assertEqual(config["generations"], 5000)
        self.assertEqual(config["max_depth"], 4)
        self.assertTrue(config["terminal_weighting"])
        self.assertFalse(config["collect_metrics"])
        self.assertNotIn("someFutureKey", config)
        # Keys the file does not mention keep their default.
        self.assertEqual(config["population_size"], main.default_config()["population_size"])

    # ── typed flags over the file ───────────────────────────────────────────
    def test_typed_flag_equal_to_the_default_still_overrides_the_file(self):
        # The bug: 200 is the built-in default of --generations, so the old
        # code took "--generations 200" for "not given" and kept the file's 5000.
        self.assertEqual(main.default_config()["generations"], 200)
        config, _ = self.resolve("-c", "lab", "--generations", "200")
        self.assertEqual(config["generations"], 200)

    def test_typed_zero_and_empty_string_override_the_file(self):
        config, _ = self.resolve("-c", "lab", "--clone-retries", "0", "--metrics-prefix", "")
        self.assertEqual(config["clone_retries"], 0)
        self.assertEqual(config["metrics_prefix"], "")

    def test_typed_flag_different_from_the_default_overrides_the_file(self):
        config, _ = self.resolve("-c", "lab", "--generations", "10", "--population", "30")
        self.assertEqual(config["generations"], 10)
        self.assertEqual(config["population_size"], 30)
        self.assertEqual(config["max_depth"], 4)          # untouched file value

    def test_flag_order_does_not_matter(self):
        config, _ = self.resolve("--generations", "200", "-c", "lab")
        self.assertEqual(config["generations"], 200)

    # ── booleans ────────────────────────────────────────────────────────────
    def test_boolean_set_by_the_file_can_be_switched_off(self):
        config, _ = self.resolve("-c", "lab", "--no-terminal-weighting", "--no-fixed-bounds")
        self.assertFalse(config["terminal_weighting"])
        self.assertFalse(config["fixed_bounds"])

    def test_boolean_flags_still_switch_on(self):
        config, _ = self.resolve("--fixed-bounds", "--track-busy-until", "--save-snapshots")
        self.assertTrue(config["fixed_bounds"])
        self.assertTrue(config["track_busy_until"])
        self.assertTrue(config["save_snapshots"])
        self.assertFalse(config["terminal_weighting"])    # not typed: default

    def test_metrics_can_be_enabled_and_disabled_over_the_file(self):
        self.assertTrue(self.resolve("-c", "lab", "--enable-metrics")[0]["collect_metrics"])
        self.assertFalse(self.resolve("--disable-metrics")[0]["collect_metrics"])
        self.assertTrue(self.resolve()[0]["collect_metrics"])

    # ── the rest ────────────────────────────────────────────────────────────
    def test_no_start_prompt(self):
        self.assertFalse(self.resolve("--no-start-prompt")[1])
        self.assertFalse(self.resolve("-c", "lab", "--no-start-prompt")[1])

    def test_missing_configuration_file_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            self.resolve("-c", "does_not_exist")

    def test_shipped_configurations_load(self):
        # The real files in runConfigurations/ must stay loadable.
        shipped = os.path.join(os.getcwd(), "runConfigurations")
        for name in ("default", "robust"):
            with contextlib.redirect_stdout(io.StringIO()):
                config, _ = main.resolve_config(["-c", name, "--generations", "200"],
                                                config_dir=shipped)
            self.assertEqual(config["generations"], 200)
            self.assertEqual(config["max_depth"], 4)


if __name__ == "__main__":
    unittest.main()
