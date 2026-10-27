"""Check pull success units without importing the simulator's primitive graph."""

import ast
from functools import lru_cache
import importlib.util
from pathlib import Path
from typing import Any, Dict
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIMITIVE = ROOT / "source/isaaclab/isaaclab/motion_collect/primitives/pull.py"
CONSTANTS = ROOT / "source/isaaclab/isaaclab/motion_collect/success/constants.py"
spec = importlib.util.spec_from_file_location("pull_test_constants", CONSTANTS)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)
namespace = {
    "Any": Any, "Dict": Dict, "np": np, "lru_cache": lru_cache,
    "OPEN_SUCCESS_MIN_ANGLE_RAD": constants.OPEN_SUCCESS_MIN_ANGLE_RAD,
    "PULL_SUCCESS_MIN_DISPLACEMENT": constants.PULL_SUCCESS_MIN_DISPLACEMENT,
}
# These two helpers use no collector/Isaac modules. Execute their actual source
# so CPU tests do not initialize the large primitive import graph.
tree = ast.parse(PRIMITIVE.read_text())
helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
    "_pull_joint_type_from_usd", "_pull_success_threshold",
}]
exec(compile(ast.fix_missing_locations(ast.Module(body=helpers, type_ignores=[])), str(PRIMITIVE), "exec"), namespace)
threshold = namespace["_pull_success_threshold"]
joint_type = namespace["_pull_joint_type_from_usd"]


class PullSuccessTests(unittest.TestCase):
    def test_revolute_and_prismatic_defaults_have_different_units(self):
        self.assertEqual(threshold("revolute", {}), (1.0, "rad"))
        self.assertEqual(threshold("prismatic", {}), (0.08, "m"))

    def test_explicit_threshold_is_preserved(self):
        self.assertEqual(threshold("revolute", {"pull_success_min_angle_rad": 0.9}), (0.9, "rad"))
        self.assertEqual(threshold("revolute", {"open_success_min_angle_rad": 1.2}), (1.2, "rad"))
        self.assertEqual(threshold("prismatic", {"pull_success_min_displacement": 0.12}), (0.12, "m"))
        self.assertEqual(threshold("prismatic", {"min_displacement": 0.2}), (0.2, "m"))

    def test_wrong_units_and_invalid_threshold_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            threshold("revolute", {"min_displacement": 0.08})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            threshold("prismatic", {"min_angle_rad": 1.0})
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                threshold("revolute", {"min_angle_rad": value})

    def test_bundled_cabinet15_joint_is_revolute(self):
        try:
            from pxr import Usd  # noqa: F401
        except ImportError:
            self.skipTest("USD Python libraries are not on this CPU environment's library path")
        asset = str(ROOT / "models/all_object/cabinet15.usd")
        self.assertEqual(joint_type(asset, "joint_0"), "revolute")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            joint_type(asset, "joint_1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            joint_type(asset, "missing_joint")


if __name__ == "__main__":
    unittest.main()
