"""CPU-only checks for selecting tested results and safely staging release configs."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("config_sync_test", ROOT / "scripts/sync_benchmark_configs.py")
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class ConfigSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "outputs/run/results"
        self.manifest = self.root / "outputs/run/manifest.json"
        self.tasks = []
        write(self.root / "models/cup.usd", b"test asset")
        for index in range(50):
            name = f"task_{index:02d}"
            previous = "old_task_00" if index == 0 else name
            old = {
                "simulation": {"render_interval": 5, "dt": 0.01},
                "scene": {"num_envs": 1},
                "runtime": {"data_dir": f"data/{previous}", "headless": False, "episodes": 3,
                            "task_name": "OldRobot", "save_npy": True},
                "task": {"name": previous, "description": "incorrect copied description"},
                "objects": {"red_cup": {"usd_path": "models/cup.usd", "position_range": {"x": [0.4, 0.6]}}},
                "articulations": {"cabinet": {"usd_path": "models/cup.usd", "position_range": {"x": [0.4, 0.8]},
                                               "orientation_range": {"yaw": [-0.5, 0.5]}}},
                "primitives": [{"type": "pick", "params": {"target": "red_cup"}},
                               {"type": "place", "mode": "pose", "params": {"success_eval": True}}],
                "robot": {"cfg_name": "old.yml", "type": "old"},
                "texture_randomization": {"enabled": False},
            }
            old_raw = yaml.safe_dump(old).encode()
            task = {"task_id": name, "config_path": f"outputs/run/prepared/{name}.yaml",
                    "canonical_path": str(SYNC.TASK_ROOT / "pick" / (name + ".yaml")),
                    "previous_canonical_path": str(SYNC.TASK_ROOT / "pick" / (previous + ".yaml")),
                    "original_local_backup": f"outputs/run/original/{previous}.yaml",
                    "original_local_sha256": SYNC.digest(old_raw), "training_task": index < 20,
                    "difficulty": "Easy", "robot": f"robot_{index % 6}",
                    "env_task_name": f"Robot{index % 6}", "robot_cfg_name": f"robot_{index % 6}.yml",
                    "robot_type": f"robot_{index % 6}"}
            write(self.root / task["previous_canonical_path"], old_raw)
            write(self.root / task["original_local_backup"], old_raw)
            prepared = copy.deepcopy(old)
            prepared["task"]["name"] = name
            prepared["runtime"].update(task_name=task["env_task_name"], data_dir="outputs/run/result", headless=True)
            prepared["robot"] = {"cfg_name": task["robot_cfg_name"], "type": task["robot_type"]}
            raw = yaml.safe_dump(prepared).encode()
            write(self.root / task["config_path"], raw)
            task["config_sha256"] = SYNC.digest(raw)
            self.tasks.append(task)
            self.round(task, "round_00", prepared, 8)
        write(self.manifest, SYNC.json_bytes({"tasks": self.tasks, "task_set_resolution": "fixture",
                                            "training20_mapping": {t["task_id"]: t["task_id"] for t in self.tasks[:20]}}))
        write(self.root / SYNC.TASK_ROOT / "example.yaml", b"example: true\n")
        write(self.root / SYNC.TASK_ROOT / "pick/example_da3.yaml", b"da3: true\n")

    def round(self, task, name, config, successes, **extra):
        path = self.results / task["task_id"] / name
        raw = yaml.safe_dump(config).encode()
        write(path / "task.yaml", raw)
        report = {"task_id": task["task_id"], "robot": task["robot"], "completed": True,
                  "num_envs": 5, "episodes": 4,
                  "config_sha256": SYNC.digest(raw), "source_config_sha256": task["config_sha256"],
                  "range_scale": 1.0,
                  "trials": [{"trial": i + 1, "success": i < successes, "program_error": False} for i in range(20)],
                  "summary": {"requested_trials": 20, "successes": successes, "success_rate": successes / 20}}
        report.update(extra)
        write(path / "report.json", SYNC.json_bytes(report))
        return path

    def plan(self):
        return SYNC.build_plan(self.root, self.manifest, self.results)

    def test_latest_round_and_original_runtime_with_portable_provenance(self):
        task = self.tasks[0]
        prepared = yaml.safe_load((self.root / task["config_path"]).read_text())
        final = copy.deepcopy(prepared)
        final["objects"]["red_cup"]["position_range"]["x"] = [0.45, 0.55]
        final["articulations"]["cabinet"]["position_range"]["x"] = [0.5, 0.7]
        final["simulation"]["render_interval"] = 30
        self.round(task, "round_01", final, 2, range_scale=0.5)
        self.round(task, "round_99.archived", prepared, 20)
        plan, files = self.plan()
        self.assertEqual(plan["ready_tasks"], 50)
        self.assertFalse(plan["blockers"])
        row = plan["tasks"][0]
        self.assertEqual(row["evaluation"]["success_rate_percent"], 10)
        self.assertEqual(row["initial_evaluation"]["success_rate_percent"], 40)
        release = yaml.safe_load(files[task["canonical_path"]])
        self.assertEqual(release["objects"], final["objects"])
        self.assertEqual(release["runtime"]["episodes"], 3)
        self.assertFalse(release["runtime"]["headless"])
        self.assertEqual(release["runtime"]["data_dir"], "data/r2s2r_data/task_00")
        self.assertEqual(release["simulation"]["render_interval"], 5)
        self.assertEqual(release["robot"], final["robot"])
        self.assertIn("red cup", release["task"]["description"])
        self.assertTrue(release["texture_randomization"]["enabled"])
        portable = json.loads(files[str(SYNC.PORTABLE_MANIFEST)])
        self.assertEqual(len(portable["tasks"]), 50)
        self.assertNotIn("outputs/", json.dumps(portable))
        self.assertEqual(portable["tasks"][0]["evaluation"]["tested_source_sha256"], task["config_sha256"])
        self.assertEqual(portable["tasks"][0]["num_envs"], 5)
        self.assertEqual(portable["tasks"][0]["initial_ranges"]["articulations"]["cabinet"]["position_range"]["x"], [0.4, 0.8])
        self.assertEqual(portable["tasks"][0]["final_ranges"]["articulations"]["cabinet"]["position_range"]["x"], [0.5, 0.7])
        # Building and writing a preview leaves every canonical file untouched.
        old = self.root / task["previous_canonical_path"]
        before = old.read_bytes()
        SYNC.write_preview(self.root / "outputs/release_preview", plan, files)
        self.assertEqual(old.read_bytes(), before)
        self.assertFalse((self.root / task["canonical_path"]).exists())

    def test_archived_results_remove_stale_preview_and_never_count(self):
        plan, files = self.plan()
        output = self.root / "outputs/release_preview"
        SYNC.write_preview(output, plan, files)
        task = self.tasks[0]
        old_result = self.results / task["task_id"]
        archived = self.root / "outputs/diagnostics" / task["task_id"]
        archived.parent.mkdir(parents=True)
        old_result.rename(archived)
        new_plan, new_files = self.plan()
        SYNC.write_preview(output, new_plan, new_files)
        self.assertEqual(new_plan["ready_tasks"], 49)
        self.assertFalse((output / "files" / task["canonical_path"]).exists())
        self.assertFalse((output / "files" / SYNC.PORTABLE_MANIFEST).exists())

    def test_incomplete_latest_or_hash_mismatch_cannot_fall_back(self):
        task = self.tasks[0]
        config = yaml.safe_load((self.root / task["config_path"]).read_text())
        path = self.round(task, "round_01", config, 20, completed=False)
        plan, files = self.plan()
        self.assertEqual(plan["ready_tasks"], 49)
        self.assertNotIn(str(SYNC.PORTABLE_MANIFEST), files)
        with self.assertRaises(ValueError):
            SYNC.apply_plan(self.root, self.root / "outputs/release_preview", plan, files)
        report = json.loads((path / "report.json").read_text())
        report["completed"] = True
        report["config_sha256"] = "wrong"
        (path / "report.json").write_bytes(SYNC.json_bytes(report))
        plan, _ = self.plan()
        self.assertIn("hash", plan["blockers"][0]["reason"])

    def test_apply_backs_up_replacements_and_second_apply_is_noop(self):
        plan, files = self.plan()
        output = self.root / "outputs/release_preview"
        backup = SYNC.apply_plan(self.root, output, plan, files)
        task = self.tasks[0]
        self.assertFalse((self.root / task["previous_canonical_path"]).exists())
        self.assertTrue((self.root / task["canonical_path"]).exists())
        self.assertEqual(SYNC.digest((self.root / backup / task["previous_canonical_path"]).read_bytes()),
                         task["original_local_sha256"])
        self.assertTrue((self.root / SYNC.TASK_ROOT / "pick/example_da3.yaml").exists())
        again, new_files = self.plan()
        self.assertFalse(again["changes"])
        self.assertIsNone(SYNC.apply_plan(self.root, output, again, new_files))

    def test_concurrent_source_edits_are_not_overwritten(self):
        plan, files = self.plan()
        path = self.root / self.tasks[0]["previous_canonical_path"]
        path.write_bytes(path.read_bytes() + b"# user edit\n")
        with self.assertRaisesRegex(ValueError, "File changed after preview"):
            SYNC.apply_plan(self.root, self.root / "outputs/release_preview", plan, files)
        second, _ = self.plan()
        self.assertTrue(any("unreviewed" in b["reason"] for b in second["blockers"]))

    def test_legacy_checkpoint_example_keeps_task_identity_explicit(self):
        name = "scripts/imitation_learning/dp/run_play.sh"
        content = ('# banana bowl legacy\npython play.py \\\n'
                   '  --checkpoint bowl_checkpoint.pth \\\n'
                   '  --stat bowl_stats.json \\\n'
                   '  --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml \\\n'
                   '  --task BowlTask\n')
        write(self.root / name, content.encode())
        updates = SYNC.reference_updates(self.root)
        text = updates[name].decode()
        self.assertIn('${DP_CHECKPOINT}', text)
        self.assertIn('${DP_TASK_CONFIG}', text)
        self.assertNotIn("pick_banana_place_plate.yaml", text)
        self.assertEqual((self.root / name).read_text(), content)


if __name__ == "__main__":
    unittest.main()
