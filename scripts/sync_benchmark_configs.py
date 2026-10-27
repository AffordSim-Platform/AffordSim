"""Stage the final tested benchmark configurations; write canonical files only with --apply.

The default command is a dry run. Its preview includes file hashes, metadata
corrections and initial/final ranges. It never changes prepared experiment inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path("source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task")
ROUND = re.compile(r"round_[0-9]{2,}$")
PHYSICAL_STAGES = {"place", "pour", "hang", "open", "close", "pull", "push"}
PORTABLE_MANIFEST = Path("benchmarks/affordsim_50.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def local(root, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves repository: {value}")
    return path


def ranges(config):
    return {section: {key: {field: copy.deepcopy(obj.get(field)) for field in
                           ("position_range", "orientation_range")}
                      for key, obj in (config.get(section) or {}).items()}
            for section in ("objects", "articulations")}


def object_name(key):
    if key == "pink_basket":
        return "pink scoop"
    label = re.sub(r"([a-z])([0-9])", r"\1 \2", key.replace("_", " "))
    for old, new in {"bigbowl": "large bowl", "bellpepper": "bell pepper",
                     "75mmcup": "75 mm cup", "95mmcup": "95 mm cup",
                     "mugcup": "mug", "papercup": "paper cup", "soupcan": "soup can",
                     "chocolatebox": "chocolate box", "sugarbox": "sugar box",
                     "coffeemachine": "coffee machine", "woodshelf": "wood shelf",
                     "cuttingboard": "cutting board", "coffeecan": "coffee can",
                     "scrubbottle": "scrub bottle", "doraemoncup": "Doraemon cup",
                     "starbuckscup": "Starbucks cup", "cheezbox": "cheese box"}.items():
        label = label.replace(old, new)
    return label


def describe(config):
    """Describe actual primitive targets, avoiding copied text from another task."""
    steps = []
    held = None
    for stage in config["primitives"]:
        kind, mode, params = stage["type"], stage.get("mode"), stage.get("params", {})
        target = params.get("target")
        noun = "the " + object_name(target) if target else "the target"
        if kind == "pick":
            held = noun
            steps.append("pick up " + noun)
        elif kind == "place":
            if mode == "pose":
                steps.append(f"place {held or 'the object'} at the specified position")
            else:
                relation = "in" if mode in {"in", "affordance"} else "on"
                if target and any(word in target for word in ("bowl", "cup", "basket", "pan", "microwave")):
                    relation = "in"
                if target and any(word in target for word in ("plate", "shelf", "base")):
                    relation = "on"
                steps.append(f"place {held or 'the object'} {relation} {noun}")
            held = None
        elif kind == "pour":
            steps.append(f"pour from {held or 'the held container'} into {noun}")
        elif kind == "hang":
            steps.append(f"hang {held or 'the held object'} on {noun}")
            held = None
        elif kind == "pull":
            steps.append("pull " + noun + " open")
        elif kind == "push":
            steps.append("push " + noun + " to the specified position")
        elif kind in {"open", "close"}:
            steps.append(kind + " " + noun)
        elif kind != "return_home":
            raise ValueError(f"No description rule for primitive {kind}")
    result = "; then ".join(steps)
    return result[0].upper() + result[1:] + "."


def validate_report(root, task, path, source_hash):
    report = json.loads(path.read_text())
    trials = report.get("trials", [])
    if not report.get("completed") or report.get("error") or report.get("timed_out"):
        raise ValueError("latest round is incomplete or has a program error")
    if report.get("task_id") != task["task_id"] or report.get("robot") != task["robot"]:
        raise ValueError("report task/robot does not match manifest")
    num_envs, episodes = report.get("num_envs", 0), report.get("episodes", 0)
    if num_envs <= 0 or episodes <= 0 or num_envs * episodes != 20:
        raise ValueError("reported parallel environments and episode count do not total 20 attempts")
    if task.get("num_envs") is not None and num_envs != task["num_envs"]:
        raise ValueError("report parallel environment count differs from task manifest")
    if (len(trials) != 20 or len({t["trial"] for t in trials}) != 20
            or any(t.get("program_error") for t in trials)
            or report.get("summary", {}).get("requested_trials") != 20):
        raise ValueError("round does not contain exactly 20 valid unique attempts")
    if report.get("source_config_sha256") != source_hash:
        raise ValueError("report was generated from a different prepared configuration")
    config_path = path.parent / "task.yaml"
    raw = config_path.read_bytes()
    if report.get("config_sha256") != digest(raw):
        raise ValueError("tested task.yaml no longer matches the report hash")
    config = yaml.safe_load(raw)
    if (config["task"]["name"] != task["task_id"]
            or config["runtime"]["task_name"] != task["env_task_name"]
            or config["robot"]["cfg_name"] != task["robot_cfg_name"]
            or config["robot"]["type"] != task["robot_type"]):
        raise ValueError("tested YAML task/robot differs from manifest")
    successes = sum(bool(t.get("success")) for t in trials)
    summary = report.get("summary", {})
    if summary.get("successes") != successes or abs(summary.get("success_rate", -1) - successes / 20) > 1e-9:
        raise ValueError("reported success rate disagrees with trial outcomes")
    return config, {"round": path.parent.name, "range_scale": report.get("range_scale", 1.0),
                    "trials": 20, "successes": successes, "success_rate_percent": successes * 5,
                    "num_envs": num_envs, "episodes_per_environment": episodes,
                    "tested_config_sha256": digest(raw), "tested_source_sha256": source_hash,
                    "report_sha256": digest(path.read_bytes())}


def release_config(original, tested, task):
    """Keep tested task mechanics; restore normal collection/output options."""
    result = copy.deepcopy(tested)
    result["simulation"] = copy.deepcopy(original.get("simulation", {}))
    result["scene"] = copy.deepcopy(original.get("scene", {}))
    result["runtime"] = copy.deepcopy(original.get("runtime", {}))
    result["runtime"]["data_dir"] = f"data/r2s2r_data/{task['task_id']}"
    result["runtime"]["task_name"] = task["env_task_name"]
    result["task"]["name"] = task["task_id"]
    result["task"]["description"] = describe(tested)
    result["enable_objects_randomization"] = True
    result.setdefault("texture_randomization", {}).update(enabled=True, avoid_repeats=True)
    return result


def validate_release(root, data):
    names = set(data.get("objects", {})) | set(data.get("articulations", {}))
    for obj in list(data.get("objects", {}).values()) + list(data.get("articulations", {}).values()):
        if not local(root, obj["usd_path"]).is_file():
            raise ValueError(f"Missing asset: {obj['usd_path']}")
    for stage in data["primitives"]:
        params = stage.get("params", {})
        if params.get("target") and params["target"] not in names:
            raise ValueError(f"Unknown primitive target: {params['target']}")
        if stage["type"] in PHYSICAL_STAGES and not params.get("success_eval"):
            raise ValueError(f"Physical success check disabled: {stage['type']}")

    def scan(value, field=""):
        if isinstance(value, dict):
            for key, child in value.items():
                scan(child, key)
        elif isinstance(value, list):
            for child in value:
                scan(child, field)
        elif isinstance(value, str) and field not in {"prim_path", "description"}:
            if value.startswith(("/", "outputs/")):
                raise ValueError(f"Nonportable release value in {field}: {value}")
    scan(data)


def reference_updates(root):
    """Only prepare these edits; DA3 and historical checkpoint identities stay intact."""
    updates = {}
    replacements = {
        "README.md": [("--task_name pick_banana_place_bowl \\", "--task_name pick_banana_place_plate \\")],
        "scripts/test_backgrounds.py": [("pick/pick_banana_place_bowl.yaml", "pick/pick_banana_place_plate.yaml")],
        "source/isaaclab/isaaclab/motion_collect/scene_create/config.py": [
            ("pick/pick_banana_place_bowl.yaml", "pick/pick_banana_place_plate.yaml")],
        "tools/combine_hdf5.py": [("train_hdf5/pick_banana_place_bowl/", "train_hdf5/pick_banana_place_plate/")],
    }
    for name, pairs in replacements.items():
        path = root / name
        if not path.is_file():
            continue
        before = path.read_text()
        after = before
        for old, new in pairs:
            after = after.replace(old, new)
        if after != before:
            updates[name] = after.encode()

    name = "scripts/imitation_learning/dp/run_play.sh"
    path = root / name
    if path.is_file() and "pick/pick_banana_place_bowl.yaml" in path.read_text():
        content = path.read_text()
        parameters = {"checkpoint": "DP_CHECKPOINT", "stat": "DP_STAT",
                      "task_config": "DP_TASK_CONFIG", "task": "DP_TASK_ID"}
        for flag, variable in parameters.items():
            content, count = re.subn(r"(--" + flag + r"\s+)\S+", lambda m: m[1] + '"${' + variable + '}"', content)
            if count != 1:
                raise ValueError(f"Review legacy DP example manually: expected one --{flag}")
        note = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                "# Supply a matching checkpoint, statistics file, task YAML and registered task ID.\n"
                "# Historical banana-to-bowl checkpoints require their original bowl YAML;\n"
                "# they must not be silently evaluated as the new banana-to-plate task.\n")
        for variable in parameters.values():
            note += ': "${' + variable + ':?Set ' + variable + ' for the trained task}"\n'
        updates[name] = (note + "\n" + content).encode()

    for name in ("scripts/imitation_learning/dp/run_train.sh", "scripts/imitation_learning/Pi0.5/run_play.sh"):
        path = root / name
        if not path.is_file():
            continue
        content = path.read_text()
        old = "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml"
        if old not in content:
            continue
        for line in content.splitlines():
            if old in line and not line.lstrip().startswith("#"):
                raise ValueError(f"Unreviewed active legacy bowl command: {name}")
        note = ("# Historical bowl examples below retain their original checkpoint/dataset/task IDs.\n"
                "# BOWL_TASK_CONFIG must point to the matching historical bowl YAML, not the plate task.\n")
        content = content.replace(old, '"${BOWL_TASK_CONFIG}"')
        lines = content.splitlines(keepends=True)
        insertion = 1 if lines and lines[0].startswith("#!") else 0
        lines.insert(insertion, note)
        updates[name] = "".join(lines).encode()
    return updates


def build_plan(root, manifest_path, results):
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    ids = {t["task_id"] for t in tasks}
    training = manifest["training20_mapping"]
    if (len(tasks) != 50 or len(ids) != 50 or len(training) != 20
            or len(set(training.values())) != 20 or not set(training.values()) <= ids):
        raise ValueError("Expected exactly 50 distinct tasks containing the 20 training tasks")
    if {t["task_id"] for t in tasks if t["training_task"]} != set(training.values()):
        raise ValueError("Training flags disagree with the 20-task mapping")
    if sorted(Counter(t["robot"] for t in tasks).values()) != [8, 8, 8, 8, 9, 9]:
        raise ValueError("Expected six assigned robots with counts 9/9/8/8/8/8")
    files, removals, records, blockers = {}, [], [], []
    original_hashes = {}
    for task in tasks:
        canonical = local(root, task["canonical_path"])
        if not canonical.is_relative_to(root / TASK_ROOT):
            raise ValueError("Canonical task destination is outside the task directory")
        original_path = local(root, task["original_local_backup"])
        original_raw = original_path.read_bytes()
        if digest(original_raw) != task["original_local_sha256"]:
            raise ValueError(f"Original backup hash mismatch: {task['task_id']}")
        original = yaml.safe_load(original_raw)
        previous = task.get("previous_canonical_path", task["canonical_path"])
        if not local(root, previous).is_relative_to(root / TASK_ROOT):
            raise ValueError("Previous canonical task path is outside the task directory")
        original_hashes[previous] = task["original_local_sha256"]
        prepared_path = local(root, task["config_path"])
        prepared_raw = prepared_path.read_bytes()
        source_hash = digest(prepared_raw)
        if source_hash != task["config_sha256"]:
            raise ValueError(f"Prepared hash differs from manifest: {task['task_id']}")
        prepared = yaml.safe_load(prepared_raw)
        record = {"task_id": task["task_id"], "config_path": task["canonical_path"],
                  **{key: task[key] for key in ("robot", "env_task_name", "robot_cfg_name", "robot_type",
                                               "training_task", "difficulty")},
                  "initial_ranges": ranges(prepared), "status": "pending"}
        try:
            paths = sorted(p for p in (results / task["task_id"]).glob("round_*/report.json")
                           if ROUND.fullmatch(p.parent.name))
            if not paths:
                raise ValueError("No current round report")
            tested, final = validate_report(root, task, paths[-1], source_hash)
            initial = None
            for path in paths:
                try:
                    _, initial = validate_report(root, task, path, source_hash)
                    break
                except ValueError:
                    continue
            release = release_config(original, tested, task)
            validate_release(root, release)
            raw = yaml.safe_dump(release, allow_unicode=True, sort_keys=False).encode()
            files[task["canonical_path"]] = raw
            if previous != task["canonical_path"]:
                removals.append(previous)
            record.update(status="ready", config_sha256=digest(raw), evaluation=final,
                          num_envs=final["num_envs"],
                          initial_evaluation=initial, final_ranges=ranges(tested),
                          description_before=tested["task"].get("description"),
                          description_after=release["task"]["description"],
                          tested_report=str(paths[-1].relative_to(root)))
        except (OSError, ValueError, KeyError) as exc:
            record["reason"] = str(exc)
            blockers.append({"task_id": task["task_id"], "reason": str(exc)})
        records.append(record)

    files.update(reference_updates(root))
    portable = {"schema_version": 1, "paths_relative_to": "repository_root", "task_count": 50,
                "episodes_per_round": 20, "task_set_resolution": manifest["task_set_resolution"],
                "range_policy": "Canonical YAMLs contain final tested ranges. Each task's initial_ranges preserves the pre-test baseline for objects and articulations; evaluation.range_scale records the selected final scale. New adaptive runs scale the current canonical ranges again, so use --no-adapt to evaluate released ranges unchanged.",
                "training20_mapping": training, "tasks": []}
    for record in records:
        if record["status"] != "ready":
            continue
        portable["tasks"].append({key: value for key, value in record.items()
                                  if key not in {"status", "tested_report", "description_before", "description_after"}})
    if not blockers:
        files[str(PORTABLE_MANIFEST)] = json_bytes(portable)

    # Refuse silent clobbering of user edits, and make an already-applied plan a no-op.
    for task in tasks:
        new = task["canonical_path"]
        previous = task.get("previous_canonical_path", new)
        for name in {previous, new}:
            path = local(root, name)
            if not path.exists():
                continue
            current = path.read_bytes()
            if current != files.get(name) and digest(current) != original_hashes.get(name):
                blockers.append({"path": name, "reason": "canonical file has unreviewed changes since original backup"})
    planned_ids = {str(Path(t["canonical_path"]).relative_to(TASK_ROOT)) for t in tasks}
    for path in (root / TASK_ROOT).rglob("*.yaml"):
        rel = str(path.relative_to(root))
        if (str(path.relative_to(root / TASK_ROOT)) not in planned_ids and rel not in removals
                and path.name != "example.yaml" and not path.stem.endswith("_da3")):
            # Missing task results do not turn a known scheduled replacement into an extra task.
            if rel not in {t.get("previous_canonical_path", t["canonical_path"]) for t in tasks}:
                blockers.append({"path": rel, "reason": "unexpected extra task YAML; preserve and review"})
    changes = []
    for name, raw in files.items():
        path = local(root, name)
        before = path.read_bytes() if path.exists() else None
        if before != raw:
            changes.append({"path": name, "operation": "write", "before_sha256": digest(before) if before is not None else None,
                            "after_sha256": digest(raw)})
    for name in removals:
        path = local(root, name)
        if path.exists():
            changes.append({"path": name, "operation": "remove_after_backup", "before_sha256": digest(path.read_bytes())})
    return {"status": "blocked" if blockers else "ready_for_explicit_apply", "applied": False,
            "task_count": 50, "ready_tasks": sum(r["status"] == "ready" for r in records),
            "blockers": blockers, "changes": changes, "tasks": records}, files


def write_preview(output, plan, files):
    output.mkdir(parents=True, exist_ok=True)
    index = output / "staged_files.json"
    old_files = json.loads(index.read_text()) if index.exists() else []
    # Earlier previews used plan.json as their generated-file index.
    if not old_files and (output / "plan.json").exists():
        old_files = [c["path"] for c in json.loads((output / "plan.json").read_text()).get("changes", [])
                     if c["operation"] == "write"]
    for name in set(old_files) - set(files):
        path = (output / "files" / name).resolve()
        if not path.is_relative_to((output / "files").resolve()):
            raise ValueError("Previous preview index contains an unsafe path")
        path.unlink(missing_ok=True)
    (output / "plan.json").write_bytes(json_bytes(plan))
    for name, raw in files.items():
        path = output / "files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    index.write_bytes(json_bytes(sorted(files)))
    (output / "final_ranges.json").write_bytes(json_bytes(plan["tasks"]))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["task_id", "robot", "status", "initial_sr_percent",
                                               "final_sr_percent", "final_range_scale", "initial_ranges", "final_ranges"])
    writer.writeheader()
    for row in plan["tasks"]:
        writer.writerow({"task_id": row["task_id"], "robot": row["robot"], "status": row["status"],
                         "initial_sr_percent": (row.get("initial_evaluation") or {}).get("success_rate_percent"),
                         "final_sr_percent": row.get("evaluation", {}).get("success_rate_percent"),
                         "final_range_scale": row.get("evaluation", {}).get("range_scale"),
                         "initial_ranges": json.dumps(row["initial_ranges"], sort_keys=True),
                         "final_ranges": json.dumps(row.get("final_ranges"), sort_keys=True)})
    (output / "final_ranges.csv").write_text(stream.getvalue())


def apply_plan(root, output, plan, files):
    if plan["blockers"] or plan["ready_tasks"] != 50:
        raise ValueError("Cannot apply: all 50 tasks must have a valid final round and no conflicts")
    if not plan["changes"]:
        return None
    for change in plan["changes"]:
        path = local(root, change["path"])
        current = digest(path.read_bytes()) if path.exists() else None
        if current != change["before_sha256"]:
            raise ValueError(f"File changed after preview: {change['path']}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = output.parent / "canonical_sync_backup" / stamp
    for change in plan["changes"]:
        path = local(root, change["path"])
        if path.exists():
            target = backup / change["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "plan.json").write_bytes(json_bytes(plan))
    # Task files and references first, portable manifest last.
    ordered = sorted(plan["changes"], key=lambda c: c["path"] == str(PORTABLE_MANIFEST))
    for change in ordered:
        path = local(root, change["path"])
        if change["operation"] == "remove_after_backup":
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".sync.tmp")
            temporary.write_bytes(files[change["path"]])
            if path.exists():
                temporary.chmod(path.stat().st_mode)
            temporary.replace(path)
    return str(backup.relative_to(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/benchmark_paper50/manifest.json")
    parser.add_argument("--results-dir", default="outputs/benchmark_paper50/results")
    parser.add_argument("--output-dir", default="outputs/benchmark_paper50/release_preview")
    parser.add_argument("--apply", action="store_true", help="Apply only after reviewing the dry-run preview")
    args = parser.parse_args()
    output = local(ROOT, args.output_dir)
    if not output.is_relative_to(ROOT / "outputs"):
        parser.error("--output-dir must be inside this checkout's outputs/ directory")
    plan, files = build_plan(ROOT, local(ROOT, args.manifest), local(ROOT, args.results_dir))
    write_preview(output, plan, files)
    if args.apply:
        plan["backup"] = apply_plan(ROOT, output, plan, files)
        plan["applied"] = True
        (output / "plan.json").write_bytes(json_bytes(plan))
    print(json.dumps({key: plan[key] for key in ("status", "applied", "ready_tasks")}, indent=2))
    print(f"Preview: {output / 'plan.json'}")


if __name__ == "__main__":
    main()
