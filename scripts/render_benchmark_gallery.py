"""Render trial GIFs and a 10-by-5 task gallery from recorded benchmark results.

This utility reads trial PNG and lossless WebP frames. Publication to logs/composite.gif is explicit
and requires complete, collision-assessed results for all 50 tasks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import textwrap
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gallery_benchmark_runner', ROOT / 'scripts/run_benchmark_50.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
VERSION = 2
ROUND_PATTERN = re.compile(r"round_\d{2,}$")
TASK_PATTERN = re.compile(r"[A-Za-z0-9_-]+$")


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sample_indices(count, limit):
    if count <= 0:
        return []
    if limit == 1:
        return [0]
    return [round(i * (count - 1) / (limit - 1)) for i in range(limit)]


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def trial_status(trial):
    if trial.get("program_error"):
        return "PROGRAM ERROR"
    classification = runner.collision_flyaway_status(trial)
    if classification == 'unknown':
        return 'COLLISION UNKNOWN'
    if classification == 'excluded':
        return 'EXCLUDED: FLYAWAY'
    return "SUCCESS" if trial.get("success") else "FAILURE"


def frame_paths(trial):
    directory = local_path(trial["directory"])
    camera = directory / "original_images/external_cam_side"
    # During lossless migration both encodings can briefly coexist. Each step
    # remains one frame; prefer the verified replacement when present.
    by_step = {path.stem: path for path in camera.glob("step_*.png")}
    by_step.update({path.stem: path for path in camera.glob("step_*.webp")})
    frames = [by_step[key] for key in sorted(by_step)]
    if not frames and (directory / "initial.png").is_file():
        frames = [directory / "initial.png"]
    return frames


def frame_signature(frames):
    return [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in frames]


def tile(frame, width, image_height, caption_height, lines, color="white"):
    image = Image.new("RGB", (width, image_height + caption_height), "#171b22")
    if frame is not None:
        with Image.open(frame) as source:
            resized = ImageOps.contain(source.convert("RGB"), (width, image_height), Image.Resampling.LANCZOS)
        image.paste(resized, ((width - resized.width) // 2, (image_height - resized.height) // 2))
    else:
        draw = ImageDraw.Draw(image)
        draw.text((10, image_height // 2 - 7), "NO RECORDED FRAME", font=font(13), fill="#999999")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((6, image_height + 3 + 13 * i), line, font=font(11), fill=color)
    return image


def color_for(status):
    return "#ffad66" if "ERROR" in status else ("#ff9292" if "FAIL" in status else "#e5e9ef")


def save_gif(frames, destination, duration=100):
    from animation_export import save_stable_gif
    save_stable_gif(frames, destination, duration)
    for frame in frames:
        frame.close()


def render_trial(task, round_name, trial, output):
    frames = frame_paths(trial)
    destination = output / task["task_id"] / round_name / f"trial_{int(trial['trial']):03d}.gif"
    cache_path = destination.with_suffix(".gif.json")
    status = trial_status(trial)
    annotation = task.get("physics_annotation", "")
    signature = sha({"version": VERSION, "frames": frame_signature(frames), "status": status,
                     "task": task["task_id"], "robot": task["robot"], "trial": trial["trial"],
                     "physics_annotation": annotation})
    record = {"trial": trial["trial"], "status": status, "source_directory": trial["directory"],
              "source_frames": len(frames), "gif": str(destination) if frames else None,
              "signature": signature}
    if not frames:
        return record
    if destination.is_file() and cache_path.is_file():
        if json.loads(cache_path.read_text()).get("signature") == signature:
            return record
    caption = f"{task['robot']} | Trial {trial['trial']} | {status}"
    if annotation:
        caption += f" | {annotation}"
    labels = [task["task_id"], caption]
    animation = []
    for index in sample_indices(len(frames), min(120, len(frames))):
        image = tile(frames[index], 320, 210, 30, labels, color_for(status))
        animation.append(image)
    save_gif(animation, destination)
    write_json(cache_path, record)
    return record


def read_rounds(task, results):
    rounds = []
    directory = results / task["task_id"]
    for path in sorted(directory.glob("round_*/report.json")):
        if not ROUND_PATTERN.fullmatch(path.parent.name):
            continue  # An archived earlier attempt is never a current result.
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # A writer may be replacing a report while a run is active.
        trials = report.get("trials", [])
        summary = report.get("summary", {})
        requested = summary.get('requested_trials')
        computed = runner.summarize(trials, requested if isinstance(requested, int) else len(trials))
        unique = len({trial["trial"] for trial in trials}) == len(trials)
        complete = bool(report.get("completed") and unique and not report.get("error")
                        and not report.get("timed_out") and report.get('returncode') == 0
                        and computed['program_error_trials'] == 0 and len(trials) == requested)
        if all('collision_flyaway' in t for t in trials):
            complete = complete and all(summary.get(k) == computed.get(k) for k in (
                'successes', 'success_rate', 'valid_trials', 'excluded_trials', 'collision_unknown_trials'))
        rounds.append({
            "round": path.parent.name, "report_path": str(path), "report": report,
            "completed": complete, "valid_trials": computed['valid_trials'],
            "requested_trials": requested, "excluded_trials": computed['excluded_trials'],
            "collision_unknown_trials": computed['collision_unknown_trials'],
            "program_errors": computed['program_error_trials'],
            "successes": computed['successes'], "success_rate": computed['success_rate'],
            "range_scale": report.get("range_scale", 1.0),
        })
    return rounds


def rate_text(rate):
    return "N/A" if rate is None else f"{rate * 100:.0f}%"


def task_record(task, results):
    rounds = read_rounds(task, results)
    completed = [entry for entry in rounds if entry["completed"]]
    final = completed[-1] if completed else (rounds[-1] if rounds else None)
    latest = rounds[-1] if rounds else None
    representative = None
    if final:
        candidates = final["report"].get("trials", [])
        # Prefer a successful completed trajectory, then a real failure, then a
        # clearly labelled program error. Missing frames are never fabricated.
        for status in ("SUCCESS", "FAILURE", "EXCLUDED: FLYAWAY", "COLLISION UNKNOWN", "PROGRAM ERROR"):
            representative = next((trial for trial in candidates if trial_status(trial) == status and frame_paths(trial)), None)
            if representative:
                break
    state = "NOT RUN" if not latest else ("COMPLETE" if latest["completed"] else "INCOMPLETE")
    if latest and latest["program_errors"]:
        state = "PROGRAM ERROR"
    return {
        "task_id": task["task_id"], "robot": task["robot"], "yaml": task["config_path"],
        "physics_annotation": task.get("physics_annotation", ""),
        "canonical_yaml": task.get("canonical_path"), "state": state,
        "initial_success_rate": rounds[0]["success_rate"] if rounds else None,
        "final_success_rate": final["success_rate"] if final else None,
        "final_round": final["round"] if final else None,
        "final_valid_trials": final["valid_trials"] if final else 0,
        "final_requested_trials": final["requested_trials"] if final else 0,
        "final_excluded_trials": final["excluded_trials"] if final else 0,
        "final_collision_unknown_trials": final["collision_unknown_trials"] if final else 0,
        "final_successes": final["successes"] if final else 0,
        "latest_round": latest["round"] if latest else None,
        "latest_completed": latest["completed"] if latest else False,
        "representative_trial_number": representative["trial"] if representative else None,
        "representative_status": trial_status(representative) if representative else None,
        "representative_trial": representative,
        "rounds": rounds,
    }


def publication_errors(tasks, records):
    errors = []
    if len(tasks) != 50:
        errors.append(f"Expected 50 manifest tasks, found {len(tasks)}")
    for task, record in zip(tasks, records):
        if not record["rounds"]:
            errors.append(f"{task['task_id']}: no reports")
            continue
        latest = record["rounds"][-1]
        report = latest["report"]
        if not latest['completed'] or latest['requested_trials'] not in (10, 20):
            errors.append(f"{task['task_id']}: latest round is incomplete or has an invalid trial count")
        if latest['collision_unknown_trials']:
            errors.append(f"{task['task_id']}: {latest['collision_unknown_trials']} collision assessments are unknown")
        if report.get("task_id") != task["task_id"] or report.get("robot") != task["robot"]:
            errors.append(f"{task['task_id']}: report task/robot does not match manifest")
        if not record["representative_trial"]:
            errors.append(f"{task['task_id']}: no representative recorded image")
        source_hash = report.get("source_config_sha256")
        if source_hash and source_hash != hashlib.sha256(local_path(task["config_path"]).read_bytes()).hexdigest():
            errors.append(f"{task['task_id']}: source YAML changed since evaluation")
    return errors


def render_gallery(records, output):
    if len(records) != 50:
        raise ValueError("The 10-column, 5-row gallery requires exactly 50 manifest tasks")
    destination = output / "gallery_10x5.gif"
    cache_path = destination.with_suffix(".gif.json")
    sources = [frame_paths(record["representative_trial"]) if record["representative_trial"] else [] for record in records]
    signature = sha({"version": VERSION, "gallery_layout": 2, "sources": [frame_signature(paths) for paths in sources],
                     "records": [{k: r[k] for k in ("task_id", "robot", "state", "final_success_rate", "final_round", "final_valid_trials", "final_successes", "final_excluded_trials", "final_collision_unknown_trials", "physics_annotation")} for r in records]})
    if destination.is_file() and cache_path.is_file() and json.loads(cache_path.read_text()).get("signature") == signature:
        return destination
    timelines = [sample_indices(len(paths), 60) for paths in sources]
    animation = []
    for number in range(60):
        canvas = Image.new("RGB", (2560, 1260), "#171b22")
        for i, (record, paths, timeline) in enumerate(zip(records, sources, timelines)):
            labels = textwrap.wrap(record["task_id"], width=39)[:2]
            labels += [f"{record['robot']} | SR {rate_text(record['final_success_rate'])} ({record['final_successes']}/{record['final_valid_trials']})"]
            trial = record["representative_trial"]
            status = trial_status(trial) if trial else record["state"]
            state = "" if record["state"] == "COMPLETE" else f"{record['state']} | "
            labels += [f"{record['physics_annotation']} | {state}{status}".lstrip(" |")]
            panel = tile(paths[timeline[number]] if paths else None, 256, 192, 60, labels, color_for(status))
            canvas.paste(panel, ((i % 10) * 256, (i // 10) * 252))
            panel.close()
        animation.append(canvas)
        if number == 0:
            canvas.save(output / "gallery_10x5.png")
    save_gif(animation, destination)
    write_json(cache_path, {"signature": signature, "normalized_frames": 60, "fps": 10,
                            "task_order": [record["task_id"] for record in records]})
    return destination


def publish_gallery(gallery, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup = destination.with_name(f"{destination.stem}.before_benchmark_{stamp}{destination.suffix}")
        shutil.copy2(destination, backup)
    else:
        backup = None
    temporary = destination.with_name(destination.stem + ".publishing.gif")
    shutil.copy2(gallery, temporary)
    temporary.replace(destination)
    return str(backup) if backup else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmarks/affordsim_50.json")
    parser.add_argument("--results-dir", default="outputs/local_benchmark/results")
    parser.add_argument("--output-dir", default="outputs/local_benchmark/media")
    parser.add_argument("--task", action="append", help="Generate only this task's recorded trial and representative GIFs; repeatable.")
    parser.add_argument("--publish", action="store_true", help="After validating all 50 tasks, back up and replace logs/composite.gif.")
    parser.add_argument("--gallery-only", action="store_true", help="Render the overview without re-encoding each trial GIF")
    parser.add_argument("--workers", type=int, default=4, help="Independent trial GIF encoders (default: 4).")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.publish and args.task:
        parser.error("--publish requires the full manifest; omit --task")
    if args.gallery_only and args.task:
        parser.error("--gallery-only cannot be combined with --task")
    manifest_path, results, output = map(local_path, (args.manifest, args.results_dir, args.output_dir))
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest.get("tasks", [])
    ids = [task["task_id"] for task in tasks]
    if not tasks or len(ids) != len(set(ids)) or any(not TASK_PATTERN.fullmatch(task_id) for task_id in ids):
        parser.error("Manifest must contain unique, safe task IDs")
    selected = set() if args.gallery_only else set(args.task or ids)
    if selected - set(ids):
        parser.error(f"Unknown task IDs: {sorted(selected - set(ids))}")
    output.mkdir(parents=True, exist_ok=True)
    records = [task_record(task, results) for task in tasks]
    errors = publication_errors(tasks, records)
    if args.publish and errors:
        parser.error("Publication refused:\n" + "\n".join(errors))
    for task, record in zip(tasks, records):
        task_output = output / task["task_id"]
        task_index_path = task_output / "index.json"
        if task["task_id"] not in selected:
            if task_index_path.is_file():
                previous = json.loads(task_index_path.read_text())
                identity = ("final_round", "representative_trial_number", "representative_status")
                if all(previous.get(key) == record.get(key) for key in identity):
                    record["representative_gif"] = previous.get("representative_gif")
            continue
        media = []
        jobs = [(round_record["round"], trial) for round_record in record["rounds"]
                for trial in round_record["report"].get("trials", [])]
        with ThreadPoolExecutor(max_workers=args.workers) as encoders:
            items = encoders.map(lambda job: render_trial(task, job[0], job[1], output), jobs)
            for (round_name, _), item in zip(jobs, items):
                item["round"] = round_name
                media.append(item)
        representative = record["representative_trial"]
        record["representative_gif"] = None
        if representative:
            item = next(item for item in media if item["round"] == record["final_round"] and item["trial"] == representative["trial"])
            if item["gif"]:
                destination = task_output / "representative.gif"
                sidecar = destination.with_suffix(".gif.json")
                if not destination.is_file() or not sidecar.is_file() or json.loads(sidecar.read_text()).get("signature") != item["signature"]:
                    shutil.copy2(item["gif"], destination)
                    write_json(sidecar, item)
                record["representative_gif"] = str(destination)
        # Omit bulky execution metadata from the browsable media index.
        task_index = {key: value for key, value in record.items() if key not in {"rounds", "representative_trial"}}
        task_index["trials"] = media
        write_json(task_index_path, task_index)
        print(f"MEDIA {task['task_id']}: {len(media)} recorded trials, {record['state']}", flush=True)
    summaries = []
    for record in records:
        summary = {key: value for key, value in record.items() if key not in {"rounds", "representative_trial"}}
        summary["rounds"] = [{key: value for key, value in item.items() if key != "report"} for item in record["rounds"]]
        summaries.append(summary)
    write_json(output / "summary.json", {"manifest": str(manifest_path), "tasks": summaries,
                                        "publication_ready": not errors, "publication_errors": errors})
    fields = ["task_id", "robot", "physics_annotation", "yaml", "state", "initial_success_rate", "final_success_rate", "final_round", "final_requested_trials", "final_valid_trials", "final_excluded_trials", "final_collision_unknown_trials", "final_successes", "representative_gif"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    if not args.task and len(tasks) == 50:
        gallery = render_gallery(records, output)
        print(f"GALLERY {gallery}", flush=True)
        if args.publish:
            backup = publish_gallery(gallery, ROOT / "logs/composite.gif")
            write_json(output / "publication.json", {"published": str(ROOT / "logs/composite.gif"), "backup": backup,
                                                     "source_gallery": str(gallery), "manifest": str(manifest_path)})
            print(f"PUBLISHED {ROOT / 'logs/composite.gif'}; previous GIF: {backup}", flush=True)


if __name__ == "__main__":
    main()
