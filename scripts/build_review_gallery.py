"""Build a 5-by-10 benchmark GIF from the previews actually shown in a review page.

Usage: python scripts/build_review_gallery.py --review outputs/.../review \
    --web-assets ../AffordSim-page/assets

The three task-name substitutions follow the 20-task main-paper list. The
main-paper classification of organize_clutter_into_baskets is Hard; to keep
the 10/30/10 split, composite_2pickplaces is grouped with the other two-pick
task in Medium. Source previews are read only; outputs is never published.
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from PIL import Image, ImageSequence


TIERS = {
    "easy": """pick_banana_place_plate pick_bottle_place_table pick_bowl_place_plate
        pick_carrot_place_bowl pick_carrot_place_plate pick_chocolatebox_place_sugarbox
        pick_cube_place_cup pick_leek_place_cuttingboard pick_pen_place_doraemoncup
        pick_soupcan_place_table""".split(),
    "medium": """close_microwave open_microwave hang_cup pick_cup_place_coffeemachine
        pick_cup_place_woodshelf pick_green_bellpepper_place_pan pick_kettle_place_base
        pick_lid_place_pot pick_mugcup_place_coffeemachine pick_mugcup_place_woodshelf
        pick_papercup_place_table pick_scrubbottle_place_woodshelf pick_soupcan_place_microwave
        pour_basket pour_bluekettle pour_mugcup pour_pan pour_papercup pour_whitekettle
        pour_whitekettle4 pull_cabinet pull_cabinet7 pull_cabinet15 push_cheezbox
        push_coffeecan stack_cube stack_four_cube_pyramid stack_papercup
        composite_2picks_to_plate composite_2pickplaces""".split(),
    "hard": """hang_mugcup stack_four_cube composite_load_microwave organize_clutter_into_baskets
        composite_pour_then_hang composite_pour_then_place_coffee composite_double_pour_kettle
        composite_unload_microwave store_cups_on_plates composite_push_pick_to_shelf""".split(),
}


class ReviewParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards = []
        self.card = None
        self.capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.card = {}
        elif self.card is not None:
            if tag == "h2":
                self.capture = "id"
            elif tag == "p" and "status" in attrs.get("class", "").split():
                self.card["review_status"] = attrs["class"].split()[-1]
            elif tag == "p" and "robot" not in self.card:
                self.capture = "robot"
            elif tag == "img":
                self.card["source_gif"] = attrs.get("src")

    def handle_data(self, data):
        if self.capture and self.card is not None:
            self.card[self.capture] = data.removeprefix("机械臂：").strip()

    def handle_endtag(self, tag):
        if tag in ("h2", "p"):
            self.capture = None
        elif tag == "article" and self.card is not None:
            self.cards.append(self.card)
            self.card = None


def skill_for(task):
    if task.startswith("pick_"):
        return "pick_place"
    if task in ("open_microwave", "close_microwave"):
        return "open_close"
    if task.startswith(("pull_", "push_")):
        return "pull_push"
    if task.startswith("hang_"):
        return "hang"
    if task.startswith("pour_"):
        return "pour"
    if task.startswith("stack_"):
        return "stack"
    if task.startswith("composite_") or task in ("store_cups_on_plates", "organize_clutter_into_baskets"):
        return "long_horizon"
    raise ValueError(f"No skill category for {task}")


def manifest(review):
    parser = ReviewParser()
    parser.feed((review / "index.html").read_text(encoding="utf-8"))
    cards = {card["id"]: card for card in parser.cards}
    ordered = [task for tasks in TIERS.values() for task in tasks]
    if len(cards) != 50 or len(set(ordered)) != 50 or set(cards) != set(ordered):
        raise ValueError(f"Review / task list mismatch: missing={set(ordered)-set(cards)}, extra={set(cards)-set(ordered)}")
    summary = json.loads((review / "summary.json").read_text(encoding="utf-8"))
    by_id = {task["task_id"]: task for task in summary["tasks"]}
    tasks = []
    for difficulty, ids in TIERS.items():
        for task_id in ids:
            card = cards[task_id]
            relative = Path(card["source_gif"])
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 5 or relative.parts[1] != task_id:
                raise ValueError(f"Invalid source GIF path for {task_id}: {relative}")
            source = review / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            attempt = relative.parts[2]
            trial_match = re.fullmatch(r"trial_(\d+)", relative.parts[3])
            if not trial_match:
                raise ValueError(f"Unrecognized trial path: {relative}")
            trial = int(trial_match.group(1))
            matches = [a for a in by_id[task_id]["attempts"] if a["attempt"] == attempt and a["trial"] == trial]
            if len(matches) != 1 or matches[0]["gif"] != str(relative) or matches[0]["status"] != card["review_status"]:
                raise ValueError(f"Review summary disagrees with displayed preview: {task_id}")
            if card["review_status"] not in ("success", "out_of_view"):
                raise ValueError(f"Displayed preview is not accepted: {task_id} {card['review_status']}")
            report = json.loads((review / matches[0]["report"]).read_text(encoding="utf-8"))
            records = [t for t in report["trials"] if t["trial"] == trial]
            if len(records) != 1 or not records[0].get("success") or records[0].get("collision_flyaway", {}).get("excluded"):
                raise ValueError(f"Displayed preview is not a valid physical success: {task_id}")
            tasks.append({"id": task_id, "difficulty": difficulty, "skills": [skill_for(task_id)], **card})
    return tasks


def sampled_frames(path, count):
    """Sample a 10-second GIF uniformly, respecting every source frame duration."""
    frames = []
    elapsed = 0
    with Image.open(path) as image:
        for frame in ImageSequence.Iterator(image):
            end = elapsed + max(1, frame.info.get("duration", 0))
            while len(frames) < count and len(frames) * 10000 / count < end:
                original = frame.convert("RGB")
                original.thumbnail((128, 96), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (128, 96), (20, 30, 32))
                tile.paste(original, ((128-original.width)//2, (96-original.height)//2))
                frames.append(tile)
            elapsed = end
    if elapsed != 10000 or len(frames) != count:
        raise ValueError(f"Expected a 10-second source GIF: {path}, duration={elapsed}, frames={len(frames)}")
    return frames


def temporal_palette(sampled):
    """Build one palette from all tasks and every sampled point in time.

    A shared palette with dithering disabled gives equal RGB pixels the same
    index throughout the animation, avoiding palette-induced flashing.
    """
    if len(sampled) != 50 or any(len(frames) != 50 for frames in sampled):
        raise ValueError("Expected fifty tasks with fifty frames each")
    atlas = Image.new("RGB", (64 * 50, 48 * 50))
    for task_index, frames in enumerate(sampled):
        for time_index, frame in enumerate(frames):
            sample = frame.resize((64, 48), Image.Resampling.NEAREST)
            atlas.paste(sample, (task_index * 64, time_index * 48))
    palette = atlas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    atlas.close()
    return palette


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--web-assets", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    review, assets = args.review.resolve(), args.web_assets.resolve()
    tasks = manifest(review)
    assets.mkdir(parents=True, exist_ok=True)
    manifest_data = {"columns": 10, "rows": 5, "width": 1280, "height": 480,
                     "duration_ms": 10000, "source_review": "outputs/collection_50_hdf5_20261029/review/index.html",
                     "tier_note": "Main-paper 20 tasks take precedence. organize_clutter_into_baskets is Hard; composite_2pickplaces is Medium to retain 10/30/10.",
                     "tasks": tasks}
    (assets / "benchmark_50_manifest.json").write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n")
    web_tasks = [{k: task[k] for k in ("id", "difficulty", "skills")} for task in tasks]
    (assets / "benchmark-tasks.js").write_text("window.AFFORDSIM_BENCHMARK_TASKS = " + json.dumps(web_tasks, ensure_ascii=False, indent=2) + ";\n")
    print(f"Manifest: {len(tasks)} validated tasks; out of view: {[t['id'] for t in tasks if t['review_status'] == 'out_of_view']}", flush=True)
    if args.manifest_only:
        return
    sampled = []
    for i, task in enumerate(tasks, 1):
        sampled.append(sampled_frames(review / task["source_gif"], 50))
        print(f"Read {i}/50: {task['id']}", flush=True)
    output = []
    palette = temporal_palette(sampled)
    for time_index in range(50):
        grid = Image.new("RGB", (1280, 480))
        for index, frames in enumerate(sampled):
            grid.paste(frames[time_index], ((index % 10) * 128, (index // 10) * 96))
        output.append(grid.quantize(palette=palette, dither=Image.Dither.NONE))
    path = assets / "benchmark_50.gif"
    output[0].save(path, save_all=True, append_images=output[1:], duration=200, loop=0, optimize=False, disposal=2)
    print(f"Wrote {path} ({path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
