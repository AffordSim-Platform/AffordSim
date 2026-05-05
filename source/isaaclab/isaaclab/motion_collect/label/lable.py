"""Offline USD label generator.

Reads all .usd files from the asset directory, computes AABB dimensions
and bottom_offset using pxr (usd-core), and writes label.json.

No Isaac Sim / simulation required.
"""

from pxr import Usd, UsdGeom
import json
import glob
import os
from pathlib import Path

ASSET_PATH = "/home/xhf/AffordSim/models"
ALL_OBJECT_DIR = os.path.join(ASSET_PATH, "all_object")
OUTPUT_PATH = os.path.join(ALL_OBJECT_DIR, "label.json")


def compute_bbox(usd_path):
    """Open a USD file offline and return (length, width, height, bottom_offset) in metres."""
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetPseudoRoot()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(root)
    aligned = bbox.ComputeAlignedRange()
    min_pt = aligned.GetMin()
    max_pt = aligned.GetMax()

    length = max_pt[0] - min_pt[0]  # X -> length
    width = max_pt[1] - min_pt[1]   # Y -> width
    height = max_pt[2] - min_pt[2]  # Z -> height

    # bottom_offset = distance from origin to object bottom = -min_z
    bottom_offset = -min_pt[2] if min_pt[2] < 0 else 0.0

    return length, width, height, bottom_offset


def main():
    usd_files = sorted(glob.glob(os.path.join(ALL_OBJECT_DIR, "*.usd")))
    if not usd_files:
        print(f"[WARN] No .usd files found in {ALL_OBJECT_DIR}")
        return

    all_data = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            all_data = json.load(f)
    existing_labels = {entry["label"] for entry in all_data}

    added = 0
    for usd_path in usd_files:
        stem = Path(usd_path).stem  # e.g. "blue_coffeecan", "cabinet15"
        if stem in existing_labels:
            print(f"  [SKIP] {stem} already in label.json")
            continue

        l, w, h, bottom_off = compute_bbox(usd_path)

        all_data.append({
            "label": stem,
            "dimensions_cm": {
                "length": round(l * 100, 2),
                "width": round(w * 100, 2),
                "height": round(h * 100, 2),
            },
            "bottom_offset_m": round(bottom_off, 4),
            "scale": [1.0, 1.0, 1.0],
        })
        existing_labels.add(stem)
        added += 1
        print(f"  {stem}: L={l * 100:.2f} W={w * 100:.2f} H={h * 100:.2f} cm, offset={bottom_off:.4f} m")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"\nAdded {added} new objects, total {len(all_data)} in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
