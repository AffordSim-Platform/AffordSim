#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from PIL import Image


def find_image_root(data_dir: Path) -> Path | None:
    candidates = sorted(data_dir.rglob("original_images"))
    if not candidates:
        candidates = sorted(data_dir.rglob("camera_images"))
    return candidates[-1] if candidates else None


def iter_camera_dirs(image_root: Path) -> Iterable[tuple[str, Path]]:
    for camera in ("external_cam_side", "external_cam_front", "wrist_cam"):
        camera_root = image_root / camera
        if not camera_root.exists():
            continue
        env_dirs = sorted(camera_root.glob("env*_ep*"))
        if not env_dirs:
            continue
        yield camera, env_dirs[-1]


def export_gif(frame_dir: Path, out_path: Path, fps: int) -> tuple[bool, int]:
    frame_paths = sorted(frame_dir.glob("step_*.png"))
    if not frame_paths:
        return False, 0

    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as img:
            frames.append(img.convert("RGB"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000 / fps)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    for frame in frames:
        frame.close()
    return True, len(frame_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export task GIFs from a motionplan batch run.")
    parser.add_argument("--run-root", required=True, help="Run root containing manifest.tsv and task folders.")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    manifest_path = run_root / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.tsv not found under {run_root}")

    rows = list(csv.DictReader(manifest_path.open(), delimiter="\t"))
    gif_manifest_path = run_root / "gif_manifest.tsv"
    with gif_manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["task", "status", "camera", "gif_path", "frames", "gif_status"])

        for row in rows:
            task = row["task"]
            status = row["status"]
            data_dir = Path(row["data_dir"])
            image_root = find_image_root(data_dir) if data_dir.exists() else None
            task_video_dir = run_root / task / "videos"

            if image_root is None:
                writer.writerow([task, status, "", "", 0, "missing_frames"])
                continue

            wrote_any = False
            for camera, frame_dir in iter_camera_dirs(image_root):
                gif_path = task_video_dir / f"{camera}.gif"
                ok, frame_count = export_gif(frame_dir, gif_path, fps=args.fps)
                writer.writerow(
                    [
                        task,
                        status,
                        camera,
                        str(gif_path) if ok else "",
                        frame_count,
                        "ok" if ok else "missing_frames",
                    ]
                )
                wrote_any = wrote_any or ok

            if not wrote_any:
                writer.writerow([task, status, "", "", 0, "missing_frames"])

    print(gif_manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
