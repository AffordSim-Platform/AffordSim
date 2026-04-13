"""Combine 40 individual task GIFs into a single 5x8 grid GIF, compressed."""

import os
from PIL import Image

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
OUTPUT_PATH = os.path.join(LOGS_DIR, "composite.gif")

# 5 columns x 8 rows = 40 cells
COLS, ROWS = 8, 5
# Target size per cell (downscale from 640x480)
CELL_W, CELL_H = 192, 144
# Total frames in output (loop duration control)
TARGET_FRAMES = 30
# Frame delay in ms
FRAME_DELAY = 150

# Ordered GIF filenames
GIF_FILES = [
    "1.gif",  "2_k.gif",  "3.gif",    "4.gif",    "5.gif",
    "6.gif",  "7.gif",    "8.gif",    "9.gif",    "10.gif",
    "11.gif", "12.gif",   "13_u.gif", "14.gif",   "15.gif",
    "16.gif", "17_k.gif", "18.gif",   "19.gif",   "20.gif",
    "21_k.gif","22.gif",  "23.gif",   "24.gif",   "25_u.gif",
    "26.gif", "27.gif",   "28.gif",   "29.gif",   "30.gif",
    "31.gif", "32.gif",   "33.gif",   "34.gif",   "35.gif",
    "36.gif", "37_u.gif", "38.gif",   "39.gif",   "40.gif",
]


def load_gif_frames(path, target_frames, cell_size):
    """Load a GIF file and return a list of resized PIL frames."""
    gif = Image.open(path)
    n = gif.n_frames

    # Sample target_frames evenly from the original
    indices = [round(i * (n - 1) / (target_frames - 1)) for i in range(target_frames)]
    frames = []
    for idx in indices:
        gif.seek(idx)
        frame = gif.convert("RGB").resize(cell_size, Image.LANCZOS)
        frames.append(frame)
    return frames


def main():
    canvas_w = COLS * CELL_W
    canvas_h = ROWS * CELL_H

    print(f"Output size: {canvas_w}x{canvas_h}, {TARGET_FRAMES} frames")
    print(f"Loading {len(GIF_FILES)} GIFs...")

    # Load all GIF frames
    all_gif_frames = []
    for i, fname in enumerate(GIF_FILES):
        path = os.path.join(LOGS_DIR, fname)
        print(f"  [{i+1:2d}/40] Loading {fname}...", end=" ", flush=True)
        frames = load_gif_frames(path, TARGET_FRAMES, (CELL_W, CELL_H))
        all_gif_frames.append(frames)
        print("done")

    # Compose each output frame
    print("Compositing frames...")
    output_frames = []
    for f_idx in range(TARGET_FRAMES):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        for g_idx, gif_frames in enumerate(all_gif_frames):
            row = g_idx // COLS
            col = g_idx % COLS
            x = col * CELL_W
            y = row * CELL_H
            canvas.paste(gif_frames[f_idx], (x, y))
        # Quantize to 128 colors for compression
        canvas_p = canvas.quantize(colors=64, method=0)
        output_frames.append(canvas_p)
        if (f_idx + 1) % 10 == 0:
            print(f"  Frame {f_idx+1}/{TARGET_FRAMES}")

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    output_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=output_frames[1:],
        duration=FRAME_DELAY,
        loop=0,
        optimize=True,
    )

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Done! Output: {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
