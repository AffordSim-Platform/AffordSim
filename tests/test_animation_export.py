"""A static textured patch must not shimmer when the rest of a GIF changes."""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "animation_export", Path(__file__).resolve().parents[1] / "scripts/animation_export.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_static_rgb_texture_has_identical_colors_in_every_gif_frame(tmp_path):
    y, x = np.mgrid[:64, :128]
    fixed = np.stack([(x * 3 + y) % 256, (x + y * 4) % 256, (x * 2 + y * 3) % 256], -1).astype(np.uint8)
    frames = []
    for color in [(220, 40, 30), (20, 180, 220), (60, 240, 30)]:
        pixels = fixed.copy()
        pixels[:, 64:] = color
        frames.append(Image.fromarray(pixels))
    destination = tmp_path / "stable.gif"
    module.save_stable_gif(frames, destination)
    gif = Image.open(destination)
    assert gif.n_frames == 3
    first = np.array(gif.convert("RGB"))[:, :64]
    for index in range(1, gif.n_frames):
        gif.seek(index)
        np.testing.assert_array_equal(np.array(gif.convert("RGB"))[:, :64], first)
    # Export must not alter the original RGB frames.
    np.testing.assert_array_equal(np.array(frames[0])[:, :64], fixed[:, :64])
