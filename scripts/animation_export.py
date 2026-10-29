"""GIF export with one color mapping for the entire animation."""
from PIL import Image


GIF_PALETTE_VERSION = 2


def shared_palette(frames, colors=256):
    """Sample across time so unchanged RGB values cannot change palette mapping."""
    count = min(32, len(frames))
    atlas = Image.new("RGB", (160 * 8, 120 * ((count + 7) // 8)))
    for slot in range(count):
        index = round(slot * (len(frames) - 1) / max(count - 1, 1))
        sample = frames[index].convert("RGB")
        sample.thumbnail((160, 120))
        atlas.paste(sample, ((slot % 8) * 160, (slot // 8) * 120))
        sample.close()
    palette = atlas.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    atlas.close()
    return palette


def save_stable_gif(frames, destination, duration=100):
    if not frames:
        raise ValueError("Cannot encode an animation without frames")
    destination.parent.mkdir(parents=True, exist_ok=True)
    palette = shared_palette(frames)
    indexed = []
    try:
        for frame in frames:
            rgb = frame.convert("RGB")
            indexed.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
            rgb.close()
        temporary = destination.with_name(destination.stem + ".tmp.gif")
        indexed[0].save(temporary, save_all=True, append_images=indexed[1:],
                        duration=duration, loop=0, disposal=2, optimize=False)
        temporary.replace(destination)
    finally:
        palette.close()
        for frame in indexed:
            frame.close()
