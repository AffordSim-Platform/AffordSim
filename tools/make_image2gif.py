import argparse
import glob
import os

from PIL import Image


def make_gif(input_dir, output_path, step=1, duration=100, loop=0):

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    image_files.sort()


    selected_files = image_files[::step]
   

    frames = [Image.open(f) for f in selected_files]

  
    base_size = frames[0].size
    frames = [f.resize(base_size) if f.size != base_size else f for f in frames]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
    print(f"GIF save: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-input_dir", default="data/r2s2r_data/pour_affordence_basket/20260409_102919/20260409_102931/camera_images/external_cam_side/env0_ep1", help="")
    parser.add_argument("-o", "--output", default=None, help=" input_dir/output.gif")
    parser.add_argument("-s", "--step", type=int, default=2, help="")
    parser.add_argument("-d", "--duration", type=int, default=100, help="")
    parser.add_argument("--loop", type=int, default=0, help="")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.input_dir, "output.gif")
    make_gif(args.input_dir, output_path, step=args.step, duration=args.duration, loop=args.loop)
