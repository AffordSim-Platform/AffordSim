import argparse
import glob
import os
import re

import h5py


def collect_episode_indices(folder: str) -> list[int]:
    """Return sorted episode indices found as episode_<i>.hdf5 in folder."""
    episode_re = re.compile(r"episode_(\d+)\.hdf5$")
    indices = []
    for f in glob.glob(os.path.join(folder, "episode_*.hdf5")):
        m = episode_re.search(f)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


parser = argparse.ArgumentParser(
    description="Merge per-episode HDF5 files into a single trajectory file."
)
parser.add_argument(
    "-n", "--num_episodes",
    type=int,
    default=(int(os.environ["COMBINE_EPISODES"]) if os.environ.get("COMBINE_EPISODES") else None),
    help="Number of episodes to combine (first N by index). "
         "Default: env COMBINE_EPISODES, or all episodes found if unset.",
)
parser.add_argument(
    "--data_dir",
    default=os.environ.get(
        "COMBINE_DATA_DIR",
        "data/r2s2r_data/hang_mugcup_franka_fr3_default/20260618_170303/hdf5",
    ),
    help="Directory containing the per-episode HDF5 files.",
)
parser.add_argument(
    "--output_file",
    default=os.environ.get(
        "COMBINE_OUTPUT_FILE",
        "data/r2s2r_data/hang_mugcup_franka_fr3_default/20260618_170303/merged_trajectories.hdf5",
    ),
    help="Path to the merged output HDF5 file.",
)
args = parser.parse_args()

data_dir = args.data_dir
output_file = args.output_file


# Episode selection:
#   --num_episodes unset  -> auto: use every episode_*.hdf5 found in data_dir
#   --num_episodes N      -> manual: keep only the first N episodes (by index)
all_indices = collect_episode_indices(data_dir)
if args.num_episodes is not None:
    num_episodes = args.num_episodes
    episode_indices = all_indices[:num_episodes]
    if len(episode_indices) < num_episodes:
        print(f"[WARN] requested {num_episodes} episodes but only {len(episode_indices)} found in {data_dir}")
else:
    episode_indices = all_indices

print(f"Combining {len(episode_indices)} episodes from {data_dir}")

with h5py.File(output_file, 'w') as h5out:
    for traj_idx, i in enumerate(episode_indices):
        episode_file = os.path.join(data_dir, f"episode_{i}.hdf5")
        group_name = f"traj_{traj_idx}"
        with h5py.File(episode_file, 'r') as h5in:
            grp = h5out.create_group(group_name)

            h5in.copy('action', grp)

            obs_grp = grp.create_group('observations')
            h5in.copy('observations/qpos', obs_grp)
            h5in.copy('observations/qvel', obs_grp)

            images_grp = obs_grp.create_group('images')
            h5in.copy('observations/images/side', images_grp)
            h5in.copy('observations/images/front', images_grp)
            h5in.copy('observations/images/wrist', images_grp)

print(f"save {output_file}")
