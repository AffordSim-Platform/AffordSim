import h5py
import os


data_dir = os.environ.get(
    "COMBINE_DATA_DIR",
    "train_hdf5/pick_banana_place_bowl/hdf5",
)
output_file = os.environ.get(
    "COMBINE_OUTPUT_FILE",
    "train_hdf5/pick_banana_place_bowl/merged_trajectories.hdf5",
)
_episodes_env = os.environ.get("COMBINE_EPISODES")
num_episodes = int(_episodes_env) if _episodes_env else 51


with h5py.File(output_file, 'w') as h5out:
    for i in range(num_episodes):  # episode_0  episode_293
        episode_file = os.path.join(data_dir, f"episode_{i}.hdf5")
        group_name = f"traj_{i}"
        with h5py.File(episode_file, 'r') as h5in:
          
            grp = h5out.create_group(group_name)
            
      
            h5in.copy('action', grp)
            
        
            obs_grp = grp.create_group('observations')
            
         
            h5in.copy('observations/qpos', obs_grp)
            h5in.copy('observations/qvel', obs_grp)
            
          
            images_grp = obs_grp.create_group('images')
            h5in.copy('observations/images/front', images_grp)
            h5in.copy('observations/images/wrist', images_grp)

print(f"save {output_file}")
