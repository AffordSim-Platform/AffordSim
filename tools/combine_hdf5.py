import h5py
import os


data_dir = "dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup/hdf5"
output_file = "dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup/merged_trajectories.hdf5"


with h5py.File(output_file, 'w') as h5out:
    for i in range(157):  # episode_0 到 episode_293
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
