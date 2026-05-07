import h5py
import sys

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"  {name}/")

path = sys.argv[1] if len(sys.argv) > 1 else "/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_new/hdf5merged_trajectories_1.hdf5"

with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys()))

    if "data" in f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")],
                       key=lambda x: int(x.split("_")[1]))
        print(f"\nTotal demos: {len(demos)}")

        # Show structure of first demo
        print(f"\n--- {demos[0]} structure ---")
        f[f"data/{demos[0]}"].visititems(print_hdf5_structure)

        # Show a few demo lengths
        print(f"\n--- Demo lengths (first 10) ---")
        for dk in demos[:10]:
            actions = f[f"data/{dk}/actions"]
            print(f"  {dk}: {actions.shape[0]} steps, action_dim={actions.shape[1]}")
    else:
        print("\nFull structure:")
        f.visititems(print_hdf5_structure)
