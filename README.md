# AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulation

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-red.svg" alt="arXiv"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-BSD--3-yellow.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg" alt="IsaacSim">
  <img src="https://img.shields.io/badge/platform-linux--64-orange.svg" alt="Platform">
</p>

<p align="center">
  <img src="logs/composite.gif" alt="AffordSim Task Gallery" width="100%">
</p>

## Introduction

**AffordSim** is the first simulation framework that integrates open-vocabulary 3D affordance prediction into the robotic manipulation data generation pipeline. Existing simulation platforms do not incorporate object affordance information into trajectory generation, making it impossible to automatically generate semantically correct trajectories for tasks that require precise interaction with specific functional regions — such as grasping a mug by its handle, pouring from a cup's rim, or hanging a mug on a hook.

AffordSim addresses this by using **VoxAfford**, an open-vocabulary 3D affordance detector that enhances MLLM output tokens with multi-scale geometric features, to predict affordance maps on object point clouds and guide grasp pose estimation toward task-relevant functional regions. Built on NVIDIA Isaac Sim, AffordSim features:

- **Affordance-aware trajectory generation.** VoxAfford predicts 3D affordance heatmaps on object point clouds, guiding grasp pose selection toward task-relevant functional regions.
- **Automated task-scene generation.** A VLM-powered pipeline converts natural language task descriptions into simulation scenes, supporting 4 robot embodiments (Franka FR3, Panda, UR5e, Kinova) without task-specific engineering.
- **Photorealistic domain randomization.** Five randomization axes — including DA3-based 3D Gaussian reconstruction from real photographs — reduce the visual domain gap for sim-to-real transfer.
- **A 50-task benchmark** across 7 manipulation categories (grasping, placing, stacking, pushing/pulling, pouring, mug hanging, long-horizon composite) with evaluations of 4 imitation learning baselines (BC, Diffusion Policy, ACT, Pi 0.5).

<p align="center">
  <img src="logs/grasp.png" alt="Affordance-guided Grasp Planning" width="100%">
  <br>
  <em>Affordance-guided grasp planning vs. generic grasp estimation.</em>
</p>

## Authors

**Mingyang Li\***, **Haofan Xu\***, **Haowen Sun**, **Xinzhe Chen**, **Sihua Ren**, **Liqi Huang**, **Chenyang Miao**, **Xinyang Sui**, **Qiongjie Cui†**, **Zeyang Liu†**, **Xingyu Chen†**, **Xuguang Lan†**

## Paper

Our paper is available at: [main.pdf](main.pdf)

> **AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulation**
>
> Mingyang Li\*, Haofan Xu\*. *Preprint, 2025.*

## Installation

### Prerequisites

- **OS**: Linux (Ubuntu 22.04 recommended)
- **Python**: 3.11
- **GPU**: NVIDIA GPU with CUDA >=12.7 support
- **NVIDIA Isaac Sim**:  5.1

### Step 1: Clone the Repository

```bash
git clone https://github.com/yangml071/AffordSim.git
cd AffordSim
```

### Step 2: Create Conda Environment

```bash
conda create -n env_affodsim python=3.11
conda activate env_affodsim
```

### Step 3: Install IsaacSim 5.1 and IsaacLab 2.3.1

```bash
pip install isaaclab[isaacsim,all]==2.3.1 --extra-index-url https://pypi.nvidia.com
```

### Step 4: Install PyTorch

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 5: Install cuRobo

```bash
unzip curobo.zip
cd curobo
python -m pip install -e .[isaacsim] --no-build-isolation
```

### Step 6: Install AnyGrasp SDK

Follow the [AnyGrasp SDK](https://github.com/graspnet/anygrasp_sdk) installation guide:

```bash
# Install MinkowskiEngine (modified version)
mkdir dependencies && cd dependencies
conda install openblas-devel -c anaconda
export CUDA_HOME=/usr/local/cuda
git clone git@github.com:chenxi-wang/MinkowskiEngine.git
cd MinkowskiEngine
# For CUDA 12.x, uncomment the following line:
# git checkout cuda-12-1
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas_library_dirs=${CONDA_PREFIX}/lib --blas=openblas
cd ../..

# Install AnyGrasp dependencies
pip install -r requirements.txt

# Build PointNet2
cd pointnet2
python setup.py install
cd ..
```

> **Note:** AnyGrasp SDK requires a license. Please refer to the [license registration](https://github.com/graspnet/anygrasp_sdk#license-registration) section to obtain your license file.

### Step 7: Download Assets

Download the simulation models (robot URDFs/USDs, object meshes, textures, etc.) from HuggingFace:

```bash
# Download the assets
huggingface-cli download yangml071/simulation-models --local-dir models --repo-type model
```

> **Note:** This is a private repository. You must be granted access before downloading. Please contact the authors to request access.


## Usage

### Data Collection

Collect affordance-aware manipulation trajectories from natural language task descriptions:

```bash
python scripts/run_data_collect_imporve.py \
    --task_type task_gen \
    --config xxx.yaml \
    --num_envs 2 \
    --bot franka_fr3 \
    --task_name pick_banana_place_in_plate \
    --headless
```

Key arguments:
| Argument | Description |
|---|---|
| `--task_type` | Task type (`task_gen` for generated tasks) |
| `--config` | Path to the task configuration YAML file |
| `--num_envs` | Number of parallel environments |
| `--bot` | Robot embodiment (`franka_fr3`, `franka_panda`, `ur5e`, `kinova`) |
| `--task_name` | Name of the manipulation task |
| `--headless` | Run without GUI (for large-scale collection) |

### Training Imitation Learning Policies

AffordSim supports multiple imitation learning backends.

#### Data Preprocessing

Before training, convert collected raw data into the HDF5 format required by the IL backends:

```bash
# Step 1: Convert numpy trajectories to HDF5
python tools/np2hdf5.py

# Step 2: Combine multiple HDF5 files into a single training dataset
python tools/combine_hdf5.py

# Step 3: Convert to Robomimic-compatible format with dataset statistics
python tools/convert_hdf5_for_robomimic.py \
    --src <origin_hdf5_path> \
    --dst <target_hdf5_path> \
    --stat <dataset_stat.json>
```

#### Training

**Behavioral Cloning (BC):**
```bash
python scripts/imitation_learning/robomimic/train.py \
    --dataset <target_hdf5_path> \
    --stat <dataset_stat.json> \
    --epochs 1000 \
    --save_every 50 \
    --eval_task_config <path_to_task_yaml> \
    --eval_task <name_to_eval_task> \
    --eval_num_rollouts 5 \
    --eval_horizon 300
```

**Diffusion Policy:**
```bash
python scripts/imitation_learning/diffusion_policy/train.py \
    --dataset <target_hdf5_path> \
    --stat <dataset_stat.json> \
    --epochs 1000 \
    --save_every 50 \
    --eval_task_config <path_to_task_yaml> \
    --eval_task <name_to_eval_task> \
    --eval_num_rollouts 5 \
    --eval_horizon 300
```

**ACT (Action Chunking Transformers):**
```bash
python scripts/imitation_learning/act/train.py \
    --dataset <target_hdf5_path> \
    --stat <dataset_stat.json> \
    --epochs 1000 \
    --save_every 50 \
    --eval_task_config <path_to_task_yaml> \
    --eval_task <name_to_eval_task> \
    --eval_num_rollouts 5 \
    --eval_horizon 300
```

> **Note:**
> - `<path_to_task_yaml>`: Task config files are located in `source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/` (subdirectories: `pick`, `pour`, `hang`, `stack`, `pull_push`, `open_close`, `composite`).
> - `<name_to_eval_task>`: Registered task IDs can be found in [source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/\_\_init\_\_.py](source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/__init__.py) (e.g., `Pick_Place_banana-v0`, `Pick_cupwithhandle_Pour-v0`). To add a custom task, register it following the same pattern in this file.

### Reinforcement Learning

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task FrankaFr3-OurGripper-IKRel-VisPick-v0 \
    --num_envs 512 \
    --enable_cameras \
    --headless
```

### Evaluation

Evaluate trained policies:

**BC:**
```bash
python scripts/imitation_learning/robomimic/play_bc.py \
    --checkpoint <checkpoint_path> \
    --stat <dataset_stat.json> \
    --task <name_to_eval_task> \
    --task_config <path_to_task_yaml> \
    --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless
```

**Diffusion Policy:**
```bash
python scripts/imitation_learning/diffusion_policy/play_dp.py \
    --checkpoint <checkpoint_path> \
    --stat <dataset_stat.json> \
    --task <name_to_eval_task> \
    --task_config <path_to_task_yaml> \
    --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless
```

**ACT:**
```bash
python scripts/imitation_learning/act/play_act.py \
    --checkpoint <checkpoint_path> \
    --stat <dataset_stat.json> \
    --task <name_to_eval_task> \
    --task_config <path_to_task_yaml> \
    --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless
```

## Cross-Embodiment Support

AffordSim supports 4 robot embodiments with automatic kinematic adaptation — no task-specific tuning required:

<p align="center">
  <img src="logs/cross_embodiment.png" alt="Cross-Embodiment Support" width="100%">
</p>

| Robot | Trajectory Generation Success Rate |
|---|---|
| Franka FR3 | 94% |
| Franka Panda | 92% |
| UR5e | 83% |
| Kinova | 95% |

## Domain Randomization

AffordSim applies domain randomization along five axes to enable zero-shot sim-to-real transfer:

<p align="center">
  <img src="logs/domain_randomization.png" alt="Domain Randomization" width="100%">
</p>

1. **Background texture** — random textures on workspace and surrounding surfaces
2. **Lighting** — randomized number, position, intensity, and color of light sources
3. **Object texture** — PBR material property perturbation
4. **Object pose** — small random perturbations to initial positions and orientations
5. **DA3 Gaussian background** — photorealistic backgrounds from 3D Gaussian Splatting reconstruction of real scenes

## Benchmark Evaluation

Each task is trained on 300 demonstrations with object pose randomization applied during data collection; all evaluation rollouts also include object pose randomization to test generalization.

<table>
<thead>
<tr><th>Category</th><th>Task</th><th>BC</th><th>ACT</th><th>DP</th><th>Pi 0.5</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Grasping</td><td>pick_banana</td><td>53</td><td>63</td><td>87</td><td><b>93</b></td></tr>
<tr><td>pick_red_cup</td><td>17</td><td>77</td><td>63</td><td><b>80</b></td></tr>
<tr><td rowspan="3">Placing</td><td>pick_banana_place_plate</td><td>37</td><td>60</td><td>84</td><td><b>93</b></td></tr>
<tr><td>pick_cup_place_shelf</td><td>5</td><td>7</td><td>3</td><td><b>47</b></td></tr>
<tr><td>pick_kettle_place_coffee_machine</td><td>27</td><td>43</td><td>49</td><td><b>65</b></td></tr>
<tr><td rowspan="2">Stacking</td><td>stack_two_blocks</td><td>30</td><td>43</td><td>47</td><td><b>77</b></td></tr>
<tr><td>stack_three_blocks</td><td>13</td><td>27</td><td>28</td><td><b>53</b></td></tr>
<tr><td rowspan="2">Push/Pull</td><td>push_box_to_target</td><td>23</td><td>63</td><td>67</td><td><b>78</b></td></tr>
<tr><td>pull_drawer_open</td><td>37</td><td>53</td><td>58</td><td><b>86</b></td></tr>
<tr><td rowspan="3">Pouring</td><td>pour_basket_into_bowl</td><td>20</td><td>86</td><td>94</td><td><b>99</b></td></tr>
<tr><td>pour_pan_into_bowl</td><td>7</td><td>33</td><td>87</td><td><b>92</b></td></tr>
<tr><td>pour_cup_into_bowl</td><td>1</td><td>24</td><td>36</td><td><b>43</b></td></tr>
<tr><td rowspan="2">Mug Hanging</td><td>hang_mug_on_rack</td><td>0</td><td>10</td><td>17</td><td><b>47</b></td></tr>
<tr><td>hang_mug_on_hook</td><td>0</td><td>7</td><td>13</td><td><b>33</b></td></tr>
<tr><td rowspan="3">Long Horizon</td><td>pick_cup_pour_place_coffee_machine</td><td>0</td><td>3</td><td>8</td><td><b>16</b></td></tr>
<tr><td>pick_cup_pour_hang_on_rack</td><td>0</td><td>1</td><td>3</td><td><b>21</b></td></tr>
<tr><td>open_microwave_place_can</td><td>0</td><td>0</td><td>2</td><td><b>13</b></td></tr>
<tr><td colspan="2"><b>Average (all 17 tasks)</b></td><td><b>16</b></td><td><b>35</b></td><td><b>44</b></td><td><b>61</b></td></tr>
</tbody>
</table>

**Key findings:**
- **Grasping** tasks achieve the highest success rates (53–93%), confirming that basic pick-up manipulation is largely solved by current methods.
- **Placing** and **Push/Pull** tasks show moderate-to-high success (3–93%), with failures primarily occurring during precision-placement or target-reaching rather than the initial grasp.
- **Affordance-demanding** tasks show a sharp performance drop: pouring into a wide basket achieves 20–99% success, but pouring into a narrow cup drops to 1–43%. Mug hanging achieves at most 47% success even for the best baseline. These results highlight that affordance-demanding tasks—where success depends on functional-region interaction precision—remain significantly more challenging.
- **Pi 0.5** achieves the highest average performance (61%), followed by DP (44%) and ACT (35%). BC lags significantly behind (16%), suggesting that foundation-model-based, diffusion-based, and action-chunking architectures better capture multimodal action distributions.
- **Sub-task analysis** reveals that grasping succeeds at 63–100% across pouring tasks, while the downstream pouring action fails dramatically for small containers—localizing the bottleneck to fine-grained, affordance-dependent manipulation rather than the initial grasp.

## Project Structure

```
AffordSim/
├── source/
│   ├── isaaclab/                 # Core framework (simulation, assets, sensors, controllers)
│   │   └── isaaclab/motion_collect/  # Affordance-aware motion collection module
│   ├── isaaclab_tasks/           # 30+ task environment implementations
│   ├── isaaclab_assets/          # Robot and sensor asset definitions
│   ├── isaaclab_rl/              # Reinforcement learning utilities
│   └── isaaclab_mimic/           # Imitation learning (cuRobo integration)
├── scripts/
│   ├── run_data_collect.py       # Main data collection entry point
│   ├── run_eval.py               # Evaluation script
│   ├── reinforcement_learning/   # RL training scripts (RSL-RL, SKRL, RL Games, SB3)
│   └── imitation_learning/       # IL training scripts (BC, Diffusion Policy, ACT, RDT)
├── curobo/                       # GPU-accelerated motion planning library
├── backgrounds/                  # DA3 Gaussian background assets
├── isaaclab.sh                   # Build and execution interface
└── main.pdf                      # Paper
```

## License

The AffordSim framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). See [`docs/licenses`](docs/licenses) for dependency licenses.

## Citation

If you find AffordSim useful in your research, please cite:

```bibtex
@article{li2025affordsim,
  title={AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulation},
  author={Mingyang Li and Haofan Xu},
  year={2025}
}
```

## Acknowledgement

AffordSim is built on top of [Isaac Lab](https://github.com/isaac-sim/IsaacLab). We gratefully acknowledge the Isaac Lab team for their foundational framework.
