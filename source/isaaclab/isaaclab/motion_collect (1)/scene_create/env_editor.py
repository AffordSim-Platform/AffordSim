# env_editor.py
"""
Interactive Environment Editor for Isaac Lab.
Main entry point for scene generation and editing.
"""

import argparse
from isaaclab.app import AppLauncher

# Must launch app before importing Isaac Lab modules
parser = argparse.ArgumentParser(description="Interactive Environment Creator")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import simulation modules
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from sim_config import SceneCfg
from scene_system import LLMClient, SceneGenerator, SceneManager, export_to_yaml


def get_randomization_config() -> dict:
    """Interactive randomization configuration."""
    print("\n--- Randomization Config ---")
    print("Enter position range (meters): Front Back Left Right")
    print("Example: 0.05 0.05 0.1 0.1")
    
    pos_input = input("Range: ").strip() or "0.05 0.05 0.05 0.05"
    try:
        vals = [float(x) for x in pos_input.split()]
        if len(vals) != 4:
            raise ValueError("Need 4 values")
        
        config = {
            "pos_delta": {
                "front": vals[0], "back": vals[1],
                "left": vals[2], "right": vals[3]
            }
        }
        
        print("\nEnter rotation range (radians): min max")
        rot_input = input("Yaw range: ").strip() or "0 0"
        config["rot_range"] = [float(x) for x in rot_input.split()]
        
        return config
    except Exception as e:
        print(f"[ERROR] Invalid input: {e}")
        return None


def main():
    try:
        # 1. Initialize simulation
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([2, 0.0, 1], [0.5, 0, 0])
        scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
        sim.reset()
        print("[INFO] Simulation initialized.")

        # 2. Initialize scene system
        llm = LLMClient()
        generator = SceneGenerator(llm)
        manager = SceneManager(scene, generator)
        
        sim_dt = sim.get_physics_dt()

        # Main loop
        while True:
            # === Phase 1: Scene Source ===
            print("\n" + "="*40)
            print("SCENE GENERATION")
            print("="*40)
            print("1: Generate from text (default)")
            print("2: Load from YAML")
            
            mode = input("Choice [1/2]: ").strip()
            
            if mode == "2":
                # Load YAML
                default_path = "/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick.yaml"
                yaml_path = input(f"YAML path [{default_path}]: ").strip() or default_path
                
                if not manager.load_from_yaml(yaml_path):
                    continue
                manager.sync_to_simulation()
            else:
                # Generate from text
                default_prompt = "create a desktop scene with a kettle, a self, a banana, a bowl, and one random object. The banana is in the bowl."
                prompt = input(f"Scene description:\n> ").strip() or default_prompt
                
                scene_data = generator.generate_scene(prompt)
                if not scene_data:
                    print("[ERROR] Generation failed.")
                    continue
                    
                manager.update_scene(scene_data)
                manager.sync_to_simulation()

            # Render frames
            print("Updating simulation...")
            for _ in range(50):
                sim.step()

            # === Phase 2: Edit Loop ===
            editing = True
            while editing:
                print("\n" + "-"*40)
                print("EDIT MODE")
                print("-"*40)
                print("1: Save & Exit")
                print("2: Add object")
                print("3: Remove object")
                print("4: Regenerate scene")
                print("5: Modify object position")
                print("6: Modify object rotation")

                choice = input("Choice: ").strip()
                
                if choice == "1":
                    # Save workflow
                    default_task = "Pick up the banana and place it in the bowl"
                    task = input(f"Task description:\n> ").strip() or default_task
                    
                    parsed = llm.parse_task_instruction(task)
                    print(f"[INFO] Parsed: target={parsed['target']}, container={parsed['container']}, relation={parsed['relation']}")

                    target_key = manager.find_key_by_label(parsed['target'])
                    if not target_key:
                        print(f"[WARN] Target '{parsed['target']}' not found in scene")
                    
                    # Randomization
                    random_config = None
                    if input("Enable randomization? [y/N]: ").strip().lower() == 'y':
                        random_config = get_randomization_config()

                    # Domain randomization settings
                    print("\n--- Domain Randomization ---")
                    enable_random_lights = input("Enable random lights? [y/N]: ").strip().lower() == 'y'

                    image_noise_std = 0
                    noise_input = input("Image noise std [0]: ").strip()
                    if noise_input:
                        image_noise_std = float(noise_input)

                    image_speckle_noise_std = 0.5
                    speckle_input = input("Image speckle noise std [0.5]: ").strip()
                    if speckle_input:
                        image_speckle_noise_std = float(speckle_input)

                    enable_texture_rand = input("Enable texture randomization? [Y/n]: ").strip().lower() != 'n'

                    visual_config = {
                        'enable_random_lights': enable_random_lights,
                        'image_noise_std': image_noise_std,
                        'image_speckle_noise_std': image_speckle_noise_std,
                        'enable_texture_randomization': enable_texture_rand,
                    }

                    # Export
                    export_to_yaml(manager.current_scene, task, random_config, visual_config)
                    
                    print("\n✔ Done. Exiting...")
                    print("Complete shutdown finished")
                    simulation_app.close()
                    return

                elif choice == "2":
                    label = input("Object name: ").strip()
                    if label:
                        if manager.add_object(label) is not False:
                            manager.sync_to_simulation()
                            for _ in range(50):
                                sim.step()

                elif choice == "3":
                    label = input("Object name: ").strip()
                    if label:
                        manager.remove_object(label)
                        manager.sync_to_simulation()
                        for _ in range(50):
                            sim.step()

                elif choice == "4":
                    editing = False  # Return to generation phase

                elif choice == "5":
                    if not manager.current_scene:
                        print("[WARN] Scene is empty.")
                    else:
                        print("\nCurrent objects:")
                        for idx, obj in enumerate(manager.current_scene):
                            pos = obj['position']
                            print(f"  {idx}: {obj.get('label', 'unknown')}  (x={pos['x']:.3f}, y={pos['y']:.3f})")
                        try:
                            sel = int(input("Select object index: ").strip())
                            if 0 <= sel < len(manager.current_scene):
                                new_x = float(input("New x coordinate: ").strip())
                                new_y = float(input("New y coordinate: ").strip())
                                manager.current_scene[sel]['position']['x'] = round(new_x, 3)
                                manager.current_scene[sel]['position']['y'] = round(new_y, 3)
                                print(f"[SUCCESS] Moved '{manager.current_scene[sel].get('label', 'unknown')}' to ({new_x:.3f}, {new_y:.3f})")
                                manager.sync_to_simulation()
                                for _ in range(50):
                                    sim.step()
                            else:
                                print("[WARN] Invalid index.")
                        except ValueError:
                            print("[WARN] Invalid input.")

                elif choice == "6":
                    if not manager.current_scene:
                        print("[WARN] Scene is empty.")
                    else:
                        print("\nCurrent objects:")
                        for idx, obj in enumerate(manager.current_scene):
                            import math as _math
                            cur_deg = _math.degrees(obj.get('_angle', 0.0))
                            print(f"  {idx}: {obj.get('label', 'unknown')}  (yaw={cur_deg:.1f}°)")
                        try:
                            sel = int(input("Select object index: ").strip())
                            if 0 <= sel < len(manager.current_scene):
                                import math as _math
                                delta_deg = float(input("Rotate around Z axis (degrees): ").strip())
                                delta_rad = _math.radians(delta_deg)
                                obj = manager.current_scene[sel]
                                new_angle = obj.get('_angle', 0.0) + delta_rad
                                obj['_angle'] = new_angle
                                qw = _math.cos(new_angle / 2.0)
                                qz = _math.sin(new_angle / 2.0)
                                obj['orientation'] = {'w': round(qw, 4), 'x': 0.0, 'y': 0.0, 'z': round(qz, 4)}
                                print(f"[SUCCESS] Rotated '{obj.get('label', 'unknown')}' by {delta_deg:.1f}° -> yaw={_math.degrees(new_angle):.1f}°")
                                manager.sync_to_simulation()
                                for _ in range(50):
                                    sim.step()
                            else:
                                print("[WARN] Invalid index.")
                        except ValueError:
                            print("[WARN] Invalid input.")

                scene.update(sim_dt)

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except:
            pass


if __name__ == "__main__":
    main()
