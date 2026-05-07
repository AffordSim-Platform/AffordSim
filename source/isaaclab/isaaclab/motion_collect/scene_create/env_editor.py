# env_editor.py
"""
Interactive Environment Editor for Isaac Lab.
Main entry point for scene generation and editing.
"""

import argparse
import builtins
import os
import queue
import re
import threading

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Interactive Environment Creator")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--restore-state", type=str, default="")
parser.add_argument("--initial-yaml", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


class _AsyncTerminalInput:
    """Read terminal input on a background thread while the main thread keeps sim stepping."""

    def __init__(self):
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="scene-editor-input", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._requests.get()
            if item is None:
                return
            prompt, responses = item
            try:
                responses.put(builtins.input(prompt))
            except BaseException as exc:
                responses.put(exc)

    def read(self, prompt: str = ""):
        responses = queue.Queue(maxsize=1)
        self._requests.put((prompt, responses))
        return responses

    def close(self):
        self._requests.put(None)


_terminal_input = _AsyncTerminalInput()
_active_runtime = None


def _set_active_runtime(runtime: dict | None):
    global _active_runtime
    _active_runtime = runtime


def _step_active_runtime_once():
    runtime = _active_runtime
    if runtime is None:
        return

    scene = runtime.get("scene")
    sim = runtime.get("sim")
    sim_dt = runtime.get("sim_dt")
    if scene is not None and sim_dt is not None:
        scene.update(sim_dt)
    if sim is not None:
        sim.step()


def input(prompt: str = "") -> str:
    """Drop-in input replacement that does not pause the Isaac Sim main loop."""
    responses = _terminal_input.read(prompt)
    while True:
        try:
            result = responses.get_nowait()
        except queue.Empty:
            if _active_runtime is None:
                result = responses.get()
                break
            _step_active_runtime_once()
            continue
        break

    if isinstance(result, BaseException):
        raise result
    return result


import config as scene_create_config
from editor_runtime import (
    create_runtime,
    destroy_runtime,
    extract_scene_labels_from_yaml,
    load_restore_state,
    restart_editor_fresh,
    restart_editor_with_state,
    restart_editor_with_yaml,
    sync_and_step_runtime,
)
from live_randomization import (
    apply_random_lights_preview,
    apply_random_texture_preview,
)
from scene_core import LLMClient, SceneGenerator
from task_pipeline import compile_primitives_for_yaml, export_to_yaml


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
                "front": vals[0],
                "back": vals[1],
                "left": vals[2],
                "right": vals[3],
            }
        }

        print("\nEnter rotation range (radians): min max")
        rot_input = input("Yaw range: ").strip() or "0 0"
        config["rot_range"] = [float(x) for x in rot_input.split()]
        return config
    except Exception as e:
        print(f"[ERROR] Invalid input: {e}")
        return None


def _choose_ambiguous_label(alias: str, candidates: list[str], context_name: str) -> str:
    """Ask the user to choose one strict label for an ambiguous fuzzy alias."""
    print(f"[WARN] {context_name} term '{alias}' matches multiple labels:")
    for index, label in enumerate(candidates, start=1):
        print(f"  {index}: {label}")
    while True:
        choice = input(f"Choose label for '{alias}' [1-{len(candidates)}]: ").strip()
        try:
            selected_index = int(choice)
        except ValueError:
            print("[WARN] Please enter a number.")
            continue
        if 1 <= selected_index <= len(candidates):
            return candidates[selected_index - 1]
        print("[WARN] Selection out of range.")


def resolve_label_reference_input(raw_text: str, candidate_labels: list[str], context_name: str) -> str:
    """Resolve a parsed object reference phrase to one strict label."""
    current_text = raw_text.strip()

    while True:
        matches = _collect_containing_labels(current_text, candidate_labels)
        if len(matches) == 1:
            resolved = matches[0]
            if resolved != current_text:
                print(f"[INFO] Normalized {context_name}: {resolved}")
            return resolved
        if len(matches) > 1:
            chosen = _choose_ambiguous_label(current_text, matches, context_name)
            print(f"[INFO] Normalized {context_name}: {chosen}")
            return chosen

        print(f"[WARN] {context_name} contains terms with 0 label matches: {current_text}")

        current_text = input(f"Please re-enter {context_name}:\n> ").strip()


def _normalize_reference_text(text: str) -> str:
    """Normalize free text for phrase-presence checks."""
    lowered = str(text).lower().replace("_", " ").replace("-", " ")
    lowered = re.sub(r"([a-z])(\d)", r"\1 \2", lowered)
    lowered = re.sub(r"(\d)([a-z])", r"\1 \2", lowered)
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    tokens = lowered.split()
    if tokens:
        last = tokens[-1]
        if last.endswith("ies") and len(last) > 3:
            tokens[-1] = last[:-3] + "y"
        elif last.endswith(("ses", "xes", "zes", "ches", "shes")) and len(last) > 3:
            tokens[-1] = last[:-2]
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 1:
            tokens[-1] = last[:-1]
    return " ".join(tokens)


def _collect_containing_labels(raw_text: str, candidate_labels: list[str]) -> list[str]:
    """Return labels whose normalized names contain the user's normalized phrase."""
    norm_phrase = _normalize_reference_text(raw_text)
    compact_phrase = norm_phrase.replace(" ", "")
    if not compact_phrase:
        return []

    matches = []
    for label in candidate_labels:
        norm_label = _normalize_reference_text(label)
        compact_label = norm_label.replace(" ", "")
        if (
            norm_phrase == norm_label
            or compact_phrase == compact_label
            or norm_phrase in norm_label
            or compact_phrase in compact_label
        ):
            matches.append(label)
    return sorted(set(matches))


def _find_user_mentioned_ambiguous_alias(raw_text: str, target_label: str, candidate_labels: list[str]) -> str | None:
    """Return the longest ambiguous phrase from raw_text that could refer to target_label."""
    if target_label not in candidate_labels:
        return None

    tokens = _normalize_reference_text(raw_text).split()
    phrases = []
    max_len = min(3, len(tokens))
    for size in range(1, max_len + 1):
        for start in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[start : start + size])
            if phrase and phrase not in phrases:
                phrases.append(phrase)

    ambiguous_phrases = []
    has_unique_user_mention = False
    for phrase in phrases:
        labels = _collect_containing_labels(phrase, candidate_labels)
        if target_label not in labels:
            continue
        if len(labels) == 1 and labels[0] == target_label:
            has_unique_user_mention = True
            break
        if len(labels) > 1:
            ambiguous_phrases.append(phrase)

    if has_unique_user_mention or not ambiguous_phrases:
        return None

    return max(ambiguous_phrases, key=lambda item: (len(_normalize_reference_text(item)), item))


def resolve_scene_spec_labels(parsed_scene: dict, candidate_labels: list[str], raw_instruction: str) -> dict:
    """Resolve scene parser object mentions to strict labels after LLM parsing."""
    resolved_objects = []
    seen_labels = set()

    for raw_label in parsed_scene.get("specified_objects", []) or []:
        if not str(raw_label).strip():
            continue
        raw_label = str(raw_label)
        ambiguous_alias = _find_user_mentioned_ambiguous_alias(raw_instruction, raw_label, candidate_labels)
        label = resolve_label_reference_input(
            ambiguous_alias or raw_label,
            candidate_labels,
            "scene object",
        )
        if label not in seen_labels:
            resolved_objects.append(label)
            seen_labels.add(label)

    resolved_relationships = []
    for rel in parsed_scene.get("relationships", []) or []:
        child_raw = str(rel.get("child", "")).strip()
        parent_raw = str(rel.get("parent", "")).strip()
        if not child_raw or not parent_raw:
            continue

        child_alias = _find_user_mentioned_ambiguous_alias(raw_instruction, child_raw, candidate_labels)
        parent_alias = _find_user_mentioned_ambiguous_alias(raw_instruction, parent_raw, candidate_labels)

        child = resolve_label_reference_input(child_alias or child_raw, candidate_labels, "relationship child")
        parent = resolve_label_reference_input(parent_alias or parent_raw, candidate_labels, "relationship parent")

        if child not in seen_labels:
            resolved_objects.append(child)
            seen_labels.add(child)
        if parent not in seen_labels:
            resolved_objects.append(parent)
            seen_labels.add(parent)

        resolved_relationships.append(
            {
                "child": child,
                "relation": rel.get("relation"),
                "parent": parent,
            }
        )

    return {
        "specified_objects": resolved_objects,
        "random_objects_count": int(parsed_scene.get("random_objects_count", 0) or 0),
        "relationships": resolved_relationships,
    }


def resolve_task_plan_labels(parsed_plan: dict, scene_labels: list[str], raw_instruction: str) -> dict:
    """Resolve parsed task targets to strict scene labels after LLM parsing."""
    resolved_primitives = []
    for primitive in parsed_plan.get("primitives", []) or []:
        if not isinstance(primitive, dict):
            continue
        resolved = dict(primitive)
        primitive_type = str(resolved.get("type", "")).strip().lower()
        requested_mode = str(resolved.get("mode", "")).strip().lower()
        has_position_xy = isinstance(resolved.get("position_xy"), list) and len(resolved.get("position_xy")) == 2
        if primitive_type == "place" and not requested_mode and has_position_xy:
            resolved["mode"] = "pose"
            requested_mode = "pose"
        if primitive_type == "place" and requested_mode == "pose":
            resolved.pop("target", None)
            resolved_primitives.append(resolved)
            continue
        target = resolved.get("target")
        if isinstance(target, str) and target.strip():
            ambiguous_alias = _find_user_mentioned_ambiguous_alias(raw_instruction, target, scene_labels)
            resolved["target"] = resolve_label_reference_input(
                ambiguous_alias or target,
                scene_labels,
                "task target",
            )
        queries = resolved.get("affordance_queries")
        if isinstance(queries, list):
            resolved_queries = []
            for entry in queries:
                if not isinstance(entry, dict):
                    continue
                resolved_entry = dict(entry)
                obj_ref = resolved_entry.get("object")
                if isinstance(obj_ref, str) and obj_ref.strip():
                    ambiguous_alias = _find_user_mentioned_ambiguous_alias(
                        raw_instruction, obj_ref, scene_labels
                    )
                    resolved_entry["object"] = resolve_label_reference_input(
                        ambiguous_alias or obj_ref,
                        scene_labels,
                        "affordance query object",
                    )
                resolved_queries.append(resolved_entry)
            resolved["affordance_queries"] = resolved_queries
        resolved_primitives.append(resolved)
    return {"primitives": resolved_primitives}


def main():
    runtime = None
    scene_initialized = False
    script_path = os.path.abspath(__file__)
    try:
        llm = LLMClient()
        generator = SceneGenerator(llm)

        if args_cli.restore_state:
            restored_scene, pending_add_label = load_restore_state(args_cli.restore_state)
            restore_labels = [obj.get("label") for obj in restored_scene if obj.get("label")]
            if pending_add_label and pending_add_label not in restore_labels:
                restore_labels.append(pending_add_label)
            runtime = create_runtime(args_cli, restore_labels, generator, runtime)
            manager = runtime["manager"]
            manager.update_scene(restored_scene)
            if pending_add_label:
                result = manager.add_object(pending_add_label)
                if result is False:
                    print(
                        f"[WARN] Scene state was restored, but '{pending_add_label}' was not added. "
                        "The current table layout has no valid free placement that satisfies bounds and collision checks."
                    )
                    print(
                        "[INFO] Remove or move an object and try adding it again if you still want this asset in scene."
                    )
                elif result == "requires_scene_rebuild":
                    raise RuntimeError(
                        f"Restart flow still missing requested label '{pending_add_label}' in runtime."
                    )
            sync_and_step_runtime(runtime)
            print("[INFO] Scene state restored after restart.")
            scene_initialized = True
        elif args_cli.initial_yaml:
            yaml_path = args_cli.initial_yaml
            try:
                active_labels = extract_scene_labels_from_yaml(yaml_path)
                runtime = create_runtime(args_cli, active_labels, generator, runtime)
                manager = runtime["manager"]
                if manager.load_from_yaml(yaml_path):
                    sync_and_step_runtime(runtime)
                    print("[INFO] YAML object layout loaded after restart.")
                    scene_initialized = True
                else:
                    runtime = create_runtime(args_cli, [], generator, runtime)
                    sync_and_step_runtime(runtime)
                    print("[WARN] YAML load failed after restart. Showing empty tabletop preview.")
            except Exception as e:
                print(f"[ERROR] Failed to load initial YAML object layout after restart: {e}")
                runtime = create_runtime(args_cli, [], generator, runtime)
                sync_and_step_runtime(runtime)
                print("[WARN] Showing empty tabletop preview.")
        else:
            print(f"[INFO] Using fixed scene: {scene_create_config.DEFAULT_SCENE_NAME}")
            print("[INFO] Loading empty scene preview...")
            runtime = create_runtime(args_cli, [], generator, runtime)
            sync_and_step_runtime(runtime)
            _set_active_runtime(runtime)
            print("[INFO] Empty scene is ready. Use the simulator viewport to inspect it.")

        while True:
            if runtime is not None:
                _set_active_runtime(runtime)
                manager = runtime["manager"]
                sim = runtime["sim"]

            if not scene_initialized:
                print("\n" + "=" * 40)
                print("SCENE GENERATION")
                print("=" * 40)
                print("1: Generate from text (default)")
                print("2: Load from YAML")

                mode = input("Choice [1/2]: ").strip()

                if mode == "2":
                    default_path = scene_create_config.DEFAULT_LOAD_YAML_PATH
                    yaml_path = input(f"YAML path [{default_path}]: ").strip() or default_path

                    try:
                        active_labels = extract_scene_labels_from_yaml(yaml_path)
                    except FileNotFoundError:
                        print(f"[ERROR] File not found: {yaml_path}")
                        continue
                    except Exception as e:
                        print(f"[ERROR] Failed to inspect YAML object labels: {e}")
                        continue

                    restart_editor_with_yaml(script_path, yaml_path)
                else:
                    default_prompt = (
                        "Could you help me set up a casual tabletop scene, please? "
                        "I'd like a blue kettle, a white kettle, a shelf, a banana, a yellow bowl, "
                        "a white plate, a white cup, a sugar box, and that doraemon cup. "
                        "The banana should be inside the yellow bowl, and the white cup should be on top of the wooden shelf. "
                        "Also, put the blue kettle to the left of the sugar box, the white plate to the right of the yellow bowl, "
                        "and the doraemon cup in front of the sugar box. "
                        "Oh, and the yellow bowl should be behind the wooden shelf, and the sugar box should be near the white plate."
                    )
                    prompt = input("Scene description:\n> ").strip() or default_prompt
                    parsed_scene = llm.parse_scene_instruction(prompt)
                    parsed_scene = resolve_scene_spec_labels(
                        parsed_scene,
                        list(generator.objects_db.keys()),
                        prompt,
                    )
                    scene_data = generator.generate_scene_from_parsed(parsed_scene)
                    if not scene_data:
                        print("[ERROR] Generation failed.")
                        sync_and_step_runtime(runtime)
                        continue

                    restart_editor_with_state(script_path, scene_data)

                print("Updating simulation...")
                for _ in range(50):
                    sim.step()
                scene_initialized = True

            editing = True
            while editing:
                print("\n" + "-" * 40)
                print("EDIT MODE")
                print("-" * 40)
                print("1: Save & Exit")
                print("2: Add object")
                print("3: Remove object")
                print("4: Regenerate scene")
                print("5: Modify object position")
                print("6: Modify object rotation")

                choice = input("Choice: ").strip()

                if choice == "1":
                    default_task = (
                        "Please pick up the banana, then pour into the bowl, and after that "
                        "put the banana in the yellow bowl, then finally pick the white cup."
                    )
                    task = input("Task description:\n> ").strip() or default_task
                    scene_labels = manager.get_scene_labels()
                    if not scene_labels:
                        print("[WARN] Scene is empty. Cannot export a task.")
                        continue
                    parsed = llm.parse_task_primitives(task, scene_labels)
                    parsed = resolve_task_plan_labels(parsed, scene_labels, task)
                    print(f"[INFO] Parsed primitives: {parsed.get('primitives', [])}")

                    compiled_primitives, compile_errors = compile_primitives_for_yaml(manager.current_scene, parsed)
                    if compile_errors:
                        print("[ERROR] Task export blocked due to invalid primitive plan:")
                        for err in compile_errors:
                            print(f"  - {err}")
                        continue

                    print("\n--- Domain Randomization ---")
                    enable_random_lights = input("Enable random lights? [y/N]: ").strip().lower() == "y"
                    if enable_random_lights:
                        if apply_random_lights_preview(runtime):
                            sync_and_step_runtime(runtime, num_steps=10)

                    enable_texture_rand = input("Enable texture randomization? [Y/n]: ").strip().lower() != "n"
                    if enable_texture_rand:
                        if apply_random_texture_preview(num_envs=args_cli.num_envs):
                            sync_and_step_runtime(runtime, num_steps=10)

                    random_config = None
                    if input("Enable randomization? [y/N]: ").strip().lower() == "y":
                        random_config = get_randomization_config()

                    image_noise_std = 0
                    noise_input = input("Image noise std [0]: ").strip()
                    if noise_input:
                        image_noise_std = float(noise_input)

                    image_speckle_noise_std = 0.5
                    speckle_input = input("Image speckle noise std [0.5]: ").strip()
                    if speckle_input:
                        image_speckle_noise_std = float(speckle_input)
                    visual_config = {
                        "enable_random_lights": enable_random_lights,
                        "image_noise_std": image_noise_std,
                        "image_speckle_noise_std": image_speckle_noise_std,
                        "enable_texture_randomization": enable_texture_rand,
                    }

                    export_to_yaml(
                        manager.current_scene,
                        task,
                        compiled_primitives,
                        random_config,
                        visual_config,
                    )

                    print("\n Done. Exiting...")
                    print("Complete shutdown finished")
                    _set_active_runtime(None)
                    destroy_runtime(runtime)
                    runtime = None
                    simulation_app.close()
                    return

                elif choice == "2":
                    label = input("Object name: ").strip()
                    if label:
                        label = resolve_label_reference_input(
                            label,
                            list(generator.objects_db.keys()),
                            "object name",
                        )
                        result = manager.add_object(label)
                        if result == "requires_scene_rebuild":
                            restart_editor_with_state(
                                script_path,
                                manager.current_scene,
                                pending_add_label=label,
                            )
                        if result is not False:
                            sync_and_step_runtime(runtime)

                elif choice == "3":
                    label = input("Object name: ").strip()
                    if label:
                        scene_labels = manager.get_scene_labels()
                        if not scene_labels:
                            print("[WARN] Scene is empty.")
                            continue
                        label = resolve_label_reference_input(label, scene_labels, "object name")
                        manager.remove_object(label)
                        sync_and_step_runtime(runtime)

                elif choice == "4":
                    restart_editor_fresh(script_path)

                elif choice == "5":
                    if not manager.current_scene:
                        print("[WARN] Scene is empty.")
                    else:
                        print("\nCurrent objects:")
                        for idx, obj in enumerate(manager.current_scene):
                            pos = obj["position"]
                            print(f"  {idx}: {obj.get('label', 'unknown')}  (x={pos['x']:.3f}, y={pos['y']:.3f})")
                        try:
                            sel = int(input("Select object index: ").strip())
                            if 0 <= sel < len(manager.current_scene):
                                new_x = float(input("New x coordinate: ").strip())
                                new_y = float(input("New y coordinate: ").strip())
                                ok, reason = manager.try_move_object(sel, new_x, new_y)
                                if ok:
                                    print(
                                        f"[SUCCESS] Moved '{manager.current_scene[sel].get('label', 'unknown')}' "
                                        f"to ({new_x:.3f}, {new_y:.3f})"
                                    )
                                    sync_and_step_runtime(runtime)
                                else:
                                    print(f"[WARN] Move rejected: {reason}")
                            else:
                                print("[WARN] Invalid index.")
                        except ValueError:
                            print("[WARN] Invalid input.")

                elif choice == "6":
                    if not manager.current_scene:
                        print("[WARN] Scene is empty.")
                    else:
                        import math as _math

                        print("\nCurrent objects:")
                        for idx, obj in enumerate(manager.current_scene):
                            cur_deg = _math.degrees(obj.get("_angle", 0.0))
                            print(f"  {idx}: {obj.get('label', 'unknown')}  (yaw={cur_deg:.1f} deg)")
                        try:
                            sel = int(input("Select object index: ").strip())
                            if 0 <= sel < len(manager.current_scene):
                                delta_deg = float(input("Rotate around Z axis (degrees): ").strip())
                                ok, reason = manager.try_rotate_object(sel, delta_deg)
                                if ok:
                                    obj = manager.current_scene[sel]
                                    print(
                                        f"[SUCCESS] Rotated '{obj.get('label', 'unknown')}' by {delta_deg:.1f} deg "
                                        f" to  yaw={_math.degrees(obj.get('_angle', 0.0)):.1f} deg"
                                    )
                                    sync_and_step_runtime(runtime)
                                else:
                                    print(f"[WARN] Rotation rejected: {reason}")
                            else:
                                print("[WARN] Invalid index.")
                        except ValueError:
                            print("[WARN] Invalid input.")

                if runtime is not None:
                    runtime["scene"].update(runtime["sim_dt"])

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        _set_active_runtime(None)
        destroy_runtime(runtime)
        try:
            simulation_app.close()
        except Exception:
            pass
        _terminal_input.close()


if __name__ == "__main__":
    main()
