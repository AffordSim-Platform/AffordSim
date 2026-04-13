# scene_system.py
"""
Unified scene generation and management system.
Consolidates: llm_client, scene_generator, scene_manager, exporter, utils

Dimension Convention (from label.json):
- 'length' -> X-coordinate in world space (dim_x)
- 'width'  -> Y-coordinate in world space (dim_y)  
- 'height' -> Z-coordinate (height)
"""

import os
import json
import math
import random
import torch
from openai import OpenAI
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

import config


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def make_flow_list(data_list):
    """Create inline YAML list [a, b, c]"""
    seq = CommentedSeq(data_list)
    seq.fa.set_flow_style()
    return seq


def get_obb_corners(x, y, dim_x, dim_y, angle):
    """
    Calculate rotated rectangle corners.
    
    Args:
        x, y: Center position in world coordinates
        dim_x: X extent (from 'length' in JSON)
        dim_y: Y extent (from 'width' in JSON)
        angle: Rotation angle in radians
    """
    c, s = math.cos(angle), math.sin(angle)
    hx, hy = dim_x / 2.0, dim_y / 2.0
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(lx * c - ly * s + x, lx * s + ly * c + y) for lx, ly in corners]


def check_obb_collision(pos1, dim1, angle1, pos2, dim2, angle2):
    """
    SAT collision detection for oriented bounding boxes.
    
    Args:
        pos: (x, y) center position
        dim: (dim_x, dim_y) extents in world space
        angle: rotation in radians
    
    Returns:
        True if colliding, False otherwise
    """
    corners1 = get_obb_corners(pos1[0], pos1[1], dim1[0], dim1[1], angle1)
    corners2 = get_obb_corners(pos2[0], pos2[1], dim2[0], dim2[1], angle2)
    
    for corners in [corners1, corners2]:
        for i in range(4):
            p1, p2 = corners[i], corners[(i + 1) % 4]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            length = math.hypot(*edge)
            if length < 1e-6:
                continue
            axis = (-edge[1] / length, edge[0] / length)
            proj1 = [x * axis[0] + y * axis[1] for x, y in corners1]
            proj2 = [x * axis[0] + y * axis[1] for x, y in corners2]
            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False
    return True


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """Handles LLM-based instruction parsing."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

    def _call_llm(self, system_prompt: str, user_message: str) -> dict:
        """Generic LLM call with JSON response parsing."""
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return {}

    def parse_scene_instruction(self, instruction: str) -> dict:
        """
        Parse scene generation instruction.
        
        Returns:
            {
                "specified_objects": ["banana", "bowl", ...],
                "random_objects_count": 0,
                "relationships": [{"child": "banana", "relation": "inside", "parent": "bowl"}, ...]
            }
        """
        system_prompt = """You are a Scene Graph Parser. Extract:
1. `specified_objects`: List of explicit object names.
2. `random_objects_count`: Count of random objects.
3. `relationships`: Spatial constraints.
   Format: {"child": "fork", "relation": "left_of", "parent": "plate"}
   
   Supported relations:
   - "inside", "on_top" (Vertical/Container relations)
   - "left_of", "right_of", "front", "behind" (Directional relations)

Example: "Plate with a banana inside, and a fork on the left of the plate."
Output:
{
    "specified_objects": ["plate", "banana", "fork"],
    "relationships": [
        {"child": "banana", "relation": "inside", "parent": "plate"},
        {"child": "fork", "relation": "left_of", "parent": "plate"}
    ]
}"""
        result = self._call_llm(system_prompt, instruction)
        print(f"[LLM] Parsed scene instruction: {result}")
        return {
            "specified_objects": result.get("specified_objects", []),
            "random_objects_count": result.get("random_objects_count", 0),
            "relationships": result.get("relationships", [])
        }

    def parse_task_instruction(self, instruction: str) -> dict:
        """
        Parse pick-and-place task instruction.

        Returns:
            {"target": "banana", "container": "bowl", "relation": "in"}
        """
        system_prompt = """Extract three fields:
1. 'target': Object to pick up.
2. 'container': Object to place into/onto (or null).
3. 'relation': The spatial relation between target and container, either "in" or "on". Use "in" for placing inside a container (e.g. bowl, box, cup). Use "on" for placing on top of a surface (e.g. plate, shelf, table). Default to "in" if unclear.
Return ONLY a JSON object with keys "target", "container", "relation"."""
        result = self._call_llm(system_prompt, instruction)
        return {
            "target": result.get("target"),
            "container": result.get("container"),
            "relation": result.get("relation", "in")
        }


# =============================================================================
# SCENE GENERATOR
# =============================================================================

class SceneGenerator:
    """Generates scene layouts from natural language instructions."""
    
    # Common synonyms for object matching
    SYNONYMS = {
        'cup': ['mug', 'cup'],
        'mug': ['mug', 'cup'],
        'can': ['can', 'canned'],
        'canned': ['can', 'canned'],
        'box': ['box'],
        'bottle': ['bottle'],
        'bowl': ['bowl'],
        'plate': ['plate', 'dish'],
        'dish': ['plate', 'dish'],
        'fruit': ['fruit'],
        'apple': ['apple'],
        'banana': ['banana'],
        'pitcher': ['pitcher', 'jug'],
        'jug': ['pitcher', 'jug'],
        'tuna': ['tuna'],
        'spam': ['spam'],
        'soup': ['soup'],
        'mustard': ['mustard'],
        'sugar': ['sugar'],
        'coffee': ['coffee'],
        'cheez-it': ['cheez-it', 'cheezit', 'cheese'],
        'jell-o': ['jell-o', 'jello', 'gelatin'],
    }
    
    # Words to ignore when matching (colors, generic descriptors)
    IGNORE_WORDS = {'red', 'blue', 'yellow', 'green', 'white', 'black', 'brown', 
                    'orange', 'purple', 'pink', 'gray', 'grey', 'metal', 'plastic',
                    'wooden', 'glass', 'ceramic', 'small', 'large', 'big', 'little'}
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.objects_db = self._load_objects_db()

    def _load_objects_db(self) -> dict:
        """Load object database and index by multiple keys for flexible matching."""
        try:
            with open(config.ALL_OBJECTS_JSON, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            # Store raw data for reference
            self._raw_objects = all_data
            
            mapping = {}
            for item in all_data:
                full_label = item.get('label', item.get('type', '')).lower()
                if not full_label:
                    continue
                
                # Index by full label
                mapping.setdefault(full_label, []).append(item)
                
                # Extract and index by keywords
                keywords = self._extract_keywords(full_label)
                for keyword in keywords:
                    mapping.setdefault(keyword, []).append(item)
                    
                    # Also add synonyms
                    if keyword in self.SYNONYMS:
                        for syn in self.SYNONYMS[keyword]:
                            if syn != keyword:
                                mapping.setdefault(syn, []).append(item)
            
            return mapping
        except Exception as e:
            print(f"[ERROR] Loading objects data: {e}")
            return {}

    def _extract_keywords(self, label: str) -> list:
        """Extract meaningful keywords from a verbose label."""
        words = label.lower().replace('-', ' ').replace("'", '').split()
        # Filter out color and generic descriptor words
        keywords = [w for w in words if w not in self.IGNORE_WORDS and len(w) > 2]
        return keywords

    def _find_best_match(self, query: str) -> list:
        """
        Find best matching objects for a user query.
        
        Matching priority:
        1. Exact full label match
        2. Exact keyword match
        3. Partial/substring match
        4. Synonym match
        
        Returns:
            List of matching object data dictionaries
        """
        query_lower = query.lower().strip()
        
        # 1. Exact match in database
        if query_lower in self.objects_db:
            return self.objects_db[query_lower]
        
        # 2. Check if query is a substring of any key or vice versa
        for key, items in self.objects_db.items():
            if query_lower in key or key in query_lower:
                return items
        
        # 3. Multi-word query: try matching all words
        query_words = [w for w in query_lower.replace('-', ' ').split() 
                       if w not in self.IGNORE_WORDS and len(w) > 2]
        
        if query_words:
            # Find objects that match all query words
            candidates = []
            for item in self._raw_objects:
                label_lower = item.get('label', '').lower()
                label_keywords = self._extract_keywords(label_lower)
                
                # Check if all query words match (as substring or synonym)
                matches_all = True
                for qw in query_words:
                    word_matched = False
                    for lk in label_keywords:
                        if qw in lk or lk in qw:
                            word_matched = True
                            break
                        # Check synonyms
                        if qw in self.SYNONYMS:
                            if any(syn in lk or lk in syn for syn in self.SYNONYMS[qw]):
                                word_matched = True
                                break
                    if not word_matched:
                        matches_all = False
                        break
                
                if matches_all:
                    candidates.append(item)
            
            if candidates:
                return candidates
        
        # 4. No match found
        return []

    def _find_instance_key(self, query: str, instances: dict) -> str:
        """
        Find the instance key that best matches the query.
        Used for matching relationship labels to actual instance keys.
        
        Args:
            query: The label to search for
            instances: Dict of current instances
            
        Returns:
            Matching instance key or None
        """
        query_lower = query.lower().strip()
        
        # 1. Exact key match
        if query_lower in instances:
            return query_lower
        
        # 2. Check if query matches any instance's actual label or data label
        for key, inst in instances.items():
            # Check against instance key
            if query_lower in key or key in query_lower:
                return key
            
            # Check against actual label from data
            data_label = inst['data'].get('label', '').lower()
            if query_lower in data_label or data_label in query_lower:
                return key
            
            # Check keywords
            data_keywords = self._extract_keywords(data_label)
            query_keywords = self._extract_keywords(query_lower)
            
            # If any query keyword matches any data keyword
            for qk in query_keywords:
                for dk in data_keywords:
                    if qk in dk or dk in qk:
                        return key
                    # Check synonyms
                    if qk in self.SYNONYMS:
                        if any(syn in dk or dk in syn for syn in self.SYNONYMS[qk]):
                            return key
        
        return None

    def generate_scene(self, instruction: str) -> list:
        """
        Generate complete scene from instruction.
        
        Returns:
            List of placed object dictionaries
        """
        # 1. Parse instruction
        parsed = self.llm.parse_scene_instruction(instruction)
        specified = parsed.get("specified_objects", [])
        random_count = parsed.get("random_objects_count", 0)
        relationships = parsed.get("relationships", [])

        # 2. Create object instances
        instances = {}
        label_to_key = {}  # Map user's label to the key used in instances
        
        for label in specified:
            label_lower = label.lower()
            matches = self._find_best_match(label_lower)
            
            if matches:
                chosen = random.choice(matches)
                actual_label = chosen.get('label', label_lower).lower()
                # Use user's label as key for relationship matching
                instances[label_lower] = {
                    'id': label_lower,
                    'data': chosen,
                    'parent': None,
                    'relation': None,
                    '_actual_label': actual_label  # Store actual label for reference
                }
                label_to_key[label_lower] = label_lower
                print(f"[MATCH] '{label}' -> '{chosen.get('label', 'unknown')}'")
            else:
                print(f"[WARN] No match found for '{label}'")

        # Add random objects
        # Get unique objects not already selected
        used_ids = {inst['data']['id'] for inst in instances.values()}
        available = [item for item in self._raw_objects if item['id'] not in used_ids]
        
        if random_count > 0 and available:
            selected = random.sample(available, min(random_count, len(available)))
            for item in selected:
                label = item.get('label', 'unknown').lower()
                logic_id = f"{label}_rand"
                instances[logic_id] = {
                    'id': logic_id,
                    'data': item,
                    'parent': None,
                    'relation': None
                }

        # 3. Apply relationships (with flexible matching for relationship labels)
        for rel in relationships:
            child = rel.get('child', '').lower()
            parent = rel.get('parent', '').lower()
            
            # Find matching keys in instances (user might use different terms)
            child_key = self._find_instance_key(child, instances)
            parent_key = self._find_instance_key(parent, instances)
            
            if child_key and parent_key:
                instances[child_key]['parent'] = parent_key
                instances[child_key]['relation'] = rel.get('relation')
            else:
                if not child_key:
                    print(f"[WARN] Relationship child '{child}' not found in scene")
                if not parent_key:
                    print(f"[WARN] Relationship parent '{parent}' not found in scene")

        # 4. Topological sort for placement order
        placement_queue = []
        unplaced = list(instances.values())
        
        for _ in range(10):
            placed_ids = {item['id'] for item in placement_queue}
            remaining = []
            for item in unplaced:
                if item['parent'] is None or item['parent'] in placed_ids:
                    placement_queue.append(item)
                else:
                    remaining.append(item)
            unplaced = remaining
            if not unplaced:
                break
        placement_queue.extend(unplaced)  # Force-place any remaining

        # 5. Execute placement
        scene = []
        placed_map = {}

        for idx, item in enumerate(placement_queue):
            parent = placed_map.get(item['parent'])
            
            placement = None
            if parent and item['relation']:
                placement = self._place_relative(item['data'], parent, item['relation'], scene)
            
            if placement is None:
                placement = self._place_on_table(item['data'], scene)

            if placement:
                placement['placement_order'] = idx + 1
                placement['_logic_id'] = item['id']
                scene.append(placement)
                placed_map[item['id']] = placement
            else:
                print(f"[WARN] Failed to place '{item['id']}'")

        print(f"[INFO] Generated scene with {len(scene)} objects.")
        return scene

    def _place_on_table(self, obj_data: dict, current_scene: list, max_attempts: int = 200) -> dict:
        """Find valid position on table surface."""
        dim_x = obj_data['dimensions_cm']['length'] / 100.0
        dim_y = obj_data['dimensions_cm']['width'] / 100.0
        height = obj_data['dimensions_cm']['height'] / 100.0

        # Use bottom_offset if available, otherwise fall back to height/2
        bottom_offset = obj_data.get('bottom_offset_m', height / 2.0)
        placement_z = 0.025 + bottom_offset + 0.005
        
        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * math.pi)
            radius = max(dim_x, dim_y) / 2.0 + config.MARGIN
            
            x = random.uniform(config.TABLE_MIN_X + radius, config.TABLE_MAX_X - radius)
            y = random.uniform(config.TABLE_MIN_Y + radius, config.TABLE_MAX_Y - radius)

            if not self._check_collision((x, y), (dim_x, dim_y), angle, current_scene):
                return self._create_placement(obj_data, x, y, placement_z, angle, dim_x, dim_y)
        return None

    def _place_relative(self, child_data: dict, parent: dict, relation: str, current_scene: list) -> dict:
        """Place object relative to parent based on relation type."""
        p_pos = parent['position']
        p_dim_x, p_dim_y = parent.get('_dims', (0.1, 0.1))
        p_height = parent.get('_height', 0.05)
        p_bottom_offset = parent.get('_bottom_offset', p_height / 2.0)
        p_angle = parent.get('_angle', 0.0)

        c_dim_x = child_data['dimensions_cm']['length'] / 100.0
        c_dim_y = child_data['dimensions_cm']['width'] / 100.0
        c_height = child_data['dimensions_cm']['height'] / 100.0
        c_bottom_offset = child_data.get('bottom_offset_m', c_height / 2.0)

        max_attempts = 50
        ignore_parent = False
        # Default: place on table using bottom_offset
        base_z = 0.025 + c_bottom_offset + 0.005

        # Define sampling strategy based on relation
        if relation in ["inside", "on_top"]:
            ignore_parent = True
            # Parent top z = parent_origin_z + (full_height - bottom_offset)
            p_top_z = p_pos['z'] + (p_height - p_bottom_offset)
            if relation == "inside":
                # Place child so bottom is at parent's mid-height
                p_mid_z = p_pos['z'] - p_bottom_offset + p_height * 0.5
                base_z = p_mid_z + c_bottom_offset
            else:  # on_top
                # Place child so bottom sits on parent's top
                base_z = p_top_z + c_bottom_offset + 0.002
            
            def sample():
                return 0, 0  # Center placement
                
        elif relation in ["left_of", "right_of", "front", "behind"]:
            gap_min, gap_max = 0.02, 0.15
            
            def sample():
                gap = random.uniform(gap_min, gap_max)
                jx = random.uniform(-p_dim_x / 2, p_dim_x / 2)
                jy = random.uniform(-p_dim_y / 2, p_dim_y / 2)
                
                if relation == "left_of":
                    return jx, -((p_dim_y + c_dim_y) / 2 + gap)
                elif relation == "right_of":
                    return jx, (p_dim_y + c_dim_y) / 2 + gap
                elif relation == "front":
                    return (p_dim_x + c_dim_x) / 2 + gap, jy
                else:  # behind
                    return -((p_dim_x + c_dim_x) / 2 + gap), jy
        else:
            return None

        # Try placement
        for _ in range(max_attempts):
            dx, dy = sample()
            angle = random.uniform(0, 2 * math.pi)
            
            # Transform to world coordinates
            x = p_pos['x'] + dx * math.cos(p_angle) - dy * math.sin(p_angle)
            y = p_pos['y'] + dx * math.sin(p_angle) + dy * math.cos(p_angle)

            # Boundary check
            radius = max(c_dim_x, c_dim_y) / 2.0
            if not (config.TABLE_MIN_X + radius < x < config.TABLE_MAX_X - radius and
                    config.TABLE_MIN_Y + radius < y < config.TABLE_MAX_Y - radius):
                continue

            # Collision check
            ignore_id = parent['id'] if ignore_parent else None
            if not self._check_collision((x, y), (c_dim_x, c_dim_y), angle, current_scene, ignore_id):
                return self._create_placement(child_data, x, y, base_z, angle, c_dim_x, c_dim_y)

        return None

    def _check_collision(self, pos: tuple, dims: tuple, angle: float, 
                         scene: list, ignore_id: str = None) -> bool:
        """Check collision against all objects in scene."""
        margin_dims = (dims[0] + config.MARGIN, dims[1] + config.MARGIN)
        
        for obj in scene:
            if ignore_id and obj['id'] == ignore_id:
                continue
            
            ex_dims = obj.get('_dims', (0.1, 0.1))
            ex_pos = (obj['position']['x'], obj['position']['y'])
            ex_angle = obj.get('_angle', 0.0)
            ex_margin = (ex_dims[0] + config.MARGIN, ex_dims[1] + config.MARGIN)

            if check_obb_collision(pos, margin_dims, angle, ex_pos, ex_margin, ex_angle):
                return True
        return False

    def _create_placement(self, obj_data: dict, x: float, y: float, z: float,
                          angle: float, dim_x: float, dim_y: float) -> dict:
        """Create standardized placement dictionary."""
        qw = math.cos(angle / 2.0)
        qz = math.sin(angle / 2.0)
        h = obj_data.get('dimensions_cm', {}).get('height', 5.0) / 100.0
        bottom_offset = obj_data.get('bottom_offset_m', h / 2.0)

        return {
            'id': obj_data['id'],
            'label': obj_data.get('label', 'unknown'),
            'position': {'x': round(x, 3), 'y': round(y, 3), 'z': round(z, 3)},
            'orientation': {'w': round(qw, 4), 'x': 0.0, 'y': 0.0, 'z': round(qz, 4)},
            '_dims': (dim_x, dim_y),
            '_height': h,
            '_bottom_offset': bottom_offset,
            '_angle': angle
        }


# =============================================================================
# SCENE MANAGER
# =============================================================================

class SceneManager:
    """Manages scene state and simulation synchronization."""
    
    def __init__(self, sim_scene, generator: SceneGenerator):
        self.sim_scene = sim_scene
        self.generator = generator
        self.current_scene = []
        
        with open(config.ALL_OBJECTS_JSON, 'r') as f:
            self.all_objects_data = json.load(f)

    def update_scene(self, scene_data: list):
        """Update current scene description."""
        self.current_scene = scene_data

    def add_object(self, label: str):
        """Add object to scene by label using flexible matching."""
        # Use generator's improved matching
        candidates = self.generator._find_best_match(label)
        
        if not candidates:
            keywords = sorted(set(
                kw for o in self.all_objects_data
                for kw in self.generator._extract_keywords(o.get('label', ''))
            ))
            print(f"[WARN] Object '{label}' not found. Available: {', '.join(keywords)}")
            return False

        chosen = candidates[0]
        print(f"[MATCH] '{label}' -> '{chosen.get('label', 'unknown')}'")

        # Rebuild scene context with dimensions
        context = []
        for item in self.current_scene:
            matched = next((x for x in self.all_objects_data if x['id'] == item['id']), None)
            temp = item.copy()
            if matched and 'dimensions_cm' in matched:
                d = matched['dimensions_cm']
                temp['_dims'] = (d.get('length', 10) / 100, d.get('width', 10) / 100)
                temp['_height'] = d.get('height', 5) / 100
            context.append(temp)

        new_item = self.generator._place_on_table(chosen, context, max_attempts=2000)
        if new_item:
            new_item['_logic_id'] = label.lower().strip()
            self.current_scene.append(new_item)
            print(f"[SUCCESS] Added '{chosen.get('label', label)}'")
            return True
        print(f"[WARN] No space for '{label}'")
        return False

    def remove_object(self, label: str):
        """Remove object from scene by label using flexible matching."""
        label_lower = label.lower().strip()
        initial = len(self.current_scene)
        
        # Find object to remove using flexible matching
        to_remove = None
        for obj in self.current_scene:
            obj_label = obj.get('label', '').lower()
            # Check exact match
            if label_lower == obj_label:
                to_remove = obj
                break
            # Check substring match
            if label_lower in obj_label or obj_label in label_lower:
                to_remove = obj
                break
            # Check keyword match
            obj_keywords = self.generator._extract_keywords(obj_label)
            query_keywords = self.generator._extract_keywords(label_lower)
            for qk in query_keywords:
                for ok in obj_keywords:
                    if qk in ok or ok in qk:
                        to_remove = obj
                        break
                if to_remove:
                    break
            if to_remove:
                break
        
        if to_remove:
            self.current_scene.remove(to_remove)
            print(f"[SUCCESS] Removed '{to_remove.get('label', label)}'")
            return True
        
        print(f"[WARN] '{label}' not found in scene. Current objects:")
        for obj in self.current_scene:
            print(f"       - {obj.get('label', 'unknown')}")
        return False

    def sync_to_simulation(self):
        """Write current scene state to Isaac Lab simulation."""
        from sim_config import SIM_OBJECT_IDS, SIM_NUM_OBJECTS

        objects_to_place = {obj['id']: obj for obj in self.current_scene}
        placed_ids = set(objects_to_place.keys())

        for i in range(1, SIM_NUM_OBJECTS + 1):
            obj_name = f"object_{i}"
            obj_id = SIM_OBJECT_IDS[i - 1]  # actual file stem, e.g. "001", "019"

            if obj_name not in self.sim_scene.rigid_objects:
                continue

            sim_obj = self.sim_scene.rigid_objects[obj_name]

            if obj_id in placed_ids:
                data = objects_to_place[obj_id]
                pos = [data['position']['x'], data['position']['y'], data['position']['z']]
                rot = [data['orientation']['w'], data['orientation']['x'],
                       data['orientation']['y'], data['orientation']['z']]
            else:
                pos = [-10.0, -10.0 - i, 1.5]
                rot = [1.0, 0.0, 0.0, 0.0]

            pose = torch.tensor([pos + rot], dtype=torch.float32, device=sim_obj.device)
            sim_obj.write_root_pose_to_sim(pose)
            sim_obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=sim_obj.device))

    def find_key_by_label(self, label: str) -> str:
        """Find object key by label for export using flexible matching."""
        if not label:
            return None
            
        label_lower = label.lower().strip()
        
        for obj in self.current_scene:
            obj_label = obj.get('label', '').lower()
            logic_id = obj.get('_logic_id', obj_label)
            
            # Exact or substring match
            if label_lower in obj_label or obj_label in label_lower:
                return logic_id
            
            # Keyword match
            obj_keywords = self.generator._extract_keywords(obj_label)
            query_keywords = self.generator._extract_keywords(label_lower)
            
            for qk in query_keywords:
                for ok in obj_keywords:
                    if qk in ok or ok in qk:
                        return logic_id
                    # Check synonyms
                    if qk in self.generator.SYNONYMS:
                        if any(syn in ok or ok in syn for syn in self.generator.SYNONYMS[qk]):
                            return logic_id
        
        return None

    def load_from_yaml(self, yaml_path: str) -> bool:
        """Load scene from YAML configuration."""
        if not os.path.exists(yaml_path):
            print(f"[ERROR] File not found: {yaml_path}")
            return False
            
        try:
            import yaml as pyyaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                cfg = pyyaml.safe_load(f)
            
            objects_dict = cfg.get("objects", {})
            if not objects_dict:
                print("[WARN] No objects in YAML")
                return False

            reconstructed = []
            for idx, (obj_key, obj_val) in enumerate(objects_dict.items()):
                usd_path = obj_val.get('usd_path', '')
                obj_id = os.path.basename(usd_path).split('.')[0]
                
                # Extract label
                label = "unknown"
                for tag in obj_val.get('semantic_tags', []):
                    if len(tag) == 2 and tag[0] == 'class':
                        label = tag[1]
                        break
                
                # Get dimensions from database
                matched = next((x for x in self.all_objects_data if x['id'] == obj_id), None)
                if matched and 'dimensions_cm' in matched:
                    d = matched['dimensions_cm']
                    dim_x = d.get('length', 10) / 100
                    dim_y = d.get('width', 10) / 100
                    height = d.get('height', 5) / 100
                else:
                    dim_x, dim_y, height = 0.1, 0.1, 0.05
                
                # Extract pose
                rot = obj_val.get('init_rot', [1, 0, 0, 0])
                pos = obj_val.get('init_pos', [0, 0, 0])
                angle = 2 * math.atan2(rot[3], rot[0])

                reconstructed.append({
                    'id': obj_id,
                    'label': label,
                    'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
                    'orientation': {'w': rot[0], 'x': rot[1], 'y': rot[2], 'z': rot[3]},
                    '_dims': (dim_x, dim_y),
                    '_height': height,
                    '_angle': angle,
                    'placement_order': idx + 1,
                    '_logic_id': obj_key
                })

            self.current_scene = reconstructed
            print(f"[SUCCESS] Loaded {len(reconstructed)} objects from YAML")
            return True
            
        except Exception as e:
            print(f"[ERROR] Loading YAML: {e}")
            import traceback
            traceback.print_exc()
            return False


# =============================================================================
# EXPORTER
# =============================================================================

def export_to_yaml(scene: list, task: str, random_config: dict = None, visual_config: dict = None):
    """
    Export scene configuration to YAML file.

    Args:
        scene: List of placed objects
        task: Task description string
        random_config: Optional randomization settings
        visual_config: Optional visual domain randomization settings
    """
    print(f"[INFO] Saving to: {config.SAVE_YAML_PATH}")
    
        yaml_data = {
            'simulation': {'decimation': 10, 'dt': 0.00833333, 'render_interval': 2, 'episode_length_s': 20},
            'scene': {'num_envs': 1, 'env_spacing': 10},
            'task': {'name': 'pick task', 'description': task},
            'runtime': {
                'headless': False, 'episodes': 50, 'save_npy': True, 'save_camera_images': True,
                'save_composite_images': False, 'save_mask_images': False, 'save_original_images': False,
                'enable_motion_planning': True,
                'data_dir': "/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/data/r2s2r_data/pick",
                'task_name': "FrankaFr3-OurGripper-Joint-VisPick-v0"
        },
        'robot': {'cfg_name': 'franka_fr3_our.yml', 'type': 'franka_fr3_default'},
        'offsets': {'grasp': make_flow_list([0, 0, 0.0]), 'place': make_flow_list([0, 0, 0.0])},
    }

    # Build objects dict
    objects_dict = {}
    for obj in scene:
        raw_key = obj.get('_logic_id', f"{obj['label']}_{obj['id']}")
        import re
        obj_key = re.sub(r'[^a-zA-Z0-9_]', '_', raw_key)
        semantic_label = obj['label'].replace(' ', '_')
        p, o = obj['position'], obj['orientation']
        h = float(obj.get('_height', 0.05))  # stored in meters

        entry = {
            'usd_path': f"{config.ASSET_ROOT}/{obj['id']}.usd",
            'semantic_tags': [make_flow_list(['class', semantic_label])],
            'init_pos': make_flow_list([p['x'], p['y'], p['z']]),
            'init_rot': make_flow_list([o['w'], o['x'], o['y'], o['z']]),
            'height': round(h, 3)
        }

        if random_config:
            d = random_config['pos_delta']
            entry['position_range'] = {
                'x': make_flow_list([round(p['x'] - d['back'], 3), round(p['x'] + d['front'], 3)]),
                'y': make_flow_list([round(p['y'] - d['right'], 3), round(p['y'] + d['left'], 3)]),
                'z': make_flow_list([round(p['z'], 3), round(p['z'], 3)])
            }
            entry['orientation_range'] = {'yaw': make_flow_list(random_config.get('rot_range', [0, 0]))}

        objects_dict[obj_key] = entry

    yaml_data['objects'] = objects_dict
    if visual_config is None:
        visual_config = {}
    yaml_data['enable_random_lights'] = visual_config.get('enable_random_lights', False)
    yaml_data['image_noise_std'] = visual_config.get('image_noise_std', 0)
    yaml_data['image_speckle_noise_std'] = visual_config.get('image_speckle_noise_std', 0.5)
    yaml_data['enable_objects_randomization'] = bool(random_config)
    yaml_data['enable_objects_randomization_curriculum'] = False
    yaml_data['texture_randomization'] = {
        'enabled': visual_config.get('enable_texture_randomization', True),
        'texture_dir': config.TEXTURE_DIR,
        'targets': {
            'table': {
                'prim_path': '/World/envs/env_{i}/table/Xform_01/Xform/leg/node_/Cube',
                'texture_dir': config.TEXTURE_DIR,
            }
        }
    }

    # Write file
    os.makedirs(os.path.dirname(config.SAVE_YAML_PATH), exist_ok=True)
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config.SAVE_YAML_PATH, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated by Scene System\n")
        yaml.dump(yaml_data, f)

    print("✓ YAML saved successfully.")
