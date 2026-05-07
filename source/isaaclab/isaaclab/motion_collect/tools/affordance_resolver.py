"""Resolve affordance NPZ paths by convention: {affordance_id}_{action}/{filename}."""

from pathlib import Path
from typing import Optional


# action  to  (directory suffix, filename)
ACTION_FILE_MAP = {
    "pick":      ("pick",    "grasp_result.npz"),
    "pick_hang": ("pick",    "grasp_result_hang.npz"),
    "pick_4dir": ("pick",    "grasp_result_4dir.npz"),
    "hang":      ("hang",    "affordance_clusters.npz"),
    "pourout":   ("pourout", "affordance_clusters.npz"),
    "pourin":    ("pourin",  "affordance_clusters.npz"),
    "place":     ("place",   "affordance_clusters.npz"),
    "open":      ("open",    "grasp_result.npz"),
    "pull":      ("pull",    "grasp_result.npz"),
    "push":      ("push",    "grasp_result.npz"),
}

# project root: Real2Sim2Real-Pipeline-auto_task_aff/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent


def resolve_affordance_path(
    object_name: str,
    action: str,
    project_root: Optional[Path] = None,
) -> Optional[str]:
    """Resolve an affordance NPZ path from object name and action type.

    Args:
        object_name: The object key used in the YAML (e.g. "banana", "bluekettle").
        action: One of "pick", "pick_4dir", "pourout", "pourin", "pull", "push".
        project_root: Override for the project root directory.

    Returns:
        Absolute path string if the file exists, None otherwise.
    """
    if action not in ACTION_FILE_MAP:
        print(f"  [WARN] Unknown affordance action: '{action}'")
        return None

    root = project_root or _PROJECT_ROOT
    suffix, filename = ACTION_FILE_MAP[action]
    path = root / "pc_dataset_generate" / "pc" / f"{object_name}_{suffix}" / filename

    if path.exists():
        return str(path)

    print(f"  [WARN] Affordance file not found: {path}")
    return None
