"""Physical end-state checks, independent of commanded end-effector poses."""
import numpy as np

from ..tools.pose_utils import _quat_wxyz_to_matrix


def evaluate_relation_poses(source_pos, source_quat, target_pos, target_quat, rule,
                            source_speed=0.0, target_speed=0.0):
    """Check actual centers, vertical separation, orientation and settling.

    Heights/radii are explicit task geometry, in metres. No grasp-offset
    compensation is applied: success describes objects, not the gripper.
    """
    if any(v is None for v in (source_pos, source_quat, target_pos, target_quat)):
        return False, {"geometry_valid": False}
    delta = np.asarray(source_pos) - np.asarray(target_pos)
    xy = float(np.linalg.norm(delta[:2]))
    dz = float(delta[2])
    source_up = float(_quat_wxyz_to_matrix(np.asarray(source_quat))[2, 2])
    target_up = float(_quat_wxyz_to_matrix(np.asarray(target_quat))[2, 2])
    radius = float(rule["xy_radius"])
    z_min, z_max = map(float, rule["z_range"])
    upright = float(rule.get("upright_cos", -1.0))
    speed_limit = float(rule.get("max_speed", 0.08))
    values = [xy, dz, source_up, target_up, source_speed, target_speed]
    passed = bool(np.isfinite(values).all() and xy <= radius and z_min <= dz <= z_max
                  and source_up >= upright and target_up >= upright
                  and source_speed <= speed_limit and target_speed <= speed_limit)
    return passed, {"geometry_valid": passed, "center_xy_dist": xy, "center_z_delta": dz,
                    "source_up": source_up, "target_up": target_up,
                    "source_speed": float(source_speed), "target_speed": float(target_speed),
                    "xy_radius": radius, "z_range": [z_min, z_max]}


def evaluate_scene_relation(collector, env_idx, rule):
    source, target = rule["source"], rule["target"]
    a, aq = collector._get_object_pose_global(source, env_idx)
    b, bq = collector._get_object_pose_global(target, env_idx)

    def speed(name):
        key = collector._find_object_in_scene(name)
        if key is None:
            return float("inf")
        data = collector.scene[key].data
        linear = data.root_lin_vel_w[env_idx].detach().cpu().numpy()
        return float(np.linalg.norm(linear))

    return evaluate_relation_poses(a, aq, b, bq, rule, speed(source), speed(target))
