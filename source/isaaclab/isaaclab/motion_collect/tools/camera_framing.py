"""Per-frame object visibility checks using the unmodified runtime camera."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation


def box_corners(bounds):
    return np.asarray(list(product(*zip(bounds[0], bounds[1]))), dtype=float)


def camera_projection(points, position, orientation, intrinsic_matrix):
    q = np.asarray(orientation)
    rot = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    camera = (np.asarray(points) - np.asarray(position)) @ rot
    pixels = camera @ np.asarray(intrinsic_matrix).reshape(3, 3).T
    return pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-9), camera[:, 2]



class FrameVisibilityMonitor:
    """Test object bounds at every recorded control step; does not test occlusion."""
    def __init__(self, collector):
        self.collector = collector
        self.config = collector.task_config.visibility_check
        self.reset()

    def reset(self):
        self.frames = 0
        self.results = [{"all_objects_in_frame": True, "out_of_frame": {}}
                        for _ in range(self.collector.num_envs)]

    def sample(self):
        if not self.config:
            return
        self.frames += 1
        camera = self.collector.scene['external_cam_side'].data
        height, width = camera.image_shape
        camera_positions = camera.pos_w.detach().cpu().numpy()
        camera_rotations = camera.quat_w_ros.detach().cpu().numpy()
        intrinsics = camera.intrinsic_matrices.detach().cpu().numpy()
        for i in range(self.collector.num_envs):
            for name, bounds in self.config["object_bounds"].items():
                pos, quat = self.collector._get_object_pose_global(name, i)
                if pos is None:
                    outside = True
                else:
                    rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
                    points = box_corners(bounds) @ rot.T + np.asarray(pos)
                    pixels, depth = camera_projection(points, camera_positions[i], camera_rotations[i], intrinsics[i])
                    outside = bool(depth.min() <= 0 or pixels[:, 0].min() < 0 or pixels[:, 1].min() < 0
                                   or pixels[:, 0].max() >= width or pixels[:, 1].max() >= height)
                if outside:
                    self.results[i]["all_objects_in_frame"] = False
                    entry = self.results[i]["out_of_frame"].setdefault(name, {"first_step": self.frames, "steps": 0})
                    entry["steps"] += 1

    def summary(self, env_idx):
        return {"checked_control_steps": self.frames, "criterion": "object bounding boxes inside side-camera frustum",
                "occlusion_tested": False, **self.results[env_idx]}
