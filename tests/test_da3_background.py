"""Camera-calibration checks that do not launch or import Isaac Sim.

Run with ``python -m unittest discover -s tests -p 'test_da3_background.py'``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/da3_background.py"
)
SPEC = importlib.util.spec_from_file_location("da3_background_under_test", MODULE_PATH)
DA3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DA3)


class Da3BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="da3-camera-test-")
        self.addCleanup(self.temp.cleanup)
        self.scene = Path(self.temp.name)
        Image.new("RGB", (80, 60), (20, 40, 60)).save(self.scene / "view.png")
        # Camera at (1,2,3), rotated +90 degrees around world Z. OpenCV camera
        # right points along world +Y; camera down points along world -X.
        self.camera = {
            "index": 12,
            "angle_deg": 90.0,
            "render": "view.png",
            "render_hw": [60, 80],
            "intrinsics": [[50.0, 0.0, 31.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]],
            "pose_user": {"pos": [1.0, 2.0, 3.0], "rot_wxyz": [2**-0.5, 0.0, 0.0, 2**-0.5]},
            "extrinsics_w2c_user": [
                [0.0, 1.0, 0.0, -2.0],
                [-1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, -3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        self.write_cameras([self.camera])

    def write_cameras(self, cameras):
        (self.scene / "orbit_renders_meta.json").write_text(
            json.dumps({"cameras": cameras}), encoding="utf-8"
        )

    def path_camera(self):
        camera = copy.deepcopy(self.camera)
        for key in ("index", "render", "render_hw"):
            camera.pop(key)
        camera["image_size"] = [80, 60]
        return camera

    def write_mapping(self, cameras):
        path = self.scene / "metadata" / "background_cameras.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": 2,
            "image_path_base": "..",
            "cameras": cameras,
        }), encoding="utf-8")
        return path

    def test_image_path_selects_exact_calibration_and_accepts_absolute_path(self):
        for folder in ("scene_a", "scene_b"):
            (self.scene / folder).mkdir()
            Image.new("RGB", (80, 60)).save(self.scene / folder / "view.png")
        first = self.path_camera()
        second = self.path_camera()
        second["pose_user"] = {"pos": [5, 0, 0], "rot_wxyz": [1, 0, 0, 0]}
        second["extrinsics_w2c_user"] = [
            [1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        ]
        mapping = self.write_mapping({"scene_b/view.png": second, "scene_a/view.png": first})
        loaded = DA3.load_da3_background_image("scene_b/view.png", mapping)
        self.assertEqual(loaded["pos"], (5.0, 0.0, 0.0))
        self.assertEqual(loaded["image_path"], str(self.scene / "scene_b/view.png"))
        self.assertEqual(loaded, DA3.load_da3_background_image(self.scene / "scene_b/view.png", mapping))
        self.assertEqual(DA3.load_da3_background_image("scene_a/view.png", mapping)["pos"], (1.0, 2.0, 3.0))

    def test_image_mapping_scales_intrinsics_and_preserves_world_projection(self):
        mapping = self.write_mapping({"view.png": self.path_camera()})
        loaded = DA3.load_da3_background_image("view.png", mapping, output_size=(320, 120))
        intrinsic = np.asarray(loaded["intrinsic_matrix"]).reshape(3, 3)
        np.testing.assert_allclose(intrinsic, [[200, 0, 124], [0, 120, 48], [0, 0, 1]])
        camera_point = np.asarray(loaded["extrinsics_w2c_user"]) @ [0, 3, 7, 1]
        pixels = intrinsic @ camera_point[:3]
        np.testing.assert_allclose(pixels[:2] / pixels[2], [174, 78])
        self.assertEqual(loaded["convention"], "ros")

    def test_image_mapping_never_falls_back_for_unlisted_or_missing_image(self):
        mapping = self.write_mapping({"view.png": self.path_camera()})
        with self.assertRaisesRegex(ValueError, "No DA3 calibration for image path"):
            DA3.load_da3_background_image("another_scene/view.png", mapping)
        (self.scene / "view.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "rendered background image not found"):
            DA3.load_da3_background_image("view.png", mapping)

    def test_image_mapping_rejects_duplicate_keys_and_path_aliases(self):
        camera = self.path_camera()
        mapping = self.write_mapping({"view.png": camera, "./view.png": camera})
        with self.assertRaisesRegex(ValueError, "Duplicate DA3 image path"):
            DA3.load_da3_background_image("view.png", mapping)
        mapping.write_text(
            '{"schema_version": 2, "image_path_base": "..", "cameras": {'
            + '"view.png": ' + json.dumps(camera) + ', "view.png": ' + json.dumps(camera) + '}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            DA3.load_da3_background_image("view.png", mapping)

    def test_image_mapping_rejects_mismatched_size_pose_and_nonrigid_transform(self):
        for field, message in (("size", "does not match image size"), ("pose", "inconsistent"), ("matrix", "rigid")):
            with self.subTest(field=field):
                camera = self.path_camera()
                if field == "size":
                    camera["image_size"] = [60, 80]
                elif field == "pose":
                    camera["pose_user"]["pos"][0] += 1
                else:
                    camera["extrinsics_w2c_user"][3][0] = 0.1
                mapping = self.write_mapping({"view.png": camera})
                with self.assertRaisesRegex(ValueError, message):
                    DA3.load_da3_background_image("view.png", mapping)

    def test_explicit_index_is_not_a_list_offset(self):
        other = copy.deepcopy(self.camera)
        other["index"] = 3
        self.write_cameras([other, self.camera])
        loaded = DA3.load_da3_background(self.scene, 12)
        self.assertEqual(loaded["view_index"], 12)
        self.assertEqual(loaded["image_path"], str((self.scene / "view.png").resolve()))
        self.write_cameras([self.camera, other])
        self.assertEqual(DA3.load_da3_background(self.scene, 12), loaded)

    def test_world_to_camera_inverse_and_ros_wxyz(self):
        loaded = DA3.load_da3_background(self.scene, 12, output_size=(80, 60))
        self.assertEqual(loaded["convention"], "ros")
        np.testing.assert_allclose(loaded["pos"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(loaded["rot"], [2**-0.5, 0.0, 0.0, 2**-0.5])
        extrinsic = np.array(loaded["extrinsics_w2c_user"])
        # A point four metres along camera +Z lands at the principal point.
        np.testing.assert_allclose(extrinsic @ [1, 2, 7, 1], [0, 0, 4, 1])
        # One unit camera-right and one unit camera-down corresponds to the
        # world point (0,3,7), not to world (2,3,7).
        np.testing.assert_allclose(extrinsic @ [0, 3, 7, 1], [1, 1, 4, 1])

    def test_intrinsics_scale_with_both_image_dimensions(self):
        loaded = DA3.load_da3_background(self.scene, 12, output_size=(320, 120))
        intrinsic = np.array(loaded["intrinsic_matrix"]).reshape(3, 3)
        np.testing.assert_allclose(intrinsic, [[200, 0, 124], [0, 120, 48], [0, 0, 1]])
        self.assertEqual(loaded["source_size"], (80, 60))
        self.assertEqual(loaded["output_size"], (320, 120))
        projection = intrinsic @ [1, 1, 4]
        np.testing.assert_allclose(projection[:2] / projection[2], [174, 78])

    def test_rejects_extrinsic_pose_or_convention_mismatch(self):
        for field in ("translation", "quaternion"):
            with self.subTest(field=field):
                camera = copy.deepcopy(self.camera)
                if field == "translation":
                    camera["extrinsics_w2c_user"][0][3] += 0.1
                else:
                    # Camera-to-world OpenGL rotation for this pose is not a
                    # valid substitute for the requested ROS/OpenCV rotation.
                    camera["pose_user"]["rot_wxyz"] = [0, 2**-0.5, 2**-0.5, 0]
                self.write_cameras([camera])
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    DA3.load_da3_background(self.scene, 12)

    def test_missing_render_does_not_silently_select_another_view(self):
        camera = copy.deepcopy(self.camera)
        camera["render"] = "missing.png"
        self.write_cameras([camera])
        with self.assertRaisesRegex(FileNotFoundError, "rendered background image not found"):
            DA3.load_da3_background(self.scene, 12)

    def test_rejects_ambiguous_index_and_wrong_image_dimensions(self):
        self.write_cameras([self.camera, self.camera])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            DA3.load_da3_background(self.scene, 12)
        camera = copy.deepcopy(self.camera)
        camera["render_hw"] = [80, 60]
        self.write_cameras([camera])
        with self.assertRaisesRegex(ValueError, "does not match image size"):
            DA3.load_da3_background(self.scene, 12)


if __name__ == "__main__":
    unittest.main()
