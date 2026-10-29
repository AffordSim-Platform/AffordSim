from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

from isaaclab.motion_collect.success.relations import evaluate_relation_poses
from isaaclab.motion_collect.primitives.place import primitive_compute_place_pose
from isaaclab.motion_collect.primitives.verify import primitive_verify
from isaaclab.motion_collect.tools.camera_framing import box_corners, camera_projection

UP = [1., 0., 0., 0.]
STACK = dict(xy_radius=.018,z_range=[.041,.062],upright_cos=.94,max_speed=.08)

@pytest.mark.parametrize('position,rotation,speed,expected', [
    ([.50,.10,.10],UP,0.,True),
    ([.50,.10,.05],UP,0.,False), # collapsed on the table, still XY aligned
    ([.55,.10,.10],UP,0.,False), # beside the tower
    ([.50,.10,.10],[.7071,.7071,0,0],0.,False), # fallen over
    ([.50,.10,.10],UP,.5,False), # transient mid-flight match
])
def test_stack_requires_supported_upright_settled_cubes(position,rotation,speed,expected):
    ok,_=evaluate_relation_poses(position,rotation,[.5,.1,.05],UP,STACK,speed,0.)
    assert ok == expected

def test_plate_accepts_offset_object_but_not_outside_or_airborne():
    rule=dict(xy_radius=.105,z_range=[-.015,.10])
    for pos,expected in [([.06,0,.025],True),([.15,0,.025],False),([.06,0,.30],False)]:
        ok,_=evaluate_relation_poses(pos,UP,[0,0,0],UP,rule)
        assert ok == expected

def test_center_compensation_and_overhead_approach():
    c=SimpleNamespace(num_envs=1,task_config=SimpleNamespace(place_offset=[0,0,0]),
        get_current_ee_pose=lambda:(np.array([[0,0,.3]]),np.array([[0,1,0,0]])),
        _get_object_pose_global=lambda *a:(np.array([.6,.1,.07]),np.array(UP)),
        _get_env_origins=lambda:np.zeros((1,3)),
        _get_object_entry_from_scene_desc=lambda *a:{'height':.06})
    ctx=dict(post_pick_success_mask=[True],scene_desc=[],all_hang_below=[.04],
        all_pick_quat=np.array([[0,1,0,0]]),held_object_offset_in_ee=[np.array([.05,-.02,.04])])
    assert primitive_compute_place_pose(c,'on',dict(target='plate',align_object_center=True,
        on_height_margin=.025,approach_height=.07),ctx)
    pose=ctx['place_pose'];np.testing.assert_allclose(pose['positions'][0,:2],[.55,.08])
    np.testing.assert_allclose(pose['pre_positions'][0]-pose['positions'][0],[0,0,.07])
    assert pose['execution_modes']==['overhead']
    assert primitive_compute_place_pose(c,'on',dict(target='plate',align_object_center=True,
        center_height_above_target=.025),ctx)
    # Downward EE rotates the held local +Z offset to world -Z.
    np.testing.assert_allclose(ctx['place_pose']['positions'][0,2],.07+.025+.04)

def test_final_verifier_rechecks_earlier_supports_and_never_revives_failure():
    c=SimpleNamespace(num_envs=2,control_gripper_batch=lambda *a,**k:None)
    ctx=dict(env_alive_mask=[True,False],task_success_mask=[True,False],plan_success_mask=[True,True])
    rules=[dict(source='red',target='blue',**STACK),dict(source='yellow',target='green',**STACK)]
    def evaluate(collector,i,rule):
        # The final cube looks right but an earlier support has collapsed.
        return rule['source']=='yellow', {'geometry_valid':rule['source']=='yellow'}
    with patch('isaaclab.motion_collect.primitives.verify.evaluate_scene_relation',side_effect=evaluate):
        assert not primitive_verify(c,None,dict(relations=rules,samples=2),ctx)
    assert ctx['env_alive_mask']==[False,False]

def test_return_to_pick_preserves_each_grasp_and_held_object_orientation():
    c=SimpleNamespace(num_envs=2,_containers_registry={},task_config=SimpleNamespace(place_offset=[0,0,0]),
        get_current_ee_pose=lambda:(np.zeros((2,3)),np.array([[0,0,1,0],[0,1,0,0]])))
    original_positions=np.array([[.43,-.29,.11],[.45,-.28,.12]])
    original_rotations=np.array([[0,1,0,0],[0,0,1,0]])
    ctx=dict(post_pick_success_mask=[True,True],all_pick_pos=original_positions.copy(),
        all_pick_quat=original_rotations.copy())
    params=dict(return_to_pick_pose=True,return_height_offset=.01,approach_height=.07)
    assert primitive_compute_place_pose(c,'pose',params,ctx)
    pose=ctx['place_pose']
    np.testing.assert_allclose(pose['positions'],original_positions+[0,0,.01])
    np.testing.assert_allclose(pose['pre_positions'],original_positions+[0,0,.08])
    # Symmetric gripper rotations are not interchangeable while holding a cup.
    np.testing.assert_allclose(pose['orientations'],original_rotations)
    np.testing.assert_allclose(ctx['all_pick_pos'],original_positions)
    np.testing.assert_allclose(ctx['all_pick_quat'],original_rotations)
    with pytest.raises(ValueError,match='cannot be combined'):
        primitive_compute_place_pose(c,'pose',dict(params,position=[.5,0,.1]),ctx)

def test_projection_uses_camera_calibration_without_modifying_it():
    position=[0.,0.,0.];quat=[1.,0.,0.,0.]
    intrinsic=[600.,0.,320.,0.,600.,240.,0.,0.,1.]
    pixels,depth=camera_projection([[0,0,1],[.1,.1,1]],position,quat,intrinsic)
    np.testing.assert_allclose(pixels,[[320,240],[380,300]])
    np.testing.assert_allclose(depth,[1,1])
    assert intrinsic == [600.,0.,320.,0.,600.,240.,0.,0.,1.]
