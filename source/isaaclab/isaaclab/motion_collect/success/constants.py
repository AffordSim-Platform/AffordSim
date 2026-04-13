# =============================================================================
# Success evaluation threshold constants for each primitive
# These constants are only used when success_eval is enabled in the YAML config
# =============================================================================

# --- place primitive ---
# Place success distance threshold (meters):
# - Non-affordance mode: max XY distance between object final position and target place position
# - Affordance mode: max distance between placed object's place affordance and target place affordance
PLACE_SUCCESS_XY_THRESHOLD = 0.03
# Requires pick z >= place z + Z_MARGIN to prevent penetration
PLACE_SUCCESS_Z_MARGIN = 0.0

# --- common (place / pour / push / pull / hang) ---
# Boundary tolerance (meters) when checking if object falls within position_range.x / position_range.y from YAML
POSITION_RANGE_TOLERANCE = 0.01

# --- pour primitive ---
# Threshold for container local z-axis component in world z direction; at or below this value counts as successful pour
# 0.0 -> container axis must be horizontal or lower; negative -> requires more thorough pouring
POUR_SUCCESS_TILT_THRESHOLD = 0.0

# ---- pour process evaluation (per-step sampling during tilt interpolation) ----
# Max overshoot distance (meters) of pour tip world XY from container pour center
# Actual tolerance radius = tilt_horizontal_offset + POUR_PROCESS_XY_TOLERANCE
POUR_PROCESS_XY_TOLERANCE = 0.1
# Allowed reverse rise per step for z_axis_world_z (tolerates physics jitter / IK drift)
# z_axis_world_z meaning: component of cup local +z in world z direction
# 1.0 -> fully upright; 0.0 -> horizontal; <0 -> inverted; ideal process decreases monotonically
# Note: total tilt angle is ensured by endpoint check POUR_SUCCESS_TILT_THRESHOLD; this only enforces process continuity
POUR_PROCESS_TILT_MONOTONIC_SLACK = 0.01

# --- push primitive ---
# Max allowed XY distance (meters) between object final position and target push position
PUSH_SUCCESS_XY_THRESHOLD = 0.05
# Min component of object local z-axis in world z direction; below this value the object is considered toppled
# 1.0 -> fully upright; 0.8 -> allows ~37 degree tilt; 0.0 -> allows lying flat
PUSH_SUCCESS_TILT_THRESHOLD = 0.8

# --- pull primitive ---
# Min displacement (meters) of the corresponding prismatic joint on the articulation during pull
PULL_SUCCESS_MIN_DISPLACEMENT = 0.08
# Joint name used for pull evaluation (prismatic joint of cabinet1 drawer)
PULL_SUCCESS_JOINT_NAME = "joint_0"

# --- open primitive ---
# Min opening angle (radians) of revolute joint on articulation for open, default ~57 degrees
OPEN_SUCCESS_MIN_ANGLE_RAD = 1.0
# Joint name used for open evaluation (revolute joint of microwave door)
OPEN_SUCCESS_JOINT_NAME = "microjoint"

# --- close primitive ---
# Max residual angle (radians) of revolute joint on articulation for close, default ~14 degrees
CLOSE_SUCCESS_MAX_ANGLE_RAD = 0.25
# Joint name used for close evaluation (revolute joint of microwave door)
CLOSE_SUCCESS_JOINT_NAME = "microjoint"

# --- hang primitive ---
# Max XY distance (meters) between object final position and midpoint of mount point to rack center line
HANG_SUCCESS_XY_THRESHOLD = 0.05
# Min Z height (meters) in env local frame required for the object, prevents falling to ground
HANG_SUCCESS_MIN_Z = 0.08

# --- hold primitive ---
# Min Z height (meters) in env local frame required for the object, used to determine if lifted
HOLD_SUCCESS_MIN_Z = 0.10

# --- pick primitive ---
# Min lift above initial z (meters) required for a successful grasp
PICK_SUCCESS_LIFT_THRESHOLD = 0.02
