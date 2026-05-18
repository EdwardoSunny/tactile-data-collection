import numpy as np
from scipy.spatial.transform import Rotation as R

def Trans(x, y, z):
    T = np.eye(4)
    T[0:3, 3] = [x, y, z]
    return T

def RotX(angle):
    T = np.eye(4)
    T[0:3, 0:3] = R.from_euler('x', angle).as_matrix()
    return T

def RotZ(angle):
    T = np.eye(4)
    T[0:3, 0:3] = R.from_euler('z', angle).as_matrix()
    return T

def get_xarm7_forward_kinematics(angles_deg):
    """
    Offline forward kinematics for xArm7 (returns the transform of link7 relative to link_base).
    angles_deg: list of 7 joint angles in degrees.
    """
    angles = np.radians(angles_deg)

    # URDF Kinematics parameters (xyz and rpy for each joint)
    params = [
        # joint1
        ([0, 0, 0.267], [0, 0, 0]),
        # joint2
        ([0, 0, 0], [-1.5708, 0, 0]),
        # joint3
        ([0, -0.293, 0], [1.5708, 0, 0]),
        # joint4
        ([0.0525, 0, 0], [1.5708, 0, 0]),
        # joint5
        ([0.0775, -0.3425, 0], [1.5708, 0, 0]),
        # joint6
        ([0, 0, 0], [1.5708, 0, 0]),
        # joint7
        ([0.076, 0.097, 0], [-1.5708, 0, 0])
    ]

    T_out = np.eye(4)
    for i in range(7):
        xyz, rpy = params[i]

        # Translation
        T_trans = Trans(*xyz)

        # Fixed RPY Rotation
        T_rot = np.eye(4)
        T_rot[0:3, 0:3] = R.from_euler('xyz', rpy).as_matrix()

        # Joint Variable Rotation (Z-axis)
        T_joint = RotZ(angles[i])

        # Accumulate transform
        T_out = T_out @ T_trans @ T_rot @ T_joint

    return T_out

def get_finger_transforms(angles, gripper_pos):
    # Pure offline Python DH kinematics for xArm7
    T_flange = get_xarm7_forward_kinematics(angles)

    # The xArm gripper uses mechanical linkages. The finger movement is not purely linear.
    # We replicate the exact four-bar linkage geometry from the xArm URDF.
    # Joint limits are 0 to 0.85 rad. Gripper pos is 0 to 850.
    theta = abs(850 - gripper_pos) / 1000.0

    # Left finger chain from gripper base (flange)
    # T_L1 (drive_joint): origin xyz="0 0.035 0.059098", revolute around x
    T_L1 = Trans(0, 0.035, 0.059098) @ RotX(theta)
    # T_L2 (left_finger_joint): origin xyz="0 0.035465 0.042039", revolute around -x
    T_L2 = Trans(0, 0.035465, 0.042039) @ RotX(-theta)

    T_left_finger = T_flange @ T_L1 @ T_L2

    # Right finger chain from gripper base (flange)
    # T_R1 (right_outer_knuckle_joint): origin xyz="0 -0.035 0.059098", revolute around -x
    T_R1 = Trans(0, -0.035, 0.059098) @ RotX(-theta)
    # T_R2 (right_finger_joint): origin xyz="0 -0.035465 0.042039", revolute around x
    T_R2 = Trans(0, -0.035465, 0.042039) @ RotX(theta)

    T_right_finger = T_flange @ T_R1 @ T_R2

    return T_left_finger, T_right_finger

# Define constant matrices for sensors
left_finger_T_left_sensor = np.eye(4)
left_finger_T_left_sensor[0:3, 0:3] = R.from_euler("XZ", [-90, -90], degrees=True).as_matrix()
left_finger_T_left_sensor[0:3, 3] = [0, -0.019-0.0066, 0.037]

right_finger_T_right_sensor = np.eye(4)
right_finger_T_right_sensor[0:3, 0:3] = R.from_euler("XZ", [90, 90], degrees=True).as_matrix()
right_finger_T_right_sensor[0:3, 3] = [0, 0.019+0.0066, 0.037]
