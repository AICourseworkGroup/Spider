import numpy as np


def axis_angle_rotation_matrix(axis, angle):
    """Axis-angle rotation matrix.

    Input:
      axis: 3-element array-like
      angle: scalar radians
    Returns:
      3x3 numpy rotation matrix
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.identity(3)
    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [x * x * C + c,   x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c,   y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c]
    ])
    return R


def rotate_vector(v, axis, angle):
    """
    Rotate a 3D vector about axis by angle (radians).

    v may be 1D array-like of length 3.
    Returns the rotated vector as a 1D numpy array.
    """
    v = np.array(v, dtype=float)
    R = axis_angle_rotation_matrix(axis, angle)
    return (R @ v.T).T


def forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths):
    """
    Compute forward kinematics for a single leg.

    Translation of the MATLAB function:
      [j1, j2, j3, j4] = forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths)

    Inputs:
      base_pos: length-3 array-like [x,y,z] position of leg base on body
      base_angle: scalar angle around body ellipse where leg base is located (radians)
      joint_angles: length-3 array-like [theta1, theta2, theta3] in radians
      segment_lengths: length-3 array-like [coxa, femur, tibia]

    Outputs:
      j1, j2, j3, j4: numpy arrays (3,) for joint positions in 3D
    """
    # Joint angles
    theta1 = float(joint_angles[0])
    theta2 = float(joint_angles[1])
    theta3 = float(joint_angles[2])

    # Segment lengths
    L1 = float(segment_lengths[0])
    L2 = float(segment_lengths[1])
    L3 = float(segment_lengths[2])

    # Joint 1: leg base on body
    j1 = np.array(base_pos, dtype=float)

    # --- Compute Coxa direction with elevation ---
    coxa_elevation = np.deg2rad(30.0)  # fixed 30 degree upward pitch for coxa

    # Horizontal direction of coxa in XY plane based on base_angle + theta1
    coxa_horiz_dir = np.array([np.cos(base_angle + theta1), np.sin(base_angle + theta1), 0.0])

    # Rotation axis for pitch up: perpendicular to coxa horizontal direction in XY plane
    rot_axis = np.cross(coxa_horiz_dir, np.array([0.0, 0.0, 1.0]))

    # Rotation matrix around rot_axis by coxa_elevation
    R = axis_angle_rotation_matrix(rot_axis, coxa_elevation)

    # Rotate horizontal coxa direction upward
    coxa_dir = (R @ coxa_horiz_dir.T).T

    # Joint 2 position: end of coxa segment
    j2 = j1 + L1 * coxa_dir

    # --- Femur rotation ---
    # Perpendicular to coxa_dir and vertical axis
    femur_rot_axis = np.cross(coxa_dir, np.array([0.0, 0.0, 1.0]))
    femur_norm = np.linalg.norm(femur_rot_axis)
    if femur_norm == 0:
        femur_rot_axis = np.array([0.0, 1.0, 0.0])
    else:
        femur_rot_axis = femur_rot_axis / femur_norm

    # Femur direction vector starts aligned with coxa_dir
    femur_dir = rotate_vector(coxa_dir, femur_rot_axis, theta2)

    # Joint 3 position: end of femur segment
    j3 = j2 + L2 * femur_dir

    # --- Tibia rotation ---
    tibia_rot_axis = np.cross(femur_dir, np.array([0.0, 0.0, 1.0]))
    tibia_norm = np.linalg.norm(tibia_rot_axis)
    if tibia_norm == 0:
        tibia_rot_axis = np.array([0.0, 1.0, 0.0])
    else:
        tibia_rot_axis = tibia_rot_axis / tibia_norm

    tibia_dir = rotate_vector(femur_dir, tibia_rot_axis, theta3)

    # Joint 4 position: end of tibia segment (foot)
    j4 = j3 + L3 * tibia_dir

    return j1, j2, j3, j4


if __name__ == '__main__':
    base_pos = [1.0, 0.0, 0.0]
    base_angle = 0.0
    joint_angles = [0.0, -np.pi/4, -np.pi/2]
    segment_lengths = [1.2, 0.7, 1.0]
    j1, j2, j3, j4 = forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths)
    print('j1, j2, j3, j4:')
    print(j1, j2, j3, j4)
