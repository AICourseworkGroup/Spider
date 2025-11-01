import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

''' 

DUE TO UNFAMILIARITY WITH MATPLOTLIB ANIMATION, THIS FILE WAS 90%+ TRANSLATED AND CREATED FROM MATLAB CODE USING CLAUDE SONNET 3.5 IN CO-PILOT IN VSCODE

'''


def axis_angle_rotation_matrix(axis, angle):
    """
    Helper function to compute a 3x3 rotation matrix from an axis and an angle.
    """
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.identity(3) # Return identity matrix if axis is zero
    axis = axis / norm
    
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    R = np.array([
        [ x*x*C + c,   x*y*C - z*s, x*z*C + y*s ],
        [ y*x*C + z*s, y*y*C + c,   y*z*C - x*s ],
        [ z*x*C - y*s, z*y*C + x*s, z*z*C + c ]
    ])
    return R

def rotate_vector(v, axis, angle):
    """
    Helper function to rotate a 3D vector around an axis by an angle.
    """
    R = axis_angle_rotation_matrix(axis, angle)
    v_rot = (R @ v.T).T 
    return v_rot

def forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths):
    """
    This is the CORRECT 3D kinematics function translated from your MATLAB file.
    It USES theta2 and theta3 to create bends.
    """
    
    # Unpack joint angles
    theta1, theta2, theta3 = joint_angles
    
    # Unpack segment lengths
    L1, L2, L3 = segment_lengths
    
    # Joint 1: leg base on body
    j1 = np.array(base_pos)
    
    # --- Compute Coxa direction with elevation ---
    coxa_elevation = np.deg2rad(30)  # fixed 30 degree upward pitch for coxa
    
    # Horizontal direction of coxa in XY plane based on base_angle + theta1
    coxa_horiz_dir = np.array([np.cos(base_angle + theta1), np.sin(base_angle + theta1), 0])
    
    # Rotation axis for pitch up: perpendicular to coxa horizontal direction
    rot_axis = np.cross(coxa_horiz_dir, np.array([0, 0, 1]))
    
    # Rotate horizontal coxa direction upward
    coxa_dir = rotate_vector(coxa_horiz_dir, rot_axis, coxa_elevation)
    
    # Joint 2 position: end of coxa segment
    j2 = j1 + L1 * coxa_dir
    
    # --- Femur rotation (THIS USES THETA_2) ---
    femur_rot_axis = np.cross(coxa_dir, np.array([0, 0, 1]))
    
    if np.linalg.norm(femur_rot_axis) == 0:
         # if coxa is vertical, pick an arbitrary axis, e.g., Y-axis
        femur_rot_axis = np.array([0, 1, 0])
    else:
        femur_rot_axis = femur_rot_axis / np.linalg.norm(femur_rot_axis)
    
    femur_dir = rotate_vector(coxa_dir, femur_rot_axis, theta2)
    
    # Joint 3 position: end of femur segment
    j3 = j2 + L2 * femur_dir
    
    # --- Tibia rotation (THIS USES THETA_3) ---
    tibia_rot_axis = np.cross(femur_dir, np.array([0, 0, 1]))
    
    if np.linalg.norm(tibia_rot_axis) == 0:
        tibia_rot_axis = np.array([0, 1, 0])
    else:
        tibia_rot_axis = tibia_rot_axis / np.linalg.norm(tibia_rot_axis)
    
    tibia_dir = rotate_vector(femur_dir, tibia_rot_axis, theta3)
    
    # Joint 4 position: end of tibia segment (foot)
    j4 = j3 + L3 * tibia_dir
    
    return j1, j2, j3, j4


def plot_spider_pose(angles, ax=None):
    """
    Plot a static 3D spider pose based on joint angles.

    Input:
      angles: 1x24 NumPy array of joint angles in radians
              [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
      ax: Optional matplotlib 3D axis to plot on. If None, creates a new figure.
    """
    
    # --- Parameters ---
    n_legs = 8
    segment_lengths = np.array([1.2, 0.7, 1.0])  # [Coxa, Femur, Tibia]
    a = 1.5; b = 1.0  # Ellipse axes for body (oval shape)
    
    # Base angles (MATLAB's deg2rad is numpy's deg2rad)
    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])
    base_angles = np.concatenate((left_leg_angles, right_leg_angles))
    
    leg_labels = ['L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1']
    
    # --- Validate input ---
    if len(angles) != n_legs * 3:
        raise ValueError('Input angles must be a 1x24 vector (3 angles per leg for 8 legs).')
    
    # --- Setup figure if not provided ---
    if ax is None:
        fig = plt.figure(figsize=(10, 8), facecolor='w')
        ax = fig.add_subplot(111, projection='3d')
    
    ax.set_facecolor('w') # Set figure background to white
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for 3D plot
    ax.grid(True)
    
    # Set view angle (MATLAB's view(45, 45))
    ax.view_init(elev=45, azim=45)
    
    # Set limits (MATLAB's xlim, ylim, zlim)
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-2, 2])
    
    # --- Plot body (oval shape) ---
    t = np.linspace(0, 2*np.pi, 100)
    body_x = a * np.cos(t)
    body_y = b * np.sin(t)
    body_z = np.zeros_like(t)
    ax.plot(body_x, body_y, body_z, color='k', linestyle='-', linewidth=3)
    
    # --- Head marker (front of spider at +X) ---
    ax.plot([a + 0.2], [0], [0], marker='^', color='r', markersize=10, markerfacecolor='r')
    
    print('--- Spider Pose ---')
    
    # --- Loop over legs ---
    for i in range(n_legs):
        # Indices for this leg's angles (Python uses 0-based indexing)
        # MATLAB idx = (i-1)*3 + 1. Python idx = i * 3
        idx = i * 3
        
        # Get angles from the NumPy array
        theta1 = angles[idx]    # Coxa/Hip
        theta2 = angles[idx+1]  # Femur/Thigh
        theta3 = angles[idx+2]  # Tibia/Shin
        
        # Print joint angles
        print(f"Leg {leg_labels[i]}: theta1 = {theta1:.3f} rad, theta2 = {theta2:.3f} rad, theta3 = {theta3:.3f} rad")
        
        # --- Compute leg base position on body ellipse ---
        angle = base_angles[i]
        x_base = a * np.cos(angle)
        y_base = b * np.sin(angle)
        base_pos = np.array([x_base, y_base, 0])
        
        # --- Compute FK for this leg ---
        # Note: forward_leg_kinematics2 must be defined and available
        j1, j2, j3, j4 = forward_leg_kinematics2(base_pos, angle, 
                                                 [theta1, theta2, theta3], segment_lengths)
        
        # --- Plot leg segments (MATLAB's plot3) ---
        
        # Coxa (Black)
        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], 
                color='k', linestyle='-', linewidth=2)
        
        # Femur (Blue)
        ax.plot([j2[0], j3[0]], [j2[1], j3[1]], [j2[2], j3[2]], 
                color='b', linestyle='-', linewidth=2)
        
        # Tibia (Red)
        ax.plot([j3[0], j4[0]], [j3[1], j4[1]], [j3[2], j4[2]], 
                color='r', linestyle='-', linewidth=2)
        
        # Foot (Red Circle Marker)
        ax.plot([j4[0]], [j4[1]], [j4[2]], 
                marker='o', color='r', markersize=5, markerfacecolor='r')
        
        # --- Label leg (MATLAB's text) ---
        offset = 0.2
        label_pos = base_pos + offset * np.array([np.cos(angle), np.sin(angle), 0])
        ax.text(label_pos[0], label_pos[1], label_pos[2] + 0.05, leg_labels[i], 
                fontsize=12, fontweight='bold')

    # --- Finalize plot (MATLAB's hold off) ---
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # Define a symmetric stance (same example values as before)
    T1_rad = np.deg2rad(0)    # Small Yaw
    T2_rad = np.deg2rad(-45)   # Femur Bend
    T3_rad = np.deg2rad(-90)   # Tibia Bend

    L1 = np.array([T1_rad, T2_rad , T3_rad])
    L2 = np.array([T1_rad, T2_rad , T3_rad])
    L3 = np.array([T1_rad, T2_rad , T3_rad])
    L4 = np.array([T1_rad, T2_rad , T3_rad])
    R1 = np.array([-T1_rad, T2_rad, T3_rad])
    R2 = np.array([-T1_rad, T2_rad, T3_rad])
    R3 = np.array([-T1_rad, T2_rad, T3_rad])
    R4 = np.array([-T1_rad, T2_rad, T3_rad])

    angles = np.concatenate([
        L1, L2, L3, L4,   # L1 to L4
        R1, R2, R3, R4    # R1 to R4
    ])

    plot_spider_pose(angles)
    

    