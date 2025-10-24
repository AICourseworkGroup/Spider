import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

# NOTE: You MUST implement the forward_leg_kinematics2 function
# The function must accept: (base_pos, base_angle, leg_angles, segment_lengths)
# The function must return: j1, j2, j3, j4 (all 3-element NumPy arrays)
def forward_leg_kinematics2(base_pos, base_angle, leg_angles, segment_lengths):
    """
    PLACEHOLDER: You must replace this with the correct 3D forward kinematics 
    for the spider leg model. This version simply extends the leg straight 
    out from the body base for demonstration purposes.
    """
    
    # Unpack values
    L_coxa, L_femur, L_tibia = segment_lengths
    theta1, theta2, theta3 = leg_angles
    
    # --- Placeholder Logic (REPLACE THIS) ---
    j1 = np.array(base_pos)
    
    # Calculate total length for a stretched-out leg for a simple placeholder
    L_total = L_coxa + L_femur + L_tibia
    
    # Direction vector based on body base angle and coxa angle (theta1)
    # Assumes a 2D rotation for the base joints for simplification
    angle_total = base_angle + theta1
    
    # Very simple extension from the base for a placeholder
    j2_dir = np.array([L_coxa * np.cos(angle_total), L_coxa * np.sin(angle_total), 0])
    j2 = j1 + j2_dir 

    j3_dir = np.array([L_femur * np.cos(angle_total), L_femur * np.sin(angle_total), 0])
    j3 = j2 + j3_dir 

    j4_dir = np.array([L_tibia * np.cos(angle_total), L_tibia * np.sin(angle_total), 0])
    j4 = j3 + j4_dir 

    return j1, j2, j3, j4
# --------------------------------------------------------------------------

def plot_spider_pose(angles):
    """
    Plot a static 3D spider pose based on joint angles.

    Input:
      angles: 1x24 NumPy array of joint angles in radians
              [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
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
    
    # --- Setup figure (MATLAB's figure, clf, axis equal, grid, hold on, view) ---
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
    T1_rad = np.deg2rad(5)    # Small Yaw
    T2_rad = np.deg2rad(45)   # Femur Bend
    T3_rad = np.deg2rad(10)   # Tibia Bend

    left_set = np.array([T1_rad, T2_rad, T3_rad])
    right_set = np.array([-T1_rad, T2_rad, T3_rad])

    angles = np.concatenate([
        left_set, left_set, left_set, left_set,   # L1 to L4
        right_set, right_set, right_set, right_set # R4 to R1
    ])

    plot_spider_pose(angles)
    

    