import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from forward_leg_kinematics import forward_leg_kinematics2

"""
Direct translation of plotSpiderPose.matlab into Python.
Preserves plotting logic and prints exactly like the MATLAB version.
"""


def plot_spider_pose(angles):
    """Plot a static 3D spider pose based on joint angles.

    Input:
      angles: 1x24 numpy array or list of joint angles in radians
              [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
    """
    # Parameters
    n_legs = 8
    segment_lengths = [1.2, 0.7, 1.0]  # [Coxa, Femur, Tibia]
    a = 1.5
    b = 1.0

    # Base angles (L1 front-left to L4 rear-left, R4 rear-right to R1 front-right)
    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])
    base_angles = np.concatenate((left_leg_angles, right_leg_angles))

    leg_labels = ['L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1']

    # Validate input
    if len(angles) != n_legs * 3:
        raise ValueError('Input angles must be a 1x24 vector (3 angles per leg for 8 legs).')

    # Setup figure
    fig = plt.figure(1)
    fig.clf()
    fig.set_facecolor('w')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=45, azim=45)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-2, 2])
    ax.grid(True)

    # Plot body (oval shape)
    t = np.linspace(0, 2*np.pi, 100)
    body_x = a * np.cos(t)
    body_y = b * np.sin(t)
    body_z = np.zeros_like(t)
    ax.plot(body_x, body_y, body_z, color='k', linestyle='-', linewidth=3)

    # Head marker (front of spider at +X)
    ax.plot([a + 0.2], [0], [0], marker='^', color='r', markersize=10, markerfacecolor='r')

    print('--- Spider Pose ---')

    # Loop over legs
    for i in range(n_legs):
        idx = i * 3
        theta1 = angles[idx]
        theta2 = angles[idx + 1]
        theta3 = angles[idx + 2]

        print(f"Leg {leg_labels[i]}: theta1 = {theta1:.3f} rad, theta2 = {theta2:.3f} rad, theta3 = {theta3:.3f} rad")

        # Compute leg base position on body ellipse
        angle = base_angles[i]
        x_base = a * np.cos(angle)
        y_base = b * np.sin(angle)
        base_pos = np.array([x_base, y_base, 0.0])

        # Compute FK for this leg
        j1, j2, j3, j4 = forward_leg_kinematics2(base_pos, angle, [theta1, theta2, theta3], segment_lengths)

        # Plot leg segments
        # Coxa (Black)
        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], color='k', linestyle='-', linewidth=2)

        # Femur (Blue)
        ax.plot([j2[0], j3[0]], [j2[1], j3[1]], [j2[2], j3[2]], color='b', linestyle='-', linewidth=2)

        # Tibia (Red)
        ax.plot([j3[0], j4[0]], [j3[1], j4[1]], [j3[2], j4[2]], color='r', linestyle='-', linewidth=2)

        # Foot (Red Circle Marker)
        ax.plot([j4[0]], [j4[1]], [j4[2]], marker='o', color='r', markersize=5, markerfacecolor='r')

        # Label leg
        offset = 0.2
        label_pos = base_pos + offset * np.array([np.cos(angle), np.sin(angle), 0.0])
        ax.text(label_pos[0], label_pos[1], label_pos[2] + 0.05, leg_labels[i], fontsize=12, fontweight='bold')

    plt.show()

