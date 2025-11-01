import numpy as np
from plot_spider_pose import plot_spider_pose

if __name__ == '__main__':
    # Example usage translated from the MATLAB file
    a = np.deg2rad(0.0)
    b = np.deg2rad(-45.0)
    c = np.deg2rad(-40.0)

    L1 = np.array([a, b, c])
    L2 = np.array([a, b, c])
    L3 = np.array([a, b, c])
    L4 = np.array([a, b, c])
    R4 = np.array([-a, b, c])
    R3 = np.array([-a, b, c])
    R2 = np.array([-a, b, c])
    R1 = np.array([-a, b, c])

    angles = np.concatenate([L1, L2, L3, L4, R4, R3, R2, R1])

    plot_spider_pose(angles)
