import numpy as np
from plot_spider_pose import plot_spider_pose

if __name__ == '__main__':
    # Example usage translated from the MATLAB file
    T1_rad = np.deg2rad(0.0)
    T2_rad = np.deg2rad(-45.0)
    T3_rad = np.deg2rad(-40.0)

    L1 = np.array([T1_rad, T2_rad, T3_rad])
    L2 = np.array([T1_rad, T2_rad, T3_rad])
    L3 = np.array([T1_rad, T2_rad, T3_rad])
    L4 = np.array([T1_rad, T2_rad, T3_rad])
    R1 = np.array([-T1_rad, T2_rad, T3_rad])
    R2 = np.array([-T1_rad, T2_rad, T3_rad])
    R3 = np.array([-T1_rad, T2_rad, T3_rad])
    R4 = np.array([-T1_rad, T2_rad, T3_rad])

    angles = np.concatenate([L1, L2, L3, L4, R1, R2, R3, R4])

    plot_spider_pose(angles)
