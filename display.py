
from SpiderPlot import plot_spider_pose
import numpy as np

# Generate a compatible random angles array (1x24)
# Each joint angle: theta1 (yaw), theta2 (femur), theta3 (tibia)
# Reasonable ranges: theta1 [-pi/4, pi/4], theta2 [-pi/2, 0], theta3 [-pi, -pi/4]
def generate_random_angles():
    n_legs = 8
    angles = []
    for _ in range(n_legs):
        theta1 = np.random.uniform(-np.pi/4, np.pi/4)
        theta2 = np.random.uniform(-np.pi/2, 0)
        theta3 = np.random.uniform(-np.pi, -np.pi/4)
        angles.extend([theta1, theta2, theta3])
    return np.array(angles)

angles = generate_random_angles()

def display_300():
    for i in range(300):
        plot_spider_pose(generate_random_angles())

display_300()