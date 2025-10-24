from SpiderPlot import plot_spider_pose
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

''' 

DUE TO UNFAMILIARITY WITH MATPLOTLIB ANIMATION, THIS FILE WAS 90%+ CREATED USING CLAUDE SONNET 3.5 IN CO-PILOT IN VSCODE

'''

def generate_random_angles():
    n_legs = 8
    angles = []
    for leg in range(n_legs):
        theta1 = np.random.uniform(-np.pi/4, np.pi/4)
        theta2 = np.random.uniform(-np.pi/2, 0)
        theta3 = np.random.uniform(-np.pi, -np.pi/4)
        angles.extend([theta1, theta2, theta3])
    return np.array(angles)



# Create figure and 3D axis once
fig = plt.figure(figsize=(10, 8), facecolor='w')
ax = fig.add_subplot(111, projection='3d')

# Set up the 3D view
ax.set_facecolor('w')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1, 1, 1])
ax.grid(True)
ax.view_init(elev=45, azim=45)
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-2, 2])

def update(frame):
    ax.clear()  # Clear only the axis, not the entire figure
    
    # Set up the 3D view again (needed after clear)
    ax.set_facecolor('w')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    ax.view_init(elev=45, azim=45)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-2, 2])
    
    # Plot the spider using the existing axis
    angles = generate_random_angles()
    plot_spider_pose(angles, ax=ax)
    ax.set_title(f'Frame {frame + 1}')
    
# Create animation
ani = FuncAnimation(fig, update, frames=300, interval=100, repeat=False)

# Show the plot in a single window
plt.show()