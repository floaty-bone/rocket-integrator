import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Function to create rotation matrix from Euler angles (ZYX convention)
def rotation_matrix(roll, pitch, yaw):
    """Create rotation matrix from roll, pitch, yaw angles"""
    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

# Define the cuboid vertices (centered at origin)
def create_cuboid(length, width, height):
    """Create vertices of a cuboid"""
    l, w, h = length/2, width/2, height/2
    vertices = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],  # bottom
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]       # top
    ])
    return vertices

# Define the faces of the cuboid
faces = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [2, 3, 7, 6],  # back
    [0, 3, 7, 4],  # left
    [1, 2, 6, 5]   # right
]

# Create the cuboid
cuboid_vertices = create_cuboid(length=4, width=2, height=1) #returns 8x3 array of vertices

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Animation parameters
n_frames = 200
t = np.linspace(0, 4*np.pi, n_frames)

# Example trajectory: rotating and moving
positions = np.column_stack([
    2 * np.cos(t/2),      # x position
    2 * np.sin(t/2),      # y position
    0.5 * np.sin(t) + 3   # z position
])

angles = np.column_stack([
    t * 0.5,              # roll
    t * 0.3,              # pitch
    t * 0.2               # yaw
])

def update(frame):
    ax.clear()
    
    # Get current position and orientation
    pos = positions[frame]
    roll, pitch, yaw = angles[frame]
    
    # Rotate and translate vertices
    R = rotation_matrix(roll, pitch, yaw)
    rotated_vertices = (R @ cuboid_vertices.T).T + pos
    
    # Create the faces for plotting
    cuboid_faces = [[rotated_vertices[j] for j in face] for face in faces]
    
    # Plot the cuboid
    poly = Poly3DCollection(cuboid_faces, alpha=0.7, edgecolor='black', linewidths=1)
    poly.set_facecolor(['cyan', 'cyan', 'blue', 'blue', 'red', 'red'])
    ax.add_collection3d(poly)
    
    # Draw coordinate axes at the cuboid center
    axis_length = 1.5
    axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    rotated_axes = (R @ axes.T).T
    
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    for i, (axis, color, label) in enumerate(zip(rotated_axes, colors, labels)):
        ax.quiver(pos[0], pos[1], pos[2], 
                 axis[0], axis[1], axis[2], 
                 color=color, arrow_length_ratio=0.3, linewidth=2, label=label)
    
    # Draw trajectory
    ax.plot(positions[:frame+1, 0], positions[:frame+1, 1], 
            positions[:frame+1, 2], 'gray', alpha=0.3, linewidth=1)
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 6])
    ax.set_title(f'3D Rigid Body Animation (Frame {frame}/{n_frames})')
    ax.legend(loc='upper right')
    
    return ax,

# Create animation
anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

plt.tight_layout()
plt.show()