# starship\main.py
from integrator import *
# ============================================================================
# SIMULATION SETUP - TWEAK THESE VALUES!
# ============================================================================
def get_forceMoment():
    """Define the forces and moments applied to the rigid body
    Returns: [Fx, Fy, Fz, Mx, My, Mz]
    
    Try these examples:
    - Pure rotation around X: [0, 0, 0, 30, 0, 0]
    - Pure rotation around Y: [0, 0, 0, 0, 30, 0]
    - Pure rotation around Z: [0, 0, 0, 0, 0, 30]
    - Force in X direction: [1000, 0, 0, 0, 0, 0]
    - Complex motion: [500, 0, 0, 10, 5, 3]
    """
    return np.array([0, 0, 0, 650, 700, 550], dtype="float64")

# Simulation parameters
sim_time = 100.0  # seconds
step_size = 0.001  # integration step
sample_rate = 50  # save every Nth step for animation (to reduce memory)

# Initial state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
initial_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0.01, 0.01], dtype="float64")

# Cuboid dimensions (adjust to match your rocket/object)
cuboid_length = 4.0  # X direction
cuboid_width = 2.0   # Y direction
cuboid_height = 1.0  # Z direction

# Inertia matrix (adjust for your object)
inertia_matrix = np.array([
    [300, 0, 0],
    [0, 100, 0],
    [0, 0, 30]
], dtype="float64")

# ============================================================================
# RUN SIMULATION
# ============================================================================
def run_simulation(time, step, sample_rate):
    """Run the RK4 integrator and collect trajectory data"""
    integrator = Rk4_integrator(
        get_forceMoment=get_forceMoment,
        inertia_matrix=inertia_matrix,
        l=4, 
        a=2,
        mass=85000,
        g=0,
        step_size=step
    )
    
    n_steps = int(time / step)
    n_saved = n_steps // sample_rate + 1
    
    states = np.zeros((n_saved, 13), dtype="float64")
    state = initial_state.copy()
    states[0] = state
    
    saved_idx = 1
    for i in range(1, n_steps):
        state = integrator.forward(state)
        if i % sample_rate == 0 and saved_idx < n_saved:
            states[saved_idx] = state
            saved_idx += 1
    
    return states

print("Running simulation...")
trajectory = run_simulation(sim_time, step_size, sample_rate)
quat=trajectory[-2, 3:7]
print(np.linalg.norm(quat))
print(trajectory[-2])
print(quat)
print(f"Simulation complete! Generated {len(trajectory)} frames")

# ============================================================================
# ANIMATION
# ============================================================================
def create_cuboid(length, width, height):
    """Create vertices of a cuboid centered at origin"""
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
cuboid_vertices = create_cuboid(cuboid_length, cuboid_width, cuboid_height)

# Set up the figure and 3D axis
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Calculate axis limits based on trajectory
positions = trajectory[:, 0:3]
max_range = np.max([
    np.abs(positions[:, 0]).max(),
    np.abs(positions[:, 1]).max(),
    np.abs(positions[:, 2]).max()
]) + 5

def update(frame):
    ax.clear()
    
    # Get current state
    state = trajectory[frame]
    pos = state[0:3]  # position
    quat = state[3:7]  # quaternion [w, x, y, z]
    
    # Rotate and translate vertices
    R = quat_to_rotmat(quat)
    rotated_vertices = (R @ cuboid_vertices.T).T + pos
    
    # Create the faces for plotting
    cuboid_faces = [[rotated_vertices[j] for j in face] for face in faces]
    
    # Plot the cuboid
    poly = Poly3DCollection(cuboid_faces, alpha=0.7, edgecolor='black', linewidths=1.5)
    poly.set_facecolor(['cyan', 'cyan', 'blue', 'blue', 'red', 'red'])
    ax.add_collection3d(poly)
    
    # Draw body-fixed coordinate axes
    axis_length = 2.0
    axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    rotated_axes = (R @ axes.T).T
    
    colors = ['r', 'g', 'b']
    labels = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
    for i, (axis, color, label) in enumerate(zip(rotated_axes, colors, labels)):
        ax.quiver(pos[0], pos[1], pos[2], 
                 axis[0], axis[1], axis[2], 
                 color=color, arrow_length_ratio=0.2, linewidth=2.5, label=label)
    
    # Draw trajectory
    ax.plot(positions[:frame+1, 0], positions[:frame+1, 1], 
            positions[:frame+1, 2], 'gray', alpha=0.4, linewidth=1.5)
    
    # Set labels and limits
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    time_elapsed = frame * sample_rate * step_size
    ax.set_title(f'Rigid Body Dynamics | Time: {time_elapsed:.2f}s / {sim_time:.2f}s', fontsize=14)
    #ax.legend(loc='upper right', fontsize=10)
    
    return ax,

# Create animation
print("Creating animation...")
anim = FuncAnimation(fig, update, frames=len(trajectory), interval=20, blit=False)

plt.tight_layout()
plt.show()