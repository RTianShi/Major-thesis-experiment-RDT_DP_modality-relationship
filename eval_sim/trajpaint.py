import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

path_a = "/home/hjl/RoboticsDiffusionTransformer/eef_traj/PickCube-v1_原用例/PickCube-v1_ep0017.npy"
path_b = "/home/hjl/RoboticsDiffusionTransformer/eef_traj/PickCube-v1_l_none/PickCube-v1_ep0017.npy"
meta_a = "/home/hjl/RoboticsDiffusionTransformer/eef_traj/PickCube-v1_原用例/PickCube-v1_ep0017_meta.npz"
meta_b = "/home/hjl/RoboticsDiffusionTransformer/eef_traj/PickCube-v1_l_none/PickCube-v1_ep0017_meta.npz"

def load_traj(path):
    traj = np.load(path)
    traj = np.asarray(traj)
    if traj.ndim == 3 and traj.shape[1] == 1 and traj.shape[2] == 3:
        traj = traj[:, 0, :]
    elif traj.ndim == 2 and traj.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected traj shape: {traj.shape} in {path}")
    return traj

traj_a = load_traj(path_a)
traj_b = load_traj(path_b)

def load_meta(path):
    z = np.load(path)
    cube = np.asarray(z["cube_pos"]).reshape(-1)
    goal = np.asarray(z["goal_pos"]).reshape(-1)
    return cube, goal

cube_a, goal_a = load_meta(meta_a)
cube_b, goal_b = load_meta(meta_b)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    traj_a[:, 0],
    traj_a[:, 1],
    traj_a[:, 2],
    label="Baseline (Source)",
    linewidth=2.5,
    alpha=0.8,
)
ax.plot(
    traj_b[:, 0],
    traj_b[:, 1],
    traj_b[:, 2],
    label="MR-Silence (Mutated)",
    color="crimson",
    linestyle="--",
    linewidth=2,
)

# 2D shadow projection on the ground (XY plane).
z_floor = min(traj_a[:, 2].min(), traj_b[:, 2].min())
ax.plot(
    traj_a[:, 0],
    traj_a[:, 1],
    zs=z_floor,
    zdir="z",
    color="gray",
    alpha=0.3,
    linestyle=":",
)
ax.plot(
    traj_b[:, 0],
    traj_b[:, 1],
    zs=z_floor,
    zdir="z",
    color="gray",
    alpha=0.3,
    linestyle=":",
)

# Cube and goal positions (single legend entry each).
ax.scatter(cube_a[0], cube_a[1], cube_a[2], s=80, marker="o", label="Cube Position")
ax.scatter(goal_a[0], goal_a[1], goal_a[2], s=80, marker="^", label="Goal Position")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("EEF Trajectories (3D)")
ax.legend()
plt.tight_layout()
plt.show()
