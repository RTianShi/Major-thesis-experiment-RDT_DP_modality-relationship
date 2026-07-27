import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os


def _is_xyz_array(x):
    try:
        a = np.asarray(x, dtype=float)
    except Exception:
        return False
    if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] >= 2:
        return True
    return False


def find_first_traj(obj):
    # search for first array-like Nx3 in the object (recursively)
    if _is_xyz_array(obj):
        return np.asarray(obj, dtype=float)
    if isinstance(obj, dict):
        for k, v in obj.items():
            res = find_first_traj(v)
            if res is not None:
                return res
    if isinstance(obj, list):
        for v in obj:
            res = find_first_traj(v)
            if res is not None:
                return res
    return None


def find_meta_pos(obj, candidates):
    if isinstance(obj, dict):
        for key in candidates:
            if key in obj:
                try:
                    a = np.asarray(obj[key], dtype=float).reshape(-1)
                    if a.size >= 3:
                        return a[:3]
                except Exception:
                    pass
        for v in obj.values():
            res = find_meta_pos(v, candidates)
            if res is not None:
                return res
    if isinstance(obj, list):
        for v in obj:
            res = find_meta_pos(v, candidates)
            if res is not None:
                return res
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    traj = find_first_traj(data)
    cube = find_meta_pos(data, ["cube_pos", "cube", "object_pos", "obj_pos", "cube_pos_world"])
    goal = find_meta_pos(data, ["goal_pos", "goal", "target_pos", "dst_pos", "goal_pos_world"])
    return traj, cube, goal


def plot_two(traj_a, traj_b, cube_a=None, goal_a=None, out_path=None, show=False, labels=("A", "B"), view="3d"):
    view = str(view).lower()
    if view == "topdown":
        fig, ax = plt.subplots()
        if traj_a is not None:
            ax.plot(traj_a[:, 0], traj_a[:, 1], label=f"{labels[0]}", linewidth=2.5, alpha=0.85)
            ax.scatter(traj_a[0, 0], traj_a[0, 1], s=35, marker="o", color=ax.lines[-1].get_color())
            ax.scatter(traj_a[-1, 0], traj_a[-1, 1], s=55, marker="x", color=ax.lines[-1].get_color())
        if traj_b is not None:
            ax.plot(traj_b[:, 0], traj_b[:, 1], label=f"{labels[1]}", color="crimson", linestyle="--", linewidth=2)
            ax.scatter(traj_b[0, 0], traj_b[0, 1], s=35, marker="o", color="crimson")
            ax.scatter(traj_b[-1, 0], traj_b[-1, 1], s=55, marker="x", color="crimson")

        if cube_a is not None:
            ax.scatter(cube_a[0], cube_a[1], s=80, marker="o", label="Cube Position")
        if goal_a is not None:
            ax.scatter(goal_a[0], goal_a[1], s=80, marker="^", label="Goal Position")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("EEF Trajectories (Top-down XY)")
        ax.legend()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if traj_a is not None:
            ax.plot(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2], label=f"{labels[0]}", linewidth=2.5, alpha=0.8)
        if traj_b is not None:
            ax.plot(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2], label=f"{labels[1]}", color="crimson", linestyle="--", linewidth=2)

        if traj_a is not None and traj_b is not None:
            z_floor = min(traj_a[:, 2].min(), traj_b[:, 2].min())
            ax.plot(traj_a[:, 0], traj_a[:, 1], zs=z_floor, zdir="z", color="gray", alpha=0.3, linestyle=":")
            ax.plot(traj_b[:, 0], traj_b[:, 1], zs=z_floor, zdir="z", color="gray", alpha=0.3, linestyle=":")

        if cube_a is not None:
            ax.scatter(cube_a[0], cube_a[1], cube_a[2], s=80, marker="o", label="Cube Position")
        if goal_a is not None:
            ax.scatter(goal_a[0], goal_a[1], goal_a[2], s=80, marker="^", label="Goal Position")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("EEF Trajectories (3D)")
        ax.legend()

    plt.tight_layout()
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved figure to {out_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-a", type=str, required=True)
    parser.add_argument("--json-b", type=str, required=False)
    parser.add_argument("--out", type=str, default=None, help="Output png path")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--label-a", type=str, default="Baseline (Source)")
    parser.add_argument("--label-b", type=str, default="MR (Mutated)")
    parser.add_argument("--view", type=str, default="3d", choices=["3d", "topdown"], help="Plot view mode.")
    args = parser.parse_args()

    traj_a, cube_a, goal_a = load_json(args.json_a)
    if traj_a is None:
        raise RuntimeError(f"Cannot find a Nx3 trajectory array inside {args.json_a}")

    traj_b = None
    cube_b = None
    goal_b = None
    if args.json_b:
        traj_b, cube_b, goal_b = load_json(args.json_b)
        if traj_b is None:
            raise RuntimeError(f"Cannot find a Nx3 trajectory array inside {args.json_b}")

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.json_a))[0]
        if args.json_b:
            other = os.path.splitext(os.path.basename(args.json_b))[0]
            out_path = os.path.join("./", f"{base}_vs_{other}.png")
        else:
            out_path = os.path.join("./", f"{base}.png")

    # Prefer showing cube/goal from first file if available
    cube = cube_a if cube_a is not None else cube_b
    goal = goal_a if goal_a is not None else goal_b

    plot_two(
        traj_a,
        traj_b,
        cube,
        goal,
        out_path=out_path,
        show=args.show,
        labels=(args.label_a, args.label_b),
        view=args.view,
    )


if __name__ == "__main__":
    main()
