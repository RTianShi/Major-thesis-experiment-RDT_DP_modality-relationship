from typing import Callable, List, Type
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
import torch
from collections import deque
from PIL import Image
import cv2
import json
from datetime import datetime
import copy
import re

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from eval_sim_pushcube.mr_dp import get_lang, get_vision, get_proprio  # registers PushCube DP MRs
from eval_sim_pushcube.env_mr_dp import get_env  # registers PushCube DP env MRs
from eval_sim_pushcube.custom_envs import *  # noqa: F401,F403  # registers PushCube custom envs

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--show", action="store_true", help="Show real-time rendering with OpenCV window.")
    parser.add_argument("--vis", action="store_true", help="Alias of --show for backward compatibility.")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save videos.")
    parser.add_argument("--video-fps", type=int, default=25, help="FPS for saved videos.")
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="/home/hjl/RoboticsDiffusionTransformer/eef_traj_dp/PushCube",
        help="Base directory to save end-effector trajectories. The default creates a unique timestamped subfolder under /home/hjl/RoboticsDiffusionTransformer/eef_traj_dp/PushCube.",
    )
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default=None, help="Alias of --video-dir for backward compatibility.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Random seed for the environment.")
    parser.add_argument("--mr-type", type=str, default=None, help="Mutation type label for JSON output.")
    parser.add_argument("--mr-config", type=str, default="configs/mr_eval_pushcube.yaml", help="Path to MR config YAML.")

    return parser.parse_args()

task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeCubeCenter-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeCubeCorner011-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeEmptyGrasp-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeGhostLift-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeCubeYaw000-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeCubeYaw045-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeGoalZ005-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeGoalZ028-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeInvisibleHeld-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeScale150-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeSceneTrans000-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeSceneTrans04N04-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeBlueCube-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeBlueCylinder-v1": "Grasp a blue cylinder and move it to a target goal position.",
    "PickCubeBlueTriangularPrism-v1": "Grasp a blue triangular prism and move it to a target goal position.",
    "PickCubeRedSphereBlueCube-v1": "Grasp a red cube and move it to a target goal position.",
    "PickCubeWoodTable-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCubeVisualDebunking-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeLargeRedCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}
import random
import os

def _slugify(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    safe = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)

def _to_uint8_rgb(img):
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    while hasattr(img, "ndim") and img.ndim > 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 4 and img.shape[-1] == 3:
        img = img[0]
    if img.dtype != np.uint8:
        max_val = float(np.max(img)) if img.size > 0 else 0.0
        if max_val <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img

def _to_numpy_1d(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    return x

def _extract_gripper_width(obs):
    """
    返回:
      gripper_width: float 或 None
      gripper_finger_qpos: [left, right] 或 None
    """
    try:
        qpos = _to_numpy_1d(obs["agent"]["qpos"])
        if qpos is None or qpos.size < 2:
            return None, None
        left = float(qpos[-2])
        right = float(qpos[-1])
        return float(left + right), [left, right]
    except Exception:
        return None, None


def _extract_robot_joint_torques(env):
    """Best-effort read of robot joint torques/forces.

    ManiSkill/SAPIEN wrappers differ across versions/backends.
    Returns a 1D list[float] or None when unavailable.
    """
    robot = getattr(getattr(getattr(env, "unwrapped", env), "agent", None), "robot", None)
    if robot is None:
        return None
    for method_name in ("get_qf", "get_qforces", "get_qforce", "get_drive_forces"):
        method = getattr(robot, method_name, None)
        if callable(method):
            try:
                qf = method()
                arr = _to_numpy_1d(qf)
                if arr is None or arr.size == 0:
                    continue
                return [float(x) for x in arr.tolist()]
            except Exception:
                continue
    return None


def _tcp_pos_xyz(env):
    try:
        p = env.unwrapped.agent.tcp.pose.p
    except Exception:
        return None
    arr = _to_numpy_1d(p)
    if arr is None or arr.size < 3:
        return None
    return np.asarray(arr[:3], dtype=np.float32)


def _tcp_linear_velocity_xyz(curr_tcp_xyz, prev_tcp_xyz, control_freq):
    if curr_tcp_xyz is None or prev_tcp_xyz is None:
        return None
    try:
        cf = float(control_freq)
    except Exception:
        cf = 0.0
    if not np.isfinite(cf) or cf <= 0:
        return None
    vel = (np.asarray(curr_tcp_xyz, dtype=np.float32) - np.asarray(prev_tcp_xyz, dtype=np.float32)) * cf
    return vel.astype(np.float32)


def _tcp_to_obj_distance(tcp_xyz, obj_xyz):
    if tcp_xyz is None or obj_xyz is None:
        return None
    try:
        a = np.asarray(tcp_xyz, dtype=np.float32).reshape(-1)
        b = np.asarray(obj_xyz, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if a.size < 3 or b.size < 3:
        return None
    return float(np.linalg.norm(a[:3] - b[:3]))

def _extract_cube_pos(obs, env):
    # 1) 优先从环境对象读（对 PickCube 系列最稳）
    unwrapped = getattr(env, "unwrapped", env)
    for name in ("cubeA", "cube_a", "red_cube", "cube", "obj", "object", "target_object", "source_object"):
        actor = getattr(unwrapped, name, None)
        if actor is not None and hasattr(actor, "pose"):
            p = getattr(actor.pose, "p", None)
            arr = _to_numpy_1d(p)
            if arr is not None and arr.size >= 3:
                return [float(arr[0]), float(arr[1]), float(arr[2])]

    # 2) 退化到 obs["extra"]
    if isinstance(obs, dict):
        extra = obs.get("extra", {})
        if isinstance(extra, dict):
            for k in ("cube_pos", "obj_pos", "object_pos", "cube_pose", "obj_pose", "object_pose"):
                if k in extra:
                    arr = _to_numpy_1d(extra[k])
                    if arr is not None and arr.size >= 3:
                        return [float(arr[0]), float(arr[1]), float(arr[2])]
            for k, v in extra.items():
                lk = str(k).lower()
                if ("cube" in lk or "obj" in lk or "object" in lk) and ("pos" in lk or "pose" in lk):
                    arr = _to_numpy_1d(v)
                    if arr is not None and arr.size >= 3:
                        return [float(arr[0]), float(arr[1]), float(arr[2])]
    return None


def _pose_to_quat(obj):
    if obj is None:
        return None
    pose = getattr(obj, "pose", None)
    if pose is not None and hasattr(pose, "q"):
        q = _to_numpy_1d(pose.q)
        if q is not None and q.size >= 4:
            return np.asarray(q[:4], dtype=np.float64)
    if hasattr(obj, "get_pose"):
        pose = obj.get_pose()
        if hasattr(pose, "q"):
            q = _to_numpy_1d(pose.q)
            if q is not None and q.size >= 4:
                return np.asarray(q[:4], dtype=np.float64)
    return None


def _pose_to_xyz(obj):
    if obj is None:
        return None
    if hasattr(obj, "pose"):
        return np.array(obj.pose.p, dtype=np.float32)
    if hasattr(obj, "get_pose"):
        return np.array(obj.get_pose().p, dtype=np.float32)
    return None


def _quat_to_yaw_deg(quat):
    if quat is None:
        return None
    try:
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if q.size < 4:
        return None
    w, x, y, z = q[:4]
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return None
    w, x, y, z = w / n, x / n, y / n, z / n
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))


def _find_actor_by_keywords(scene, keywords):
    if scene is None:
        return None, None
    for actor in scene.get_all_actors():
        try:
            name = actor.get_name()
        except Exception:
            name = ""
        if any(keyword in str(name).lower() for keyword in keywords):
            return actor, name
    return None, None


def _find_articulation_by_keywords(scene, keywords):
    if scene is None:
        return None, None
    try:
        articulations = scene.get_all_articulations()
    except Exception:
        return None, None
    for articulation in articulations:
        try:
            name = articulation.get_name()
        except Exception:
            name = ""
        if any(keyword in str(name).lower() for keyword in keywords):
            return articulation, name
    return None, None


def _get_cube_goal_xyz(env):
    cube_pos = None
    goal_pos = None

    for key in ["cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"]:
        if hasattr(env.unwrapped, key):
            cube_pos = _pose_to_xyz(getattr(env.unwrapped, key))
            if cube_pos is not None:
                break

    for key in ["cubeB", "cube_b", "green_cube", "goal_cube", "target_cube", "goal", "_goal", "target", "_target", "goal_site", "target_site", "goal_region"]:
        if hasattr(env.unwrapped, key):
            goal_pos = _pose_to_xyz(getattr(env.unwrapped, key))
            if goal_pos is not None:
                break

    scene = getattr(env.unwrapped, "scene", None)
    if cube_pos is None:
        actor, _ = _find_actor_by_keywords(scene, ["cubea", "red", "cube", "block", "obj"])
        cube_pos = _pose_to_xyz(actor)
    if goal_pos is None:
        actor, _ = _find_actor_by_keywords(scene, ["cubeb", "green", "goal", "target", "region"])
        goal_pos = _pose_to_xyz(actor)

    if cube_pos is None:
        articulation, _ = _find_articulation_by_keywords(scene, ["cubea", "red", "cube", "block", "obj"])
        cube_pos = _pose_to_xyz(articulation)
    if goal_pos is None:
        articulation, _ = _find_articulation_by_keywords(scene, ["cubeb", "green", "goal", "target", "region"])
        goal_pos = _pose_to_xyz(articulation)

    return cube_pos, goal_pos


def _get_stackcube_actor(env, role: str):
    unwrapped = getattr(env, "unwrapped", env)
    if role == "src":
        keys = ["cubeA", "cube_a", "red_cube", "source_object", "obj", "object", "_obj", "cube"]
        keywords = ["cubea", "red", "source", "obj", "cube", "block"]
    else:
        keys = ["cubeB", "cube_b", "green_cube", "goal_cube", "target_cube", "goal", "_goal", "target", "_target"]
        keywords = ["cubeb", "green", "goal", "target", "cube", "block", "region"]

    for key in keys:
        actor = getattr(unwrapped, key, None)
        if actor is not None:
            return actor

    scene = getattr(unwrapped, "scene", None)
    actor, _ = _find_actor_by_keywords(scene, keywords)
    if actor is not None:
        return actor
    articulation, _ = _find_articulation_by_keywords(scene, keywords)
    return articulation


def _extract_stackcube_cube_state(obs, env, role: str):
    actor = _get_stackcube_actor(env, role)
    pos = _pose_to_xyz(actor)
    quat = _pose_to_quat(actor)
    yaw_deg = _quat_to_yaw_deg(quat)

    if pos is None and role == "src":
        pos = _extract_cube_pos(obs, env)

    if quat is None and isinstance(obs, dict):
        extra = obs.get("extra", {})
        if isinstance(extra, dict):
            quat_keys = (
                ["red_cube_quat", "cubeA_quat", "src_cube_quat", "cube_quat"]
                if role == "src"
                else ["green_cube_quat", "cubeB_quat", "dst_cube_quat", "goal_cube_quat"]
            )
            for key in quat_keys:
                if key in extra:
                    quat = _to_numpy_1d(extra[key])
                    if quat is not None and quat.size >= 4:
                        quat = np.asarray(quat[:4], dtype=np.float64)
                        yaw_deg = _quat_to_yaw_deg(quat)
                        break

    pos_list = pos.tolist() if pos is not None else None
    quat_list = quat.tolist() if quat is not None else None
    return pos_list, quat_list, yaw_deg


def _refresh_obs(env):
    return env.get_obs()


def _to_bool_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    arr = np.array(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0].item())


def _current_is_grasped(env):
    cube = None
    for key in ["cubeA", "cube_a", "red_cube", "cube", "obj", "object", "_obj"]:
        candidate = getattr(env.unwrapped, key, None)
        if candidate is not None:
            cube = candidate
            break
    if cube is None:
        return False
    agent = getattr(env.unwrapped, "agent", None)
    if agent is None or not hasattr(agent, "is_grasping"):
        return False
    try:
        return _to_bool_scalar(agent.is_grasping(cube))
    except Exception:
        return False


def _resolve_visual_delay_steps(env, vision_cfg, fallback_fps):
    if vision_cfg.get("type") != "MR-LTSEP5":
        return 0
    if "delay_steps" in vision_cfg:
        return max(0, int(vision_cfg.get("delay_steps", 0)))
    delay_ms = float(vision_cfg.get("delay_ms", 200.0))
    control_freq = getattr(env.unwrapped, "control_freq", None)
    if control_freq is None:
        control_freq = fallback_fps
    return max(0, int(round(float(control_freq) * delay_ms / 1000.0)))


def _get_history_frame(obs_window, delay_steps):
    if obs_window is None or len(obs_window) == 0:
        return None
    src_idx = len(obs_window) - 1 - int(delay_steps)
    if src_idx < 0 or src_idx >= len(obs_window):
        return None
    return obs_window[src_idx]


def _build_policy_obs(
    env,
    obs,
    mr_cfg,
    obs_window=None,
    vision_delay_steps=0,
    *,
    step_index=0,
    previous_cube_goal_distance=None,
    cube_lifted=False,
):
    curr_cube_pos, curr_goal_pos = _get_cube_goal_xyz(env)
    if curr_cube_pos is None or curr_goal_pos is None:
        cube_goal_distance = float("nan")
    else:
        cube_goal_distance = float(np.linalg.norm(curr_cube_pos - curr_goal_pos))

    delayed_img = _get_history_frame(obs_window, vision_delay_steps) if obs_window is not None else None
    if delayed_img is None:
        delayed_img = _to_uint8_rgb(env.render())

    tcp_xyz = _tcp_pos_xyz(env)
    cube_xyz = curr_cube_pos if curr_cube_pos is not None else None
    tcp_to_obj_dist = _tcp_to_obj_distance(tcp_xyz, cube_xyz)
    contact_distance = float(mr_cfg.get("proprio", {}).get("contact_distance", 0.035))
    is_contact = bool(tcp_to_obj_dist is not None and np.isfinite(tcp_to_obj_dist) and tcp_to_obj_dist <= contact_distance)

    vision_runtime = mr_cfg["vision"].setdefault("runtime", {})
    vision_runtime["step_index"] = int(step_index)
    vision_runtime["is_grasped"] = False
    vision_runtime["is_contact"] = is_contact
    vision_runtime["tcp_to_obj_distance"] = tcp_to_obj_dist
    vision_runtime["src_cube_pos"] = curr_cube_pos.tolist() if curr_cube_pos is not None else None
    vision_runtime["dst_cube_pos"] = curr_goal_pos.tolist() if curr_goal_pos is not None else None
    vision_runtime["cube_goal_distance"] = cube_goal_distance
    vision_runtime["previous_cube_goal_distance"] = previous_cube_goal_distance
    vision_runtime["cube_lifted"] = bool(cube_lifted)
    vision_runtime["delay_steps"] = int(vision_delay_steps)

    # Optional: help CPTMP-style MRs estimate the grasp phase length.
    observed_contact_step = mr_cfg["vision"].get("_observed_contact_step", None)
    if is_contact and observed_contact_step is None:
        observed_contact_step = int(step_index)
        mr_cfg["vision"]["_observed_contact_step"] = observed_contact_step
    vision_runtime["observed_contact_step"] = observed_contact_step
    images = vis_mut([Image.fromarray(delayed_img)], mr_cfg["vision"])
    vision_image = images[0] if images else Image.fromarray(delayed_img)
    if vision_image is None:
        vision_image = Image.fromarray(delayed_img)
    img_mut = np.array(vision_image, dtype=np.uint8)
    img_tensor = torch.as_tensor(img_mut, device="cuda").float()

    gripper_width, gripper_finger_qpos = _extract_gripper_width(obs)
    proprio = obs["agent"]["qpos"][:].cuda()
    proprio_runtime = mr_cfg["proprio"].setdefault("runtime", {})
    proprio_runtime["step_index"] = int(step_index)
    proprio_runtime["is_grasped"] = False
    proprio_runtime["is_contact"] = is_contact
    proprio_runtime["tcp_to_obj_distance"] = tcp_to_obj_dist
    proprio_runtime["cube_lifted"] = bool(cube_lifted)
    proprio_runtime["cube_goal_distance"] = cube_goal_distance
    proprio_runtime["previous_cube_goal_distance"] = previous_cube_goal_distance
    proprio_runtime["gripper_width"] = gripper_width
    proprio_runtime["gripper_finger_qpos"] = gripper_finger_qpos
    proprio = prop_mut(proprio, mr_cfg["proprio"])
    proprio = (proprio - state_min) / (state_max - state_min) * 2 - 1

    policy_obs = {
        "agent_pos": proprio,
        "head_cam": img_tensor.permute(2, 0, 1).unsqueeze(0),
    }
    return policy_obs, img_mut, cube_goal_distance

def _extract_eef_yaw(env):
    """从环境获取末端执行器 yaw 角（单位：度），失败返回 None。"""
    try:
        eef_pose = env.unwrapped.agent.tcp.pose

        # 1) 优先从 4x4 变换矩阵取 yaw（最稳）
        if hasattr(eef_pose, "to_transformation_matrix"):
            mat = np.asarray(eef_pose.to_transformation_matrix())
            if mat.ndim == 3:  # 兼容 batch 形状 (1,4,4)
                mat = mat[0]
            if mat.shape[0] >= 2 and mat.shape[1] >= 2:
                yaw = np.arctan2(mat[1, 0], mat[0, 0])
                return float(np.degrees(yaw))

        # 2) 回退：从四元数计算 yaw（按 SAPIEN 常见 wxyz）
        if hasattr(eef_pose, "q"):
            q = np.asarray(eef_pose.q, dtype=np.float64).reshape(-1)
            if q.size < 4:
                return None
            if q.size > 4:
                q = q[:4]
            w, x, y, z = q
            n = np.linalg.norm([w, x, y, z])
            if n < 1e-12:
                return None
            w, x, y, z = w / n, x / n, y / n, z / n
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            return float(np.degrees(yaw))
    except Exception:
        return None

    return None

args = parse_args()
if args.vis:
    args.show = True
if args.record_dir:
    args.video_dir = args.record_dir


default_stackcube_traj_root = "/home/hjl/RoboticsDiffusionTransformer/eef_traj_dp/PushCube"
if args.traj_dir == default_stackcube_traj_root:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = _slugify(args.env_id)
    mr_type = getattr(args, "mr_type", None)
    if mr_type:
        mr_tag = _slugify(mr_type)
        args.traj_dir = os.path.join(default_stackcube_traj_root, f"{env_tag}_{mr_tag}_{ts}")
    else:
        args.traj_dir = os.path.join(default_stackcube_traj_root, f"{env_tag}_{ts}")

os.makedirs(args.traj_dir, exist_ok=True)

with open(args.mr_config, "r", encoding="utf-8") as fp:
    mr_config_data = yaml.safe_load(fp) or {}

mr_cfg = copy.deepcopy(mr_config_data.get("mr", {}))
mr_cfg.setdefault("language", {"type": "identity"})
mr_cfg.setdefault("vision", {"type": "identity"})
mr_cfg.setdefault("proprio", {"type": "identity"})
mr_cfg.setdefault("env", {"type": "identity"})

lang_mut = get_lang(mr_cfg["language"]["type"])
vis_mut = get_vision(mr_cfg["vision"]["type"])
prop_mut = get_proprio(mr_cfg["proprio"]["type"])
env_cfg = mr_cfg["env"]
env_mut = get_env(env_cfg["type"])

seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

env_id = args.env_id
env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    sim_backend=args.sim_backend
)

import hydra
import dill

checkpoint_path = args.pretrained_path
print(f"Loading policy from {checkpoint_path}. Task is {task2lang[env_id]}")

run_config_path = os.path.join(args.traj_dir, "run_config.json")
run_config = {
    "args": vars(args),
    "mr_config": {
        "path": args.mr_config,
        "abs_path": os.path.abspath(args.mr_config),
        "content": mr_config_data,
    },
}

def get_policy(output_dir, device):
    
    # load checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

policy = get_policy('./', device = 'cuda')
MAX_EPISODE_STEPS = 400
total_episodes = args.num_traj 
success_count = 0 
base_seed = 20241201
instr = task2lang[env_id]
import tqdm

DATA_STAT = {'state_min': [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0, 0.0], 'state_max': [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033, 0.03999999910593033], 'action_min': [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0], 'action_max': [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0], 'action_std': [0.2199309915304184, 0.18780815601348877, 0.13044124841690063, 0.30669933557510376, 0.1340624988079071, 0.24968451261520386, 0.9589747190475464, 0.9827960729598999], 'action_mean': [-0.00885344110429287, 0.5523102879524231, -0.007564723491668701, -2.0108158588409424, 0.004714342765510082, 2.615924596786499, 0.08461848646402359, -0.19301606714725494]}

state_min = torch.tensor(DATA_STAT['state_min']).cuda()
state_max = torch.tensor(DATA_STAT['state_max']).cuda()
action_min = torch.tensor(DATA_STAT['action_min']).cuda()
action_max = torch.tensor(DATA_STAT['action_max']).cuda()

def _extract_cube_yaw_deg(obs, env):
    """尽力提取 cube yaw（度），失败返回 None。"""
    _, _, yaw_deg = _extract_stackcube_cube_state(obs, env, role="src")
    return yaw_deg

def _infer_cube_yaw_from_env_id(env_id: str):
    m = re.search(r"Yaw(\d{3})", str(env_id))
    return float(int(m.group(1))) if m else None

def _use_contact_grasp_mode(mr_type: str) -> bool:
    normalized = str(mr_type or "").strip().lower().replace("_", "-")
    return normalized in {"mr-sesp1", "mr-sesp-1"}

def _first_index_ge(seq, thresh: float):
    for i, v in enumerate(seq):
        if v is not None and float(v) >= thresh:
            return int(i)
    return None

def _first_index_le(seq, thresh: float):
    for i, v in enumerate(seq):
        if v is not None and float(v) <= thresh:
            return int(i)
    return None

vision_history_len = 1
vision_delay_steps = _resolve_visual_delay_steps(env, mr_cfg["vision"], args.video_fps)
obs_window_len = max(1, vision_delay_steps + 1)

for episode in tqdm.trange(total_episodes):
    obs_window = deque(maxlen=obs_window_len)
    obs, _ = env.reset(seed = episode + base_seed)
    env_mut(env, env_cfg)
    obs = _refresh_obs(env)
    policy.reset()
    mr_cfg["vision"]["runtime"] = {}
    mr_cfg["proprio"]["runtime"] = {}
    # Clear per-episode internal MR state (kept on the cfg dict across steps).
    # Do this only for LTSEP1 to avoid impacting other proprio MRs.
    proprio_type = str(mr_cfg.get("proprio", {}).get("type", "identity") or "identity")
    proprio_type_norm = proprio_type.strip().lower().replace("_", "-")
    if "ltsep1" in proprio_type_norm:
        mr_cfg["proprio"].pop("_phantom_fired_step", None)
        mr_cfg["proprio"].pop("_phantom_fired_steps", None)
    mr_cfg["vision"].pop("_observed_grasp_step", None)
    mr_cfg["vision"].pop("_stacking_blindness_active", None)
    mr_cfg["vision"].pop("_stacking_blindness_approach_counter", None)
    mr_cfg["vision"].pop("_cptmp1_tgrasp_steps", None)
    mr_cfg["vision"].pop("_cptmp1_visible_until_step", None)
    mr_cfg["vision"].pop("_cptmp1_blind_until_step", None)

    initial_img = _to_uint8_rgb(env.render())
    obs_window.append(initial_img)
    policy_obs, _, _ = _build_policy_obs(
        env,
        obs,
        mr_cfg,
        obs_window=obs_window,
        vision_delay_steps=vision_delay_steps,
        step_index=0,
        previous_cube_goal_distance=None,
        cube_lifted=False,
    )
    obs_window.append(policy_obs)

    global_steps = 0
    video_frames = []
    # Core per-step logging for push tasks (no grasp required)
    eef_traj = []  # keep key name eef_traj for backward compatibility with existing downstream tooling
    eef_vel_traj = []
    joint_torques_traj = []
    cube_pos_traj = []
    src_cube_pos_traj = []
    dst_cube_pos_traj = []
    target_pos_traj = []
    tcp_to_obj_distance_traj = []
    is_contact_traj = []
    contact_frame_index = None  # 0-based index into trajectories
    contact_eef_pos = None
    gripper_width_traj = []
    gripper_finger_qpos_traj = []
    done = False
    info = {"success": False}

    # Record initial anchor positions (PushCube: obj + goal_region)
    initial_cube_xyz, initial_goal_xyz = _get_cube_goal_xyz(env)
    initial_cube_pos = None if initial_cube_xyz is None else [float(x) for x in np.asarray(initial_cube_xyz, dtype=np.float32).reshape(-1)[:3].tolist()]
    initial_dst_cube_pos = None if initial_goal_xyz is None else [float(x) for x in np.asarray(initial_goal_xyz, dtype=np.float32).reshape(-1)[:3].tolist()]
    initial_cube_z = float(initial_cube_pos[2]) if initial_cube_pos is not None and len(initial_cube_pos) >= 3 else None
    prev_cube_goal_distance = None
    last_known_target_pos = initial_dst_cube_pos

    control_freq = getattr(env.unwrapped, "control_freq", None)
    if control_freq is None:
        control_freq = float(args.video_fps)
    last_eef_xyz = None

    gripper_action_cmd_traj = []   # keep for debugging; push tasks may ignore it
    while global_steps < MAX_EPISODE_STEPS and not done:
        obs = obs_window[-1]
        actions = policy.predict_action(obs)
        actions = actions['action_pred'].squeeze(0)
        actions = (actions + 1) / 2 * (action_max - action_min) + action_min
        actions = actions.detach().cpu().numpy()
        actions = actions[:8]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            global_steps += 1

            current_img = _to_uint8_rgb(env.render())
            obs_window.append(current_img)
            # PushCube: compute cube + goal positions from stable env sources.
            current_cube_xyz, current_goal_xyz = _get_cube_goal_xyz(env)
            current_cube_pos = None if current_cube_xyz is None else [
                float(x) for x in np.asarray(current_cube_xyz, dtype=np.float32).reshape(-1)[:3].tolist()
            ]
            current_dst_cube_pos = None if current_goal_xyz is None else [
                float(x) for x in np.asarray(current_goal_xyz, dtype=np.float32).reshape(-1)[:3].tolist()
            ]
            if current_dst_cube_pos is None and last_known_target_pos is not None:
                current_dst_cube_pos = last_known_target_pos
            elif current_dst_cube_pos is not None:
                last_known_target_pos = current_dst_cube_pos
            current_src_cube_pos = current_cube_pos
            cube_lifted = False
            if current_cube_pos is not None and len(current_cube_pos) >= 3 and initial_cube_z is not None:
                cube_lifted = float(current_cube_pos[2]) > float(initial_cube_z) + 0.015
            policy_obs, img, _ = _build_policy_obs(
                env,
                obs,
                mr_cfg,
                obs_window=obs_window,
                vision_delay_steps=vision_delay_steps,
                step_index=global_steps,
                previous_cube_goal_distance=prev_cube_goal_distance,
                cube_lifted=cube_lifted,
            )
            obs_window.append(policy_obs)
            eef_xyz = _tcp_pos_xyz(env)
            if eef_xyz is None:
                eef_xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            else:
                eef_xyz = np.array(eef_xyz, dtype=np.float32).reshape(-1)[:3]
            eef_traj.append(eef_xyz)

            eef_vel = _tcp_linear_velocity_xyz(eef_xyz, last_eef_xyz, control_freq)
            eef_vel_traj.append(None if eef_vel is None else [float(x) for x in eef_vel.tolist()])
            last_eef_xyz = np.array(eef_xyz, dtype=np.float32)

            joint_torques = _extract_robot_joint_torques(env)
            joint_torques_traj.append(joint_torques)

            tcp_to_obj_dist = _tcp_to_obj_distance(
                eef_xyz,
                None if current_cube_pos is None else np.asarray(current_cube_pos, dtype=np.float32),
            )
            tcp_to_obj_distance_traj.append(tcp_to_obj_dist)
            contact_distance = float(mr_cfg.get("proprio", {}).get("contact_distance", 0.035))
            is_contact = bool(
                tcp_to_obj_dist is not None
                and np.isfinite(tcp_to_obj_dist)
                and tcp_to_obj_dist <= contact_distance
            )
            is_contact_traj.append(is_contact)
            if contact_frame_index is None and is_contact:
                contact_frame_index = len(is_contact_traj) - 1
                contact_eef_pos = [float(x) for x in np.asarray(eef_xyz, dtype=np.float32).tolist()]
            gripper_cmd = float(action[-1]) if action.shape[0] > 0 else None

            # 仅记录原始序列，不在循环中判定抓取时刻
            gripper_action_cmd_traj.append(gripper_cmd)

            gripper_width, gripper_finger_qpos = _extract_gripper_width(obs)
            gripper_width_traj.append(gripper_width)
            gripper_finger_qpos_traj.append(gripper_finger_qpos)

            cube_pos_traj.append(current_cube_pos)
            src_cube_pos_traj.append(current_src_cube_pos)
            dst_cube_pos_traj.append(current_dst_cube_pos)
            # Analyzer expects target_pos to exist for LTSEP1 rules.
            target_pos_traj.append(current_dst_cube_pos)
            curr_goal_dist = mr_cfg["proprio"].get("runtime", {}).get("cube_goal_distance")
            prev_cube_goal_distance = curr_goal_dist
            if args.save_video:
                 video_frames.append(img)
            if args.show:
                 cv2.imshow("maniskill", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                 cv2.waitKey(1)
            if terminated or truncated:
                 assert "success" in info, sorted(info.keys())
                 if info['success']:
                     done = True
                     success_count += 1
                     break 

    if args.save_video and video_frames:
        os.makedirs(args.video_dir, exist_ok=True)
        h, w = video_frames[0].shape[:2]
        out_path = os.path.join(args.video_dir, f"{env_id}_ep{episode+1:04d}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (w, h))
        for frame in video_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    if eef_traj:
        os.makedirs(args.traj_dir, exist_ok=True)
        eef_arr = np.stack(eef_traj, axis=0)
        diffs = np.diff(eef_arr, axis=0)
        total_path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(eef_arr) > 1 else 0.0
        # For push tasks, reuse the legacy grasp_frame_index field as the first contact frame.
        grasp_frame_index = contact_frame_index

        mr_eval_extra = {}
        if "ltsep1" in str(mr_cfg.get("proprio", {}).get("type", "") or "").strip().lower().replace("_", "-"):
            mr_eval_extra["proprio_phantom_fired_steps"] = mr_cfg.get("proprio", {}).get("_phantom_fired_steps")

        result = {
            "episode_id": int(episode + 1),
            "seed": int(episode + base_seed),
            "metrics": {
                "env_success": bool(info["success"]),
            },
            "mr_eval": {
                "mr_type": args.mr_type,
                "pair_key": f"{episode + base_seed}",
                "grasp_frame_index": grasp_frame_index,
                "contact_frame_index": contact_frame_index,
                "contact_eef_pos": contact_eef_pos,
                "initial_cube_pos": initial_cube_pos,
                "initial_target_pos": initial_dst_cube_pos,
                **mr_eval_extra,
            },
            "data_availability": {
                "grasp_frame_index_available": grasp_frame_index is not None,
                "contact_frame_index_available": contact_frame_index is not None,
                "joint_torques_available": any(v is not None for v in joint_torques_traj),
            },
            "trajectory": {
                "total_steps": int(global_steps),
                "initial_cube_pos": initial_cube_pos,
                "initial_target_pos": initial_dst_cube_pos,
                "eef_path": eef_arr.tolist(),
                "eef_vel": eef_vel_traj,
                "joint_torques": joint_torques_traj,
                "gripper_width": gripper_width_traj,
                "gripper_finger_qpos": gripper_finger_qpos_traj,
                "gripper_action_cmd": gripper_action_cmd_traj,
                "cube_pos": cube_pos_traj,
                "src_cube_pos": src_cube_pos_traj,
                "dst_cube_pos": dst_cube_pos_traj,
                "target_pos": target_pos_traj,
                "tcp_to_obj_distance": tcp_to_obj_distance_traj,
                "is_contact": is_contact_traj,
                "total_path_length_meters": total_path_length,
            },
        }
        json_path = os.path.join(args.traj_dir, f"{env_id}_ep{episode+1:04d}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")

if args.show:
    cv2.destroyAllWindows()

success_rate = success_count / total_episodes * 100
print(f"Tested {total_episodes} episodes, success rate: {success_rate:.2f}%")
log_file = f"results_dp_{checkpoint_path.split('/')[-1].split('.')[0]}.txt"
with open(log_file, 'a') as f:
    f.write(f"{args.env_id}:{seed}:{success_count}\n")

run_config["summary"] = {
    "env_id": env_id,
    "total_episodes": int(total_episodes),
    "success_count": int(success_count),
    "success_rate": float(success_rate),
}
os.makedirs(args.traj_dir, exist_ok=True)
with open(run_config_path, "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)
