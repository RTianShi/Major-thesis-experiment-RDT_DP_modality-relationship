from typing import Callable, List, Type
import sys
sys.path.append('/')
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2
import json
import copy
import re

from eval_sim_stackcube.mr import get_lang, get_vision, get_proprio  # registers MRs
from eval_sim_stackcube.env_mr import get_env  # registers StackCube env MRs
from eval_sim_stackcube.custom_envs import *  # noqa: F401,F403  # registers stackcube custom envs
from scripts.grasp_event import detect_grasp_event_index


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="StackCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument("--text-encoder-device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device for T5 text encoder only.")
    parser.add_argument("--show", action="store_true", help="Show real-time rendering with OpenCV window.")
    parser.add_argument("--save-video", action="store_true", help="Save episode videos to disk.")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save videos.")
    parser.add_argument("--video-fps", type=int, default=25, help="FPS for saved videos.")
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="eef_traj/StackCube",
        help="Base directory to save end-effector trajectories. The default creates a unique timestamped subfolder under eef_traj/StackCube. Use 'auto' for the legacy root-level auto naming.",
    )
    parser.add_argument("--mr-type", type=str, default=None, help="Mutation type label for JSON output.")
    parser.add_argument("--mr-config", type=str, default="configs/mr_eval_stackcube.yaml", help="Path to MR config YAML.")
    return parser.parse_args()


import random
import os
from datetime import datetime


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


def _extract_cube_pos(obs, env):
    unwrapped = getattr(env, "unwrapped", env)
    for name in ("cubeA", "cube_a", "red_cube", "cube", "obj", "object", "target_object", "source_object"):
        actor = getattr(unwrapped, name, None)
        if actor is not None and hasattr(actor, "pose"):
            p = getattr(actor.pose, "p", None)
            arr = _to_numpy_1d(p)
            if arr is not None and arr.size >= 3:
                return [float(arr[0]), float(arr[1]), float(arr[2])]

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


def _resolve_hf_snapshot_path(repo_id: str) -> str:
    repo_dir = os.path.expanduser(
        os.path.join("~", ".cache", "huggingface", "hub", f"models--{repo_id.replace('/', '--')}")
    )
    ref_path = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            snapshot_id = f.read().strip()
        snapshot_path = os.path.join(repo_dir, "snapshots", snapshot_id)
        if os.path.isdir(snapshot_path):
            return snapshot_path
    return repo_id


def _pose_to_xyz(obj):
    if obj is None:
        return None
    if hasattr(obj, "pose"):
        return np.array(obj.pose.p, dtype=np.float32)
    if hasattr(obj, "get_pose"):
        return np.array(obj.get_pose().p, dtype=np.float32)
    return None


def _find_actor_by_keywords(scene, keywords):
    if scene is None:
        return None, None
    for actor in scene.get_all_actors():
        try:
            name = actor.get_name()
        except Exception:
            name = ""
        name_l = name.lower()
        if any(k in name_l for k in keywords):
            return actor, name
    return None, None


def _find_articulation_by_keywords(scene, keywords):
    if scene is None:
        return None, None
    try:
        arts = scene.get_all_articulations()
    except Exception:
        return None, None
    for art in arts:
        try:
            name = art.get_name()
        except Exception:
            name = ""
        name_l = name.lower()
        if any(k in name_l for k in keywords):
            return art, name
    return None, None


def _get_cube_goal_xyz(env):
    cube_pos = None
    goal_pos = None
    cube_name = None
    goal_name = None

    for key in ["obj", "object", "_obj", "cube"]:
        if hasattr(env.unwrapped, key):
            cube_pos = _pose_to_xyz(getattr(env.unwrapped, key))
            if cube_pos is not None:
                cube_name = key
                break

    for key in ["goal", "_goal", "target", "_target", "goal_site", "target_site", "goal_region"]:
        if hasattr(env.unwrapped, key):
            goal_pos = _pose_to_xyz(getattr(env.unwrapped, key))
            if goal_pos is not None:
                goal_name = key
                break

    scene = getattr(env.unwrapped, "scene", None)
    if cube_pos is None:
        actor, name = _find_actor_by_keywords(scene, ["cube", "block", "obj"])
        cube_pos = _pose_to_xyz(actor)
        cube_name = name
    if goal_pos is None:
        actor, name = _find_actor_by_keywords(scene, ["goal", "target", "region"])
        goal_pos = _pose_to_xyz(actor)
        goal_name = name

    if cube_pos is None:
        art, name = _find_articulation_by_keywords(scene, ["cube", "block", "obj"])
        cube_pos = _pose_to_xyz(art)
        cube_name = name
    if goal_pos is None:
        art, name = _find_articulation_by_keywords(scene, ["goal", "target", "region"])
        goal_pos = _pose_to_xyz(art)
        goal_name = name

    return cube_pos, goal_pos, cube_name, goal_name


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
    for key in ["cube", "obj", "object", "_obj"]:
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


def _extract_eef_yaw(env):
    """从环境获取末端执行器 yaw 角（单位：度），失败返回 None。"""
    try:
        eef_pose = env.unwrapped.agent.tcp.pose

        if hasattr(eef_pose, "to_transformation_matrix"):
            mat = np.asarray(eef_pose.to_transformation_matrix())
            if mat.ndim == 3:
                mat = mat[0]
            if mat.shape[0] >= 2 and mat.shape[1] >= 2:
                yaw = np.arctan2(mat[1, 0], mat[0, 0])
                return float(np.degrees(yaw))

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


def _get_history_frame(obs_window, history_len, delay_steps, history_idx):
    src_idx = len(obs_window) - history_len - delay_steps + history_idx
    if src_idx < 0 or src_idx >= len(obs_window):
        return None
    return obs_window[src_idx]


args = parse_args()

def _slugify(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    safe = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)

default_stackcube_traj_root = os.path.join("eef_traj", "StackCube")
if args.traj_dir in {"auto", default_stackcube_traj_root}:
  
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = _slugify(args.env_id)
    mr_type = getattr(args, "mr_type", None)
    base_dir = "eef_traj" if args.traj_dir == "auto" else default_stackcube_traj_root
    if mr_type:
        mr_tag = _slugify(mr_type)
        args.traj_dir = os.path.join(base_dir, f"{env_tag}_{mr_tag}_{ts}")
    else:
        args.traj_dir = os.path.join(base_dir, f"{env_tag}_{ts}")


os.makedirs(args.traj_dir, exist_ok=True)

with open(args.mr_config, "r", encoding="utf-8") as fp:
    mr_config_data = yaml.safe_load(fp) or {}

run_config_path = os.path.join(args.traj_dir, "run_config.json")
run_config = {
    "args": vars(args),
    "mr_config": {
        "path": args.mr_config,
        "abs_path": os.path.abspath(args.mr_config),
        "content": mr_config_data,
    },
}
with open(run_config_path, "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)

mr_cfg = copy.deepcopy(mr_config_data["mr"])

lang_mut = get_lang(mr_cfg["language"]["type"])
vis_mut = get_vision(mr_cfg["vision"]["type"])
prop_mut = get_proprio(mr_cfg["proprio"]["type"])
env_cfg = mr_cfg.get("env", {"type": "identity"})
env_mut = get_env(env_cfg["type"])

# set random seeds
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeColorDistractor-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeLargeRedCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeMugDistractor-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeShapeDistractor-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeVisualDebunking-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeRemoveRedCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCubeWeightlessIron-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

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

config_path = 'configs/base.yaml'
with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
pretrained_text_encoder_name_or_path = _resolve_hf_snapshot_path("google/t5-v1_1-xxl")
pretrained_vision_encoder_name_or_path = _resolve_hf_snapshot_path("google/siglip-so400m-patch14-384")

pretrained_path = args.pretrained_path
policy = create_model(
    args=config, 
    device="cuda",
    text_encoder_device=args.text_encoder_device,
    dtype=torch.bfloat16,
    pretrained=pretrained_path,
    pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
)

text_embed = policy.encode_instruction(task2lang[env_id])
mr_cfg["language"]["encoder"] = policy.encode_instruction
mr_cfg["language"]["text"] = task2lang[env_id]
text_embed = lang_mut(text_embed, mr_cfg["language"])
text_embed_path = f"text_embed_{env_id}.pt"
torch.save(text_embed, text_embed_path)
print(f"Saved text embedding to {text_embed_path}")

MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  
success_count = 0  
success_episode_ids = []
vision_history_len = 2
vision_delay_steps = _resolve_visual_delay_steps(env, mr_cfg["vision"], args.video_fps)
obs_window_len = max(vision_history_len, vision_history_len + vision_delay_steps)

base_seed = 20241201
import tqdm
for episode in tqdm.trange(total_episodes):
    obs_window = deque(maxlen=obs_window_len)
    obs, _ = env.reset(seed = episode + base_seed)
    env_mut(env, env_cfg)
    obs = _refresh_obs(env)
    policy.reset()

    cube_pos, goal_pos, cube_name, goal_name = _get_cube_goal_xyz(env)
    if cube_pos is None:
        cube_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    if goal_pos is None:
        goal_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    red_cube_initial = cube_pos.copy()
    green_goal = goal_pos.copy()
    initial_cube_height = float(red_cube_initial[2]) if np.isfinite(red_cube_initial[2]) else float("nan")

    initial_goal_point = goal_pos
    goal_point = initial_goal_point.tolist() if initial_goal_point is not None else None
    initial_src_cube_pos, initial_src_cube_quat, initial_src_cube_yaw_deg = _extract_stackcube_cube_state(
        obs, env, role="src"
    )
    initial_dst_cube_pos, initial_dst_cube_quat, initial_dst_cube_yaw_deg = _extract_stackcube_cube_state(
        obs, env, role="dst"
    )
    if goal_point is None or (isinstance(goal_point, list) and np.any(np.isnan(goal_point))):
        if initial_dst_cube_pos is not None:
            goal_point = list(initial_dst_cube_pos)
    initial_cube_pos = initial_src_cube_pos
    initial_cube_z = (
        float(initial_cube_pos[2]) if initial_cube_pos is not None and len(initial_cube_pos) >= 3 else None
    )
    prev_cube_goal_distance = None

    img = _to_uint8_rgb(env.render())
    for _ in range(vision_history_len - 1):
        obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs['agent']['qpos'][:, :-1]

    global_steps = 0
    video_frames = []
    eef_traj = []
    gripper_width_traj = []
    gripper_finger_qpos_traj = []
    cube_pos_traj = []
    src_cube_pos_traj = []
    dst_cube_pos_traj = []
    eef_yaw_traj = []
    cube_yaw_traj = []
    src_cube_yaw_traj = []
    dst_cube_yaw_traj = []
    src_cube_quat_traj = []
    dst_cube_quat_traj = []
    gripper_action_cmd_traj = []
    inferred_cube_yaw_deg = _infer_cube_yaw_from_env_id(env_id)
    initial_cube_yaw_deg = initial_src_cube_yaw_deg
    if initial_cube_yaw_deg is None:
        initial_cube_yaw_deg = inferred_cube_yaw_deg

    success_time = 0
    done = False
    mr_cfg["vision"].pop("_observed_grasp_step", None)
    mr_cfg["vision"].pop("_stacking_blindness_active", None)
    mr_cfg["vision"].pop("_stacking_blindness_approach_counter", None)
    mr_cfg["vision"].pop("_cptmp1_tgrasp_steps", None)
    mr_cfg["vision"].pop("_cptmp1_visible_until_step", None)
    mr_cfg["vision"].pop("_cptmp1_blind_until_step", None)

    while global_steps < MAX_EPISODE_STEPS and not done:
        curr_cube_pos, curr_goal_pos, _, _ = _get_cube_goal_xyz(env)
        if curr_cube_pos is None or curr_goal_pos is None:
            cube_goal_distance = float("nan")
        else:
            cube_goal_distance = float(np.linalg.norm(curr_cube_pos - curr_goal_pos))
        previous_cube_goal_distance = mr_cfg["proprio"].get("_previous_cube_goal_distance", None)
        is_grasped = _current_is_grasped(env)
        observed_grasp_step = mr_cfg["vision"].get("_observed_grasp_step", None)
        if is_grasped and observed_grasp_step is None:
            observed_grasp_step = global_steps
            mr_cfg["vision"]["_observed_grasp_step"] = observed_grasp_step

        cube_height = float(curr_cube_pos[2]) if curr_cube_pos is not None else float("nan")
        lift_height_threshold = float(mr_cfg["proprio"].get("lift_height_threshold", 0.02))
        cube_lifted = (
            is_grasped
            and np.isfinite(cube_height)
            and np.isfinite(initial_cube_height)
            and cube_height > initial_cube_height + lift_height_threshold
        )

        image_arrs = []
        for history_idx in range(vision_history_len):
            window_img = _get_history_frame(
                obs_window, vision_history_len, vision_delay_steps, history_idx
            )
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        images = [Image.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        mr_cfg["vision"]["runtime"] = {
            "cube_goal_distance": cube_goal_distance,
            "previous_cube_goal_distance": previous_cube_goal_distance,
            "step_index": global_steps,
            "is_grasped": is_grasped,
            "cube_lifted": cube_lifted,
            "observed_grasp_step": observed_grasp_step,
        }
        images = vis_mut(images, mr_cfg["vision"])
        mr_cfg["proprio"]["runtime"] = {
            "is_grasped": is_grasped,
            "cube_goal_distance": cube_goal_distance,
            "previous_cube_goal_distance": previous_cube_goal_distance,
            "cube_height": cube_height,
            "initial_cube_height": initial_cube_height,
            "cube_lifted": cube_lifted,
            "step_index": global_steps,
        }
        mr_cfg["proprio"]["_previous_cube_goal_distance"] = cube_goal_distance
        prop_in = torch.as_tensor(proprio)
        prop_in = prop_mut(prop_in, mr_cfg["proprio"])
        proprio = prop_in

        actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
        actions = actions[::4, :]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            img = _to_uint8_rgb(env.render())
            obs_window.append(img)
            proprio = obs['agent']['qpos'][:, :-1]
            eef_xyz = env.unwrapped.agent.tcp.pose.p
            eef_traj.append(np.array(eef_xyz, dtype=np.float32))
            current_src_cube_pos, current_src_cube_quat, current_src_cube_yaw_deg = _extract_stackcube_cube_state(
                obs, env, role="src"
            )
            current_dst_cube_pos, current_dst_cube_quat, current_dst_cube_yaw_deg = _extract_stackcube_cube_state(
                obs, env, role="dst"
            )
            eef_yaw = _extract_eef_yaw(env)
            gripper_cmd = float(action[-1]) if action.shape[0] > 0 else None
            gripper_action_cmd_traj.append(gripper_cmd)

            gripper_width, gripper_finger_qpos = _extract_gripper_width(obs)
            cube_pos = current_src_cube_pos
            cube_yaw_deg = current_src_cube_yaw_deg
            if cube_yaw_deg is None:
                cube_yaw_deg = inferred_cube_yaw_deg

            gripper_width_traj.append(gripper_width)
            gripper_finger_qpos_traj.append(gripper_finger_qpos)
            cube_pos_traj.append(cube_pos)
            src_cube_pos_traj.append(current_src_cube_pos)
            dst_cube_pos_traj.append(current_dst_cube_pos)
            cube_yaw_traj.append(cube_yaw_deg)
            src_cube_yaw_traj.append(cube_yaw_deg)
            dst_cube_yaw_traj.append(current_dst_cube_yaw_deg)
            src_cube_quat_traj.append(current_src_cube_quat)
            dst_cube_quat_traj.append(current_dst_cube_quat)
            eef_yaw_traj.append(eef_yaw)
            curr_goal_dist = mr_cfg["proprio"].get("runtime", {}).get("cube_goal_distance")
            prev_cube_goal_distance = curr_goal_dist
            if args.save_video:
                video_frames.append(img)
            if args.show:
                cv2.imshow("maniskill", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            global_steps += 1
            if terminated or truncated:
                assert "success" in info, sorted(info.keys())
                if info['success']:
                    success_count += 1
                    success_episode_ids.append(int(episode + 1))
                    done = True
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

        final_cube_pos, _, _, _ = _get_cube_goal_xyz(env)
        if final_cube_pos is None:
            final_cube_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        eef_arr = np.stack(eef_traj, axis=0)
        diffs = np.diff(eef_arr, axis=0)
        total_path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(eef_arr) > 1 else 0.0
        eef_yaw_valid = int(sum(v is not None for v in eef_yaw_traj))
        cube_yaw_valid = int(sum(v is not None for v in cube_yaw_traj))

        grasp_mode = "contact" if _use_contact_grasp_mode(args.mr_type) else "strict"
        grasp_frame_index = detect_grasp_event_index(
            gripper_action_cmd_traj,
            gripper_width_traj,
            eef_traj,
            cube_pos_traj,
            grasp_mode=grasp_mode,
        )
        gripper_fully_closed_frame_index = _first_index_le(gripper_action_cmd_traj, -0.95)
        gripper_fully_opened_frame_index = _first_index_ge(gripper_action_cmd_traj, 0.95)

        eef_yaw_at_grasp = None
        if grasp_frame_index is not None and 0 <= grasp_frame_index < len(eef_yaw_traj):
            eef_yaw_at_grasp = eef_yaw_traj[grasp_frame_index]

        env_success = bool(info["success"]) if isinstance(info["success"], (bool, np.bool_)) else bool(np.array(info["success"]).item())

        result = {
            "episode_id": int(episode + 1),
            "seed": int(episode + base_seed),
            "metrics": {
                "env_success": env_success,
            },
            "mr_eval": {
                "mr_type": args.mr_type,
                "pair_key": f"{episode + base_seed}",
                "expected_delta_yaw_deg": 45.0,
                "grasp_frame_index": grasp_frame_index,
                "initial_cube_yaw_deg": initial_cube_yaw_deg,
                "initial_src_cube_yaw_deg": initial_cube_yaw_deg,
                "initial_dst_cube_yaw_deg": initial_dst_cube_yaw_deg,
                "initial_src_cube_quat": initial_src_cube_quat,
                "initial_dst_cube_quat": initial_dst_cube_quat,
                "eef_yaw_at_grasp": eef_yaw_at_grasp,
                "goal_point": goal_point,
                "gripper_fully_closed_frame_index": gripper_fully_closed_frame_index,
                "gripper_fully_opened_frame_index": gripper_fully_opened_frame_index,
            },
            "data_availability": {
                "eef_yaw_deg_measured": eef_yaw_valid > 0,
                "cube_yaw_deg_measured_or_inferred": cube_yaw_valid > 0,
                "cube_yaw_inferred_from_env_id": inferred_cube_yaw_deg is not None,
                "grasp_frame_index_available": grasp_frame_index is not None,
                "initial_cube_yaw_available": initial_cube_yaw_deg is not None,
                "goal_point_available": goal_point is not None,
            },
            "trajectory": {
                "total_steps": int(global_steps),
                "eef_path": eef_arr.tolist(),
                "gripper_width": gripper_width_traj,
                "gripper_finger_qpos": gripper_finger_qpos_traj,
                "gripper_action_cmd": gripper_action_cmd_traj,
                "cube_pos": cube_pos_traj,
                "src_cube_pos": src_cube_pos_traj,
                "dst_cube_pos": dst_cube_pos_traj,
                "goal_point": goal_point,
                "eef_yaw_deg": eef_yaw_traj,
                "cube_yaw_deg": cube_yaw_traj,
                "src_cube_yaw_deg": src_cube_yaw_traj,
                "dst_cube_yaw_deg": dst_cube_yaw_traj,
                "src_cube_quat": src_cube_quat_traj,
                "dst_cube_quat": dst_cube_quat_traj,
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
print(f"Success rate: {success_rate}%")
run_config["summary"] = {
    "env_id": env_id,
    "total_episodes": int(total_episodes),
    "success_count": int(success_count),
    "success_rate": float(success_rate),
    "success_episode_ids": success_episode_ids,
}
with open(run_config_path, "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)
