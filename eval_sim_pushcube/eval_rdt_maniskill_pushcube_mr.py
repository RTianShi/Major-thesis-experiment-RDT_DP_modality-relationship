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

from eval_sim_pushcube.mr import get_lang, get_vision, get_proprio  # registers MRs
from eval_sim_pushcube.env_mr import get_env  # registers PushCube env MRs
from eval_sim_pushcube.custom_envs import *  # noqa: F401,F403  # registers pushcube custom envs


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help=f"Environment to run motion planning solver on. ")
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
        default="eef_traj/PushCube",
        help="Base directory to save end-effector trajectories. The default creates a unique timestamped subfolder under eef_traj/PushCube. Use 'auto' for the legacy root-level auto naming.",
    )
    parser.add_argument("--mr-type", type=str, default=None, help="Mutation type label for JSON output.")
    parser.add_argument("--mr-config", type=str, default="configs/mr_eval_pushcube.yaml", help="Path to MR config YAML.")
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
    pos = None
    if hasattr(obj, "pose"):
        pos = np.array(obj.pose.p, dtype=np.float32)
    elif hasattr(obj, "get_pose"):
        pos = np.array(obj.get_pose().p, dtype=np.float32)
    if pos is None:
        return None
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim == 0:
        return None
    pos = pos.reshape(-1)
    if pos.size < 3:
        padded = np.full(3, np.nan, dtype=np.float32)
        padded[:pos.size] = pos
        return padded
    return pos[:3].copy()


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

    for key in ["task_obj", "task_object", "obj", "object", "_obj", "cube"]:
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


def _current_tcp_to_task_distance(env):
    task_actor = None
    for key in ["task_obj", "task_object", "obj", "object", "_obj", "cube"]:
        candidate = getattr(env.unwrapped, key, None)
        if candidate is not None:
            task_actor = candidate
            break
    agent = getattr(env.unwrapped, "agent", None)
    if task_actor is None or agent is None or not hasattr(agent, "tcp"):
        return float("nan")
    try:
        tcp_pos = np.asarray(agent.tcp.pose.p, dtype=np.float32).reshape(-1)
        task_pos = np.asarray(task_actor.pose.p, dtype=np.float32).reshape(-1)
    except Exception:
        return float("nan")
    if tcp_pos.size < 3 or task_pos.size < 3:
        return float("nan")
    return float(np.linalg.norm(tcp_pos[:3] - task_pos[:3]))


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

default_pushcube_traj_root = os.path.join("eef_traj", "PushCube")
if args.traj_dir in {"auto", default_pushcube_traj_root}:
  
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = _slugify(args.env_id)
    mr_type = getattr(args, "mr_type", None)
    base_dir = "eef_traj" if args.traj_dir == "auto" else default_pushcube_traj_root
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
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it.",
    "PushCubeYellowCube-v1": "Push and move the yellow star-prism to a goal region in front of it.",
    "PushCubeDCRB2-v1": "Push and move the yellow cube to a goal region in front of it.",
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

    img = _to_uint8_rgb(env.render())
    for _ in range(vision_history_len - 1):
        obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs['agent']['qpos'][:, :-1]

    global_steps = 0
    video_frames = []
    eef_traj = []
    eef_vel_traj = []
    joint_torques_traj = []
    cube_pos_traj = []
    src_cube_pos_traj = []
    dst_cube_pos_traj = []
    target_pos_traj = []
    tcp_to_obj_distance_traj = []
    is_contact_traj = []
    contact_frame_index = None
    contact_eef_pos = None
    gripper_width_traj = []
    gripper_finger_qpos_traj = []
    gripper_action_cmd_traj = []

    initial_cube_pos = None if cube_pos is None else [float(x) for x in cube_pos.reshape(-1)[:3].tolist()]
    initial_dst_cube_pos = None if goal_pos is None else [float(x) for x in goal_pos.reshape(-1)[:3].tolist()]
    initial_cube_z = float(initial_cube_pos[2]) if initial_cube_pos is not None and len(initial_cube_pos) >= 3 else None
    prev_cube_goal_distance = None
    last_known_target_pos = initial_dst_cube_pos

    control_freq = getattr(env.unwrapped, "control_freq", None)
    if control_freq is None:
        control_freq = float(args.video_fps)
    last_eef_xyz = None

    success_time = 0
    done = False
    mr_cfg["vision"].pop("_observed_grasp_step", None)
    mr_cfg["vision"].pop("_stacking_blindness_active", None)
    mr_cfg["vision"].pop("_transient_mask_start_step", None)
    mr_cfg["proprio"].pop("_phantom_fired_step", None)

    while global_steps < MAX_EPISODE_STEPS and not done:
        curr_cube_pos, curr_goal_pos, _, _ = _get_cube_goal_xyz(env)
        if curr_cube_pos is None or curr_goal_pos is None:
            cube_goal_distance = float("nan")
        else:
            cube_goal_distance = float(np.linalg.norm(curr_cube_pos - curr_goal_pos))
        previous_cube_goal_distance = mr_cfg["proprio"].get("_previous_cube_goal_distance", None)
        is_grasped = _current_is_grasped(env)
        tcp_to_obj_distance = _current_tcp_to_task_distance(env)
        observed_grasp_step = mr_cfg["vision"].get("_observed_grasp_step", None)
        if is_grasped and observed_grasp_step is None:
            observed_grasp_step = global_steps
            mr_cfg["vision"]["_observed_grasp_step"] = observed_grasp_step

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
            "tcp_to_obj_distance": tcp_to_obj_distance,
            "step_index": global_steps,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "control_freq": float(getattr(env.unwrapped, "control_freq", 20.0)),
            "is_grasped": is_grasped,
            "observed_grasp_step": observed_grasp_step,
        }
        images = vis_mut(images, mr_cfg["vision"])

        cube_height = float(curr_cube_pos[2]) if curr_cube_pos is not None else float("nan")
        lift_height_threshold = float(mr_cfg["proprio"].get("lift_height_threshold", 0.02))
        cube_lifted = (
            is_grasped
            and np.isfinite(cube_height)
            and np.isfinite(initial_cube_height)
            and cube_height > initial_cube_height + lift_height_threshold
        )
        mr_cfg["proprio"]["runtime"] = {
            "is_grasped": is_grasped,
            "tcp_to_obj_distance": tcp_to_obj_distance,
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
            current_cube_xyz, current_goal_xyz, _, _ = _get_cube_goal_xyz(env)
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

            gripper_cmd = float(action[-1]) if getattr(action, "shape", None) is not None and action.shape[0] > 0 else None
            gripper_action_cmd_traj.append(gripper_cmd)

            gripper_width, gripper_finger_qpos = _extract_gripper_width(obs)
            gripper_width_traj.append(gripper_width)
            gripper_finger_qpos_traj.append(gripper_finger_qpos)

            cube_pos_traj.append(current_cube_pos)
            src_cube_pos_traj.append(current_src_cube_pos)
            dst_cube_pos_traj.append(current_dst_cube_pos)
            target_pos_traj.append(current_dst_cube_pos)
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

        eef_arr = np.stack(eef_traj, axis=0)
        diffs = np.diff(eef_arr, axis=0)
        total_path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(eef_arr) > 1 else 0.0
        env_success = bool(info["success"]) if isinstance(info["success"], (bool, np.bool_)) else bool(
            np.array(info["success"]).item()
        )

        grasp_frame_index = contact_frame_index
        mr_eval_extra = {}
        if "ltsep1" in str(mr_cfg.get("proprio", {}).get("type", "") or "").strip().lower().replace("_", "-"):
            mr_eval_extra["proprio_phantom_fired_steps"] = mr_cfg.get("proprio", {}).get("_phantom_fired_steps")

        result = {
            "episode_id": int(episode + 1),
            "seed": int(episode + base_seed),
            "metrics": {
                "env_success": env_success,
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
