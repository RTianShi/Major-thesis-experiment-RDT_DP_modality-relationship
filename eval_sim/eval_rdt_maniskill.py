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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. ")
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
    parser.add_argument("--text-encoder-device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for T5 text encoder only.")
    parser.add_argument("--show", action="store_true", help="Show real-time rendering with OpenCV window.")
    parser.add_argument("--save-video", action="store_true", help="Save episode videos to disk.")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save videos.")
    parser.add_argument("--video-fps", type=int, default=25, help="FPS for saved videos.")
    parser.add_argument("--traj-dir", type=str, default="eef_traj", help="Directory to save end-effector trajectories.")
    parser.add_argument("--mr-type", type=str, default="unknown", help="Mutation type label for JSON output.")
    return parser.parse_args()

import random
import os


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

    # Try common env attributes first.
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

# set cuda 
args = parse_args()
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
pretrained_text_encoder_name_or_path =  "google/t5-v1_1-xxl"
pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
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
text_embed_path = f"text_embed_{env_id}.pt"
torch.save(text_embed, text_embed_path)
print(f"Saved text embedding to {text_embed_path}")

MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  
success_count = 0  

base_seed = 20241201
import tqdm
for episode in tqdm.trange(total_episodes):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + base_seed)
    policy.reset()

    cube_pos, goal_pos, cube_name, goal_name = _get_cube_goal_xyz(env)
    if cube_pos is None:
        cube_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    if goal_pos is None:
        goal_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    red_cube_initial = cube_pos.copy()
    green_goal = goal_pos.copy()

    img = _to_uint8_rgb(env.render())
    obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs['agent']['qpos'][:, :-1]

    global_steps = 0
    video_frames = []
    eef_traj = []

    success_time = 0
    done = False

    while global_steps < MAX_EPISODE_STEPS and not done:
        image_arrs = []
        for window_img in obs_window:
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        images = [Image.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
        # Take 8 steps since RDT is trained to predict interpolated 64 steps(actual 14 steps)
        actions = actions[::4, :]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            eef_xyz = env.unwrapped.agent.tcp.pose.p  # (x, y, z)
            img = _to_uint8_rgb(env.render())
            obs_window.append(img)
            proprio = obs['agent']['qpos'][:, :-1]
            eef_xyz = env.unwrapped.agent.tcp.pose.p
            eef_traj.append(np.array(eef_xyz, dtype=np.float32))
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
        # Final cube position (after episode ends)
        final_cube_pos, _, _, _ = _get_cube_goal_xyz(env)
        if final_cube_pos is None:
            final_cube_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        eef_arr = np.stack(eef_traj, axis=0)
        diffs = np.diff(eef_arr, axis=0)
        total_path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(eef_arr) > 1 else 0.0

        # Compute metrics
        env_success = bool(info["success"]) if isinstance(info["success"], (bool, np.bool_)) else bool(np.array(info["success"]).item())
        if np.any(np.isnan(final_cube_pos)) or np.any(np.isnan(green_goal)):
            real_target_distance = float("nan")
            semantic_success = False
        else:
            real_target_distance = float(np.linalg.norm(final_cube_pos - green_goal))
            semantic_success = real_target_distance < 0.025

        result = {
            "episode_id": int(episode + 1),
            "seed": int(episode + base_seed),
            "mr_type": getattr(args, "mr_type", "unknown"),
            "metrics": {
                "env_success": env_success,
                "real_target_distance": real_target_distance,
                "semantic_success": bool(semantic_success),
            },
            "positions": {
                "red_cube_initial": red_cube_initial.tolist(),
                "green_goal": green_goal.tolist(),
                "blue_cube_initial": None,
                "red_cube_final": final_cube_pos.tolist(),
            },
            "trajectory": {
                "total_steps": int(global_steps),
                "eef_path": eef_arr.tolist(),
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
