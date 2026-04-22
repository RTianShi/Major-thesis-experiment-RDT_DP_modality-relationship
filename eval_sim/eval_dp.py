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
from eval_sim.mr import get_lang, get_vision, get_proprio  # registers MRs
from eval_sim.env_mr import get_env  # registers env MRs
from eval_sim.custom_envs import *  # noqa: F401,F403  # registers custom envs

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. ")
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
        default="/home/hjl/RoboticsDiffusionTransformer/eef_traj_dp/PickCube",
        help="Base directory to save end-effector trajectories. The default creates a unique timestamped subfolder under /home/hjl/RoboticsDiffusionTransformer/eef_traj_dp/PickCube.",
    )
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default=None, help="Alias of --video-dir for backward compatibility.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Random seed for the environment.")
    parser.add_argument("--mr-type", type=str, default=None, help="Mutation type label for JSON output.")
    parser.add_argument("--mr-config", type=str, default="configs/mr_eval.yaml", help="Path to MR config YAML.")

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
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
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

def _extract_cube_pos(obs, env):
    # 1) 优先从环境对象读（对 PickCube 系列最稳）
    unwrapped = getattr(env, "unwrapped", env)
    for name in ("cube", "obj", "object", "target_object", "source_object"):
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


default_pickcube_traj_root = "/home/hjl/RoboticsDiffusionTransformer/eef_traj_dp/PickCube"
if args.traj_dir == default_pickcube_traj_root:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = _slugify(args.env_id)
    mr_type = getattr(args, "mr_type", None)
    if mr_type:
        mr_tag = _slugify(mr_type)
        args.traj_dir = os.path.join(default_pickcube_traj_root, f"{env_tag}_{mr_tag}_{ts}")
    else:
        args.traj_dir = os.path.join(default_pickcube_traj_root, f"{env_tag}_{ts}")

os.makedirs(args.traj_dir, exist_ok=True)

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

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
import hydra
import dill

checkpoint_path = args.pretrained_path
print(f"Loading policy from {checkpoint_path}. Task is {task2lang[env_id]}")

run_config_path = os.path.join(args.traj_dir, "run_config.json")
run_config = {
    "args": vars(args),
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
    try:
        unwrapped = getattr(env, "unwrapped", env)
        for name in ("cube", "obj", "object", "target_object", "source_object"):
            actor = getattr(unwrapped, name, None)
            if actor is not None and hasattr(actor, "pose") and hasattr(actor.pose, "q"):
                q = np.asarray(actor.pose.q, dtype=np.float64).reshape(-1)
                if q.size >= 4:
                    w, x, y, z = q[:4]
                    n = np.linalg.norm([w, x, y, z])
                    if n > 1e-12:
                        w, x, y, z = w / n, x / n, y / n, z / n
                        siny_cosp = 2.0 * (w * z + x * y)
                        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                        return float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))
    except Exception:
        pass
    return None

def _infer_cube_yaw_from_env_id(env_id: str):
    m = re.search(r"Yaw(\d{3})", str(env_id))
    return float(int(m.group(1))) if m else None

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

def _detect_open_to_close_transition(cmds, open_thresh=0.1, close_thresh=-0.1):
    prev = None
    for i, c in enumerate(cmds):
        if c is None:
            continue
        c = float(c)
        if prev is not None and prev >= open_thresh and c <= close_thresh:
            return int(i)
        prev = c
    return None

for episode in tqdm.trange(total_episodes):
    
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + base_seed)
   
    img = _to_uint8_rgb(env.render())
    img_tensor = torch.as_tensor(img, device="cuda").float()
    proprio = obs['agent']['qpos'][:].cuda()
    proprio = (proprio - state_min) / (state_max - state_min) * 2 - 1
    obs_window.append({
        'agent_pos': proprio,
        "head_cam": img_tensor.permute(2, 0, 1).unsqueeze(0),
    })
    obs_window.append({
        'agent_pos': proprio,
        "head_cam": img_tensor.permute(2, 0, 1).unsqueeze(0),
    })

    global_steps = 0
    video_frames = []
    eef_traj = []
    gripper_width_traj = []
    gripper_finger_qpos_traj = []
    cube_pos_traj = []
    eef_yaw_traj = []  # 实时记录每步 yaw(度)
    cube_yaw_traj = []  # 新增：cube yaw 序列（可能含 None）
    done = False
    info = {"success": False}

    # 兜底推断（可能仍为 None）
    inferred_cube_yaw_deg = _infer_cube_yaw_from_env_id(env_id)

    # 新增：记录 reset 后（第一步动作前）的物体初始偏航角
    initial_cube_yaw_deg = _extract_cube_yaw_deg(obs, env)
    if initial_cube_yaw_deg is None:
        initial_cube_yaw_deg = inferred_cube_yaw_deg

    gripper_action_cmd_traj = []   # 每步夹爪控制指令（原始）
    open_thresh = 0.1
    close_thresh = -0.1

    while global_steps < MAX_EPISODE_STEPS and not done:
        obs = obs_window[-1]
        actions = policy.predict_action(obs)
        actions = actions['action_pred'].squeeze(0)
        actions = (actions + 1) / 2 * (action_max - action_min) + action_min
        actions = actions.detach().cpu().numpy()
        actions = actions[:8]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            gripper_cmd = float(action[-1]) if action.shape[0] > 0 else None

            obs, reward, terminated, truncated, info = env.step(action)
            global_steps += 1  # 改为每步都计数，避免仅 --show 时计数

            img = _to_uint8_rgb(env.render())
            img_tensor = torch.as_tensor(img, device="cuda").float()
            proprio = obs['agent']['qpos'][:].cuda()
            proprio = (proprio - state_min) / (state_max - state_min) * 2 - 1
            obs_window.append({
                'agent_pos': proprio,
                "head_cam": img_tensor.permute(2, 0, 1).unsqueeze(0),
            }) 
            eef_xyz = env.unwrapped.agent.tcp.pose.p
            eef_traj.append(np.array(eef_xyz, dtype=np.float32))
            step_index = len(eef_traj) - 1

            # 新增：每步获取末端执行器yaw
            eef_yaw = _extract_eef_yaw(env)

            # 仅记录原始序列，不在循环中判定抓取时刻
            gripper_action_cmd_traj.append(gripper_cmd)

            gripper_width, gripper_finger_qpos = _extract_gripper_width(obs)
            cube_pos = _extract_cube_pos(obs, env)
            cube_yaw_deg = _extract_cube_yaw_deg(obs, env)
            if cube_yaw_deg is None:
                cube_yaw_deg = inferred_cube_yaw_deg  # 兜底推断（可能仍为 None）

            if gripper_width is not None:
                gripper_width_traj.append(gripper_width)
            if gripper_finger_qpos is not None:
                gripper_finger_qpos_traj.append(gripper_finger_qpos)
            if cube_pos is not None:
                cube_pos_traj.append(cube_pos)
            cube_yaw_traj.append(cube_yaw_deg)
            eef_yaw_traj.append(eef_yaw)  # 保留：每步实时 yaw
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
        eef_yaw_valid = int(sum(v is not None for v in eef_yaw_traj))
        cube_yaw_valid = int(sum(v is not None for v in cube_yaw_traj))
        
        # 从原始序列统一推导（后处理）
        grasp_frame_index = _detect_open_to_close_transition(
            gripper_action_cmd_traj, open_thresh=open_thresh, close_thresh=close_thresh
        )
        gripper_fully_closed_frame_index = _first_index_le(gripper_action_cmd_traj, -0.95)
        gripper_fully_opened_frame_index = _first_index_ge(gripper_action_cmd_traj, 0.95)

        eef_yaw_at_grasp = None
        if grasp_frame_index is not None and 0 <= grasp_frame_index < len(eef_yaw_traj):
            eef_yaw_at_grasp = eef_yaw_traj[grasp_frame_index]

        result = {
            "episode_id": int(episode + 1),
            "seed": int(episode + base_seed),
            "metrics": {
                "env_success": bool(info["success"]),
            },
            "mr_eval": {
                "mr_type": args.mr_type,
                "pair_key": f"{episode + base_seed}",
                "expected_delta_yaw_deg": 45.0,
                "grasp_frame_index": grasp_frame_index,
                "initial_cube_yaw_deg": initial_cube_yaw_deg,  # 补回该字段
                "eef_yaw_at_grasp": eef_yaw_at_grasp,
                "gripper_fully_closed_frame_index": gripper_fully_closed_frame_index,
                "gripper_fully_opened_frame_index": gripper_fully_opened_frame_index,
            },
            "data_availability": {
                "eef_yaw_deg_measured": eef_yaw_valid > 0,
                "cube_yaw_deg_measured_or_inferred": cube_yaw_valid > 0,
                "cube_yaw_inferred_from_env_id": inferred_cube_yaw_deg is not None,
                "grasp_frame_index_available": grasp_frame_index is not None,
                "initial_cube_yaw_available": initial_cube_yaw_deg is not None,  # 新增
            },
            "trajectory": {
                "total_steps": int(global_steps),
                "eef_path": eef_arr.tolist(),
                "gripper_width": gripper_width_traj,
                "gripper_finger_qpos": gripper_finger_qpos_traj,
                "gripper_action_cmd": gripper_action_cmd_traj,  # 新增
                "cube_pos": cube_pos_traj,
                "eef_yaw_deg": eef_yaw_traj,
                "cube_yaw_deg": cube_yaw_traj,
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
