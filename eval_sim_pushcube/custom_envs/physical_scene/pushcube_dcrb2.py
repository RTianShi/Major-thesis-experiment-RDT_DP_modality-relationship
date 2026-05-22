from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose


@register_env("PushCubeDCRB2-v1", max_episode_steps=50)
class PushCubeDCRB2Env(PushCubeEnv):
    """PushCube variant with a distractor cube and a yellow target cube."""

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.06, 0.0, self.cube_half_size]),
        )

        self.yellow_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([245, 208, 66, 255]) / 255,
            name="yellow_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.06, 0.0, self.cube_half_size]),
        )
        self.obj = self.yellow_cube

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0.0, 0.0, 1e-3]),
        )

        self.task_obj = self.yellow_cube

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)

            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.06 - 0.09
            cube_xyz[..., 1] = torch.rand((b,), device=self.device) * 0.08 - 0.16
            cube_xyz[..., 2] = self.cube_half_size

            yellow_cube_xyz = cube_xyz.clone()
            yellow_cube_xyz[..., 0] = torch.clamp(cube_xyz[..., 0] + 0.10, min=-0.02, max=0.10)
            yellow_cube_xyz[..., 1] = torch.clamp(cube_xyz[..., 1] + 0.20, min=0.04, max=0.18)
            yellow_cube_xyz[..., 2] = self.cube_half_size

            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=[1, 0, 0, 0]))
            self.yellow_cube.set_pose(Pose.create_from_pq(p=yellow_cube_xyz, q=[1, 0, 0, 0]))

            target_region_xyz = yellow_cube_xyz.clone()
            target_region_xyz[..., 0] = yellow_cube_xyz[..., 0] + 0.10 + self.goal_radius
            target_region_xyz[..., 1] = 0.5 * (cube_xyz[..., 1] + yellow_cube_xyz[..., 1])
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.task_obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
            < self.goal_radius
        ) & (self.task_obj.pose.p[..., 2] < self.cube_half_size + 5e-3)
        return {"success": is_obj_placed}

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(goal_pos=self.goal_region.pose.p, obj_pose=self.task_obj.pose.raw_pose)
        return obs

    def compute_dense_reward(self, obs: Any, action, info: dict):
        tcp_push_pose = Pose.create_from_pq(
            p=self.task_obj.pose.p + torch.tensor([-self.cube_half_size - 0.02, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(self.task_obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        z_deviation = torch.abs(self.task_obj.pose.p[..., 2] - self.cube_half_size)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        reward += place_reward * z_reward * reached

        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
