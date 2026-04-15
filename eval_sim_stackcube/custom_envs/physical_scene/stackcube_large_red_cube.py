from typing import Any

import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StackCubeLargeRedCube-v1", max_episode_steps=50)
class StackCubeLargeRedCubeEnv(StackCubeEnv):
    """StackCube variant with a 1.5x larger movable red cube."""

    def __init__(
        self,
        *args,
        red_cube_scale=1.5,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.red_cube_scale = float(red_cube_scale)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cubeA_half_size = None
        self.cubeB_half_size = None
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        red_half = 0.02 * self.red_cube_scale
        green_half = 0.02
        self.cubeA_half_size = common.to_tensor([red_half] * 3, device=self.device)
        self.cubeB_half_size = common.to_tensor([green_half] * 3, device=self.device)
        self.cube_half_size = self.cubeA_half_size
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=red_half,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, red_half]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=green_half,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, green_half]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            cube_a_xyz = torch.zeros((b, 3), device=self.device)
            cube_b_xyz = torch.zeros((b, 3), device=self.device)
            cube_a_xyz[:, 2] = self.cubeA_half_size[2]
            cube_b_xyz[:, 2] = self.cubeB_half_size[2]

            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            origin_xy = torch.rand((b, 2), device=self.device) * 0.16 - 0.08
            radius_a = torch.linalg.norm(self.cubeA_half_size[:2]) + 0.001
            radius_b = torch.linalg.norm(self.cubeB_half_size[:2]) + 0.001
            cube_a_xy = origin_xy + sampler.sample(radius_a, 100)
            cube_b_xy = origin_xy + sampler.sample(radius_b, 100, verbose=False)

            cube_a_xyz[:, :2] = cube_a_xy
            cube_b_xyz[:, :2] = cube_b_xy

            cube_a_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False
            )
            cube_b_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=cube_a_xyz, q=cube_a_q))
            self.cubeB.set_pose(Pose.create_from_pq(p=cube_b_xyz, q=cube_b_q))

    def evaluate(self):
        pos_a = self.cubeA.pose.p
        pos_b = self.cubeB.pose.p
        offset = pos_a - pos_b
        xy_thresh = torch.linalg.norm(
            torch.maximum(self.cubeA_half_size[:2], self.cubeB_half_size[:2])
        ) + 0.005
        z_target = self.cubeA_half_size[2] + self.cubeB_half_size[2]
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= xy_thresh
        z_flag = torch.abs(offset[..., 2] - z_target) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pose = self.agent.tcp.pose.p
        cube_a_pos = self.cubeA.pose.p
        cube_a_to_tcp_dist = torch.linalg.norm(tcp_pose - cube_a_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cube_a_to_tcp_dist))

        cube_b_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [
                cube_b_pos[:, 0:2],
                (cube_b_pos[:, 2] + self.cubeA_half_size[2] + self.cubeB_half_size[2])[:, None],
            ]
        )
        cube_a_to_goal_dist = torch.linalg.norm(goal_xyz - cube_a_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cube_a_to_goal_dist)
        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]
        reward[info["success"]] = 8
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
