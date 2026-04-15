"""
MR-JSAP-3: 目标显式协同变异 (Explicit Co-mutation)
视觉变异：将红色方块替换为蓝色方块，绿球（目标点）保持不变
"""

import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import sapien


@register_env("PickCubeBlueCube-v1", max_episode_steps=50)
class PickCubeBlueCubeEnv(PickCubeEnv):
    """MR-JSAP-3: 蓝色方块替代红色方块"""
    
    def _load_scene(self, options: dict):
        """加载场景，只创建蓝色方块，完全替代红色方块"""
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 创建蓝色方块（完全替代原红色方块）
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=[self.cube_half_size] * 3)
        actor_builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),  # 蓝色 RGBA
        )
        self.cube = actor_builder.build(name="blue_cube")
        self.cube.set_pose(sapien.Pose(p=[0, 0, self.cube_half_size]))

        # 绿球目标点保持不变
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],  # 绿色
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
        
        # 确保没有其他立方体/红色方块被创建
        # （完全覆盖父类的_load_scene实现）

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每个episode"""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 初始化蓝色方块位置
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # 初始化绿球目标点
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
