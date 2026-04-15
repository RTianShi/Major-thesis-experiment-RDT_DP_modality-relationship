import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeRedSphereBlueCube-v1", max_episode_steps=50)
class PickCubeRedSphereBlueCubeEnv(PickCubeEnv):
    def __init__(self, *args, red_sphere_scale=1.15, **kwargs):
        self.red_sphere_scale = float(red_sphere_scale)
        self.red_sphere_radius = None
        self.red_sphere = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 0, 1, 1],
            name="blue_cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.red_sphere_radius = float(self.cube_half_size) * self.red_sphere_scale
        self.red_sphere = actors.build_sphere(
            self.scene,
            radius=self.red_sphere_radius,
            color=[1, 0, 0, 1],
            name="red_sphere",
            initial_pose=sapien.Pose(p=[0, 0, self.red_sphere_radius]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, 0] = self.cube_spawn_center[0]
            cube_xyz[:, 1] = self.cube_spawn_center[1] - self.cube_spawn_half_size * 1.2
            cube_xyz[:, 2] = self.cube_half_size
            self.cube.set_pose(Pose.create_from_pq(cube_xyz))

            sphere_xyz = torch.zeros((b, 3))
            sphere_xyz[:, 0] = self.cube_spawn_center[0]
            sphere_xyz[:, 1] = self.cube_spawn_center[1] + self.cube_spawn_half_size * 1.2
            sphere_xyz[:, 2] = self.red_sphere_radius
            self.red_sphere.set_pose(Pose.create_from_pq(sphere_xyz))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + cube_xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
