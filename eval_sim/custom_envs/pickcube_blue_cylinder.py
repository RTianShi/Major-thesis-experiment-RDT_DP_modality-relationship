import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeBlueCylinder-v1", max_episode_steps=50)
class PickCubeBlueCylinderEnv(PickCubeEnv):
    def __init__(
        self,
        *args,
        cylinder_radius_scale=0.75,
        cylinder_half_length_scale=1.0,
        **kwargs,
    ):
        self.cylinder_radius_scale = float(cylinder_radius_scale)
        self.cylinder_half_length_scale = float(cylinder_half_length_scale)
        self.cylinder_radius = None
        self.cylinder_half_length = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cylinder_radius = float(self.cube_half_size) * self.cylinder_radius_scale
        self.cylinder_half_length = (
            float(self.cube_half_size) * self.cylinder_half_length_scale
        )
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_cylinder_collision(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
        )
        actor_builder.add_cylinder_visual(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),
        )
        self.cube = actor_builder.build(name="blue_cylinder")
        self.cube.set_pose(
            sapien.Pose(p=[0, 0, self.cylinder_half_length + self.cylinder_radius])
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

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cylinder_half_length + self.cylinder_radius
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
