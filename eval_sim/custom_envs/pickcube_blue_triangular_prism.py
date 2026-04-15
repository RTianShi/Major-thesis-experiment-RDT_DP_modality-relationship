from pathlib import Path

import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


ASSET_DIR = Path(__file__).resolve().parent / "assets"
TRIANGULAR_PRISM_MESH = ASSET_DIR / "blue_triangular_prism.obj"


@register_env("PickCubeBlueTriangularPrism-v1", max_episode_steps=50)
class PickCubeBlueTriangularPrismEnv(PickCubeEnv):
    def __init__(self, *args, prism_scale=1.0, **kwargs):
        self.prism_scale = float(prism_scale)
        self.prism_half_height = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        mesh_scale = [self.cube_half_size * self.prism_scale] * 3
        self.prism_half_height = float(self.cube_half_size) * self.prism_scale

        actor_builder = self.scene.create_actor_builder()
        if hasattr(actor_builder, "add_convex_collision_from_file"):
            actor_builder.add_convex_collision_from_file(
                filename=str(TRIANGULAR_PRISM_MESH),
                scale=mesh_scale,
            )
        elif hasattr(actor_builder, "add_collision_from_file"):
            actor_builder.add_collision_from_file(
                filename=str(TRIANGULAR_PRISM_MESH),
                scale=mesh_scale,
            )
        else:
            raise AttributeError(
                "ActorBuilder does not support mesh collision loading for the triangular prism MR."
            )
        actor_builder.add_visual_from_file(
            filename=str(TRIANGULAR_PRISM_MESH),
            scale=mesh_scale,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),
        )
        self.cube = actor_builder.build(name="blue_triangular_prism")
        self.cube.set_pose(sapien.Pose(p=[0, 0, self.prism_half_height]))

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
            xyz[:, 2] = self.prism_half_height
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
