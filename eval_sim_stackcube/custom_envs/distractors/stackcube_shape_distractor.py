import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


def _scalar_value(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].detach().cpu().item())
    return float(value)


def _distractor_xy(device):
    # Match StackCube's fixed sampling region and keep distractors outside it.
    cylinder_xy = torch.tensor([-0.16, 0.26], device=device)
    sphere_xy = torch.tensor([0.16, 0.26], device=device)
    return cylinder_xy, sphere_xy


@register_env("StackCubeShapeDistractor-v1", max_episode_steps=50)
class StackCubeShapeDistractorEnv(StackCubeEnv):
    """StackCube variant with a red cylinder and green sphere distractor."""

    def __init__(
        self,
        *args,
        cylinder_radius_scale=0.75,
        cylinder_half_length_scale=1.0,
        sphere_radius_scale=1.15,
        **kwargs,
    ):
        self.cylinder_radius_scale = float(cylinder_radius_scale)
        self.cylinder_half_length_scale = float(cylinder_half_length_scale)
        self.sphere_radius_scale = float(sphere_radius_scale)
        self.red_cylinder = None
        self.green_sphere = None
        self.cylinder_radius = None
        self.cylinder_half_length = None
        self.sphere_radius = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        cube_half_size = _scalar_value(self.cube_half_size)
        self.cylinder_radius = cube_half_size * self.cylinder_radius_scale
        self.cylinder_half_length = (
            cube_half_size * self.cylinder_half_length_scale
        )
        cylinder_builder = self.scene.create_actor_builder()
        cylinder_builder.add_cylinder_collision(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
        )
        cylinder_builder.add_cylinder_visual(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
            material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0]),
        )
        self.red_cylinder = cylinder_builder.build(name="red_cylinder")
        self.red_cylinder.set_pose(
            sapien.Pose(p=[0, 0, self.cylinder_radius + self.cylinder_half_length])
        )

        self.sphere_radius = cube_half_size * self.sphere_radius_scale
        self.green_sphere = actors.build_sphere(
            self.scene,
            radius=self.sphere_radius,
            color=[0.0, 1.0, 0.0, 1.0],
            name="green_sphere",
            initial_pose=sapien.Pose(p=[0, 0, self.sphere_radius]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            cylinder_xyz = torch.zeros((b, 3))
            sphere_xyz = torch.zeros((b, 3))
            cylinder_xy, sphere_xy = _distractor_xy(self.device)

            cylinder_xyz[:, 0] = cylinder_xy[0]
            cylinder_xyz[:, 1] = cylinder_xy[1]
            cylinder_xyz[:, 2] = self.cylinder_radius + self.cylinder_half_length

            sphere_xyz[:, 0] = sphere_xy[0]
            sphere_xyz[:, 1] = sphere_xy[1]
            sphere_xyz[:, 2] = self.sphere_radius

            self.red_cylinder.set_pose(Pose.create_from_pq(cylinder_xyz))
            self.green_sphere.set_pose(Pose.create_from_pq(sphere_xyz))
