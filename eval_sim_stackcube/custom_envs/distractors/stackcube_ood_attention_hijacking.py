import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


def _scalar_value(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].detach().cpu().item())
    return float(value)


@register_env("StackCubeMugDistractor-v1", max_episode_steps=50)
class StackCubeMugDistractorEnv(StackCubeEnv):
    """StackCube variant with a patterned ceramic mug placed near the cubes."""

    def __init__(self, *args, mug_scale=1.0, **kwargs):
        self.mug_scale = float(mug_scale)
        self.mug_actor = None
        self.mug_outer_radius = None
        self.mug_height = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        cube_half_size = _scalar_value(self.cube_half_size[0])
        self.mug_outer_radius = cube_half_size * 0.95 * self.mug_scale
        self.mug_height = cube_half_size * 2.4 * self.mug_scale
        mug_half_length = self.mug_height / 2.0
        handle_half_size = cube_half_size * 0.18 * self.mug_scale
        handle_offset_x = self.mug_outer_radius * 1.45
        handle_offset_z = mug_half_length * 0.15
        stripe_half_thickness = cube_half_size * 0.03 * self.mug_scale

        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=self.mug_outer_radius,
            half_length=mug_half_length,
        )
        builder.add_cylinder_visual(
            radius=self.mug_outer_radius,
            half_length=mug_half_length,
            material=sapien.render.RenderMaterial(base_color=[0.97, 0.95, 0.9, 1.0]),
        )

        # Simple handle.
        builder.add_box_collision(
            half_size=[handle_half_size, handle_half_size, mug_half_length * 0.65],
            pose=sapien.Pose(p=[handle_offset_x, 0.0, handle_offset_z]),
        )
        builder.add_box_visual(
            half_size=[handle_half_size, handle_half_size, mug_half_length * 0.65],
            material=sapien.render.RenderMaterial(base_color=[0.97, 0.95, 0.9, 1.0]),
            pose=sapien.Pose(p=[handle_offset_x, 0.0, handle_offset_z]),
        )

        # Patterned stripes to increase saliency.
        stripe_materials = [
            sapien.render.RenderMaterial(base_color=[0.15, 0.35, 0.9, 1.0]),
            sapien.render.RenderMaterial(base_color=[0.95, 0.55, 0.15, 1.0]),
            sapien.render.RenderMaterial(base_color=[0.15, 0.7, 0.45, 1.0]),
        ]
        stripe_offsets = [-mug_half_length * 0.45, 0.0, mug_half_length * 0.45]
        stripe_width = self.mug_outer_radius * 1.2
        for stripe_z, material in zip(stripe_offsets, stripe_materials):
            builder.add_box_visual(
                half_size=[stripe_width, stripe_half_thickness, stripe_half_thickness],
                material=material,
                pose=sapien.Pose(p=[0.0, self.mug_outer_radius * 0.95, stripe_z]),
            )
            builder.add_box_visual(
                half_size=[stripe_half_thickness, stripe_width, stripe_half_thickness],
                material=material,
                pose=sapien.Pose(p=[self.mug_outer_radius * 0.95, 0.0, stripe_z]),
            )

        self.mug_actor = builder.build(name="patterned_ceramic_mug")
        self.mug_actor.set_pose(sapien.Pose(p=[0, 0, mug_half_length]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            mug_xyz = torch.zeros((b, 3))

            # Near the manipulated cubes, but outside the nominal spawn region.
            mug_xyz[:, 0] = 0.0
            mug_xyz[:, 1] = 0.27
            mug_xyz[:, 2] = self.mug_height / 2.0

            yaw = torch.full((b, 4), 0.0, device=self.device)
            yaw[:, 0] = 1.0
            self.mug_actor.set_pose(Pose.create_from_pq(mug_xyz, yaw))
