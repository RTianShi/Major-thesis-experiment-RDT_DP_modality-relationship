import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("StackCubeVisualDebunking-v1", max_episode_steps=50)
class StackCubeVisualDebunkingEnv(StackCubeEnv):
    """StackCube variant with a loud visual background and distant distractors."""

    def __init__(self, *args, distractor_scale=1.0, **kwargs):
        self.distractor_scale = float(distractor_scale)
        self.checkerboard_panel = None
        self.yellow_sphere = None
        self.blue_cylinder = None
        self._blue_point_light = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._build_checkerboard_panel()
        self._build_far_distractors()
        self._configure_lighting()

    def _build_checkerboard_panel(self):
        builder = self.scene.create_actor_builder()
        tile_size = 0.055
        tile_half = tile_size / 2.0
        tile_thickness = 0.0005
        start_x = -tile_size * 4 + tile_half
        start_y = -tile_size * 4 + tile_half
        colors = (
            [0.94, 0.94, 0.94, 1.0],
            [0.08, 0.08, 0.08, 1.0],
        )
        for ix in range(8):
            for iy in range(8):
                color = colors[(ix + iy) % 2]
                x = start_x + ix * tile_size
                y = start_y + iy * tile_size
                builder.add_box_visual(
                    half_size=[tile_half, tile_half, tile_thickness],
                    material=sapien.render.RenderMaterial(base_color=color),
                    pose=sapien.Pose(p=[x, y, tile_thickness]),
                )
        self.checkerboard_panel = builder.build_kinematic(name="checkerboard_panel")
        self.checkerboard_panel.set_pose(sapien.Pose())

    def _build_far_distractors(self):
        sphere_radius = 0.022 * self.distractor_scale
        self.yellow_sphere = actors.build_sphere(
            self.scene,
            radius=sphere_radius,
            color=[0.98, 0.82, 0.12, 1.0],
            name="yellow_sphere_far",
            initial_pose=sapien.Pose(p=[0, 0, sphere_radius]),
        )

        cylinder_radius = 0.014 * self.distractor_scale
        cylinder_half_length = 0.032 * self.distractor_scale
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=cylinder_radius,
            half_length=cylinder_half_length,
        )
        builder.add_cylinder_visual(
            radius=cylinder_radius,
            half_length=cylinder_half_length,
            material=sapien.render.RenderMaterial(base_color=[0.12, 0.38, 0.95, 1.0]),
        )
        self.blue_cylinder = builder.build(name="blue_cylinder_far")
        self.blue_cylinder.set_pose(
            sapien.Pose(p=[0, 0, cylinder_radius + cylinder_half_length])
        )

    def _configure_lighting(self):
        scene = self.scene
        scene.set_ambient_light([0.05, 0.08, 0.18])
        try:
            self._blue_point_light = scene.add_point_light(
                position=[0.05, 0.05, 0.65],
                color=[0.25, 0.4, 1.0],
                shadow=False,
            )
        except Exception:
            self._blue_point_light = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            sphere_xyz = torch.zeros((b, 3), device=self.device)
            sphere_xyz[:, 0] = -0.24
            sphere_xyz[:, 1] = 0.31
            sphere_xyz[:, 2] = 0.022 * self.distractor_scale
            self.yellow_sphere.set_pose(Pose.create_from_pq(sphere_xyz))

            cylinder_xyz = torch.zeros((b, 3), device=self.device)
            cylinder_xyz[:, 0] = 0.24
            cylinder_xyz[:, 1] = 0.31
            cylinder_xyz[:, 2] = 0.014 * self.distractor_scale + 0.032 * self.distractor_scale
            self.blue_cylinder.set_pose(Pose.create_from_pq(cylinder_xyz))
