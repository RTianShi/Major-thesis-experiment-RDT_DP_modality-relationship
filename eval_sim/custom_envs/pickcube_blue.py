import torch
import sapien

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeBlueCube-v1", max_episode_steps=50)
class PickCubeBlueCubeEnv(PickCubeEnv):
    def __init__(
        self,
        *args,
        blue_cube_scale=1.0,
        blue_cube_offset=(0.15, 0.15),
        **kwargs,
    ):
        self.blue_cube_scale = float(blue_cube_scale)
        self.blue_cube_offset = (float(blue_cube_offset[0]), float(blue_cube_offset[1]))
        self.blue_cube_half_size = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.blue_cube_half_size = float(self.cube_half_size) * self.blue_cube_scale
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.blue_cube_half_size,
            color=[0, 0, 1, 1],
            name="blue_cube",
            initial_pose=sapien.Pose(p=[0, 0, self.blue_cube_half_size]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            offset = torch.tensor(
                [self.cube_spawn_half_size * 1, -self.cube_spawn_half_size * 4],
                device=self.device,
            )
            xyz[:, 0] = self.cube_spawn_center[0] + offset[0]
            xyz[:, 1] = self.cube_spawn_center[1] + offset[1]
            xyz[:, 2] = self.blue_cube_half_size
            self.blue_cube.set_pose(Pose.create_from_pq(xyz))
