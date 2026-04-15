import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


class _PickCubeFixedCubeXYEnv(PickCubeEnv):
    def __init__(self, *args, fixed_cube_xy=(0.0, 0.0), **kwargs):
        self.fixed_cube_xy = (float(fixed_cube_xy[0]), float(fixed_cube_xy[1]))
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            cube_xyz = self.cube.pose.p.clone()
            if cube_xyz.ndim == 1:
                cube_xyz = cube_xyz.unsqueeze(0)
            cube_xyz = cube_xyz[:b].clone()
            cube_xyz[:, 0] = self.fixed_cube_xy[0]
            cube_xyz[:, 1] = self.fixed_cube_xy[1]
            cube_xyz[:, 2] = self.cube_half_size
            self.cube.set_pose(Pose.create_from_pq(cube_xyz))


@register_env("PickCubeCubeCenter-v1", max_episode_steps=50)
class PickCubeCubeCenterEnv(_PickCubeFixedCubeXYEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fixed_cube_xy=(0.0, 0.0), **kwargs)


@register_env("PickCubeCubeCorner011-v1", max_episode_steps=50)
class PickCubeCubeCorner011Env(_PickCubeFixedCubeXYEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fixed_cube_xy=(0.1, 0.1), **kwargs)
