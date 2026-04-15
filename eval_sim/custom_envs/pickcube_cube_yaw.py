import math

import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


class _PickCubeFixedCubeYawEnv(PickCubeEnv):
    def __init__(self, *args, fixed_cube_yaw_deg=0.0, **kwargs):
        self.fixed_cube_yaw_deg = float(fixed_cube_yaw_deg)
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            cube_xyz = self.cube.pose.p.clone()
            if cube_xyz.ndim == 1:
                cube_xyz = cube_xyz.unsqueeze(0)
            b = len(env_idx)
            cube_xyz = cube_xyz[:b].clone()

            yaw_rad = math.radians(self.fixed_cube_yaw_deg)
            quat = torch.zeros((b, 4), device=self.device)
            quat[:, 0] = math.cos(yaw_rad / 2.0)
            quat[:, 3] = math.sin(yaw_rad / 2.0)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, quat))


@register_env("PickCubeCubeYaw000-v1", max_episode_steps=50)
class PickCubeCubeYaw000Env(_PickCubeFixedCubeYawEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fixed_cube_yaw_deg=0.0, **kwargs)


@register_env("PickCubeCubeYaw045-v1", max_episode_steps=50)
class PickCubeCubeYaw045Env(_PickCubeFixedCubeYawEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fixed_cube_yaw_deg=45.0, **kwargs)
