import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


class _PickCubeSceneTranslationEnv(PickCubeEnv):
    def __init__(self, *args, scene_translation_xy=(0.0, 0.0), **kwargs):
        self.scene_translation_xy = (
            float(scene_translation_xy[0]),
            float(scene_translation_xy[1]),
        )
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            dx, dy = self.scene_translation_xy

            cube_xyz = self.cube.pose.p.clone()
            if cube_xyz.ndim == 1:
                cube_xyz = cube_xyz.unsqueeze(0)
            cube_xyz = cube_xyz[:b].clone()
            cube_xyz[:, 0] += dx
            cube_xyz[:, 1] += dy
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, self.cube.pose.q[:b].clone()))

            goal_xyz = self.goal_site.pose.p.clone()
            if goal_xyz.ndim == 1:
                goal_xyz = goal_xyz.unsqueeze(0)
            goal_xyz = goal_xyz[:b].clone()
            goal_xyz[:, 0] += dx
            goal_xyz[:, 1] += dy
            self.goal_site.set_pose(
                Pose.create_from_pq(goal_xyz, self.goal_site.pose.q[:b].clone())
            )


@register_env("PickCubeSceneTrans000-v1", max_episode_steps=50)
class PickCubeSceneTrans000Env(_PickCubeSceneTranslationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, scene_translation_xy=(0.0, 0.0), **kwargs)


@register_env("PickCubeSceneTrans04N04-v1", max_episode_steps=50)
class PickCubeSceneTrans04N04Env(_PickCubeSceneTranslationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, scene_translation_xy=(0.04, -0.04), **kwargs)
