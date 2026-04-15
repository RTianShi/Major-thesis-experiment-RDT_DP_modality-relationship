import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


class _PickCubeGoalHeightEnv(PickCubeEnv):
    def __init__(self, *args, fixed_goal_z=0.05, **kwargs):
        self.fixed_goal_z = float(fixed_goal_z)
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            goal_xyz = self.goal_site.pose.p.clone()
            if goal_xyz.ndim == 1:
                goal_xyz = goal_xyz.unsqueeze(0)
            goal_xyz = goal_xyz[:b].clone()
            goal_xyz[:, 2] = self.fixed_goal_z
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


@register_env("PickCubeGoalZ005-v1", max_episode_steps=50)
class PickCubeGoalZ005Env(_PickCubeGoalHeightEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fixed_goal_z=0.05, **kwargs)


@register_env("PickCubeGoalZ028-v1", max_episode_steps=50)
class PickCubeGoalZ028Env(_PickCubeGoalHeightEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fixed_goal_z=0.28, **kwargs)
