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
    # StackCube samples the task cubes inside x in [-0.1, 0.1], y in [-0.2, 0.2].
    # Place distractors outside that band with margin so they stay on-table but avoid contact.
    pink_xy = torch.tensor([-0.16, 0.26], device=device)
    teal_xy = torch.tensor([0.16, 0.26], device=device)
    return pink_xy, teal_xy


@register_env("StackCubeColorDistractor-v1", max_episode_steps=50)
class StackCubeColorDistractorEnv(StackCubeEnv):
    """StackCube variant with pink and teal distractor cubes added to the table."""

    def __init__(self, *args, distractor_scale=1.0, **kwargs):
        self.distractor_scale = float(distractor_scale)
        self.distractor_half_size = None
        self.pink_cube = None
        self.teal_cube = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.distractor_half_size = _scalar_value(self.cube_half_size) * self.distractor_scale
        self.pink_cube = actors.build_cube(
            self.scene,
            half_size=self.distractor_half_size,
            color=[1.0, 0.45, 0.7, 1.0],
            name="pink_cube",
            initial_pose=sapien.Pose(p=[0, 0, self.distractor_half_size]),
        )
        self.teal_cube = actors.build_cube(
            self.scene,
            half_size=self.distractor_half_size,
            color=[0.0, 0.65, 0.65, 1.0],
            name="teal_cube",
            initial_pose=sapien.Pose(p=[0, 0, self.distractor_half_size]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            pink_xyz = torch.zeros((b, 3))
            teal_xyz = torch.zeros((b, 3))
            z_height = self.distractor_half_size
            pink_xy, teal_xy = _distractor_xy(self.device)

            pink_xyz[:, 0] = pink_xy[0]
            pink_xyz[:, 1] = pink_xy[1]
            pink_xyz[:, 2] = z_height

            teal_xyz[:, 0] = teal_xy[0]
            teal_xyz[:, 1] = teal_xy[1]
            teal_xyz[:, 2] = z_height

            self.pink_cube.set_pose(Pose.create_from_pq(pink_xyz))
            self.teal_cube.set_pose(Pose.create_from_pq(teal_xyz))
