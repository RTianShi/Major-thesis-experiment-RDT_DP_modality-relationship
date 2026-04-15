import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env


@register_env("PickCubeInvisibleHeld-v1", max_episode_steps=50)
class PickCubeInvisibleHeldEnv(PickCubeEnv):
    def __init__(self, *args, hide_height_threshold=0.05, **kwargs):
        self.hide_height_threshold = float(hide_height_threshold)
        self.invisible_triggered = False
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.invisible_triggered = False
        self._set_actor_visibility(self.cube, 1)

    def _set_actor_visibility(self, actor, visibility: int):
        for obj in getattr(actor, "_objs", []):
            render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            render_body.visibility = visibility

    def _cube_height(self):
        pos = self.cube.pose.p
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        pos = np.array(pos, dtype=np.float32)
        if pos.ndim > 1:
            pos = pos[0]
        return float(pos[2])

    def _trigger_invisible(self):
        if self.invisible_triggered:
            return
        self.invisible_triggered = True
        self._set_actor_visibility(self.cube, 0)
        self.scene.update_render(
            update_sensors=True,
            update_human_render_cameras=True,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        is_grasped = bool(np.array(info.get("is_grasped", False)).item())
        if not self.invisible_triggered and is_grasped and self._cube_height() > self.hide_height_threshold:
            self._trigger_invisible()
        return obs, reward, terminated, truncated, info
