import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeEmptyGrasp-v1", max_episode_steps=50)
class PickCubeEmptyGraspEnv(PickCubeEnv):
    def __init__(self, *args, empty_grasp_delay_steps=5, **kwargs):
        self.empty_grasp_triggered = False
        self.empty_grasp_delay_steps = int(empty_grasp_delay_steps)
        self._grasp_hold_counter = None
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.empty_grasp_triggered = False
        self._grasp_hold_counter = None
        self._set_actor_visibility(self.cube, 1)

    def _set_actor_visibility(self, actor, visibility: int):
        for obj in getattr(actor, "_objs", []):
            render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            render_body.visibility = visibility

    def _remove_real_cube(self):
        cube_pos = self.cube.pose.p.clone()
        cube_quat = self.cube.pose.q.clone()
        if cube_pos.ndim == 1:
            cube_pos = cube_pos.unsqueeze(0)
            cube_quat = cube_quat.unsqueeze(0)
        cube_pos = cube_pos.clone()
        cube_pos[:, 2] = -10.0
        self.cube.set_pose(Pose.create_from_pq(cube_pos, cube_quat))
        self._set_actor_visibility(self.cube, 0)
        self.scene.update_render(
            update_sensors=True,
            update_human_render_cameras=True,
        )

    def _trigger_empty_grasp(self):
        if self.empty_grasp_triggered:
            return
        self.empty_grasp_triggered = True
        self._remove_real_cube()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        is_grasped = bool(np.array(info.get("is_grasped", False)).item())
        if is_grasped and not self.empty_grasp_triggered:
            if self._grasp_hold_counter is None:
                self._grasp_hold_counter = max(self.empty_grasp_delay_steps, 0)
            elif self._grasp_hold_counter <= 0:
                self._trigger_empty_grasp()
            else:
                self._grasp_hold_counter -= 1
        elif not is_grasped and not self.empty_grasp_triggered:
            self._grasp_hold_counter = None
        return obs, reward, terminated, truncated, info
