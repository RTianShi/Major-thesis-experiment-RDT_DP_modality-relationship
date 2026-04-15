import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeGhostLift-v1", max_episode_steps=50)
class PickCubeGhostLiftEnv(PickCubeEnv):
    def __init__(self, *args, **kwargs):
        self.ghost_cube = None
        self.ghost_triggered = False
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.ghost_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="ghost_cube",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, -10]),
        )
        self.ghost_cube.hide_visual()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.ghost_triggered = False
        self.ghost_cube.hide_visual()
        ghost_xyz = self.ghost_cube.pose.p.clone()
        if ghost_xyz.ndim == 1:
            ghost_xyz = ghost_xyz.unsqueeze(0)
        ghost_xyz = ghost_xyz[: len(env_idx)].clone()
        ghost_xyz[:, 2] = -10.0
        self.ghost_cube.set_pose(Pose.create_from_pq(ghost_xyz))
        self._set_actor_visibility(self.cube, 1)

    def _set_actor_visibility(self, actor, visibility: int):
        for obj in getattr(actor, "_objs", []):
            render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            render_body.visibility = visibility

    def _sync_ghost_to_tcp(self):
        tcp_pose = self.agent.tcp.pose
        tcp_pos = tcp_pose.p.clone()
        tcp_quat = tcp_pose.q.clone()
        self.ghost_cube.set_pose(Pose.create_from_pq(tcp_pos, tcp_quat))

    def _remove_real_cube_physics(self):
        cube_pos = self.cube.pose.p.clone()
        cube_quat = self.cube.pose.q.clone()
        if cube_pos.ndim == 1:
            cube_pos = cube_pos.unsqueeze(0)
            cube_quat = cube_quat.unsqueeze(0)
        cube_pos = cube_pos.clone()
        cube_pos[:, 2] = -10.0
        self.cube.set_pose(Pose.create_from_pq(cube_pos, cube_quat))
        self._set_actor_visibility(self.cube, 0)

    def _trigger_ghost_lift(self):
        if self.ghost_triggered:
            return
        self.ghost_triggered = True
        self._remove_real_cube_physics()
        self._sync_ghost_to_tcp()
        self.ghost_cube.show_visual()
        self.scene.update_render(
            update_sensors=True,
            update_human_render_cameras=True,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if bool(np.array(info.get("is_grasped", False)).item()):
            self._trigger_ghost_lift()
        if self.ghost_triggered:
            self._sync_ghost_to_tcp()
        return obs, reward, terminated, truncated, info
