import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("StackCubeRemoveRedCube-v1", max_episode_steps=50)
class StackCubeRemoveRedCubeEnv(StackCubeEnv):
    """StackCube variant with the movable red cube removed from the scene."""

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        cube = self._get_red_cube_actor()
        if cube is None:
            raise ValueError("StackCubeRemoveRedCube-v1 could not locate the movable red cube actor")
        self._remove_actor_from_scene(cube)

    def _get_red_cube_actor(self):
        for attr in ("cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"):
            actor = getattr(self, attr, None)
            if actor is not None:
                return actor
        return None

    def _set_actor_visibility(self, actor, visibility: int):
        for obj in getattr(actor, "_objs", []):
            render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            render_body.visibility = visibility

    def _remove_actor_from_scene(self, actor):
        actor_pose = actor.pose
        actor_pos = actor_pose.p.clone()
        actor_quat = actor_pose.q.clone()
        if actor_pos.ndim == 1:
            actor_pos = actor_pos.unsqueeze(0)
            actor_quat = actor_quat.unsqueeze(0)
        actor_pos = actor_pos.clone()
        actor_pos[:, 2] = -10.0
        actor.set_pose(Pose.create_from_pq(actor_pos, actor_quat))
        self._set_actor_visibility(actor, 0)

        scene = self.scene
        if scene is not None and getattr(scene, "device", None) is not None and scene.device.type == "cuda":
            scene._gpu_apply_all()
            scene.px.gpu_update_articulation_kinematics()
            scene._gpu_fetch_all()

        scene.update_render(
            update_sensors=True,
            update_human_render_cameras=True,
        )
