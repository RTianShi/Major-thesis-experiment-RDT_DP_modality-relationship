import torch
import sapien

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
import numpy as np


@register_env("PickCubeBlueCube-v1", max_episode_steps=50)
class PickCubeBlueCubeEnv(PickCubeEnv):
    def _iter_render_materials(self, actor):
        if actor is None:
            return
        for obj in getattr(actor, "_objs", []):
            render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                for visual_part in shape.parts:
                    yield visual_part.material

    def _set_actor_base_color(self, actor, rgba):
        for material in self._iter_render_materials(actor):
            material.set_base_color(np.array(rgba, dtype=np.float32))
            material.set_base_color_texture(None)
            material.set_normal_texture(None)
            material.set_emission_texture(None)
            material.set_transmission_texture(None)
            material.set_metallic_texture(None)
            material.set_roughness_texture(None)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._set_actor_base_color(self.cube, [0, 0, 1, 1])
