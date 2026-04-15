import numpy as np
import sapien

from mani_skill.utils.registration import register_env

from .stackcube_large_red_cube import StackCubeLargeRedCubeEnv


def _iter_render_materials(actor):
    if actor is None:
        return
    for obj in getattr(actor, "_objs", []):
        render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_body is None:
            continue
        for shape in render_body.render_shapes:
            for visual_part in shape.parts:
                yield visual_part.material


@register_env("StackCubeWeightlessIron-v1", max_episode_steps=50)
class StackCubeWeightlessIronEnv(StackCubeLargeRedCubeEnv):
    """StackCube variant with a visually huge rusty iron block as the red cube."""

    def __init__(self, *args, red_cube_scale=1.8, **kwargs):
        super().__init__(*args, red_cube_scale=red_cube_scale, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        rust_base = np.array([0.58, 0.22, 0.12, 1.0], dtype=np.float32)
        rust_dark = np.array([0.32, 0.12, 0.08, 1.0], dtype=np.float32)
        for idx, material in enumerate(_iter_render_materials(self.cubeA)):
            material.set_base_color((rust_base if idx % 2 == 0 else rust_dark).tolist())
            material.set_base_color_texture(None)
            material.set_normal_texture(None)
            material.set_emission([0.0, 0.0, 0.0, 1.0])
            material.set_emission_texture(None)
            material.set_transmission(0.0)
            material.set_transmission_texture(None)
            material.set_metallic(0.82)
            material.set_metallic_texture(None)
            material.set_roughness(0.78)
            material.set_roughness_texture(None)
            if hasattr(material, "set_specular"):
                material.set_specular(0.35)
