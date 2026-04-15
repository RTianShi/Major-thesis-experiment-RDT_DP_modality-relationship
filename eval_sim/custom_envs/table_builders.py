import numpy as np
import sapien

from mani_skill.utils.scene_builder.table import TableSceneBuilder


class WoodTableSceneBuilder(TableSceneBuilder):
    def build(self):
        super().build()

        wood_color = np.array([139, 94, 60, 255], dtype=np.float32) / 255.0
        dark_wood_color = np.array([110, 72, 46, 255], dtype=np.float32) / 255.0

        for part_idx, part in enumerate(self.table._objs):
            render_body = part.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                for visual_part in shape.parts:
                    material = visual_part.material
                    # Keep the original table geometry/collision and only override visuals.
                    material.set_base_color(
                        wood_color if part_idx == 0 else dark_wood_color
                    )
                    material.set_base_color_texture(None)
                    material.set_normal_texture(None)
                    material.set_emission_texture(None)
                    material.set_transmission_texture(None)
                    material.set_metallic_texture(None)
                    material.set_roughness_texture(None)
