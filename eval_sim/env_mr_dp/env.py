import numpy as np
import torch
import sapien

from mani_skill.utils.structs.pose import Pose

from .registry import register_env


def _iter_render_materials(actor):
    if actor is None:
        return
    objs = getattr(actor, "_objs", [])
    for obj in objs:
        render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_body is None:
            continue
        for shape in render_body.render_shapes:
            for visual_part in shape.parts:
                yield visual_part.material


def _get_table_actor(env):
    table_scene = getattr(env.unwrapped, "table_scene", None)
    if table_scene is not None:
        table = getattr(table_scene, "table", None)
        if table is not None:
            return table
    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None:
        for actor in scene.get_all_actors():
            try:
                name = actor.get_name().lower()
            except Exception:
                name = ""
            if "table" in name:
                return actor
    return None


@register_env("identity")
def env_identity(env, cfg):
    # No-op mutation for environment.
    return env


@register_env("MR-JDCP-2")
def env_remove_goal_visual_anchor(env, cfg):
    goal_site = getattr(env.unwrapped, "goal_site", None)
    if goal_site is None:
        raise ValueError("MR-JDCP-2 requires env.unwrapped.goal_site")
    hidden_objects = getattr(env.unwrapped, "_hidden_objects", None)
    if hidden_objects is not None:
        env.unwrapped._hidden_objects = [obj for obj in hidden_objects if obj is not goal_site]
    goal_site.hide_visual()
    env.unwrapped.scene.update_render(update_sensors=True, update_human_render_cameras=True)
    return env


@register_env("MR-SDPP-2")
def env_spawn_cube_on_goal(env, cfg):
    cube = None
    for key in ["cube", "obj", "object", "_obj"]:
        candidate = getattr(env.unwrapped, key, None)
        if candidate is not None:
            cube = candidate
            break
    if cube is None:
        raise ValueError("MR-SDPP-2 requires a cube-like object on env.unwrapped")

    goal_site = getattr(env.unwrapped, "goal_site", None)
    if goal_site is None:
        raise ValueError("MR-SDPP-2 requires env.unwrapped.goal_site")
    if hasattr(goal_site, "pose"):
        goal_pose = goal_site.pose
    elif hasattr(goal_site, "get_pose"):
        goal_pose = goal_site.get_pose()
    else:
        raise ValueError("MR-SDPP-2 could not read goal_site pose")

    cube.set_pose(goal_pose)
    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env


@register_env("MR-SADP-1-translate_cube_xy")
def env_translate_cube_xy(env, cfg):
    dx = float(cfg.get("dx", 0.04))
    dy = float(cfg.get("dy", 0.0))
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return env

    cube = None
    for key in ["cube", "obj", "object", "_obj"]:
        candidate = getattr(env.unwrapped, key, None)
        if candidate is not None:
            cube = candidate
            break
    if cube is None:
        raise ValueError("MR-SADP-1-translate_cube_xy requires a cube-like object on env.unwrapped")

    if hasattr(cube, "pose"):
        cube_pose = cube.pose
    elif hasattr(cube, "get_pose"):
        cube_pose = cube.get_pose()
    else:
        raise ValueError("MR-SADP-1-translate_cube_xy could not read cube pose")

    cube_pos = np.array(cube_pose.p, dtype=np.float32).copy()
    cube_quat = np.array(cube_pose.q, dtype=np.float32).copy()
    if cube_pos.ndim > 1:
        cube_pos = cube_pos[0].copy()
        cube_quat = cube_quat[0].copy()
    cube_pos[0] += dx
    cube_pos[1] += dy
    cube.set_pose(Pose.create_from_pq(cube_pos, cube_quat))

    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env


@register_env("MR-SADP-1-joint_reset_noise")
def env_joint_reset_noise(env, cfg):
    scale = float(cfg.get("scale", 0.05))
    num_joints = int(cfg.get("num_joints", 7))
    if scale <= 0 or num_joints <= 0:
        return env

    robot = getattr(getattr(env.unwrapped, "agent", None), "robot", None)
    if robot is None:
        raise ValueError("MR-SADP-1-joint_reset_noise requires env.unwrapped.agent.robot")

    qpos = robot.get_qpos().clone()
    qvel = robot.get_qvel().clone()
    width = min(num_joints, qpos.shape[-1])
    noise = (torch.rand_like(qpos[..., :width]) * 2.0 - 1.0) * scale
    qpos[..., :width] = qpos[..., :width] + noise
    robot.set_qpos(qpos)
    robot.set_qvel(torch.zeros_like(qvel))

    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None and getattr(scene, "device", None) is not None and scene.device.type == "cuda":
        scene._gpu_apply_all()
        scene.px.gpu_update_articulation_kinematics()
        scene._gpu_fetch_all()

    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env


@register_env("MR-SADP-1-translate_cube_xy_and_joint_reset_noise")
def env_translate_cube_xy_and_joint_reset_noise(env, cfg):
    env_translate_cube_xy(env, cfg)
    env_joint_reset_noise(env, cfg)
    return env


@register_env("MR-SADP-1")
def env_sadp1(env, cfg):
    return env_translate_cube_xy_and_joint_reset_noise(env, cfg)


@register_env("MR-BG-1-checkerboard_table")
def env_checkerboard_table(env, cfg):
    table = _get_table_actor(env)
    if table is None:
        raise ValueError("MR-BG-1-checkerboard_table requires a table actor")

    palette = [
        np.array([0.92, 0.92, 0.92, 1.0], dtype=np.float32),
        np.array([0.08, 0.08, 0.08, 1.0], dtype=np.float32),
        np.array([0.95, 0.82, 0.18, 1.0], dtype=np.float32),
        np.array([0.12, 0.45, 0.85, 1.0], dtype=np.float32),
    ]
    for idx, material in enumerate(_iter_render_materials(table)):
        material.set_base_color(palette[idx % len(palette)])
        material.set_base_color_texture(None)
        material.set_normal_texture(None)
        material.set_emission_texture(None)
        material.set_transmission_texture(None)
        material.set_metallic_texture(None)
        material.set_roughness_texture(None)

    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env


@register_env("MR-SADP-2")
def env_sadp2(env, cfg):
    return env_checkerboard_table(env, cfg)


@register_env("MR-BG-1-darken")
def env_darken_scene(env, cfg):
    ambient = float(cfg.get("ambient", 0.08))
    ambient = max(0.0, min(ambient, 1.0))
    env.unwrapped.scene.set_ambient_light([ambient, ambient, ambient])

    table = _get_table_actor(env)
    if table is not None:
        dark_scale = float(cfg.get("table_dark_scale", 0.45))
        dark_scale = max(0.0, min(dark_scale, 1.0))
        for material in _iter_render_materials(table):
            try:
                base_color = np.array(material.base_color, dtype=np.float32)
            except Exception:
                base_color = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
            base_color[:3] *= dark_scale
            material.set_base_color(base_color)
            material.set_base_color_texture(None)

    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env
