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


def _build_visual_box(scene, half_size, rgba, name):
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=np.array(rgba, dtype=np.float32).tolist())
    builder.add_box_visual(
        half_size=[half_size, half_size, half_size],
        material=material,
    )
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, half_size]))
    return actor


def _build_collision_box(scene, half_size, rgba, name):
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=np.array(rgba, dtype=np.float32).tolist())
    builder.add_box_collision(half_size=[half_size, half_size, half_size])
    builder.add_box_visual(
        half_size=[half_size, half_size, half_size],
        material=material,
    )
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, half_size]))
    return actor


def _build_collision_cylinder(scene, radius, half_length, rgba, name):
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=np.array(rgba, dtype=np.float32).tolist())
    try:
        builder.add_cylinder_collision(radius=radius, half_length=half_length)
        builder.add_cylinder_visual(radius=radius, half_length=half_length, material=material)
    except Exception:
        builder.add_box_collision(half_size=[radius, radius, half_length])
        builder.add_box_visual(
            half_size=[radius, radius, half_length],
            material=material,
        )
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, half_length]))
    return actor


def _actor_xyz(actor):
    if actor is None:
        return None
    pose = getattr(actor, "pose", None)
    if pose is None and hasattr(actor, "get_pose"):
        pose = actor.get_pose()
    if pose is None:
        return None
    try:
        xyz = np.asarray(pose.p, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return xyz[:3].copy() if xyz.size >= 3 else None


def _get_task_anchor_positions(env):
    cube = None
    for key in ["cube", "obj", "object", "_obj"]:
        candidate = getattr(env.unwrapped, key, None)
        if candidate is not None:
            cube = candidate
            break

    cube_xyz = _actor_xyz(cube)

    goal_site = getattr(env.unwrapped, "goal_site", None)
    goal_xyz = _actor_xyz(goal_site)
    if goal_xyz is None:
        for key in ["goal", "_goal", "target", "_target", "goal_region", "target_site"]:
            candidate = getattr(env.unwrapped, key, None)
            goal_xyz = _actor_xyz(candidate)
            if goal_xyz is not None:
                break

    return cube_xyz, goal_xyz


def _sample_non_interfering_xy(env, cube_xyz, goal_xyz, cfg):
    center = np.asarray(getattr(env.unwrapped, "cube_spawn_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    extent_x = float(cfg.get("table_extent_x", 0.32))
    extent_y = float(cfg.get("table_extent_y", 0.24))
    min_distance = float(cfg.get("min_distance", 0.2))
    margin = float(cfg.get("table_margin", 0.04))
    max_tries = int(cfg.get("max_tries", 128))

    low_x = float(center[0] - extent_x + margin)
    high_x = float(center[0] + extent_x - margin)
    low_y = float(center[1] - extent_y + margin)
    high_y = float(center[1] + extent_y - margin)

    best_xy = None
    best_clearance = -1.0
    for _ in range(max_tries):
        xy = np.array(
            [np.random.uniform(low_x, high_x), np.random.uniform(low_y, high_y)],
            dtype=np.float32,
        )
        clearance = float("inf")
        if cube_xyz is not None:
            clearance = min(clearance, float(np.linalg.norm(xy - cube_xyz[:2])))
        if goal_xyz is not None:
            clearance = min(clearance, float(np.linalg.norm(xy - goal_xyz[:2])))
        if clearance > min_distance:
            return xy
        if clearance > best_clearance:
            best_clearance = clearance
            best_xy = xy
    return best_xy


def _move_actor_offstage(actor):
    if actor is None:
        return
    try:
        actor.set_pose(sapien.Pose(p=[10.0, 10.0, -10.0]))
    except Exception:
        pass


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


@register_env("MR-CMSI1")
@register_env("MR-CMSI-1")
def env_cmsi1_add_blue_cube(env, cfg):
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        raise ValueError("MR-CMSI1 requires env.unwrapped.scene")

    runtime = getattr(env.unwrapped, "_mr_cmsi1_runtime", None)
    if runtime is None:
        runtime = {}
        env.unwrapped._mr_cmsi1_runtime = runtime

    distractor = runtime.get("blue_cube")
    if distractor is None:
        half_size = float(cfg.get("blue_cube_half_size", getattr(env.unwrapped, "cube_half_size", 0.02)))
        rgba = cfg.get("blue_cube_rgba", [0.0, 0.0, 1.0, 1.0])
        distractor = _build_collision_box(scene, half_size=half_size, rgba=rgba, name="mr_cmsi1_blue_cube")
        runtime["blue_cube"] = distractor
    else:
        half_size = float(cfg.get("blue_cube_half_size", getattr(env.unwrapped, "cube_half_size", 0.02)))

    spawn_half = float(getattr(env.unwrapped, "cube_spawn_half_size", half_size))
    spawn_center = np.asarray(getattr(env.unwrapped, "cube_spawn_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    pos = np.array(
        [
            spawn_center[0] + spawn_half * 1.0,
            spawn_center[1] - spawn_half * 4.0,
            half_size,
        ],
        dtype=np.float32,
    )
    distractor.set_pose(Pose.create_from_pq(pos))

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


@register_env("MR4")
@register_env("MR-4")
@register_env("Target-Object-Relocation")
def env_mr4_target_object_relocation(env, cfg):
    cfg = dict(cfg or {})
    cfg.setdefault("dx", 0.05)
    cfg.setdefault("dy", 0.05)
    return env_translate_cube_xy(env, cfg)


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


@register_env("MR2")
@register_env("MR-2")
@register_env("Non-interfering-Object-Addition")
def env_mr2_non_interfering_object_addition(env, cfg):
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        raise ValueError("MR2 requires env.unwrapped.scene")

    runtime = getattr(env.unwrapped, "_mr2_runtime", None)
    if runtime is None:
        runtime = {}
        env.unwrapped._mr2_runtime = runtime

    cube_xyz, goal_xyz = _get_task_anchor_positions(env)
    xy = _sample_non_interfering_xy(env, cube_xyz, goal_xyz, cfg)
    if xy is None:
        raise ValueError("MR2 could not sample a non-interfering distractor position")

    distractor_type = cfg.get("distractor_type", "random")
    if distractor_type == "random":
        distractor_type = str(np.random.choice(["blue_cup", "yellow_block"]))
    distractor_type = str(distractor_type).strip().lower()

    block_half_size = float(cfg.get("yellow_block_half_size", getattr(env.unwrapped, "cube_half_size", 0.02)))
    cup_radius = float(cfg.get("blue_cup_radius", max(0.015, block_half_size * 0.7)))
    cup_half_length = float(cfg.get("blue_cup_half_length", max(0.025, block_half_size * 1.2)))

    blue_cup = runtime.get("blue_cup")
    if blue_cup is None:
        blue_cup = _build_collision_cylinder(
            scene,
            radius=cup_radius,
            half_length=cup_half_length,
            rgba=cfg.get("blue_cup_rgba", [0.1, 0.35, 0.95, 1.0]),
            name="mr2_blue_cup",
        )
        runtime["blue_cup"] = blue_cup

    yellow_block = runtime.get("yellow_block")
    if yellow_block is None:
        yellow_block = _build_collision_box(
            scene,
            half_size=block_half_size,
            rgba=cfg.get("yellow_block_rgba", [0.95, 0.82, 0.15, 1.0]),
            name="mr2_yellow_block",
        )
        runtime["yellow_block"] = yellow_block

    _move_actor_offstage(blue_cup)
    _move_actor_offstage(yellow_block)

    if distractor_type == "blue_cup":
        pos = np.array([xy[0], xy[1], cup_half_length], dtype=np.float32)
        blue_cup.set_pose(Pose.create_from_pq(pos))
        runtime["active_name"] = "mr2_blue_cup"
        runtime["active_type"] = "blue_cup"
    elif distractor_type == "yellow_block":
        pos = np.array([xy[0], xy[1], block_half_size], dtype=np.float32)
        yellow_block.set_pose(Pose.create_from_pq(pos))
        runtime["active_name"] = "mr2_yellow_block"
        runtime["active_type"] = "yellow_block"
    else:
        raise ValueError(f"MR2 unknown distractor_type: {distractor_type}")

    runtime["active_xy"] = xy.tolist()
    runtime["cube_xyz"] = None if cube_xyz is None else cube_xyz.tolist()
    runtime["goal_xyz"] = None if goal_xyz is None else goal_xyz.tolist()

    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env


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


def _apply_darken_scene(env, cfg, *, darken_table: bool):
    ambient = float(cfg.get("ambient", 0.08))
    ambient = max(0.0, min(ambient, 1.0))
    env.unwrapped.scene.set_ambient_light([ambient, ambient, ambient])

    if darken_table:
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


@register_env("MR-BG-1-darken")
def env_darken_scene(env, cfg):
    return _apply_darken_scene(env, cfg, darken_table=True)


@register_env("MR-SADP-3")
def env_sadp3(env, cfg):
    # DP 兼容版：只调暗环境光，不改桌面材质
    return _apply_darken_scene(env, cfg, darken_table=False)
