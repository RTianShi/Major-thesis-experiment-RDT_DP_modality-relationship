"""StackCube-specific env mutation module.

Add StackCube runtime and environment mutations here as they diverge from PickCube.
"""

import math
import numpy as np
import sapien
import torch

from mani_skill.utils.structs.pose import Pose

from .registry import register_env


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


def _find_actor_by_attr(env, names):
    for name in names:
        actor = getattr(env.unwrapped, name, None)
        if actor is not None:
            return actor
    return None


def _find_scene_actor_by_keywords(env, keywords):
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        return None
    try:
        actors = scene.get_all_actors()
    except Exception:
        return None

    for actor in actors:
        try:
            name = actor.get_name().lower()
        except Exception:
            name = ""
        if any(keyword in name for keyword in keywords):
            return actor
    return None


def _set_actor_base_color(actor, rgba):
    for material in _iter_render_materials(actor):
        material.set_base_color(np.array(rgba, dtype=np.float32))
        material.set_base_color_texture(None)
        material.set_normal_texture(None)
        material.set_emission_texture(None)
        material.set_transmission_texture(None)
        material.set_metallic_texture(None)
        material.set_roughness_texture(None)


def _scalar_half_extent(value, fallback=0.02):
    if value is None:
        return float(fallback)
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        return float(fallback)
    return float(tensor[0].item())


def _set_actor_surface_material(actor, cfg):
    mode = str(cfg.get("surface_mode", "ice")).strip().lower()
    if mode in {"metal", "oily_metal", "lubricated_metal"}:
        base_color = np.array(cfg.get("metal_rgba", [0.82, 0.18, 0.16, 1.0]), dtype=np.float32)
        roughness = float(cfg.get("metal_roughness", 0.06))
        metallic = float(cfg.get("metallic", 0.95))
        specular = float(cfg.get("metal_specular", 0.95))
        transmission = float(cfg.get("metal_transmission", 0.0))
    else:
        base_color = np.array(cfg.get("ice_rgba", [0.88, 0.22, 0.18, 0.9]), dtype=np.float32)
        roughness = float(cfg.get("ice_roughness", 0.02))
        metallic = float(cfg.get("metallic", 0.0))
        specular = float(cfg.get("ice_specular", 1.0))
        transmission = float(cfg.get("ice_transmission", 0.18))

    emission = np.array(cfg.get("emission", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    for material in _iter_render_materials(actor):
        material.set_base_color(base_color.tolist())
        material.set_base_color_texture(None)
        material.set_normal_texture(None)
        material.set_emission(emission.tolist())
        material.set_emission_texture(None)
        material.set_transmission(float(np.clip(transmission, 0.0, 1.0)))
        material.set_transmission_texture(None)
        material.set_metallic(float(np.clip(metallic, 0.0, 1.0)))
        material.set_metallic_texture(None)
        material.set_roughness(float(np.clip(roughness, 0.0, 1.0)))
        material.set_roughness_texture(None)
        if hasattr(material, "set_specular"):
            material.set_specular(float(np.clip(specular, 0.0, 1.0)))


def _actor_pose_tensor(actor):
    pose = actor.pose
    pos = pose.p.clone()
    quat = pose.q.clone()
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
        quat = quat.unsqueeze(0)
    return pos, quat


def _cube_xy_radius(env, actor, fallback_half_size=0.02):
    actor_name = ""
    try:
        actor_name = actor.get_name().lower()
    except Exception:
        actor_name = ""

    candidate_attrs = ["cube_half_size"]
    if "cubeb" in actor_name or "green" in actor_name:
        candidate_attrs = ["cubeB_half_size", "cube_half_size"]
    elif "cubea" in actor_name or "red" in actor_name:
        candidate_attrs = ["cubeA_half_size", "cube_half_size"]

    for attr in candidate_attrs:
        half_size = getattr(env.unwrapped, attr, None)
        if half_size is None:
            continue
        half_size = torch.as_tensor(half_size, dtype=torch.float32).reshape(-1)
        if half_size.numel() >= 2:
            return float(torch.linalg.norm(half_size[:2]).item())
    return float(np.sqrt(2.0) * fallback_half_size)


def _set_actor_pose(actor, pos, quat):
    if pos.ndim == 1:
        actor.set_pose(sapien.Pose(p=pos.tolist(), q=quat.tolist()))
        return
    actor.set_pose(Pose.create_from_pq(pos, quat))


def _zero_actor_velocity(actor, pos):
    zeros = torch.zeros_like(pos)
    if pos.ndim == 1:
        zeros = zeros.tolist()
    try:
        actor.set_linear_velocity(zeros)
    except Exception:
        pass
    try:
        actor.set_angular_velocity(zeros)
    except Exception:
        pass


def _translate_xy_within_bounds(pos_batch, dx, dy, bounds):
    x_min, y_min = [float(v) for v in bounds[0]]
    x_max, y_max = [float(v) for v in bounds[1]]
    translated = pos_batch.clone()
    translated[:, 0] = translated[:, 0] + float(dx)
    translated[:, 1] = translated[:, 1] + float(dy)
    if torch.any(translated[:, 0] < x_min) or torch.any(translated[:, 0] > x_max):
        raise ValueError(
            f"translation dx={dx:.3f} moves cube out of x bounds [{x_min:.3f}, {x_max:.3f}]"
        )
    if torch.any(translated[:, 1] < y_min) or torch.any(translated[:, 1] > y_max):
        raise ValueError(
            f"translation dy={dy:.3f} moves cube out of y bounds [{y_min:.3f}, {y_max:.3f}]"
        )
    return translated


def _feasible_translation_interval(pos_batch, bounds):
    x_min, y_min = [float(v) for v in bounds[0]]
    x_max, y_max = [float(v) for v in bounds[1]]
    dx_low = float(torch.max(torch.as_tensor(x_min, dtype=pos_batch.dtype) - pos_batch[:, 0]).item())
    dx_high = float(torch.min(torch.as_tensor(x_max, dtype=pos_batch.dtype) - pos_batch[:, 0]).item())
    dy_low = float(torch.max(torch.as_tensor(y_min, dtype=pos_batch.dtype) - pos_batch[:, 1]).item())
    dy_high = float(torch.min(torch.as_tensor(y_max, dtype=pos_batch.dtype) - pos_batch[:, 1]).item())
    return dx_low, dx_high, dy_low, dy_high


def _resolve_translation_vector(red_pos_batch, green_pos_batch, dx, dy, bounds, cfg):
    try:
        _translate_xy_within_bounds(red_pos_batch, dx, dy, bounds)
        _translate_xy_within_bounds(green_pos_batch, dx, dy, bounds)
        return float(dx), float(dy)
    except ValueError:
        pass

    auto_adjust = bool(cfg.get("auto_adjust_translation", True))
    if not auto_adjust:
        _translate_xy_within_bounds(red_pos_batch, dx, dy, bounds)
        _translate_xy_within_bounds(green_pos_batch, dx, dy, bounds)
        return float(dx), float(dy)

    red_dx_low, red_dx_high, red_dy_low, red_dy_high = _feasible_translation_interval(
        red_pos_batch, bounds
    )
    green_dx_low, green_dx_high, green_dy_low, green_dy_high = _feasible_translation_interval(
        green_pos_batch, bounds
    )
    dx_low = max(red_dx_low, green_dx_low)
    dx_high = min(red_dx_high, green_dx_high)
    dy_low = max(red_dy_low, green_dy_low)
    dy_high = min(red_dy_high, green_dy_high)

    if dx_low > dx_high or dy_low > dy_high:
        raise ValueError("MR-SEMP1 could not find any feasible shared translation region")

    resolved_dx = min(max(float(dx), dx_low), dx_high)
    resolved_dy = min(max(float(dy), dy_low), dy_high)
    min_translation_norm = float(cfg.get("min_translation_norm", 0.01))
    if np.hypot(resolved_dx, resolved_dy) < min_translation_norm:
        x_candidates = [dx_low, dx_high]
        y_candidates = [dy_low, dy_high]
        candidates = [(cx, cy) for cx in x_candidates for cy in y_candidates]
        candidates = [
            (cx, cy) for (cx, cy) in candidates if np.hypot(cx, cy) >= min_translation_norm
        ]
        if not candidates:
            raise ValueError(
                "MR-SEMP1 feasible translation exists only with near-zero magnitude; "
                f"min_translation_norm={min_translation_norm:.3f} is too large"
            )
        resolved_dx, resolved_dy = min(
            candidates,
            key=lambda item: (item[0] - float(dx)) ** 2 + (item[1] - float(dy)) ** 2,
        )
    return float(resolved_dx), float(resolved_dy)


def _yaw_quat_batch(batch_size, yaw_deg, device):
    yaw_rad = math.radians(float(yaw_deg))
    quat = torch.zeros((batch_size, 4), device=device, dtype=torch.float32)
    quat[:, 0] = math.cos(yaw_rad / 2.0)
    quat[:, 3] = math.sin(yaw_rad / 2.0)
    return quat


def _apply_scene_updates(env):
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        return
    if getattr(scene, "device", None) is not None and scene.device.type == "cuda":
        scene._gpu_apply_all()
        scene.px.gpu_update_articulation_kinematics()
        scene._gpu_fetch_all()
    scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )


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


def _ensure_sadp2_visual_assets(env, cfg):
    runtime = getattr(env.unwrapped, "_mr_sadp2_runtime", None)
    if runtime is not None:
        return runtime

    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        raise ValueError("MR-SADP2 requires env.unwrapped.scene")

    distractor_half_size = float(cfg.get("distractor_half_size", 0.016))
    distractor_colors = cfg.get(
        "distractor_colors",
        [
            [0.12, 0.42, 0.95, 1.0],
            [0.98, 0.82, 0.18, 1.0],
            [0.65, 0.25, 0.88, 1.0],
        ],
    )

    distractor_actors = []
    for idx, rgba in enumerate(distractor_colors[:3]):
        actor = _build_visual_box(
            scene,
            half_size=distractor_half_size,
            rgba=rgba,
            name=f"mr_sadp2_visual_cube_{idx}",
        )
        distractor_actors.append(actor)

    point_light = None
    try:
        point_light = scene.add_point_light(
            position=cfg.get("shadow_light_position", [0.22, -0.22, 0.75]),
            color=cfg.get("shadow_light_color", [0.9, 0.82, 0.72]),
            shadow=bool(cfg.get("shadow", True)),
        )
    except Exception:
        point_light = None

    runtime = {
        "distractor_actors": distractor_actors,
        "point_light": point_light,
        "distractor_half_size": distractor_half_size,
    }
    env.unwrapped._mr_sadp2_runtime = runtime
    return runtime


def _set_table_visual_noise(env, cfg):
    table_actor = _find_scene_actor_by_keywords(env, ("table", "desk"))
    if table_actor is not None:
        _set_actor_base_color(table_actor, cfg.get("table_rgba", [0.76, 0.68, 0.52, 1.0]))

    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None:
        ambient = cfg.get("ambient_light", [0.16, 0.15, 0.14])
        try:
            scene.set_ambient_light(ambient)
        except Exception:
            pass


def _safe_visual_distractor_positions(red_xy, green_xy, cfg):
    candidates = np.array(
        cfg.get(
            "candidate_distractor_xy",
            [
                [-0.24, 0.27],
                [0.00, 0.29],
                [0.24, 0.27],
                [-0.24, -0.27],
                [0.24, -0.27],
            ],
        ),
        dtype=np.float32,
    )
    min_clearance = float(cfg.get("distractor_clearance", 0.09))

    legal = []
    for xy in candidates:
        if np.linalg.norm(xy - red_xy) < min_clearance:
            continue
        if np.linalg.norm(xy - green_xy) < min_clearance:
            continue
        legal.append(xy)

    if len(legal) < 3:
        raise ValueError("MR-SADP2 could not find 3 legal distractor positions")
    np.random.shuffle(legal)
    return legal[:3]


def _place_sadp2_visual_distractors(env, red_xy, green_xy, cfg):
    runtime = _ensure_sadp2_visual_assets(env, cfg)
    positions = _safe_visual_distractor_positions(red_xy, green_xy, cfg)
    z_height = runtime["distractor_half_size"]

    for actor, xy in zip(runtime["distractor_actors"], positions):
        actor.set_pose(sapien.Pose(p=[float(xy[0]), float(xy[1]), z_height]))

    point_light = runtime.get("point_light")
    if point_light is not None:
        try:
            point_light.set_pose(
                sapien.Pose(p=cfg.get("shadow_light_position", [0.22, -0.22, 0.75]))
            )
        except Exception:
            pass


def _resolve_green_cube_far_side_xy(red_xy, green_xy, bounds, cfg, min_center_distance):
    x_min, y_min = [float(v) for v in bounds[0]]
    x_max, y_max = [float(v) for v in bounds[1]]
    target_far_y = float(cfg.get("target_far_y", 0.18))
    target_y = -target_far_y if green_xy[1] >= 0.0 else target_far_y
    mirrored_y = float(np.clip(target_y, y_min, y_max))
    candidate_x = float(np.clip(green_xy[0], x_min, x_max))

    candidates = [
        np.array([candidate_x, mirrored_y], dtype=np.float32),
        np.array([x_min + 0.02, mirrored_y], dtype=np.float32),
        np.array([x_max - 0.02, mirrored_y], dtype=np.float32),
        np.array([0.0, mirrored_y], dtype=np.float32),
        np.array([candidate_x, -green_xy[1]], dtype=np.float32),
    ]

    min_displacement = float(cfg.get("min_green_displacement", 0.12))
    for candidate in candidates:
        candidate[0] = float(np.clip(candidate[0], x_min, x_max))
        candidate[1] = float(np.clip(candidate[1], y_min, y_max))
        if np.linalg.norm(candidate - green_xy) < min_displacement:
            continue
        if np.linalg.norm(candidate - red_xy) < min_center_distance:
            continue
        return candidate

    raise ValueError("MR-SADP2 could not relocate the green cube to the opposite side legally")


@register_env("identity")
def env_identity(env, cfg):
    return env


@register_env("MR-B1")
@register_env("JSAP-CoMutate-Color")
def env_mr_b1_color_comutate(env, cfg):
    red_cube = _find_actor_by_attr(
        env,
        ("cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"),
    )
    green_cube = _find_actor_by_attr(
        env,
        ("cubeB", "cube_b", "green_cube", "goal_cube", "target_cube"),
    )

    if red_cube is None:
        raise ValueError("MR-B1 could not locate the movable cube actor in StackCube")
    if green_cube is None:
        raise ValueError("MR-B1 could not locate the support cube actor in StackCube")

    blue_rgba = cfg.get("blue_rgba", [0.1, 0.35, 0.95, 1.0])
    yellow_rgba = cfg.get("yellow_rgba", [0.95, 0.8, 0.15, 1.0])
    _set_actor_base_color(red_cube, blue_rgba)
    _set_actor_base_color(green_cube, yellow_rgba)

    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env


@register_env("MR-SESP2")
@register_env("Friction-ForceSynergy")
def env_mr_sesp2_surface_friction_force_synergy(env, cfg):
    red_cube = _find_actor_by_attr(
        env,
        ("cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"),
    )
    if red_cube is None:
        raise ValueError("MR-SESP2 could not locate the movable red cube actor in StackCube")

    _set_actor_surface_material(red_cube, cfg)
    env.unwrapped.scene.update_render(
        update_sensors=True,
        update_human_render_cameras=True,
    )
    return env


@register_env("MR-DRP1")
@register_env("Source Position Perturbation")
def env_mr_drp1_source_position_perturbation(env, cfg):
    red_cube = _find_actor_by_attr(
        env,
        ("cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"),
    )
    green_cube = _find_actor_by_attr(
        env,
        ("cubeB", "cube_b", "green_cube", "goal_cube", "target_cube"),
    )
    if red_cube is None:
        raise ValueError("MR-DRP1 could not locate the movable red cube actor in StackCube")
    if green_cube is None:
        raise ValueError("MR-DRP1 could not locate the support green cube actor in StackCube")

    red_pos_batch, red_quat_batch = _actor_pose_tensor(red_cube)
    green_pos_batch, _ = _actor_pose_tensor(green_cube)
    new_pos_batch = red_pos_batch.clone()

    bounds = cfg.get("placement_bounds", [[-0.1, -0.2], [0.1, 0.2]])
    x_min, y_min = [float(v) for v in bounds[0]]
    x_max, y_max = [float(v) for v in bounds[1]]
    min_displacement = float(cfg.get("min_displacement", 0.10))
    separation_margin = float(cfg.get("separation_margin", 0.01))
    max_attempts = max(1, int(cfg.get("max_attempts", 256)))
    red_radius = _cube_xy_radius(env, red_cube)
    green_radius = _cube_xy_radius(env, green_cube)
    min_center_distance = red_radius + green_radius + separation_margin

    for batch_idx in range(new_pos_batch.shape[0]):
        origin_xy = red_pos_batch[batch_idx, :2].detach().cpu().numpy()
        green_xy = green_pos_batch[batch_idx, :2].detach().cpu().numpy()
        candidate_xy = None
        for _ in range(max_attempts):
            sample_xy = np.array(
                [
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(sample_xy - origin_xy) <= min_displacement:
                continue
            if np.linalg.norm(sample_xy - green_xy) <= min_center_distance:
                continue
            candidate_xy = sample_xy
            break
        if candidate_xy is None:
            raise ValueError(
                "MR-DRP1 failed to sample a legal red-cube position satisfying "
                f"min_displacement>{min_displacement:.3f}m within {max_attempts} attempts"
            )
        new_pos_batch[batch_idx, 0] = float(candidate_xy[0])
        new_pos_batch[batch_idx, 1] = float(candidate_xy[1])

    if red_cube.pose.p.ndim == 1:
        new_pos = new_pos_batch[0]
        red_quat = red_quat_batch[0]
    else:
        new_pos = new_pos_batch
        red_quat = red_quat_batch

    _set_actor_pose(red_cube, new_pos, red_quat)
    _zero_actor_velocity(red_cube, new_pos)
    _apply_scene_updates(env)
    return env


@register_env("MR-SEMP1")
@register_env("Global Translation Equivariance")
def env_mr_semp1_global_translation_equivariance(env, cfg):
    red_cube = _find_actor_by_attr(
        env,
        ("cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"),
    )
    green_cube = _find_actor_by_attr(
        env,
        ("cubeB", "cube_b", "green_cube", "goal_cube", "target_cube"),
    )
    if red_cube is None:
        raise ValueError("MR-SEMP1 could not locate the movable red cube actor in StackCube")
    if green_cube is None:
        raise ValueError("MR-SEMP1 could not locate the support green cube actor in StackCube")

    dx = float(cfg.get("dx", 0.04))
    dy = float(cfg.get("dy", 0.0))
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return env

    bounds = cfg.get("placement_bounds", [[-0.1, -0.2], [0.1, 0.2]])
    red_pos_batch, red_quat_batch = _actor_pose_tensor(red_cube)
    green_pos_batch, green_quat_batch = _actor_pose_tensor(green_cube)

    dx, dy = _resolve_translation_vector(red_pos_batch, green_pos_batch, dx, dy, bounds, cfg)
    red_new_batch = _translate_xy_within_bounds(red_pos_batch, dx, dy, bounds)
    green_new_batch = _translate_xy_within_bounds(green_pos_batch, dx, dy, bounds)

    if red_cube.pose.p.ndim == 1:
        red_new, red_quat = red_new_batch[0], red_quat_batch[0]
    else:
        red_new, red_quat = red_new_batch, red_quat_batch

    if green_cube.pose.p.ndim == 1:
        green_new, green_quat = green_new_batch[0], green_quat_batch[0]
    else:
        green_new, green_quat = green_new_batch, green_quat_batch

    _set_actor_pose(red_cube, red_new, red_quat)
    _set_actor_pose(green_cube, green_new, green_quat)
    _zero_actor_velocity(red_cube, red_new)
    _zero_actor_velocity(green_cube, green_new)
    _apply_scene_updates(env)
    return env


@register_env("MR-SEMP2")
@register_env("Z-axis Rotation Equivariance")
def env_mr_semp2_z_axis_rotation_equivariance(env, cfg):
    red_cube = _find_actor_by_attr(
        env,
        ("cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"),
    )
    if red_cube is None:
        raise ValueError("MR-SEMP2 could not locate the movable red cube actor in StackCube")

    delta_theta_deg = float(cfg.get("delta_theta_deg", 45.0))
    red_pos_batch, _ = _actor_pose_tensor(red_cube)
    quat_batch = _yaw_quat_batch(red_pos_batch.shape[0], delta_theta_deg, red_pos_batch.device)

    if red_cube.pose.p.ndim == 1:
        red_pos = red_pos_batch[0]
        red_quat = quat_batch[0]
    else:
        red_pos = red_pos_batch
        red_quat = quat_batch

    _set_actor_pose(red_cube, red_pos, red_quat)
    _zero_actor_velocity(red_cube, red_pos)
    _apply_scene_updates(env)
    return env


@register_env("MR-SADP2")
@register_env("Visual Redundancy Immunity")
def env_mr_sadp2_visual_redundancy_immunity(env, cfg):
    red_cube = _find_actor_by_attr(
        env,
        ("cubeA", "cube_a", "red_cube", "obj", "object", "_obj", "cube"),
    )
    green_cube = _find_actor_by_attr(
        env,
        ("cubeB", "cube_b", "green_cube", "goal_cube", "target_cube"),
    )
    if red_cube is None:
        raise ValueError("MR-SADP2 could not locate the movable red cube actor in StackCube")
    if green_cube is None:
        raise ValueError("MR-SADP2 could not locate the support green cube actor in StackCube")

    bounds = cfg.get("placement_bounds", [[-0.1, -0.2], [0.1, 0.2]])
    red_pos_batch, _ = _actor_pose_tensor(red_cube)
    green_pos_batch, green_quat_batch = _actor_pose_tensor(green_cube)
    green_new_batch = green_pos_batch.clone()

    red_radius = _cube_xy_radius(env, red_cube)
    green_radius = _cube_xy_radius(env, green_cube)
    min_center_distance = red_radius + green_radius + float(cfg.get("separation_margin", 0.02))

    for batch_idx in range(green_new_batch.shape[0]):
        red_xy = red_pos_batch[batch_idx, :2].detach().cpu().numpy()
        green_xy = green_pos_batch[batch_idx, :2].detach().cpu().numpy()
        new_green_xy = _resolve_green_cube_far_side_xy(
            red_xy,
            green_xy,
            bounds=bounds,
            cfg=cfg,
            min_center_distance=min_center_distance,
        )
        green_new_batch[batch_idx, 0] = float(new_green_xy[0])
        green_new_batch[batch_idx, 1] = float(new_green_xy[1])

    if green_cube.pose.p.ndim == 1:
        green_new = green_new_batch[0]
        green_quat = green_quat_batch[0]
    else:
        green_new = green_new_batch
        green_quat = green_quat_batch

    _set_actor_pose(green_cube, green_new, green_quat)
    _zero_actor_velocity(green_cube, green_new)

    red_xy_ref = red_pos_batch[0, :2].detach().cpu().numpy()
    green_xy_ref = green_new_batch[0, :2].detach().cpu().numpy()
    _set_table_visual_noise(env, cfg)
    _place_sadp2_visual_distractors(env, red_xy_ref, green_xy_ref, cfg)
    _apply_scene_updates(env)
    return env
