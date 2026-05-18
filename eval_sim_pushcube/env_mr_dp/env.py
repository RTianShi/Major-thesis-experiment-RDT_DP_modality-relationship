"""PushCube-specific env mutation module."""

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


def _find_goal_anchor(env):
    anchor = _find_actor_by_attr(
        env,
        (
            "goal",
            "_goal",
            "target",
            "_target",
            "goal_site",
            "target_site",
            "goal_region",
        ),
    )
    if anchor is not None:
        return anchor
    return _find_scene_actor_by_keywords(env, ("goal", "target", "region"))


def _find_cube_anchor(env):
    anchor = _find_actor_by_attr(
        env,
        (
            "task_obj",
            "task_object",
            "cube",
            "obj",
            "object",
            "_obj",
        ),
    )
    if anchor is not None:
        return anchor
    return _find_scene_actor_by_keywords(env, ("star", "cube", "block", "obj"))


def _find_task_object_anchor(env):
    anchor = _find_actor_by_attr(
        env,
        (
            "task_obj",
            "task_object",
            "obj",
            "object",
            "_obj",
            "cube",
        ),
    )
    if anchor is not None:
        return anchor
    return _find_scene_actor_by_keywords(env, ("cube", "block", "obj"))


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


def _set_actor_visibility(actor, visibility):
    if actor is None:
        return False

    updated = False
    for obj in getattr(actor, "_objs", []):
        render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_body is None:
            continue
        render_body.visibility = int(visibility)
        updated = True
    return updated


def _scalar_half_extent(value, fallback=0.02):
    if value is None:
        return float(fallback)
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        return float(fallback)
    return float(tensor[0].item())


def _infer_goal_radius(env, goal_anchor, fallback=0.0):
    goal_radius = getattr(env.unwrapped, "goal_radius", None)
    if goal_radius is not None:
        try:
            return float(goal_radius)
        except Exception:
            pass

    for attr in (
        "goal_half_size",
        "goal_size",
        "goal_region_half_size",
        "goal_region_size",
        "goal_region_radius",
    ):
        if hasattr(env.unwrapped, attr):
            return float(_scalar_half_extent(getattr(env.unwrapped, attr, None), fallback=fallback))

    return float(_scalar_half_extent(getattr(goal_anchor, "half_size", None), fallback=fallback))


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
    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    pos = torch.as_tensor(pose.p, dtype=torch.float32).clone()
    quat = torch.as_tensor(pose.q, dtype=torch.float32).clone()
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


def _apply_robot_qpos(env, qpos):
    robot = getattr(getattr(env.unwrapped, "agent", None), "robot", None)
    if robot is None:
        raise ValueError("robot mutation requires env.unwrapped.agent.robot")

    qvel = robot.get_qvel().clone()
    robot.set_qpos(qpos)
    robot.set_qvel(torch.zeros_like(qvel))

    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None and getattr(scene, "device", None) is not None and scene.device.type == "cuda":
        scene._gpu_apply_all()
        scene.px.gpu_update_articulation_kinematics()
        scene._gpu_fetch_all()


def _entity_is_batched(entity):
    pose = entity.pose if hasattr(entity, "pose") else entity.get_pose()
    pos = torch.as_tensor(pose.p)
    return pos.ndim > 1


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


def _validate_xy_within_bounds(pos_batch, bounds, label):
    x_min, y_min = [float(v) for v in bounds[0]]
    x_max, y_max = [float(v) for v in bounds[1]]
    if torch.any(pos_batch[:, 0] < x_min) or torch.any(pos_batch[:, 0] > x_max):
        raise ValueError(
            f"{label} x is out of bounds [{x_min:.3f}, {x_max:.3f}]"
        )
    if torch.any(pos_batch[:, 1] < y_min) or torch.any(pos_batch[:, 1] > y_max):
        raise ValueError(
            f"{label} y is out of bounds [{y_min:.3f}, {y_max:.3f}]"
        )
    return pos_batch


def _bounds_or_current_x_window(bounds, current_pos_batch, margin=1e-4):
    if bounds is not None:
        return bounds
    x_min = float(torch.min(current_pos_batch[:, 0]).item()) - float(margin)
    x_max = float(torch.max(current_pos_batch[:, 0]).item()) + float(margin)
    y_min = -float("inf")
    y_max = float("inf")
    return [[x_min, y_min], [x_max, y_max]]


def _mirror_y_within_bounds(pos_batch, bounds, cfg, label):
    x_min, y_min = [float(v) for v in bounds[0]]
    x_max, y_max = [float(v) for v in bounds[1]]
    mirror_axis_y = float(cfg.get("mirror_axis_y", 0.0))

    mirrored = pos_batch.clone()
    mirrored[:, 1] = 2.0 * mirror_axis_y - mirrored[:, 1]

    if torch.all(mirrored[:, 0] >= x_min) and torch.all(mirrored[:, 0] <= x_max):
        if torch.all(mirrored[:, 1] >= y_min) and torch.all(mirrored[:, 1] <= y_max):
            return mirrored

    auto_adjust = bool(cfg.get("auto_adjust_mirror", True))
    if not auto_adjust:
        return _validate_xy_within_bounds(mirrored, bounds, label)

    clipped = mirrored.clone()
    clipped[:, 0] = torch.clamp(clipped[:, 0], min=x_min, max=x_max)
    clipped[:, 1] = torch.clamp(clipped[:, 1], min=y_min, max=y_max)

    min_mirror_displacement = float(cfg.get("min_mirror_displacement", 0.01))
    displacement = torch.linalg.norm(clipped[:, :2] - pos_batch[:, :2], dim=-1)
    if torch.any(displacement < min_mirror_displacement):
        raise ValueError(
            f"{label} mirror displacement is below min_mirror_displacement={min_mirror_displacement:.3f}"
        )
    return clipped


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


def _resolve_translation_vector_with_separate_bounds(
    task_pos_batch,
    goal_pos_batch,
    dx,
    dy,
    task_bounds,
    goal_bounds,
    cfg,
):
    try:
        _translate_xy_within_bounds(task_pos_batch, dx, dy, task_bounds)
        _translate_xy_within_bounds(goal_pos_batch, dx, dy, goal_bounds)
        return float(dx), float(dy)
    except ValueError:
        pass

    auto_adjust = bool(cfg.get("auto_adjust_translation", True))
    if not auto_adjust:
        _translate_xy_within_bounds(task_pos_batch, dx, dy, task_bounds)
        _translate_xy_within_bounds(goal_pos_batch, dx, dy, goal_bounds)
        return float(dx), float(dy)

    task_dx_low, task_dx_high, task_dy_low, task_dy_high = _feasible_translation_interval(
        task_pos_batch, task_bounds
    )
    goal_dx_low, goal_dx_high, goal_dy_low, goal_dy_high = _feasible_translation_interval(
        goal_pos_batch, goal_bounds
    )
    dx_low = max(task_dx_low, goal_dx_low)
    dx_high = min(task_dx_high, goal_dx_high)
    dy_low = max(task_dy_low, goal_dy_low)
    dy_high = min(task_dy_high, goal_dy_high)

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


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


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


def _build_visual_sphere(scene, radius, rgba, name):
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=np.array(rgba, dtype=np.float32).tolist())
    builder.add_sphere_visual(radius=float(radius), material=material)
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, float(radius)]))
    return actor


def _build_visual_duck(scene, scale, body_rgba, beak_rgba, name):
    builder = scene.create_actor_builder()
    body_material = sapien.render.RenderMaterial(
        base_color=np.array(body_rgba, dtype=np.float32).tolist()
    )
    beak_material = sapien.render.RenderMaterial(
        base_color=np.array(beak_rgba, dtype=np.float32).tolist()
    )

    builder.add_sphere_visual(radius=0.018 * scale, material=body_material)
    builder.add_sphere_visual(
        radius=0.012 * scale,
        material=body_material,
        pose=sapien.Pose(p=[0.018 * scale, 0.0, 0.017 * scale]),
    )
    builder.add_box_visual(
        half_size=[0.006 * scale, 0.003 * scale, 0.003 * scale],
        material=beak_material,
        pose=sapien.Pose(p=[0.031 * scale, 0.0, 0.017 * scale]),
    )
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, 0.018 * scale]))
    return actor


def _sync_proxy_actor_pose(proxy_actor, source_actor):
    source_pos, source_quat = _actor_pose_tensor(source_actor)
    _set_actor_pose(proxy_actor, source_pos, source_quat)
    _zero_actor_velocity(proxy_actor, source_pos)


def _build_visual_target_marker(
    scene,
    outer_radius,
    inner_radius,
    outer_half_length,
    inner_half_length,
    outer_rgba,
    inner_rgba,
    name,
):
    builder = scene.create_actor_builder()
    outer_material = sapien.render.RenderMaterial(
        base_color=np.array(outer_rgba, dtype=np.float32).tolist()
    )
    inner_material = sapien.render.RenderMaterial(
        base_color=np.array(inner_rgba, dtype=np.float32).tolist()
    )
    builder.add_cylinder_visual(
        radius=float(outer_radius),
        half_length=float(outer_half_length),
        material=outer_material,
    )
    builder.add_cylinder_visual(
        radius=float(inner_radius),
        half_length=float(inner_half_length),
        material=inner_material,
    )
    actor = builder.build_kinematic(name=name)
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, float(max(outer_half_length, inner_half_length))]))
    return actor


def _build_visual_star_prism(
    scene,
    arm_half_size,
    arm_length,
    height,
    rgba,
    name,
):
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=np.array(rgba, dtype=np.float32).tolist())
    for yaw_deg in (0.0, 45.0, 90.0, 135.0):
        yaw_rad = math.radians(float(yaw_deg))
        arm_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=[
                math.cos(yaw_rad / 2.0),
                0.0,
                0.0,
                math.sin(yaw_rad / 2.0),
            ],
        )
        builder.add_box_visual(
            half_size=[float(arm_length), float(arm_half_size), float(height)],
            material=material,
            pose=arm_pose,
        )
        if hasattr(builder, "add_box_collision"):
            builder.add_box_collision(
                half_size=[float(arm_length), float(arm_half_size), float(height)],
                pose=arm_pose,
            )
    actor = builder.build(name=name)
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, float(height)]))
    return actor


def _ensure_cmsi1_visual_assets(env, cfg):
    runtime = getattr(env.unwrapped, "_mr_cmsi1_runtime", None)
    if runtime is not None:
        return runtime

    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        raise ValueError("MR-CMSI1 requires env.unwrapped.scene")

    outer_radius = float(cfg.get("outer_radius", 0.04))
    inner_radius = float(cfg.get("inner_radius", 0.022))
    outer_half_length = float(cfg.get("outer_half_length", 0.004))
    inner_half_length = float(cfg.get("inner_half_length", 0.005))

    left_actor = _build_visual_target_marker(
        scene=scene,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        outer_half_length=outer_half_length,
        inner_half_length=inner_half_length,
        outer_rgba=cfg.get("left_outer_rgba", [0.12, 0.35, 0.92, 1.0]),
        inner_rgba=cfg.get("left_inner_rgba", [0.98, 0.98, 0.98, 1.0]),
        name="mr_cmsi1_left_decoy_goal",
    )
    right_actor = _build_visual_target_marker(
        scene=scene,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        outer_half_length=outer_half_length,
        inner_half_length=inner_half_length,
        outer_rgba=cfg.get("right_outer_rgba", [0.12, 0.76, 0.28, 1.0]),
        inner_rgba=cfg.get("right_inner_rgba", [0.98, 0.98, 0.98, 1.0]),
        name="mr_cmsi1_right_decoy_goal",
    )

    runtime = {
        "left_actor": left_actor,
        "right_actor": right_actor,
        "z_offset": float(cfg.get("z_offset", 0.0)),
    }
    env.unwrapped._mr_cmsi1_runtime = runtime
    return runtime


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


def _ensure_sesp2_visual_assets(env, cfg):
    runtime = getattr(env.unwrapped, "_mr_sesp2_runtime", None)
    if runtime is not None:
        return runtime

    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        raise ValueError("MR-SESP2 requires env.unwrapped.scene")

    task_anchor = _find_task_object_anchor(env)
    if task_anchor is None:
        raise ValueError("MR-SESP2 could not find a task object actor")

    base_half_size = float(cfg.get("proxy_half_size", _cube_xy_radius(env, task_anchor)))
    visual_scale = float(cfg.get("visual_scale", 1.45))
    proxy_half_size = max(base_half_size * visual_scale, base_half_size + 1e-3)
    proxy_rgba = cfg.get("proxy_rgba", [0.86, 0.16, 0.12, 1.0])
    proxy_actor = _build_visual_box(
        scene=scene,
        half_size=proxy_half_size,
        rgba=proxy_rgba,
        name="mr_sesp2_large_cube_proxy",
    )

    runtime = {
        "task_anchor": task_anchor,
        "proxy_actor": proxy_actor,
        "hide_original_visual": bool(cfg.get("hide_original_visual", True)),
    }
    env.unwrapped._mr_sesp2_runtime = runtime
    return runtime


def _ensure_scdp1_visual_assets(env, cfg):
    runtime = getattr(env.unwrapped, "_mr_scdp1_runtime", None)
    if runtime is not None:
        return runtime

    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        raise ValueError("MR-SCDP1 requires env.unwrapped.scene")

    duck_scales = cfg.get("duck_scales", [1.0, 0.9, 1.1])
    duck_body_colors = cfg.get(
        "duck_body_colors",
        [
            [0.98, 0.88, 0.12, 1.0],
            [0.95, 0.74, 0.16, 1.0],
            [0.99, 0.83, 0.24, 1.0],
        ],
    )
    duck_beak_rgba = cfg.get("duck_beak_rgba", [0.96, 0.42, 0.08, 1.0])

    ducks = []
    for idx, scale in enumerate(duck_scales[:3]):
        body_rgba = duck_body_colors[idx % len(duck_body_colors)]
        duck = _build_visual_duck(
            scene=scene,
            scale=float(scale),
            body_rgba=body_rgba,
            beak_rgba=duck_beak_rgba,
            name=f"mr_scdp1_duck_{idx}",
        )
        ducks.append(duck)

    accent_light = None
    try:
        accent_light = scene.add_point_light(
            position=cfg.get("accent_light_position", [-0.28, 0.26, 0.62]),
            color=cfg.get("accent_light_color", [0.45, 0.95, 0.55]),
            shadow=bool(cfg.get("accent_light_shadow", False)),
        )
    except Exception:
        accent_light = None

    runtime = {"ducks": ducks, "accent_light": accent_light}
    env.unwrapped._mr_scdp1_runtime = runtime
    return runtime


def _apply_sesp2_visual_state(env):
    runtime = getattr(env.unwrapped, "_mr_sesp2_runtime", None)
    if runtime is None:
        return
    task_anchor = runtime["task_anchor"]
    proxy_actor = runtime["proxy_actor"]
    if runtime.get("hide_original_visual", True):
        _set_actor_visibility(task_anchor, 0)
    _sync_proxy_actor_pose(proxy_actor, task_anchor)


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


def _set_table_checkerboard_or_wood(env, cfg):
    table_actor = _find_scene_actor_by_keywords(env, ("table", "desk"))
    if table_actor is None:
        return

    mode = str(cfg.get("table_mode", "checkerboard")).strip().lower()
    if mode in {"wood", "woodgrain", "wood_grain"}:
        palette = [
            np.array([0.52, 0.34, 0.18, 1.0], dtype=np.float32),
            np.array([0.64, 0.42, 0.23, 1.0], dtype=np.float32),
            np.array([0.45, 0.29, 0.14, 1.0], dtype=np.float32),
            np.array([0.71, 0.51, 0.30, 1.0], dtype=np.float32),
        ]
    else:
        palette = [
            np.array([0.93, 0.93, 0.93, 1.0], dtype=np.float32),
            np.array([0.09, 0.09, 0.09, 1.0], dtype=np.float32),
            np.array([0.16, 0.52, 0.24, 1.0], dtype=np.float32),
            np.array([0.92, 0.24, 0.22, 1.0], dtype=np.float32),
        ]

    for idx, material in enumerate(_iter_render_materials(table_actor)):
        material.set_base_color(palette[idx % len(palette)].tolist())
        material.set_base_color_texture(None)
        material.set_normal_texture(None)
        material.set_emission_texture(None)
        material.set_transmission_texture(None)
        material.set_metallic_texture(None)
        material.set_roughness_texture(None)


def _place_scdp1_visual_distractors(env, cfg):
    runtime = _ensure_scdp1_visual_assets(env, cfg)
    duck_positions = cfg.get(
        "duck_positions",
        [
            [-0.25, 0.24],
            [0.25, 0.24],
            [0.25, -0.24],
        ],
    )
    duck_yaws = cfg.get("duck_yaws_deg", [20.0, -35.0, 145.0])

    for idx, duck in enumerate(runtime["ducks"]):
        xy = duck_positions[idx % len(duck_positions)]
        yaw_deg = float(duck_yaws[idx % len(duck_yaws)])
        yaw_rad = math.radians(yaw_deg)
        duck.set_pose(
            sapien.Pose(
                p=[float(xy[0]), float(xy[1]), 0.02],
                q=[math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)],
            )
        )

    accent_light = runtime.get("accent_light")
    if accent_light is not None:
        try:
            accent_light.set_pose(
                sapien.Pose(p=cfg.get("accent_light_position", [-0.28, 0.26, 0.62]))
            )
        except Exception:
            pass


def _apply_scdp1_visual_state(env, cfg):
    _ensure_scdp1_visual_assets(env, cfg)
    _set_table_checkerboard_or_wood(env, cfg)
    _set_table_visual_noise(
        env,
        {
            "table_rgba": cfg.get("table_rgba", [0.66, 0.55, 0.44, 1.0]),
            "ambient_light": cfg.get("ambient_light", [0.10, 0.24, 0.12]),
        },
    )

    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None:
        try:
            scene.set_ambient_light(cfg.get("ambient_light", [0.10, 0.24, 0.12]))
        except Exception:
            pass

    _place_scdp1_visual_distractors(env, cfg)
    _apply_scene_updates(env)


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

    raise ValueError("MR-SADP2 could not relocate the goal anchor to a legal opposite-side position")


@register_env("identity")
def env_identity(env, cfg):
    return env


@register_env("MR-JDCP3")
@register_env("MR-JDCP-3")
@register_env("JDCP-Deprivative-Vision")
def env_hide_goal_region_visual(env, cfg):
    goal_anchor = _find_goal_anchor(env)
    if goal_anchor is None:
        raise ValueError(
            "MR-JDCP3 could not find a goal/target/region actor to hide"
        )

    hidden = False
    if hasattr(goal_anchor, "hide_visual"):
        try:
            goal_anchor.hide_visual()
            hidden = True
        except Exception:
            hidden = False

    hidden = _set_actor_visibility(goal_anchor, 0) or hidden

    if not hidden:
        _set_actor_base_color(goal_anchor, [0.0, 0.0, 0.0, 0.0])
        hidden = True

    _apply_scene_updates(env)
    return env


@register_env("MR-JSAP1")
@register_env("MR-JSAP-1")
@register_env("JSAP-CoMutate-Goal")
def env_move_goal_region_to_cube_left(env, cfg):
    cube_anchor = _find_cube_anchor(env)
    if cube_anchor is None:
        raise ValueError("MR-JSAP1 could not find a cube/block actor")

    goal_anchor = _find_goal_anchor(env)
    if goal_anchor is None:
        raise ValueError("MR-JSAP1 could not find a goal/target/region actor")

    cube_pos, _ = _actor_pose_tensor(cube_anchor)
    goal_pos, goal_quat = _actor_pose_tensor(goal_anchor)

    offset_xy = torch.as_tensor(
        cfg.get("goal_offset_xy", [0.0, 0.2]),
        dtype=cube_pos.dtype,
        device=cube_pos.device,
    ).reshape(-1)
    if offset_xy.numel() < 2:
        raise ValueError("MR-JSAP1 requires goal_offset_xy to contain at least two values")

    target_pos = goal_pos.clone()
    target_pos[:, 0] = cube_pos[:, 0] + float(offset_xy[0].item())
    target_pos[:, 1] = cube_pos[:, 1] + float(offset_xy[1].item())

    _set_actor_pose(goal_anchor, target_pos, goal_quat)
    _zero_actor_velocity(goal_anchor, target_pos)
    _apply_scene_updates(env)
    return env


@register_env("MR-CMSI1")
@register_env("MR-CMSI-1")
@register_env("CMSI-Contrastive-Decoy-Targets")
def env_inject_contrastive_goal_decoys(env, cfg):
    cube_anchor = _find_cube_anchor(env)
    if cube_anchor is None:
        raise ValueError("MR-CMSI1 could not find a cube/block actor")

    goal_anchor = _find_goal_anchor(env)
    if goal_anchor is None:
        raise ValueError("MR-CMSI1 could not find a goal/target/region actor")

    runtime = _ensure_cmsi1_visual_assets(env, cfg)
    cube_pos, _ = _actor_pose_tensor(cube_anchor)
    goal_pos, goal_quat = _actor_pose_tensor(goal_anchor)

    left_offset_xy = torch.as_tensor(
        cfg.get("left_offset_xy", [0.0, 0.2]),
        dtype=cube_pos.dtype,
        device=cube_pos.device,
    ).reshape(-1)
    right_offset_xy = torch.as_tensor(
        cfg.get("right_offset_xy", [0.0, -0.2]),
        dtype=cube_pos.dtype,
        device=cube_pos.device,
    ).reshape(-1)
    if left_offset_xy.numel() < 2 or right_offset_xy.numel() < 2:
        raise ValueError("MR-CMSI1 offsets must each contain at least two values")

    z_offset = float(runtime["z_offset"])
    left_pos = goal_pos.clone()
    right_pos = goal_pos.clone()
    left_pos[:, 0] = cube_pos[:, 0] + float(left_offset_xy[0].item())
    left_pos[:, 1] = cube_pos[:, 1] + float(left_offset_xy[1].item())
    right_pos[:, 0] = cube_pos[:, 0] + float(right_offset_xy[0].item())
    right_pos[:, 1] = cube_pos[:, 1] + float(right_offset_xy[1].item())
    left_pos[:, 2] = goal_pos[:, 2] + z_offset
    right_pos[:, 2] = goal_pos[:, 2] + z_offset

    _set_actor_pose(runtime["left_actor"], left_pos, goal_quat)
    _set_actor_pose(runtime["right_actor"], right_pos, goal_quat)
    _zero_actor_velocity(runtime["left_actor"], left_pos)
    _zero_actor_velocity(runtime["right_actor"], right_pos)
    _apply_scene_updates(env)
    return env


@register_env("MR-SESP2")
@register_env("MR-SESP-2")
@register_env("SESP-Physical-Semantic-Synergy")
def env_large_cube_visual_proxy(env, cfg):
    runtime = _ensure_sesp2_visual_assets(env, cfg)
    _apply_sesp2_visual_state(env)

    if not getattr(env.unwrapped, "_mr_sesp2_step_wrapped", False):
        original_step = env.step

        def _wrapped_step(action):
            result = original_step(action)
            _apply_sesp2_visual_state(env)
            _apply_scene_updates(env)
            return result

        env.step = _wrapped_step
        env.unwrapped._mr_sesp2_step_wrapped = True

    _apply_scene_updates(env)
    return env


@register_env("MR-SCDP1")
@register_env("MR-SCDP-1")
@register_env("SCDP-Visual-Background-Debunking")
def env_visual_background_debunking(env, cfg):
    _apply_scdp1_visual_state(env, cfg)

    refresh_each_step = bool(cfg.get("refresh_each_step", True))
    if refresh_each_step and not getattr(env.unwrapped, "_mr_scdp1_step_wrapped", False):
        original_step = env.step

        def _wrapped_step(action):
            result = original_step(action)
            _apply_scdp1_visual_state(env, cfg)
            return result

        env.step = _wrapped_step
        env.unwrapped._mr_scdp1_step_wrapped = True
    return env


@register_env("MR-DRP1")
@register_env("MR-DRP-1")
@register_env("DRP-Extreme-Diagonal-Perturbation")
def env_extreme_diagonal_cube_position(env, cfg):
    task_anchor = _find_task_object_anchor(env)
    if task_anchor is None:
        raise ValueError("MR-DRP1 could not find a task object actor")

    task_pos, task_quat = _actor_pose_tensor(task_anchor)
    target_xy = torch.as_tensor(
        cfg.get("target_xy", [0.1, -0.1]),
        dtype=task_pos.dtype,
        device=task_pos.device,
    ).reshape(-1)
    if target_xy.numel() < 2:
        raise ValueError("MR-DRP1 requires target_xy to contain at least two values")

    translated_task_pos = task_pos.clone()
    translated_task_pos[:, 0] = float(target_xy[0].item())
    translated_task_pos[:, 1] = float(target_xy[1].item())

    task_bounds = cfg.get(
        "task_bounds_xy",
        [[-0.10, -0.20], [0.10, 0.20]],
    )
    translated_task_pos[..., :2] = _validate_xy_within_bounds(
        translated_task_pos[..., :2], task_bounds, "MR-DRP1 task position"
    )

    _set_actor_pose(task_anchor, translated_task_pos, task_quat)
    _zero_actor_velocity(task_anchor, translated_task_pos)
    env.unwrapped._mr_drp1_runtime = {
        "target_xy": [float(v) for v in target_xy[:2].tolist()],
    }
    _apply_scene_updates(env)
    return env


@register_env("MR-DRP2-base")
@register_env("MR-DRP-2")
@register_env("DRP-Bilateral-Extreme-Entry")
def env_bilateral_extreme_entry_pose(env, cfg):
    robot = getattr(getattr(env.unwrapped, "agent", None), "robot", None)
    if robot is None:
        raise ValueError("MR-DRP2 requires env.unwrapped.agent.robot")

    pose_variant = str(cfg.get("pose_variant", "left")).strip().lower()
    current_qpos = robot.get_qpos().clone()
    width = min(int(cfg.get("num_joints", 7)), current_qpos.shape[-1])
    if width <= 0:
        return env

    left_delta = torch.as_tensor(
        cfg.get(
            "left_delta",
            [0.45, 0.12, 0.0, -0.30, 0.0, 0.22, 0.55],
        ),
        dtype=current_qpos.dtype,
        device=current_qpos.device,
    ).reshape(-1)
    right_delta = torch.as_tensor(
        cfg.get(
            "right_delta",
            [-0.45, -0.12, 0.0, -0.30, 0.0, 0.22, -0.55],
        ),
        dtype=current_qpos.dtype,
        device=current_qpos.device,
    ).reshape(-1)

    left_pose = cfg.get("left_pose", None)
    right_pose = cfg.get("right_pose", None)

    target_qpos = current_qpos.clone()
    if pose_variant in {"left", "left_biased", "source"}:
        if left_pose is not None:
            left_pose = torch.as_tensor(left_pose, dtype=target_qpos.dtype, device=target_qpos.device).reshape(-1)
            target_qpos[..., :width] = left_pose[:width]
        else:
            target_qpos[..., :width] = target_qpos[..., :width] + left_delta[:width]
    elif pose_variant in {"right", "right_biased", "derived"}:
        if right_pose is not None:
            right_pose = torch.as_tensor(right_pose, dtype=target_qpos.dtype, device=target_qpos.device).reshape(-1)
            target_qpos[..., :width] = right_pose[:width]
        else:
            target_qpos[..., :width] = target_qpos[..., :width] + right_delta[:width]
    else:
        raise ValueError(f"MR-DRP2 unknown pose_variant={pose_variant!r}")

    joint_limits = cfg.get("joint_limits", None)
    if joint_limits is not None:
        joint_limits = torch.as_tensor(
            joint_limits,
            dtype=target_qpos.dtype,
            device=target_qpos.device,
        )
        if joint_limits.ndim == 2 and joint_limits.shape[0] >= width and joint_limits.shape[1] >= 2:
            lower = joint_limits[:width, 0]
            upper = joint_limits[:width, 1]
            target_qpos[..., :width] = torch.max(
                torch.min(target_qpos[..., :width], upper),
                lower,
            )

    _apply_robot_qpos(env, target_qpos)
    env.unwrapped._mr_drp2_runtime = {
        "pose_variant": pose_variant,
        "num_joints": int(width),
    }
    _apply_scene_updates(env)
    return env


@register_env("MR-DRP2-MR")
def env_bilateral_extreme_entry_pose_mr(env, cfg):
    robot = getattr(getattr(env.unwrapped, "agent", None), "robot", None)
    if robot is None:
        raise ValueError("MR-DRP2 requires env.unwrapped.agent.robot")

    pose_variant = str(cfg.get("pose_variant", "right")).strip().lower()
    current_qpos = robot.get_qpos().clone()
    width = min(int(cfg.get("num_joints", 7)), current_qpos.shape[-1])
    if width <= 0:
        return env

    left_delta = torch.as_tensor(
        cfg.get(
            "left_delta",
            [0.45, 0.12, 0.0, -0.30, 0.0, 0.22, 0.55],
        ),
        dtype=current_qpos.dtype,
        device=current_qpos.device,
    ).reshape(-1)
    right_delta = torch.as_tensor(
        cfg.get(
            "right_delta",
            [-0.45, -0.12, 0.0, -0.30, 0.0, 0.22, -0.55],
        ),
        dtype=current_qpos.dtype,
        device=current_qpos.device,
    ).reshape(-1)

    left_pose = cfg.get("left_pose", None)
    right_pose = cfg.get("right_pose", None)

    target_qpos = current_qpos.clone()
    if pose_variant in {"left", "left_biased", "source"}:
        if left_pose is not None:
            left_pose = torch.as_tensor(left_pose, dtype=target_qpos.dtype, device=target_qpos.device).reshape(-1)
            target_qpos[..., :width] = left_pose[:width]
        else:
            target_qpos[..., :width] = target_qpos[..., :width] + left_delta[:width]
    elif pose_variant in {"right", "right_biased", "derived"}:
        if right_pose is not None:
            right_pose = torch.as_tensor(right_pose, dtype=target_qpos.dtype, device=target_qpos.device).reshape(-1)
            target_qpos[..., :width] = right_pose[:width]
        else:
            target_qpos[..., :width] = target_qpos[..., :width] + right_delta[:width]
    else:
        raise ValueError(f"MR-DRP2 unknown pose_variant={pose_variant!r}")

    joint_limits = cfg.get("joint_limits", None)
    if joint_limits is not None:
        joint_limits = torch.as_tensor(
            joint_limits,
            dtype=target_qpos.dtype,
            device=target_qpos.device,
        )
        if joint_limits.ndim == 2 and joint_limits.shape[0] >= width and joint_limits.shape[1] >= 2:
            lower = joint_limits[:width, 0]
            upper = joint_limits[:width, 1]
            target_qpos[..., :width] = torch.max(
                torch.min(target_qpos[..., :width], upper),
                lower,
            )

    _apply_robot_qpos(env, target_qpos)
    env.unwrapped._mr_drp2_runtime = {
        "pose_variant": pose_variant,
        "num_joints": int(width),
    }
    _apply_scene_updates(env)
    return env


@register_env("MR-SEMP1")
@register_env("MR-SEMP-1")
@register_env("SEMP-2D-Continuous-Translation-Equivariance")
@register_env("MR-SADP1")
@register_env("MR-SADP-1")
@register_env("SADP-Cross-Modal-Semantic-Noise")
def env_translate_task_and_goal_scene_xy(env, cfg):
    task_anchor = _find_task_object_anchor(env)
    if task_anchor is None:
        raise ValueError("MR-SEMP1 could not find a task object actor")

    goal_anchor = _find_goal_anchor(env)
    if goal_anchor is None:
        raise ValueError("MR-SEMP1 could not find a goal/target/region actor")

    dx = float(cfg.get("dx", 0.05))
    dy = float(cfg.get("dy", -0.05))
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return env

    task_bounds = cfg.get(
        "task_bounds_xy",
        [[-0.10, -0.20], [0.10, 0.20]],
    )
    goal_bounds = cfg.get(
        "goal_bounds_xy",
        [[0.00, -0.20], [0.25, 0.20]],
    )

    task_pos, task_quat = _actor_pose_tensor(task_anchor)
    goal_pos, goal_quat = _actor_pose_tensor(goal_anchor)
    resolved_dx, resolved_dy = _resolve_translation_vector_with_separate_bounds(
        task_pos[..., :2],
        goal_pos[..., :2],
        dx=dx,
        dy=dy,
        task_bounds=task_bounds,
        goal_bounds=goal_bounds,
        cfg=cfg,
    )

    translated_task_pos = task_pos.clone()
    translated_goal_pos = goal_pos.clone()
    translated_task_pos[:, 0] += resolved_dx
    translated_task_pos[:, 1] += resolved_dy
    translated_goal_pos[:, 0] += resolved_dx
    translated_goal_pos[:, 1] += resolved_dy

    _set_actor_pose(task_anchor, translated_task_pos, task_quat)
    _set_actor_pose(goal_anchor, translated_goal_pos, goal_quat)
    _zero_actor_velocity(task_anchor, translated_task_pos)
    _zero_actor_velocity(goal_anchor, translated_goal_pos)
    env.unwrapped._mr_semp1_runtime = {
        "dx": float(resolved_dx),
        "dy": float(resolved_dy),
    }
    _apply_scene_updates(env)
    return env


@register_env("MR-SADP2")
@register_env("MR-SADP-2")
@register_env("SADP-Intra-Modal-Background-Noise")
def env_sadp2_visual_background_and_material_noise(env, cfg):
    env_translate_task_and_goal_scene_xy(env, cfg)

    _set_table_checkerboard_or_wood(env, cfg)
    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None:
        try:
            scene.set_ambient_light(cfg.get("ambient_light", [0.18, 0.18, 0.18]))
        except Exception:
            pass

    task_anchor = _find_task_object_anchor(env)
    if task_anchor is None:
        raise ValueError("MR-SADP2 could not find a task object actor")

    visual_noise_mode = str(cfg.get("visual_noise_mode", "material+background")).strip().lower()
    apply_material_noise = visual_noise_mode in {"material", "material+background", "both", "all"}
    apply_background_noise = visual_noise_mode in {"background", "material+background", "both", "all"}
    apply_camera_noise = bool(cfg.get("apply_camera_noise", False))

    _set_actor_surface_material(
        task_anchor,
        {
            "surface_mode": cfg.get("surface_mode", "metal") if apply_material_noise else "ice",
            "metal_rgba": cfg.get("metal_rgba", [0.78, 0.78, 0.82, 1.0]),
            "metal_roughness": cfg.get("metal_roughness", 0.08),
            "metallic": cfg.get("metallic", 0.98),
            "metal_specular": cfg.get("metal_specular", 0.95),
            "metal_transmission": cfg.get("metal_transmission", 0.0),
            "ice_rgba": cfg.get("ice_rgba", [0.88, 0.22, 0.18, 0.9]),
            "ice_roughness": cfg.get("ice_roughness", 0.02),
            "ice_specular": cfg.get("ice_specular", 1.0),
            "ice_transmission": cfg.get("ice_transmission", 0.18),
        },
    )

    if not apply_material_noise:
        # Keep the block visually close to the baseline if only background perturbation is desired.
        _set_actor_base_color(task_anchor, cfg.get("task_rgba", [0.88, 0.22, 0.18, 0.9]))

    if apply_background_noise:
        try:
            scene.set_ambient_light(cfg.get("ambient_light", [0.18, 0.18, 0.18]))
        except Exception:
            pass

    if apply_camera_noise:
        camera_noise = cfg.get("camera_noise", {})
        env.unwrapped._mr_sadp2_camera_noise = {
            "enabled": True,
            "mode": str(camera_noise.get("mode", "gaussian_blur")),
            "radius": float(camera_noise.get("radius", 2.0)),
            "fill_value": int(camera_noise.get("fill_value", 0)),
        }
    else:
        env.unwrapped._mr_sadp2_camera_noise = {"enabled": False}

    env.unwrapped._mr_sadp2_runtime = {
        "dx": float(cfg.get("dx", 0.05)),
        "dy": float(cfg.get("dy", -0.05)),
        "table_mode": str(cfg.get("table_mode", "checkerboard")),
        "visual_noise_mode": visual_noise_mode,
        "apply_material_noise": bool(apply_material_noise),
        "apply_background_noise": bool(apply_background_noise),
        "apply_camera_noise": bool(apply_camera_noise),
    }
    _apply_scene_updates(env)
    return env


@register_env("MR-SEMP2")
@register_env("MR-SEMP-2")
@register_env("SEMP-Y-Axis-Mirror-Equivariance")
def env_mirror_task_and_goal_about_x_axis(env, cfg):
    task_anchor = _find_task_object_anchor(env)
    if task_anchor is None:
        raise ValueError("MR-SEMP2 could not find a task object actor")

    goal_anchor = _find_goal_anchor(env)
    if goal_anchor is None:
        raise ValueError("MR-SEMP2 could not find a goal/target/region actor")

    task_pos, task_quat = _actor_pose_tensor(task_anchor)
    goal_pos, goal_quat = _actor_pose_tensor(goal_anchor)
    task_bounds = _bounds_or_current_x_window(
        cfg.get("task_bounds_xy", None),
        task_pos[..., :2],
    )
    goal_bounds = _bounds_or_current_x_window(
        cfg.get("goal_bounds_xy", None),
        goal_pos[..., :2],
    )
    mirrored_task_pos = task_pos.clone()
    mirrored_goal_pos = goal_pos.clone()
    mirrored_task_pos[..., :2] = _mirror_y_within_bounds(
        task_pos[..., :2],
        bounds=task_bounds,
        cfg=cfg,
        label="MR-SEMP2 mirrored task position",
    )
    mirrored_goal_pos[..., :2] = _mirror_y_within_bounds(
        goal_pos[..., :2],
        bounds=goal_bounds,
        cfg=cfg,
        label="MR-SEMP2 mirrored goal position",
    )

    _set_actor_pose(task_anchor, mirrored_task_pos, task_quat)
    _set_actor_pose(goal_anchor, mirrored_goal_pos, goal_quat)
    _zero_actor_velocity(task_anchor, mirrored_task_pos)
    _zero_actor_velocity(goal_anchor, mirrored_goal_pos)
    env.unwrapped._mr_semp2_runtime = {
        "mirrored": True,
        "mirror_axis_y": float(cfg.get("mirror_axis_y", 0.0)),
        "task_source_xy": [float(v) for v in task_pos[0, :2].tolist()],
        "task_mirrored_xy": [float(v) for v in mirrored_task_pos[0, :2].tolist()],
        "goal_source_xy": [float(v) for v in goal_pos[0, :2].tolist()],
        "goal_mirrored_xy": [float(v) for v in mirrored_goal_pos[0, :2].tolist()],
        "task_delta_xy": [float(v) for v in (mirrored_task_pos[0, :2] - task_pos[0, :2]).tolist()],
        "goal_delta_xy": [float(v) for v in (mirrored_goal_pos[0, :2] - goal_pos[0, :2]).tolist()],
    }
    _apply_scene_updates(env)
    return env


@register_env("MR-SEMP3")
@register_env("MR-SEMP-3")
@register_env("SSGEP-State-Space-Geometric-Equivariance-Pattern")
def env_rotate_task_object_about_z(env, cfg):
    task_anchor = _find_task_object_anchor(env)
    if task_anchor is None:
        raise ValueError("MR-SEMP3 could not find a task object actor")

    goal_anchor = _find_goal_anchor(env)
    if goal_anchor is None:
        raise ValueError("MR-SEMP3 could not find a goal/target/region actor")

    task_pos, task_quat = _actor_pose_tensor(task_anchor)
    goal_pos, goal_quat = _actor_pose_tensor(goal_anchor)

    yaw_deg = float(cfg.get("yaw_deg", 45.0))
    relative = bool(cfg.get("relative", True))
    rotate_goal = bool(cfg.get("rotate_goal", False))
    maintain_relative_xy = bool(cfg.get("maintain_relative_xy", True))
    task_bounds = cfg.get("task_bounds_xy", [[-0.35, -0.30], [0.35, 0.30]])
    goal_bounds = cfg.get("goal_bounds_xy", [[-0.35, -0.30], [0.35, 0.30]])
    rotation_center_xy = torch.as_tensor(
        cfg.get("rotation_center_xy", [0.0, 0.0]),
        dtype=task_pos.dtype,
        device=task_pos.device,
    ).reshape(-1)
    if rotation_center_xy.numel() < 2:
        raise ValueError("MR-SEMP3 rotation_center_xy must contain at least two values")

    yaw_quat = _yaw_quat_batch(
        batch_size=task_quat.shape[0],
        yaw_deg=yaw_deg,
        device=task_quat.device,
    ).to(dtype=task_quat.dtype)
    rotated_quat = _quat_multiply(yaw_quat, task_quat) if relative else yaw_quat

    rotated_task_pos = task_pos.clone()
    rotated_goal_pos = goal_pos.clone()

    if maintain_relative_xy:
        center_x = float(rotation_center_xy[0].item())
        center_y = float(rotation_center_xy[1].item())
        theta = math.radians(float(yaw_deg))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        def _rotate_xy(pos_batch):
            rel_x = pos_batch[:, 0] - center_x
            rel_y = pos_batch[:, 1] - center_y
            out = pos_batch.clone()
            out[:, 0] = center_x + cos_t * rel_x - sin_t * rel_y
            out[:, 1] = center_y + sin_t * rel_x + cos_t * rel_y
            return out

        rotated_task_pos = _rotate_xy(task_pos)
        if rotate_goal:
            rotated_goal_pos = _rotate_xy(goal_pos)
    elif rotate_goal:
        rotated_goal_pos = goal_pos.clone()
        rotated_goal_pos[:, :2] = task_pos[:, :2]

    rotated_task_pos[..., :2] = _validate_xy_within_bounds(
        rotated_task_pos[..., :2],
        bounds=task_bounds,
        label="MR-SEMP3 rotated task position",
    )
    if rotate_goal:
        rotated_goal_pos[..., :2] = _validate_xy_within_bounds(
            rotated_goal_pos[..., :2],
            bounds=goal_bounds,
            label="MR-SEMP3 rotated goal position",
        )

    _set_actor_pose(task_anchor, rotated_task_pos, rotated_quat)
    _zero_actor_velocity(task_anchor, rotated_task_pos)

    if rotate_goal:
        _set_actor_pose(goal_anchor, rotated_goal_pos, goal_quat)
        _zero_actor_velocity(goal_anchor, rotated_goal_pos)

    env.unwrapped._mr_semp3_runtime = {
        "yaw_deg": float(yaw_deg),
        "relative": bool(relative),
        "rotate_goal": bool(rotate_goal),
        "maintain_relative_xy": bool(maintain_relative_xy),
        "rotation_center_xy": [float(v) for v in rotation_center_xy[:2].tolist()],
        "task_source_xy": [float(v) for v in task_pos[0, :2].tolist()],
        "task_rotated_xy": [float(v) for v in rotated_task_pos[0, :2].tolist()],
        "goal_source_xy": [float(v) for v in goal_pos[0, :2].tolist()],
        "goal_rotated_xy": [float(v) for v in rotated_goal_pos[0, :2].tolist()],
    }
    _apply_scene_updates(env)
    return env


@register_env("MR-SESP1")
@register_env("MR-SESP-1")
@register_env("SESP-Spatial-Translation-Equivariance")
def env_translate_task_and_goal_scene_y(env, cfg):
    task_anchor = _find_task_object_anchor(env)
    if task_anchor is None:
        raise ValueError("MR-SESP1 could not find a task object actor")

    goal_anchor = _find_goal_anchor(env)
    if goal_anchor is None:
        raise ValueError("MR-SESP1 could not find a goal/target/region actor")

    dx = float(cfg.get("dx", 0.0))
    dy = float(cfg.get("dy", -0.1))
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return env

    translation_bounds = cfg.get(
        "translation_bounds_xy",
        [[-0.35, -0.30], [0.35, 0.30]],
    )

    task_pos, task_quat = _actor_pose_tensor(task_anchor)
    goal_pos, goal_quat = _actor_pose_tensor(goal_anchor)
    resolved_dx, resolved_dy = _resolve_translation_vector(
        task_pos[..., :2],
        goal_pos[..., :2],
        dx=dx,
        dy=dy,
        bounds=translation_bounds,
        cfg=cfg,
    )

    translated_task_pos = task_pos.clone()
    translated_goal_pos = goal_pos.clone()
    translated_task_pos[:, 0] += resolved_dx
    translated_task_pos[:, 1] += resolved_dy
    translated_goal_pos[:, 0] += resolved_dx
    translated_goal_pos[:, 1] += resolved_dy

    _set_actor_pose(task_anchor, translated_task_pos, task_quat)
    _set_actor_pose(goal_anchor, translated_goal_pos, goal_quat)
    _zero_actor_velocity(task_anchor, translated_task_pos)
    _zero_actor_velocity(goal_anchor, translated_goal_pos)
    env.unwrapped._mr_sesp1_runtime = {
        "dx": float(resolved_dx),
        "dy": float(resolved_dy),
    }
    _apply_scene_updates(env)
    return env
