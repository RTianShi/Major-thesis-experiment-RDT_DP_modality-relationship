"""StackCube-specific proprio mutations."""

import math
import torch
import numpy as np

from .registry import register_proprio


def _clone_tensor(x):
    return x.clone() if torch.is_tensor(x) else torch.as_tensor(x).clone()


def _resolve_indices(last_dim, indices):
    resolved = []
    for idx in indices:
        idx = int(idx)
        if idx < 0:
            idx = last_dim + idx
        if 0 <= idx < last_dim:
            resolved.append(idx)
    return sorted(set(resolved))


def _clamp_by_limits(values, lower, upper):
    if lower is not None:
        values = torch.maximum(values, lower)
    if upper is not None:
        values = torch.minimum(values, upper)
    return values


@register_proprio("MR6")
@register_proprio("MR-6")
@register_proprio("Action-Optimality-Completeness")
def prop_mr6_action_optimality_completeness(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_goal_distance = runtime.get("cube_goal_distance", None)
    previous_cube_goal_distance = runtime.get("previous_cube_goal_distance", None)
    step_index = int(runtime.get("step_index", 0))
    if not is_grasped:
        return proprio
    if cube_goal_distance is None or previous_cube_goal_distance is None:
        return proprio

    out = _clone_tensor(proprio)
    progress_epsilon = float(cfg.get("progress_epsilon", 1e-4))
    moving_toward_goal = float(cube_goal_distance) < float(previous_cube_goal_distance) - progress_epsilon
    if not moving_toward_goal:
        return out

    trigger_distance = float(cfg.get("trigger_distance", 0.10))
    if float(cube_goal_distance) > trigger_distance:
        return out

    detour_cycle = max(1, int(cfg.get("detour_cycle", 6)))
    pause_cycle = max(1, int(cfg.get("pause_cycle", 8)))
    arm_indices = _resolve_indices(out.shape[-1], cfg.get("arm_indices", list(range(min(7, out.shape[-1])))))
    if not arm_indices:
        return out

    mode = str(cfg.get("mode", "pause")).strip().lower()
    if mode == "detour":
        detour_bias = cfg.get("detour_bias", [0.05, -0.04, 0.03, -0.02, 0.01, -0.01, 0.0])
        num_arm_dims = min(len(detour_bias), len(arm_indices))
        if num_arm_dims > 0 and (step_index % detour_cycle) < max(1, detour_cycle // 2):
            bias = torch.as_tensor(detour_bias[:num_arm_dims], dtype=out.dtype, device=out.device)
            out[..., arm_indices[:num_arm_dims]] = out[..., arm_indices[:num_arm_dims]] + bias
    else:
        pause_scale = float(cfg.get("pause_scale", 0.985))
        if (step_index % pause_cycle) == 0:
            out[..., arm_indices] = out[..., arm_indices] * pause_scale
    return out


@register_proprio("MR-SESP1")
@register_proprio("Size-ApertureTorqueSynergy")
def prop_mr_a1_size_aperture_torque_synergy(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_lifted = bool(runtime.get("cube_lifted", False))
    if not is_grasped and not cube_lifted:
        return proprio

    out = _clone_tensor(proprio)
    last_dim = out.shape[-1]

    arm_indices = _resolve_indices(last_dim, cfg.get("arm_indices", list(range(min(7, last_dim)))))
    gripper_indices = _resolve_indices(last_dim, cfg.get("gripper_indices", [-2, -1]))
    if not arm_indices and not gripper_indices:
        return out

    # DP uses raw qpos and normalizes afterwards, so keep perturbations small and joint-aware.
    aperture_delta = float(cfg.get("aperture_delta", 0.006))
    grasp_aperture_scale = float(cfg.get("grasp_aperture_scale", 1.0))
    lift_aperture_scale = float(cfg.get("lift_aperture_scale", 0.35))
    arm_joint_bias = cfg.get(
        "arm_joint_bias",
        [-0.015, 0.010, -0.020, 0.015, -0.010, 0.008, -0.005],
    )
    transit_torque_scale = float(cfg.get("transit_torque_scale", 1.35))
    stationary_torque_scale = float(cfg.get("stationary_torque_scale", 0.85))
    gripper_lower = float(cfg.get("gripper_lower", 0.0))
    gripper_upper = float(cfg.get("gripper_upper", 0.04))

    cube_goal_distance = runtime.get("cube_goal_distance", None)
    previous_cube_goal_distance = runtime.get("previous_cube_goal_distance", None)
    progress_epsilon = float(cfg.get("progress_epsilon", 1e-4))
    moving_toward_goal = (
        cube_goal_distance is not None
        and previous_cube_goal_distance is not None
        and float(cube_goal_distance) < float(previous_cube_goal_distance) - progress_epsilon
    )

    if gripper_indices and (is_grasped or cube_lifted):
        aperture_scale = grasp_aperture_scale if is_grasped else lift_aperture_scale
        per_finger_delta = aperture_delta * aperture_scale / max(1, len(gripper_indices))
        finger_delta = torch.full(
            (*out.shape[:-1], len(gripper_indices)),
            per_finger_delta,
            dtype=out.dtype,
            device=out.device,
        )
        updated = out[..., gripper_indices] + finger_delta
        lower = torch.full_like(updated, gripper_lower)
        upper = torch.full_like(updated, gripper_upper)
        out[..., gripper_indices] = _clamp_by_limits(updated, lower, upper)

    if arm_indices and cube_lifted:
        num_arm_dims = min(len(arm_joint_bias), len(arm_indices))
        if num_arm_dims > 0:
            torque_tensor = torch.as_tensor(
                arm_joint_bias[:num_arm_dims], dtype=out.dtype, device=out.device
            )
            torque_scale = transit_torque_scale if moving_toward_goal else stationary_torque_scale
            out[..., arm_indices[:num_arm_dims]] = (
                out[..., arm_indices[:num_arm_dims]] + torque_tensor * torque_scale
            )
    return out


@register_proprio("MR-SESP2")
@register_proprio("Friction-ForceSynergy")
def prop_mr_sesp2_friction_force_synergy(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_lifted = bool(runtime.get("cube_lifted", False))
    cube_goal_distance = runtime.get("cube_goal_distance", None)
    previous_cube_goal_distance = runtime.get("previous_cube_goal_distance", None)
    if not is_grasped and not cube_lifted:
        return proprio

    step_index = int(runtime.get("step_index", 0))
    out = _clone_tensor(proprio)
    last_dim = out.shape[-1]

    gripper_indices = _resolve_indices(last_dim, cfg.get("gripper_indices", [-2, -1]))
    arm_indices = _resolve_indices(last_dim, cfg.get("arm_indices", list(range(min(7, last_dim)))))
    tactile_indices = cfg.get("tactile_indices", None)

    progress_epsilon = float(cfg.get("progress_epsilon", 1e-4))
    moving_toward_goal = (
        cube_goal_distance is not None
        and previous_cube_goal_distance is not None
        and float(cube_goal_distance) < float(previous_cube_goal_distance) - progress_epsilon
    )

    # 夹爪：摩擦越强，闭合/回弹扰动越明显
    if gripper_indices:
        slip_amplitude = float(cfg.get("slip_amplitude", 0.012))
        slip_frequency = float(cfg.get("slip_frequency", 0.9))
        friction_bias = float(cfg.get("friction_bias", 0.004))
        lift_bias = float(cfg.get("lift_bias", 0.006))
        goal_bias = float(cfg.get("goal_bias", 0.003))
        lower = float(cfg.get("gripper_lower", 0.0))
        upper = float(cfg.get("gripper_upper", 0.04))

        base_bias = friction_bias
        if cube_lifted:
            base_bias += lift_bias
        if moving_toward_goal:
            base_bias += goal_bias

        phase = step_index * slip_frequency
        carrier = math.sin(phase) * slip_amplitude
        jitter = torch.randn_like(out[..., gripper_indices]) * (slip_amplitude * 0.2)
        updated = out[..., gripper_indices] + base_bias + carrier + jitter
        out[..., gripper_indices] = torch.clamp(updated, lower, upper)

    # 机械臂：抓住并抬起时增加轻微力矩偏置
    if arm_indices and (cube_lifted or moving_toward_goal):
        arm_joint_bias = cfg.get(
            "arm_joint_bias",
            [-0.010, 0.008, -0.014, 0.010, -0.008, 0.006, -0.004],
        )
        num_arm_dims = min(len(arm_joint_bias), len(arm_indices))
        if num_arm_dims > 0:
            torque_scale = float(cfg.get("lift_torque_scale", 1.0))
            if moving_toward_goal:
                torque_scale *= float(cfg.get("goal_torque_scale", 1.15))
            torque_tensor = torch.as_tensor(
                arm_joint_bias[:num_arm_dims], dtype=out.dtype, device=out.device
            )
            out[..., arm_indices[:num_arm_dims]] = (
                out[..., arm_indices[:num_arm_dims]] + torque_tensor * torque_scale
            )

    # 触觉/冗余通道：保留小噪声，模拟接触抖动
    if tactile_indices:
        resolved = _resolve_indices(last_dim, tactile_indices)
        if resolved:
            tactile_noise = torch.randn_like(out[..., resolved]) * float(cfg.get("tactile_noise_scale", 0.01))
            out[..., resolved] = out[..., resolved] + tactile_noise

    return out


@register_proprio("MR-LTSEP1")
@register_proprio("Phantom-Grasp")
def prop_mr_ltsep1_phantom_grasp(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    if not is_grasped:
        return proprio

    out = _clone_tensor(proprio)
    last_dim = out.shape[-1]

    # Current DP StackCube proprio uses the last two low-dim channels for gripper state.
    # Phantom grasp should therefore spoof a "fully closed but empty" reading on both channels,
    # instead of only overwriting a legacy single open-width index.
    gripper_indices = _resolve_indices(
        last_dim,
        cfg.get(
            "gripper_indices",
            cfg.get("target_indices", [-2, -1]),
        ),
    )
    if not gripper_indices:
        legacy_gripper_open_index = int(cfg.get("gripper_open_index", -1))
        gripper_indices = _resolve_indices(last_dim, [legacy_gripper_open_index])

    gripper_closed_false_value = float(
        cfg.get(
            "gripper_closed_false_value",
            cfg.get("closed_value", 0.0),
        )
    )
    if gripper_indices:
        out[..., gripper_indices] = gripper_closed_false_value

    force_indices = cfg.get("force_indices", [])
    if force_indices:
        resolved = _resolve_indices(last_dim, force_indices)
        if resolved:
            out[..., resolved] = 0.0
    return out


@register_proprio("MR-LTSEP2")
@register_proprio("Invisible-Obstacle")
def prop_mr_ltsep2_invisible_obstacle(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_goal_distance = runtime.get("cube_goal_distance", None)
    previous_cube_goal_distance = runtime.get("previous_cube_goal_distance", None)
    if not is_grasped:
        return proprio
    if cube_goal_distance is None or previous_cube_goal_distance is None:
        return proprio

    trigger_distance = float(cfg.get("trigger_distance", 0.18))
    progress_epsilon = float(cfg.get("progress_epsilon", 1e-4))
    moving_toward_goal = float(cube_goal_distance) < float(previous_cube_goal_distance) - progress_epsilon
    if not moving_toward_goal:
        return proprio
    if float(cube_goal_distance) > trigger_distance:
        return proprio

    out = _clone_tensor(proprio)
    last_dim = out.shape[-1]

    # Current DP StackCube low-dim observation uses the tail channels for gripper state.
    # Restrict the invisible-obstacle disturbance to arm joint dimensions only.
    arm_indices = _resolve_indices(
        last_dim,
        cfg.get(
            "arm_indices",
            cfg.get("target_indices", list(range(min(7, last_dim)))),
        ),
    )
    if not arm_indices:
        return out

    torque_bias = cfg.get(
        "collision_torque",
        [-0.45, 0.35, -0.5, 0.42, -0.28, 0.22, -0.14],
    )
    num_arm_dims = min(len(torque_bias), len(arm_indices))
    if num_arm_dims > 0:
        torque_tensor = torch.as_tensor(
            torque_bias[:num_arm_dims], dtype=out.dtype, device=out.device
        )
        disturbance_scale = float(cfg.get("disturbance_scale", cfg.get("torque_scale", 1.0)))
        out[..., arm_indices[:num_arm_dims]] = (
            out[..., arm_indices[:num_arm_dims]] + torque_tensor * disturbance_scale
        )
    return out


@register_proprio("MR-LTSEP3")
@register_proprio("Weightless-Iron")
def prop_mr_ltsep3_weightless_iron(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_lifted = bool(runtime.get("cube_lifted", False))
    if not is_grasped and not cube_lifted:
        return proprio

    buoyant_torque = cfg.get(
        "buoyant_torque",
        [0.12, -0.08, 0.15, -0.1, 0.07, -0.05, 0.03],
    )
    gripper_bias = float(cfg.get("gripper_light_bias", 0.01))
    out = _clone_tensor(proprio)
    num_arm_dims = min(len(buoyant_torque), out.shape[-1])
    if num_arm_dims > 0 and cube_lifted:
        torque_tensor = torch.as_tensor(
            buoyant_torque[:num_arm_dims], dtype=out.dtype, device=out.device
        )
        out[..., :num_arm_dims] = out[..., :num_arm_dims] + torque_tensor
    if is_grasped:
        out[..., -1] = out[..., -1] + gripper_bias
    return out

@register_proprio("MR-SCDP2")
@register_proprio("Proprioceptive-Noise-Debunking")
def prop_mr_scdp2_proprioceptive_noise_debunking(proprio, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    apply_mode = str(cfg.get("apply_mode", "always")).strip().lower()
    if apply_mode == "initial" and step_index != 0:
        return proprio

    is_torch = torch.is_tensor(proprio)
    out = proprio.clone() if is_torch else np.array(proprio, copy=True)
    last_dim = int(out.shape[-1])

    # Current DP StackCube proprio is raw qpos and is normalized later.
    # It typically contains 7 arm joints followed by 2 gripper finger qpos channels.
    arm_indices = _resolve_indices(last_dim, cfg.get("arm_indices", list(range(min(7, last_dim)))))
    gripper_indices = _resolve_indices(last_dim, cfg.get("gripper_indices", [-2, -1]))

    redundant_joint_indices = _resolve_indices(last_dim, cfg.get("redundant_joint_indices", [5, 6]))
    if arm_indices:
        arm_set = set(arm_indices)
        redundant_joint_indices = [i for i in redundant_joint_indices if i in arm_set]
    if gripper_indices:
        gripper_set = set(gripper_indices)
        redundant_joint_indices = [i for i in redundant_joint_indices if i not in gripper_set]

    jitter_amplitude = float(cfg.get("jitter_amplitude", 0.08))
    jitter_frequency = float(cfg.get("jitter_frequency", 1.7))
    jitter_noise_scale = float(cfg.get("jitter_noise_scale", 0.02))

    if redundant_joint_indices:
        phase = step_index * jitter_frequency
        carrier = math.sin(phase) * jitter_amplitude
        if is_torch:
            jitter = torch.randn_like(out[..., redundant_joint_indices]) * jitter_noise_scale
            out[..., redundant_joint_indices] = out[..., redundant_joint_indices] + carrier + jitter
        else:
            jitter = np.random.randn(*out[..., redundant_joint_indices].shape) * jitter_noise_scale
            out[..., redundant_joint_indices] = out[..., redundant_joint_indices] + carrier + jitter

    # Current StackCube proprio does not include a true temperature sensor channel.
    # If pseudo-temperature indices are provided, overwrite them with elevated-but-subcritical values.
    temperature_indices = _resolve_indices(last_dim, cfg.get("temperature_indices", []))
    if temperature_indices:
        occupied = set(arm_indices) | set(gripper_indices)
        temperature_indices = [i for i in temperature_indices if i not in occupied]
    if temperature_indices:
        temperature_low = float(cfg.get("temperature_low", 0.65))
        temperature_high = float(cfg.get("temperature_high", 0.85))
        if temperature_high < temperature_low:
            temperature_high = temperature_low
        if is_torch:
            rand = torch.rand_like(out[..., temperature_indices])
            elevated = temperature_low + rand * (temperature_high - temperature_low)
            out[..., temperature_indices] = elevated
        else:
            rand = np.random.rand(*out[..., temperature_indices].shape)
            elevated = temperature_low + rand * (temperature_high - temperature_low)
            out[..., temperature_indices] = elevated

    return out
