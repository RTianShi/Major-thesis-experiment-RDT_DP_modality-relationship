"""StackCube-specific proprio mutations."""

import math
import torch

from .registry import register_proprio


@register_proprio("MR-SESP1")
@register_proprio("Size-ApertureTorqueSynergy")
def prop_mr_a1_size_aperture_torque_synergy(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_lifted = bool(runtime.get("cube_lifted", False))
    if not is_grasped and not cube_lifted:
        return proprio

    aperture_bias = float(cfg.get("aperture_bias", 0.035))
    num_gripper_dims = max(1, int(cfg.get("num_gripper_dims", 2)))
    torque_bias = cfg.get(
        "torque_bias",
        [-0.06, 0.04, -0.08, 0.06, -0.04, 0.03, -0.02],
    )

    out = proprio.clone() if torch.is_tensor(proprio) else torch.as_tensor(proprio).clone()
    gripper_width = min(num_gripper_dims, out.shape[-1])
    if is_grasped:
        out[..., -gripper_width:] = out[..., -gripper_width:] + aperture_bias

    if cube_lifted:
        num_arm_dims = min(len(torque_bias), out.shape[-1] - gripper_width)
        if num_arm_dims > 0:
            torque_tensor = torch.as_tensor(
                torque_bias[:num_arm_dims], dtype=out.dtype, device=out.device
            )
            out[..., :num_arm_dims] = out[..., :num_arm_dims] + torque_tensor
    return out


@register_proprio("MR-SESP2")
@register_proprio("Friction-ForceSynergy")
def prop_mr_sesp2_friction_force_synergy(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    if not is_grasped:
        return proprio

    step_index = int(runtime.get("step_index", 0))
    num_gripper_dims = max(1, int(cfg.get("num_gripper_dims", 2)))
    slip_amplitude = float(cfg.get("slip_amplitude", 0.012))
    slip_frequency = float(cfg.get("slip_frequency", 0.9))
    tactile_indices = cfg.get("tactile_indices", None)

    out = proprio.clone() if torch.is_tensor(proprio) else torch.as_tensor(proprio).clone()
    phase = step_index * slip_frequency
    carrier = torch.sin(
        torch.as_tensor(phase, dtype=out.dtype, device=out.device)
    ) * slip_amplitude
    high_freq = torch.randn_like(out[..., -num_gripper_dims:]) * (slip_amplitude * 0.35)
    out[..., -num_gripper_dims:] = out[..., -num_gripper_dims:] + carrier + high_freq

    if tactile_indices:
        resolved = []
        last_dim = out.shape[-1]
        for idx in tactile_indices:
            idx = int(idx)
            if idx < 0:
                idx = last_dim + idx
            if 0 <= idx < last_dim:
                resolved.append(idx)
        if resolved:
            tactile_noise = torch.randn_like(out[..., resolved]) * (slip_amplitude * 0.5)
            out[..., resolved] = out[..., resolved] + tactile_noise
    return out


@register_proprio("MR-LTSEP1")
@register_proprio("Phantom-Grasp")
def prop_mr_ltsep1_phantom_grasp(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    if not is_grasped:
        return proprio

    gripper_open_false_value = float(cfg.get("gripper_open_false_value", 0.04))
    gripper_open_index = int(cfg.get("gripper_open_index", -1))
    force_indices = cfg.get("force_indices", [])

    out = proprio.clone() if torch.is_tensor(proprio) else torch.as_tensor(proprio).clone()
    last_dim = out.shape[-1]
    if gripper_open_index < 0:
        gripper_open_index = last_dim + gripper_open_index
    if 0 <= gripper_open_index < last_dim:
        out[..., gripper_open_index] = gripper_open_false_value

    if force_indices:
        resolved = []
        for idx in force_indices:
            idx = int(idx)
            if idx < 0:
                idx = last_dim + idx
            if 0 <= idx < last_dim:
                resolved.append(idx)
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

    torque_bias = cfg.get(
        "collision_torque",
        [-0.45, 0.35, -0.5, 0.42, -0.28, 0.22, -0.14],
    )
    out = proprio.clone() if torch.is_tensor(proprio) else torch.as_tensor(proprio).clone()
    num_arm_dims = min(len(torque_bias), out.shape[-1])
    if num_arm_dims > 0:
        torque_tensor = torch.as_tensor(
            torque_bias[:num_arm_dims], dtype=out.dtype, device=out.device
        )
        out[..., :num_arm_dims] = out[..., :num_arm_dims] + torque_tensor
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
    out = proprio.clone() if torch.is_tensor(proprio) else torch.as_tensor(proprio).clone()
    num_arm_dims = min(len(buoyant_torque), out.shape[-1])
    if num_arm_dims > 0 and cube_lifted:
        torque_tensor = torch.as_tensor(
            buoyant_torque[:num_arm_dims], dtype=out.dtype, device=out.device
        )
        out[..., :num_arm_dims] = out[..., :num_arm_dims] + torque_tensor
    if is_grasped:
        out[..., -1] = out[..., -1] + gripper_bias
    return out


def _resolve_indices(last_dim, indices):
    resolved = []
    for idx in indices:
        idx = int(idx)
        if idx < 0:
            idx = last_dim + idx
        if 0 <= idx < last_dim:
            resolved.append(idx)
    return sorted(set(resolved))


@register_proprio("MR-SCDP2")
@register_proprio("Proprioceptive-Noise-Debunking")
def prop_mr_scdp2_proprioceptive_noise_debunking(proprio, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    out = proprio.clone() if torch.is_tensor(proprio) else torch.as_tensor(proprio).clone()
    last_dim = out.shape[-1]

    redundant_joint_indices = _resolve_indices(
        last_dim, cfg.get("redundant_joint_indices", [5, 6])
    )
    jitter_amplitude = float(cfg.get("jitter_amplitude", 0.08))
    jitter_frequency = float(cfg.get("jitter_frequency", 1.7))
    jitter_noise_scale = float(cfg.get("jitter_noise_scale", 0.02))

    if redundant_joint_indices:
        phase = step_index * jitter_frequency
        carrier = math.sin(phase) * jitter_amplitude
        jitter = torch.randn_like(out[..., redundant_joint_indices]) * jitter_noise_scale
        out[..., redundant_joint_indices] = out[..., redundant_joint_indices] + carrier + jitter

    # Current StackCube proprio does not include a true temperature sensor channel.
    # If pseudo-temperature indices are provided, overwrite them with elevated-but-subcritical values.
    temperature_indices = _resolve_indices(
        last_dim, cfg.get("temperature_indices", [])
    )
    if temperature_indices:
        temperature_low = float(cfg.get("temperature_low", 0.65))
        temperature_high = float(cfg.get("temperature_high", 0.85))
        if temperature_high < temperature_low:
            temperature_high = temperature_low
        rand = torch.rand_like(out[..., temperature_indices])
        elevated = temperature_low + rand * (temperature_high - temperature_low)
        out[..., temperature_indices] = elevated

    return out
