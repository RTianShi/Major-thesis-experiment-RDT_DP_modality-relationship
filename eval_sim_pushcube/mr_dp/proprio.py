"""PushCube proprio mutations."""

import numpy as np
import torch

from eval_sim.mr.proprio import *  # noqa: F401,F403

from .registry import register_proprio


def _resolve_target_indices(proprio, target_indices):
    last_dim = proprio.shape[-1]
    resolved = []
    for idx in target_indices:
        idx = int(idx)
        if idx < 0:
            idx = last_dim + idx
        if 0 <= idx < last_dim:
            resolved.append(idx)
    return sorted(set(resolved))


def _clone_like(proprio):
    if torch.is_tensor(proprio):
        return proprio.clone()
    return proprio.copy()


@register_proprio("MR-SESP1")
@register_proprio("MR-SESP-1")
@register_proprio("SESP-Spatial-Translation-Equivariance")
def prop_spatial_translation_equivariance(proprio, cfg):
    # DP 当前实现里，policy 主要消费的是 ManiSkill 的关节 qpos（约 14 维），
    # 这类 proprio 对“全局平移”等变通常天然不敏感，因此默认应当是安全 no-op。
    # 但当上游提供的是统一 state 向量（128 维，包含 eef_pos_y=31）时，
    # 这里会自动对 eef_pos_y 做平移补偿，使 MR 在该表示下可触发。

    mode = str(cfg.get("mode", "auto")).strip().lower()
    if mode in {"keep_initial_pose", "identity", "none"}:
        return proprio

    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    apply_mode = str(cfg.get("apply_mode", "always")).strip().lower()
    if apply_mode == "initial" and step_index != 0:
        return proprio

    dy = float(cfg.get("dy", -0.1))
    scale = float(cfg.get("scale", 1.0))
    delta = float(dy * scale)
    if abs(delta) < 1e-12:
        return proprio

    target_indices = cfg.get("target_indices", None)
    if target_indices is None:
        if mode == "auto":
            last_dim = int(proprio.shape[-1])
            # unified state vec mapping: eef_pos_y index is 31
            target_indices = [31] if last_dim > 31 else []
        else:
            target_indices = []

    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    if torch.is_tensor(proprio):
        out = proprio.clone()
        out[..., resolved_indices] = out[..., resolved_indices] + delta
        return out

    out = proprio.copy()
    out[..., resolved_indices] = out[..., resolved_indices] + delta
    return out


@register_proprio("MR-LTSEP1")
@register_proprio("MR-LTSEP-1")
@register_proprio("LTSEP-Vision-Proprio-Phantom-Tear")
def prop_phantom_collision_tear(proprio, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    tcp_to_obj_distance = float(runtime.get("tcp_to_obj_distance", float("nan")))
    is_contact = bool(runtime.get("is_contact", False))
    is_grasped = bool(runtime.get("is_grasped", False))

    # PushCube DP: no grasp required. Keep this MR independent of gripper state,
    # and rely on proximity/contact signals that eval_dp.py already provides.
    if is_grasped:
        return proprio

    contact_distance = float(cfg.get("contact_distance", 0.035))
    if is_contact:
        return proprio
    if torch.isfinite(torch.tensor(tcp_to_obj_distance)) and tcp_to_obj_distance <= contact_distance:
        return proprio

    # Phantom tear should happen when the TCP is close to the cube but not actually in contact.
    # This avoids constantly perturbing proprio when the robot is far away.
    phantom_min_distance = float(cfg.get("phantom_min_distance", contact_distance))
    phantom_max_distance = float(cfg.get("phantom_max_distance", contact_distance * 2.0))
    if not torch.isfinite(torch.tensor(tcp_to_obj_distance)):
        return proprio
    if tcp_to_obj_distance < phantom_min_distance or tcp_to_obj_distance > phantom_max_distance:
        return proprio

    # Optional step gating.
    # NOTE: previous versions defaulted trigger_until_step=trigger_step=0, which made this MR
    # almost never fire (only possible at step 0). We only gate when the user explicitly
    # provides trigger_step/trigger_until_step.
    trigger_step_raw = cfg.get("trigger_step", None)
    trigger_until_step_raw = cfg.get("trigger_until_step", None)
    if trigger_step_raw is not None or trigger_until_step_raw is not None:
        trigger_step = int(0 if trigger_step_raw is None else trigger_step_raw)
        trigger_until_step = int(trigger_step if trigger_until_step_raw is None else trigger_until_step_raw)
        if step_index < trigger_step or step_index > trigger_until_step:
            return proprio

    fired_step = cfg.get("_phantom_fired_step", None)
    duration_steps = max(1, int(cfg.get("duration_steps", 1)))
    retrigger = bool(cfg.get("retrigger", False))
    if fired_step is not None:
        active = step_index < int(fired_step) + duration_steps
        if active:
            pass
        elif not retrigger:
            return proprio
        else:
            fired_step = None

    target_indices = cfg.get("target_indices", None)
    if target_indices is None:
        last_dim = int(proprio.shape[-1])
        # For PushCube, perturbing gripper qpos usually has little effect on the policy.
        # Prefer perturbing a few arm joints by default.
        if 7 <= last_dim <= 32:
            target_indices = [0, 1, 2]
        else:
            target_indices = [-2, -1] if last_dim >= 2 else [-1]
    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    # DP uses raw qpos (then often normalizes). Default magnitude is intentionally moderate,
    # but can be overridden via cfg["magnitude"].
    magnitude = float(cfg.get("magnitude", 0.35))
    pulse_values = cfg.get("pulse_values", None)
    if pulse_values is not None and len(pulse_values) > 0:
        if torch.is_tensor(proprio):
            pulse = torch.as_tensor(
                pulse_values,
                dtype=proprio.dtype,
                device=proprio.device,
            ).reshape(-1)
        else:
            pulse = pulse_values
    else:
        if torch.is_tensor(proprio):
            pulse = torch.full(
                (len(resolved_indices),),
                fill_value=-abs(magnitude),
                dtype=proprio.dtype,
                device=proprio.device,
            )
        else:
            pulse = [-abs(magnitude)] * len(resolved_indices)

    if fired_step is None:
        cfg["_phantom_fired_step"] = step_index
        fired_steps = cfg.setdefault("_phantom_fired_steps", [])
        if isinstance(fired_steps, list):
            fired_steps.append(int(step_index))

    clamp_gripper = bool(cfg.get("clamp_gripper", True))
    gripper_lower = float(cfg.get("gripper_lower", 0.0))
    gripper_upper = float(cfg.get("gripper_upper", 0.04))

    out = _clone_like(proprio)
    if torch.is_tensor(out):
        pulse = pulse[: len(resolved_indices)]
        out[..., resolved_indices] = out[..., resolved_indices] + pulse
        if clamp_gripper:
            last_dim = int(out.shape[-1])
            # Heuristic: ManiSkill qpos often has two gripper joints at the tail.
            is_tail_gripper = last_dim <= 32 and all(int(i) >= last_dim - 2 for i in resolved_indices)
            if is_tail_gripper:
                out[..., resolved_indices] = torch.clamp(out[..., resolved_indices], gripper_lower, gripper_upper)
        return out

    pulse = pulse[: len(resolved_indices)]
    out[..., resolved_indices] = out[..., resolved_indices] + pulse
    if clamp_gripper:
        last_dim = int(out.shape[-1])
        is_tail_gripper = last_dim <= 32 and all(int(i) >= last_dim - 2 for i in resolved_indices)
        if is_tail_gripper:
            out[..., resolved_indices] = np.clip(out[..., resolved_indices], gripper_lower, gripper_upper)
    return out


@register_proprio("MR-LTSEP2")
@register_proprio("MR-LTSEP-2")
@register_proprio("LTSEP-Spatial-Dislocation-Tear")
def prop_spatial_dislocation_tear(proprio, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    tcp_to_obj_distance = runtime.get("tcp_to_obj_distance", None)
    is_contact = bool(runtime.get("is_contact", False))

    # PushCube DP does not rely on grasp; use contact/proximity as the default activation signal.
    activate_after_contact = bool(cfg.get("activate_after_contact", True))
    contact_distance = float(cfg.get("contact_distance", 0.08))
    fired_step = cfg.get("_ltsep2_fired_step", None)
    if activate_after_contact and fired_step is None:
        close_enough = False
        try:
            d = float(tcp_to_obj_distance)
            close_enough = np.isfinite(d) and d <= contact_distance
        except Exception:
            close_enough = False
        if not is_contact and not close_enough:
            return proprio
        cfg["_ltsep2_fired_step"] = step_index
        fired_step = step_index

    # Optional step gating (only active when user provides either bound).
    trigger_step_raw = cfg.get("trigger_step", None)
    trigger_until_step_raw = cfg.get("trigger_until_step", None)
    if trigger_step_raw is not None or trigger_until_step_raw is not None:
        trigger_step = int(0 if trigger_step_raw is None else trigger_step_raw)
        trigger_until_step = int(trigger_step if trigger_until_step_raw is None else trigger_until_step_raw)
        if step_index < trigger_step or step_index > trigger_until_step:
            return proprio

    # Which proprio dimensions to mutate.
    # DP PushCube low-dim is typically qpos with 7 arm joints + 2 gripper joints.
    target_indices = cfg.get("target_indices", None)
    if target_indices is None:
        last_dim = int(proprio.shape[-1])
        target_indices = list(range(min(9, last_dim)))
    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    # Values to inject.
    target_pose = cfg.get("target_pose", None)
    if target_pose is None:
        target_pose = cfg.get(
            "folded_pose",
            [2.8, -2.2, 2.8, -2.6, 2.4, -1.8, 2.1, 0.04, 0.04],
        )
    additive = bool(cfg.get("additive", False))
    scale = float(cfg.get("scale", 1.75))

    clamp_gripper = bool(cfg.get("clamp_gripper", True))
    gripper_lower = float(cfg.get("gripper_lower", 0.0))
    gripper_upper = float(cfg.get("gripper_upper", 0.04))

    out = _clone_like(proprio)
    if torch.is_tensor(out):
        vals = torch.as_tensor(target_pose, dtype=out.dtype, device=out.device).reshape(-1)
        width = min(len(resolved_indices), int(vals.numel()))
        if width <= 0:
            return out
        idx = resolved_indices[:width]
        if additive:
            out[..., idx] = out[..., idx] + vals[:width] * scale
        else:
            out[..., idx] = vals[:width] * scale

        if bool(cfg.get("mirror_remaining_dims", False)) and out.shape[-1] > width and width < out.shape[-1]:
            out[..., width:] = -out[..., width:]

        if clamp_gripper:
            last_dim = int(out.shape[-1])
            is_tail_gripper = last_dim <= 32 and any(int(i) >= last_dim - 2 for i in resolved_indices)
            if is_tail_gripper:
                tail_idx = [i for i in resolved_indices if int(i) >= last_dim - 2]
                if tail_idx:
                    out[..., tail_idx] = torch.clamp(out[..., tail_idx], gripper_lower, gripper_upper)
        return out

    vals = np.asarray(list(target_pose), dtype=np.asarray(out).dtype).reshape(-1)
    width = min(len(resolved_indices), int(vals.size))
    if width <= 0:
        return out
    idx = resolved_indices[:width]
    if additive:
        out[..., idx] = out[..., idx] + vals[:width] * scale
    else:
        out[..., idx] = vals[:width] * scale

    if bool(cfg.get("mirror_remaining_dims", False)) and out.shape[-1] > width and width < out.shape[-1]:
        out[..., width:] = -out[..., width:]

    if clamp_gripper:
        last_dim = int(out.shape[-1])
        is_tail_gripper = last_dim <= 32 and any(int(i) >= last_dim - 2 for i in resolved_indices)
        if is_tail_gripper:
            tail_idx = [i for i in resolved_indices if int(i) >= last_dim - 2]
            if tail_idx:
                out[..., tail_idx] = np.clip(out[..., tail_idx], gripper_lower, gripper_upper)
    return out

@register_proprio("MR-SADP2")
@register_proprio("MR-SADP-2")
@register_proprio("SADP-Intra-Modal-Background-Noise")
def prop_sadp2_safe_joint_gaussian_noise(proprio, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    apply_mode = str(cfg.get("apply_mode", "initial")).strip().lower()
    visual_noise_mode = str(runtime.get("visual_noise_mode", cfg.get("visual_noise_mode", "material+background"))).strip().lower()
    apply_material_noise = bool(runtime.get("apply_material_noise", cfg.get("apply_material_noise", True)))
    apply_background_noise = bool(runtime.get("apply_background_noise", cfg.get("apply_background_noise", True)))
    apply_camera_noise = bool(runtime.get("apply_camera_noise", cfg.get("apply_camera_noise", False)))

    if apply_mode == "initial" and step_index != 0:
        return proprio

    if not (apply_material_noise or apply_background_noise or apply_camera_noise):
        return proprio

    sigma_ratio = float(cfg.get("sigma_ratio", 0.01))
    min_sigma = float(cfg.get("min_sigma", 0.0025))
    if visual_noise_mode in {"background", "camera"}:
        sigma_ratio *= float(cfg.get("background_sigma_scale", 1.15))
        min_sigma *= float(cfg.get("background_min_sigma_scale", 1.10))
    if apply_camera_noise:
        sigma_ratio *= float(cfg.get("camera_sigma_scale", 0.75))
        min_sigma *= float(cfg.get("camera_min_sigma_scale", 0.75))
    target_indices = cfg.get("target_indices", None)
    if sigma_ratio <= 0 and min_sigma <= 0:
        return proprio

    if torch.is_tensor(proprio):
        out = proprio.clone()
        if target_indices:
            resolved_indices = _resolve_target_indices(out, target_indices)
        else:
            last_dim = int(out.shape[-1])
            resolved_indices = list(range(min(9, last_dim))) if last_dim <= 32 else [31]
        if not resolved_indices:
            return out

        base = torch.abs(out[..., resolved_indices])
        sigma = torch.clamp(base * sigma_ratio, min=min_sigma)
        noise = torch.randn_like(out[..., resolved_indices]) * sigma
        out[..., resolved_indices] = out[..., resolved_indices] + noise
        return out

    out = proprio.copy()
    if target_indices:
        resolved_indices = _resolve_target_indices(out, target_indices)
    else:
        last_dim = int(out.shape[-1])
        resolved_indices = list(range(min(9, last_dim))) if last_dim <= 32 else [31]
    if not resolved_indices:
        return out

    base = abs(out[..., resolved_indices])
    sigma = base * sigma_ratio
    sigma[sigma < min_sigma] = min_sigma
    noise = np.random.randn(*out[..., resolved_indices].shape) * sigma
    out[..., resolved_indices] = out[..., resolved_indices] + noise
    return out


@register_proprio("MR6")
@register_proprio("MR-6")
@register_proprio("Action-Optimality-Completeness")
def prop_mr6_precontact_lateral_jog(proprio, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    tcp_to_obj_distance = float(runtime.get("tcp_to_obj_distance", float("nan")))
    is_contact = bool(runtime.get("is_contact", False))
    contact_distance = float(cfg.get("contact_distance", 0.035))

    if is_contact:
        return proprio
    if torch.isfinite(torch.tensor(tcp_to_obj_distance)) and tcp_to_obj_distance <= contact_distance:
        return proprio

    fired_step = cfg.get("_mr6_fired_step", None)
    if fired_step is None:
        trigger_step = int(cfg.get("trigger_step", 1))
        if step_index < trigger_step:
            return proprio
        cfg["_mr6_fired_step"] = step_index
        fired_step = step_index

    phase = int(step_index - int(fired_step))
    if phase not in (0, 1):
        return proprio

    target_indices = cfg.get("target_indices", [0])
    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    lateral_offset = float(cfg.get("lateral_offset", 0.02))
    signed_offset = lateral_offset if phase == 0 else -lateral_offset

    if torch.is_tensor(proprio):
        out = proprio.clone()
        out[..., resolved_indices] = out[..., resolved_indices] + signed_offset
        return out

    out = proprio.copy()
    out[..., resolved_indices] = out[..., resolved_indices] + signed_offset
    return out
