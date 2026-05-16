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
    mode = str(cfg.get("mode", "keep_initial_pose")).strip().lower()
    if mode in {"keep_initial_pose", "identity", "none"}:
        return proprio

    dy = float(cfg.get("dy", -0.1))
    scale = float(cfg.get("scale", 1.0))
    target_indices = cfg.get("target_indices", [])
    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    delta = dy * scale
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
    is_grasped = bool(runtime.get("is_grasped", False))

    if is_grasped:
        return proprio

    contact_distance = float(cfg.get("contact_distance", 0.035))
    if torch.isfinite(torch.tensor(tcp_to_obj_distance)) and tcp_to_obj_distance <= contact_distance:
        return proprio

    trigger_step = int(cfg.get("trigger_step", 0))
    trigger_until_step = int(cfg.get("trigger_until_step", trigger_step))
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

    target_indices = cfg.get("target_indices", [0, 1, 2])
    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    magnitude = float(cfg.get("magnitude", 0.8))
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

    out = _clone_like(proprio)
    if torch.is_tensor(out):
        pulse = pulse[: len(resolved_indices)]
        out[..., resolved_indices] = out[..., resolved_indices] + pulse
        return out

    pulse = pulse[: len(resolved_indices)]
    out[..., resolved_indices] = out[..., resolved_indices] + pulse
    return out


@register_proprio("MR-LTSEP2")
@register_proprio("MR-LTSEP-2")
@register_proprio("LTSEP-Spatial-Dislocation-Tear")
def prop_spatial_dislocation_tear(proprio, cfg):
    target_pose = cfg.get("target_pose", None)
    if target_pose is None:
        target_pose = cfg.get(
            "folded_pose",
            [0.0, 1.35, -1.8, -0.2, 1.57, 1.1, -0.8],
        )

    if torch.is_tensor(proprio):
        out = proprio.clone()
        folded = torch.as_tensor(
            target_pose,
            dtype=out.dtype,
            device=out.device,
        ).reshape(-1)
        width = min(out.shape[-1], folded.numel())
        if width <= 0:
            return out
        out[..., :width] = folded[:width]
        if bool(cfg.get("mirror_remaining_dims", False)) and out.shape[-1] > width:
            out[..., width:] = -out[..., width:]
        return out

    out = proprio.copy()
    folded = list(target_pose)
    width = min(out.shape[-1], len(folded))
    if width <= 0:
        return out
    out[..., :width] = folded[:width]
    if bool(cfg.get("mirror_remaining_dims", False)) and out.shape[-1] > width:
        out[..., width:] = -out[..., width:]
    return out

@register_proprio("MR-SADP2")
@register_proprio("MR-SADP-2")
@register_proprio("SADP-Intra-Modal-Background-Noise")
def prop_sadp2_safe_joint_gaussian_noise(proprio, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    apply_mode = str(cfg.get("apply_mode", "initial")).strip().lower()

    if apply_mode == "initial" and step_index != 0:
        return proprio

    sigma_ratio = float(cfg.get("sigma_ratio", 0.01))
    min_sigma = float(cfg.get("min_sigma", 0.0025))
    target_indices = cfg.get("target_indices", None)
    if sigma_ratio <= 0 and min_sigma <= 0:
        return proprio

    if torch.is_tensor(proprio):
        out = proprio.clone()
        if target_indices:
            resolved_indices = _resolve_target_indices(out, target_indices)
        else:
            resolved_indices = list(range(out.shape[-1]))
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
        resolved_indices = list(range(out.shape[-1]))
    if not resolved_indices:
        return out

    base = abs(out[..., resolved_indices])
    sigma = base * sigma_ratio
    sigma[sigma < min_sigma] = min_sigma
    noise = np.random.randn(*out[..., resolved_indices].shape) * sigma
    out[..., resolved_indices] = out[..., resolved_indices] + noise
    return out
