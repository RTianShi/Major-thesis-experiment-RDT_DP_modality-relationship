import numpy as np
import torch
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


def _coerce_closed_value_like(proprio, closed_value):
    if torch.is_tensor(proprio):
        return torch.as_tensor(closed_value, dtype=proprio.dtype, device=proprio.device)
    return np.asarray(closed_value, dtype=np.asarray(proprio).dtype)


@register_proprio("identity")
def prop_identity(proprio, cfg):
    return proprio


@register_proprio("zero")
def prop_zero(proprio, cfg):
    if torch.is_tensor(proprio):
        return torch.zeros_like(proprio)
    return np.zeros_like(proprio)


@register_proprio("drop_joints")
def prop_drop_joints(proprio, cfg):
    idx = cfg.get("idx", [])
    if len(idx) == 0:
        return proprio
    if torch.is_tensor(proprio):
        out = proprio.clone()
        out[..., idx] = 0
        return out
    out = proprio.copy()
    out[..., idx] = 0
    return out


@register_proprio("MR-LTSEP2")
def prop_spoof_gripper_closed(proprio, cfg):
    runtime = cfg.get("runtime", {})
    activate_after_grasp = bool(cfg.get("activate_after_grasp", True))
    is_grasped = bool(runtime.get("is_grasped", False))
    if activate_after_grasp and not is_grasped:
        return proprio

    num_dims = int(cfg.get("num_gripper_dims", 2))
    target_indices = cfg.get("target_indices", list(range(-num_dims, 0)))
    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    closed_value = cfg.get("closed_values", runtime.get("gripper_closed_value", cfg.get("closed_value", 0.0)))
    debug = bool(cfg.get("debug", False))
    closed_value = _coerce_closed_value_like(proprio, closed_value)

    if torch.is_tensor(proprio):
        out = proprio.clone()
        before = out[..., resolved_indices].detach().cpu().numpy().copy() if debug else None
        out[..., resolved_indices] = closed_value
        if debug:
            after = out[..., resolved_indices].detach().cpu().numpy().copy()
            print(
                f"[MR-LTSEP2] is_grasped={is_grasped} indices={resolved_indices} "
                f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
            )
        return out

    out = proprio.copy()
    before = out[..., resolved_indices].copy() if debug else None
    out[..., resolved_indices] = closed_value
    if debug:
        after = out[..., resolved_indices].copy()
        print(
            f"[MR-LTSEP2] is_grasped={is_grasped} indices={resolved_indices} "
            f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
        )
    return out


@register_proprio("MR-CPTMP2")
def prop_gripper_width_noise_before_grasp(proprio, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    if is_grasped:
        return proprio

    num_dims = int(cfg.get("num_gripper_dims", 2))
    sigma = float(cfg.get("sigma", 0.08))
    debug = bool(cfg.get("debug", False))
    if num_dims <= 0 or sigma <= 0:
        return proprio

    if torch.is_tensor(proprio):
        out = proprio.clone()
        width = min(num_dims, out.shape[-1])
        before = out[..., -width:].detach().cpu().numpy().copy() if debug else None
        noise = torch.randn_like(out[..., -width:]) * sigma
        out[..., -width:] = out[..., -width:] + noise
        if debug:
            after = out[..., -width:].detach().cpu().numpy().copy()
            print(
                f"[MR-CPTMP2] is_grasped={is_grasped} sigma={sigma} width={width} "
                f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
            )
        return out

    out = proprio.copy()
    width = min(num_dims, out.shape[-1])
    before = out[..., -width:].copy() if debug else None
    noise = np.random.randn(*out[..., -width:].shape) * sigma
    out[..., -width:] = out[..., -width:] + noise
    if debug:
        after = out[..., -width:].copy()
        print(
            f"[MR-CPTMP2] is_grasped={is_grasped} sigma={sigma} width={width} "
            f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
        )
    return out


@register_proprio("MR-FPDP2")
def prop_terminal_region_state_bias_noise(proprio, cfg):
    runtime = cfg.get("runtime", {})
    cube_goal_distance = runtime.get("cube_goal_distance", None)
    trigger_distance = float(cfg.get("trigger_distance", 0.05))
    if cube_goal_distance is None or not np.isfinite(cube_goal_distance):
        return proprio
    if float(cube_goal_distance) >= trigger_distance:
        return proprio

    mean = float(cfg.get("mean", -0.15))
    variance = float(cfg.get("variance", 0.05))
    sigma = float(cfg.get("sigma", np.sqrt(max(variance, 0.0))))
    debug = bool(cfg.get("debug", False))
    target_indices = cfg.get("target_indices", [-4, -3, -2, -1])

    resolved_indices = _resolve_target_indices(proprio, target_indices)
    if not resolved_indices:
        return proprio

    if torch.is_tensor(proprio):
        out = proprio.clone()
        before = out[..., resolved_indices].detach().cpu().numpy().copy() if debug else None
        noise = torch.randn_like(out[..., resolved_indices]) * sigma + mean
        out[..., resolved_indices] = out[..., resolved_indices] + noise
        if debug:
            after = out[..., resolved_indices].detach().cpu().numpy().copy()
            print(
                f"[MR-FPDP2] cube_goal_distance={float(cube_goal_distance):.4f} "
                f"indices={resolved_indices} mean={mean} sigma={sigma} "
                f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
            )
        return out

    out = proprio.copy()
    before = out[..., resolved_indices].copy() if debug else None
    noise = np.random.randn(*out[..., resolved_indices].shape) * sigma + mean
    out[..., resolved_indices] = out[..., resolved_indices] + noise
    if debug:
        after = out[..., resolved_indices].copy()
        print(
            f"[MR-FPDP2] cube_goal_distance={float(cube_goal_distance):.4f} "
            f"indices={resolved_indices} mean={mean} sigma={sigma} "
            f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
        )
    return out
