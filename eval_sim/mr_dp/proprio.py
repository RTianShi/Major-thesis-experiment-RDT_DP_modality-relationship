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

    sigma = float(cfg.get("sigma", 0.08))
    debug = bool(cfg.get("debug", False))
    if sigma <= 0:
        return proprio

    # DP 当前使用的 low-dim 输入是 agent_pos[8]，最后两个维度对应夹爪开合状态。
    # 因此 MR-CPTMP2 在该链路下明确只扰动最后两个夹爪维度，
    # 并且仅在抓取建立前生效；一旦 is_grasped=True，立刻恢复正常输入。
    target_indices = _resolve_target_indices(proprio, cfg.get("target_indices", [-2, -1]))
    if not target_indices:
        return proprio

    if torch.is_tensor(proprio):
        out = proprio.clone()
        before = out[..., target_indices].detach().cpu().numpy().copy() if debug else None
        noise = torch.randn_like(out[..., target_indices]) * sigma
        out[..., target_indices] = out[..., target_indices] + noise
        if debug:
            after = out[..., target_indices].detach().cpu().numpy().copy()
            print(
                f"[MR-CPTMP2] is_grasped={is_grasped} sigma={sigma} indices={target_indices} "
                f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
            )
        return out

    out = proprio.copy()
    before = out[..., target_indices].copy() if debug else None
    noise = np.random.randn(*out[..., target_indices].shape) * sigma
    out[..., target_indices] = out[..., target_indices] + noise
    if debug:
        after = out[..., target_indices].copy()
        print(
            f"[MR-CPTMP2] is_grasped={is_grasped} sigma={sigma} indices={target_indices} "
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

    mean = float(cfg.get("mean", 0.15))
    variance = float(cfg.get("variance", 0.05))
    sigma = float(cfg.get("sigma", np.sqrt(max(variance, 0.0))))
    debug = bool(cfg.get("debug", False))
    # DP 当前使用的 low-dim 输入是 agent_pos[8]。
    # MR-FPDP2 在该链路下明确只扰动末尾 4 个维度，并且只在方块进入终段区域后生效。
    target_indices = _resolve_target_indices(proprio, [-4, -3, -2, -1])
    if not target_indices:
        return proprio

    if torch.is_tensor(proprio):
        out = proprio.clone()
        before = out[..., target_indices].detach().cpu().numpy().copy() if debug else None
        noise = torch.randn_like(out[..., target_indices]) * sigma + mean
        out[..., target_indices] = out[..., target_indices] + noise
        if debug:
            after = out[..., target_indices].detach().cpu().numpy().copy()
            print(
                f"[MR-FPDP2] cube_goal_distance={float(cube_goal_distance):.4f} "
                f"indices={target_indices} mean={mean} sigma={sigma} "
                f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
            )
        return out

    out = proprio.copy()
    before = out[..., target_indices].copy() if debug else None
    noise = np.random.randn(*out[..., target_indices].shape) * sigma + mean
    out[..., target_indices] = out[..., target_indices] + noise
    if debug:
        after = out[..., target_indices].copy()
        print(
            f"[MR-FPDP2] cube_goal_distance={float(cube_goal_distance):.4f} "
            f"indices={target_indices} mean={mean} sigma={sigma} "
            f"before={before.reshape(-1).tolist()} after={after.reshape(-1).tolist()}"
        )
    return out
