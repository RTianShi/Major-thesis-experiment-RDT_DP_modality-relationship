"""StackCube-specific vision mutations."""

import random

import numpy as np
from PIL import Image, ImageFilter

from .registry import register_vision


def _black_color_for_image(im, fill_value):
    if im.mode in {"1", "L", "I", "F", "P"}:
        return fill_value
    return tuple([fill_value] * len(im.getbands()))


def _blackout_target_cameras(images, camera_group_size, target_camera_indices, fill_value):
    if not target_camera_indices:
        return images

    target_camera_indices = {
        int(camera_idx) % camera_group_size for camera_idx in target_camera_indices
    }

    out = []
    for idx, im in enumerate(images):
        if im is None:
            out.append(None)
            continue
        if idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue
        out.append(Image.new(im.mode, im.size, color=_black_color_for_image(im, fill_value)))
    return out


def _target_camera_index_set(camera_group_size, target_camera_indices):
    return {
        int(camera_idx) % camera_group_size for camera_idx in target_camera_indices
    }


def _resolve_camera_group_size(images, cfg):
    camera_group_size_cfg = cfg.get("camera_group_size", None)
    if camera_group_size_cfg is None:
        camera_group_size = 1 if len(images) == 1 else 3
    else:
        camera_group_size = max(1, int(camera_group_size_cfg))
    if len(images) < camera_group_size:
        camera_group_size = max(1, len(images))
    return camera_group_size


def _resolve_fill_color(im: Image.Image, fill_color):
    """Resolve fillcolor for PIL transforms.

    Current DP path feeds a single RGB PIL image (converted from uint8 render).
    This helper keeps the MR safe across modes (RGB/RGBA/L/etc.) and accepts:
      - scalar -> broadcast to all bands
      - list/tuple -> truncated/padded to num bands
      - values in [0,1] -> treated as normalized and scaled to 0-255
    """
    if im is None:
        return fill_color

    num_bands = len(im.getbands())
    if fill_color is None:
        values = [0] * num_bands
    elif isinstance(fill_color, (int, float)):
        values = [fill_color] * num_bands
    else:
        try:
            values = list(fill_color)
        except Exception:
            values = [0] * num_bands

    if len(values) < num_bands:
        values = values + [values[-1] if values else 0] * (num_bands - len(values))
    elif len(values) > num_bands:
        values = values[:num_bands]

    resolved = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        # Heuristic: treat [0,1] floats as normalized colors.
        if 0.0 <= fv <= 1.0:
            fv = fv * 255.0
        fv = float(np.clip(fv, 0.0, 255.0))
        resolved.append(int(round(fv)))

    if num_bands == 1:
        return resolved[0]
    return tuple(resolved)


def _resolve_fill_value(fill_value, *, default=0):
    """Resolve scalar fill_value for blackout-style MRs.

    Accepts either 0-255 int, or 0-1 float (normalized) and converts to 0-255.
    """
    if fill_value is None:
        fill_value = default
    try:
        fv = float(fill_value)
    except Exception:
        fv = float(default)
    if 0.0 <= fv <= 1.0:
        fv = fv * 255.0
    return int(np.clip(int(round(fv)), 0, 255))


def _apply_brightness_and_gaussian_noise(im, cfg):
    if im is None:
        return None

    arr = np.asarray(im, dtype=np.float32)
    brightness_scale = float(cfg.get("brightness_scale", 0.92))
    brightness_bias = float(cfg.get("brightness_bias", 0.0))
    noise_std = float(cfg.get("gaussian_noise_std", 4.0))

    arr = arr * brightness_scale + brightness_bias
    if noise_std > 0.0:
        arr = arr + np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32)

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode=im.mode)


@register_vision("MR-GDIP1")
@register_vision("Full-Episode Visual Blindness")
def vis_mr_gdip1_full_episode_visual_blindness(images, cfg):
    if not images:
        return images

    camera_group_size_cfg = cfg.get("camera_group_size", None)
    if camera_group_size_cfg is None:
        # Current DP uses a single RGB frame; treat as a single-camera group by default.
        camera_group_size = 1 if len(images) == 1 else 3
    else:
        camera_group_size = max(1, int(camera_group_size_cfg))
    if len(images) < camera_group_size:
        camera_group_size = max(1, len(images))

    target_camera_indices = cfg.get("target_camera_indices", [0])
    fill_value = _resolve_fill_value(cfg.get("fill_value", 0), default=0)
    return _blackout_target_cameras(
        images,
        camera_group_size=camera_group_size,
        target_camera_indices=target_camera_indices,
        fill_value=fill_value,
    )


@register_vision("MR-FPDP1")
@register_vision("High-Frequency Edge Blurring")
def vis_mr_fpdp1_high_frequency_edge_blurring(images, cfg):
    if not images:
        return images

    camera_group_size = _resolve_camera_group_size(images, cfg)
    target_camera_indices = cfg.get("target_camera_indices", None)
    if target_camera_indices is None:
        target_camera_indices = list(range(camera_group_size))
    if not target_camera_indices:
        return images

    target_camera_indices = _target_camera_index_set(camera_group_size, target_camera_indices)
    mode = str(cfg.get("mode", "gaussian_blur")).strip().lower()
    radius = float(cfg.get("sigma", cfg.get("radius", 5.0)))

    out = []
    for idx, im in enumerate(images):
        if im is None or idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue

        if mode in {"mosaic", "pixelation", "pixelate"}:
            downsample_size = max(1, int(cfg.get("downsample_size", 16)))
            reduced = im.resize(
                (downsample_size, downsample_size),
                resample=Image.Resampling.BILINEAR,
            )
            degraded = reduced.resize(im.size, resample=Image.Resampling.NEAREST)
        else:
            degraded = im.filter(ImageFilter.GaussianBlur(radius=radius))
        out.append(degraded)
    return out


@register_vision("MR-FPDP2")
@register_vision("Spatial Downsampling")
def vis_mr_fpdp2_spatial_downsampling(images, cfg):
    if not images:
        return images

    camera_group_size = _resolve_camera_group_size(images, cfg)
    target_camera_indices = cfg.get("target_camera_indices", None)
    if target_camera_indices is None:
        target_camera_indices = list(range(camera_group_size))
    if not target_camera_indices:
        return images

    target_camera_indices = _target_camera_index_set(camera_group_size, target_camera_indices)
    downsample_size = max(1, int(cfg.get("downsample_size", 32)))
    downsample_resample = Image.Resampling.BILINEAR
    upsample_mode = str(cfg.get("upsample_mode", "bilinear")).strip().lower()
    upsample_resample = (
        Image.Resampling.NEAREST
        if upsample_mode in {"nearest", "nearest_neighbor", "nearest-neighbor"}
        else Image.Resampling.BILINEAR
    )

    out = []
    for idx, im in enumerate(images):
        if im is None or idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue

        reduced = im.resize(
            (downsample_size, downsample_size),
            resample=downsample_resample,
        )
        degraded = reduced.resize(im.size, resample=upsample_resample)
        out.append(degraded)
    return out


@register_vision("MR3")
@register_vision("MR-3")
@register_vision("Visual-Perturbation")
def vis_mr3_visual_perturbation(images, cfg):
    if not images:
        return images

    camera_group_size = _resolve_camera_group_size(images, cfg)
    target_camera_indices = cfg.get("target_camera_indices", None)
    if target_camera_indices is None:
        target_camera_indices = list(range(camera_group_size))
    if not target_camera_indices:
        return images

    target_camera_indices = _target_camera_index_set(camera_group_size, target_camera_indices)
    out = []
    for idx, im in enumerate(images):
        if im is None or idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue
        out.append(_apply_brightness_and_gaussian_noise(im, cfg))
    return out


@register_vision("MR-CPTMP1")
@register_vision("Early-Phase Approaching Blindness")
def vis_mr_cptmp1_early_phase_approaching_blindness(images, cfg):
    if not images:
        return images

    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    is_grasped = bool(runtime.get("is_grasped", False))

    visible_ratio = float(cfg.get("target_encoding_ratio", 0.1))
    blind_ratio = float(cfg.get("midflight_blink_end_ratio", 0.4))
    estimated_tgrasp_steps = max(1, int(cfg.get("estimated_tgrasp_steps", 120)))

    # Freeze the blink schedule at episode start so the blackout stays in the
    # intended mid-approach window instead of drifting when grasp is observed later.
    tgrasp_steps = int(cfg.get("_cptmp1_tgrasp_steps", estimated_tgrasp_steps))
    cfg["_cptmp1_tgrasp_steps"] = tgrasp_steps

    visible_until_step = max(1, int(round(tgrasp_steps * visible_ratio)))
    blind_until_step = max(visible_until_step + 1, int(round(tgrasp_steps * blind_ratio)))
    cfg["_cptmp1_visible_until_step"] = visible_until_step
    cfg["_cptmp1_blind_until_step"] = blind_until_step

    if is_grasped:
        return images
    if step_index < visible_until_step:
        return images
    if step_index >= blind_until_step:
        return images

    camera_group_size = _resolve_camera_group_size(images, cfg)
    target_camera_indices = cfg.get("target_camera_indices", [0])
    fill_value = _resolve_fill_value(cfg.get("fill_value", 0), default=0)
    return _blackout_target_cameras(
        images,
        camera_group_size=camera_group_size,
        target_camera_indices=target_camera_indices,
        fill_value=fill_value,
    )


@register_vision("MR-CPTMP2")
@register_vision("Critical-Stacking Blindness")
def vis_mr_cptmp2_critical_stacking_blindness(images, cfg):
    if not images:
        return images

    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_lifted = bool(runtime.get("cube_lifted", False))
    cube_goal_distance = runtime.get("cube_goal_distance", None)
    previous_cube_goal_distance = runtime.get("previous_cube_goal_distance", None)

    # Spec: define stacking phase as
    #   talign: gripper holds the red cube and it reaches the alignment region above
    #           the green cube (approaching the stacking goal)
    #   tend:   the step where the cube is released (is_grasped becomes False)
    # From talign (inclusive) until tend (exclusive/inclusive depending on env timing),
    # the visual input should be fully cut off (blackout).
    step_index = int(runtime.get("step_index", 0))
    if step_index <= 0:
        cfg.pop("_cptmp2_active", None)
        cfg.pop("_cptmp2_talign_step", None)
        cfg.pop("_cptmp2_align_counter", None)
        cfg.pop("_cptmp2_blackout_through_step", None)

    active = bool(cfg.get("_cptmp2_active", False))
    align_trigger_distance = float(cfg.get("align_trigger_distance", 0.06))
    sustain_steps = max(1, int(cfg.get("sustain_steps", 1)))
    align_counter = int(cfg.get("_cptmp2_align_counter", 0))

    blackout_through_step = cfg.get("_cptmp2_blackout_through_step", None)
    if blackout_through_step is not None:
        try:
            if int(step_index) <= int(blackout_through_step):
                camera_group_size = _resolve_camera_group_size(images, cfg)
                target_camera_indices = cfg.get("target_camera_indices", None)
                if target_camera_indices is None:
                    target_camera_indices = list(range(camera_group_size))
                fill_value = _resolve_fill_value(cfg.get("fill_value", 0), default=0)
                return _blackout_target_cameras(
                    images,
                    camera_group_size=camera_group_size,
                    target_camera_indices=target_camera_indices,
                    fill_value=fill_value,
                )
            else:
                cfg.pop("_cptmp2_blackout_through_step", None)
        except Exception:
            cfg.pop("_cptmp2_blackout_through_step", None)

    # If we are in blackout phase, keep it until the cube is released.
    if active:
        if not is_grasped:
            # Keep blackout for the release step itself, then clear for subsequent steps.
            cfg["_cptmp2_blackout_through_step"] = int(step_index)
            cfg["_cptmp2_active"] = False
            cfg["_cptmp2_align_counter"] = 0
    else:
        # User-defined talign: once the gripper holds the red cube and it is above
        # the green cube (pre-drop), blackout should start.
        src_cube_pos = runtime.get("src_cube_pos", None)
        dst_cube_pos = runtime.get("dst_cube_pos", None)

        above = False
        if src_cube_pos is not None and dst_cube_pos is not None:
            try:
                sz = float(src_cube_pos[2])
                gz = float(dst_cube_pos[2])
                min_height_above = float(cfg.get("min_height_above", 0.02))
                above = sz >= gz + min_height_above
            except Exception:
                above = False
        if not above:
            # Fallback: if we can't access Z, use a conservative scalar distance.
            above = (
                cube_goal_distance is not None
                and np.isfinite(cube_goal_distance)
                and float(cube_goal_distance) <= align_trigger_distance
            )

        # Do not require cube_lifted here: "above" already implies a stacking-phase posture.
        if is_grasped and above:
            align_counter += 1
        else:
            align_counter = 0
        cfg["_cptmp2_align_counter"] = align_counter

        if align_counter >= sustain_steps:
            active = True
            cfg["_cptmp2_active"] = True
            cfg["_cptmp2_talign_step"] = int(step_index)
            if bool(cfg.get("debug_print", False)):
                print(f"[MR-CPTMP2] talign at step={int(step_index)} (min_height_above={float(cfg.get('min_height_above', 0.02))})")

    if not active:
        return images

    camera_group_size = _resolve_camera_group_size(images, cfg)
    target_camera_indices = cfg.get("target_camera_indices", None)
    if target_camera_indices is None:
        target_camera_indices = list(range(camera_group_size))
    fill_value = _resolve_fill_value(cfg.get("fill_value", 0), default=0)
    return _blackout_target_cameras(
        images,
        camera_group_size=camera_group_size,
        target_camera_indices=target_camera_indices,
        fill_value=fill_value,
    )


@register_vision("MR-SCDP3")
@register_vision("Camera-Viewpoint-Jitter")
def vis_mr_scdp3_camera_viewpoint_jitter(images, cfg):
    # Current DP uses a single RGB frame (from env.render()) and expects the
    # mutated output to preserve size and mode.
    max_angle_deg = float(cfg.get("max_angle_deg", 1.0))
    max_shift_px = float(cfg.get("max_shift_px", 2.0))
    fill_color_cfg = cfg.get("fill_color", [0, 0, 0])

    out = []
    for im in images:
        if im is None:
            out.append(None)
            continue

        fill_color = _resolve_fill_color(im, fill_color_cfg)

        angle = random.uniform(-max_angle_deg, max_angle_deg)
        shift_x = random.uniform(-max_shift_px, max_shift_px)
        shift_y = random.uniform(-max_shift_px, max_shift_px)

        rotated = im.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill_color,
        )
        jittered = rotated.transform(
            rotated.size,
            Image.Transform.AFFINE,
            (1.0, 0.0, shift_x, 0.0, 1.0, shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill_color,
        )
        out.append(jittered)
    return out
