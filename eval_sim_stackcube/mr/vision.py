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


@register_vision("MR-GDIP1")
@register_vision("Full-Episode Visual Blindness")
def vis_mr_gdip1_full_episode_visual_blindness(images, cfg):
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = cfg.get("target_camera_indices", [0])
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    return _blackout_target_cameras(
        images,
        camera_group_size=camera_group_size,
        target_camera_indices=target_camera_indices,
        fill_value=fill_value,
    )


@register_vision("MR-FPDP1")
@register_vision("High-Frequency Edge Blurring")
def vis_mr_fpdp1_high_frequency_edge_blurring(images, cfg):
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = cfg.get("target_camera_indices", [0])
    if not target_camera_indices:
        return images

    target_camera_indices = _target_camera_index_set(
        camera_group_size, target_camera_indices
    )
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
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = cfg.get("target_camera_indices", [0])
    if not target_camera_indices:
        return images

    target_camera_indices = _target_camera_index_set(
        camera_group_size, target_camera_indices
    )
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


@register_vision("MR-CPTMP1")
@register_vision("Early-Phase Approaching Blindness")
def vis_mr_cptmp1_early_phase_approaching_blindness(images, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    is_grasped = bool(runtime.get("is_grasped", False))

    observed_grasp_step = runtime.get("observed_grasp_step", None)
    if observed_grasp_step is None:
        tgrasp_steps = max(1, int(cfg.get("estimated_tgrasp_steps", 80)))
    else:
        tgrasp_steps = max(1, int(observed_grasp_step))

    visible_ratio = float(cfg.get("target_encoding_ratio", 0.1))
    blind_ratio = float(cfg.get("midflight_blink_end_ratio", 0.4))
    visible_until_step = max(0, int(round(tgrasp_steps * visible_ratio)))
    blind_until_step = max(visible_until_step, int(round(tgrasp_steps * blind_ratio)))

    if is_grasped:
        return images
    if step_index < visible_until_step:
        return images
    if step_index >= blind_until_step:
        return images

    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = cfg.get("target_camera_indices", [0])
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    return _blackout_target_cameras(
        images,
        camera_group_size=camera_group_size,
        target_camera_indices=target_camera_indices,
        fill_value=fill_value,
    )


@register_vision("MR-CPTMP2")
@register_vision("Critical-Stacking Blindness")
def vis_mr_cptmp2_critical_stacking_blindness(images, cfg):
    runtime = cfg.get("runtime", {})
    is_grasped = bool(runtime.get("is_grasped", False))
    cube_goal_distance = runtime.get("cube_goal_distance", None)

    active = bool(cfg.get("_stacking_blindness_active", False))
    align_trigger_distance = float(cfg.get("align_trigger_distance", 0.06))

    if active and not is_grasped:
        cfg["_stacking_blindness_active"] = False
        return images

    if (not active) and is_grasped and cube_goal_distance is not None and np.isfinite(cube_goal_distance):
        if float(cube_goal_distance) <= align_trigger_distance:
            active = True
            cfg["_stacking_blindness_active"] = True

    if not active:
        return images

    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = cfg.get("target_camera_indices", [0])
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    return _blackout_target_cameras(
        images,
        camera_group_size=camera_group_size,
        target_camera_indices=target_camera_indices,
        fill_value=fill_value,
    )


@register_vision("MR-SCDP3")
@register_vision("Camera-Viewpoint-Jitter")
def vis_mr_scdp3_camera_viewpoint_jitter(images, cfg):
    max_angle_deg = float(cfg.get("max_angle_deg", 1.0))
    max_shift_px = float(cfg.get("max_shift_px", 2.0))
    fill_color = tuple(cfg.get("fill_color", [0, 0, 0]))

    out = []
    for im in images:
        if im is None:
            out.append(None)
            continue

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
