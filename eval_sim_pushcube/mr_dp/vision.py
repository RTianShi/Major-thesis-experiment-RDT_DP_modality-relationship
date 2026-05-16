"""PushCube vision mutations."""

import numpy as np
from PIL import Image

from eval_sim.mr.vision import *  # noqa: F401,F403

from .registry import register_vision


def _black_color_for_image(im, fill_value):
    if im.mode in {"1", "L", "I", "F", "P"}:
        return fill_value
    return tuple([fill_value] * len(im.getbands()))


@register_vision("MR-GDIP1")
@register_vision("GDIP-Full-Temporal-Vision-Blackout")
def vis_global_visual_deprivation(images, cfg):
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))

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


@register_vision("MR-FPDP1")
@register_vision("MR-FPDP-1")
@register_vision("FPDP-High-Frequency-Edge-Deprivation")
def vis_high_frequency_edge_deprivation(images, cfg):
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }
    radius = float(cfg.get("radius", 5.0))

    out = []
    for idx, im in enumerate(images):
        if im is None:
            out.append(None)
            continue
        if idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue
        out.append(im.filter(ImageFilter.GaussianBlur(radius=radius)))
    return out


@register_vision("MR-CPTMP1")
@register_vision("MR-CPTMP-1")
@register_vision("CPTMP-Post-Contact-Blindfold")
def vis_post_contact_blindfold(images, cfg):
    runtime = cfg.get("runtime", {})
    tcp_to_obj_distance = runtime.get("tcp_to_obj_distance", None)
    contact_distance = float(cfg.get("contact_distance", 0.035))
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }

    if tcp_to_obj_distance is None or not np.isfinite(float(tcp_to_obj_distance)):
        return images
    if float(tcp_to_obj_distance) > contact_distance:
        return images

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


@register_vision("MR-CPTMP2")
@register_vision("MR-CPTMP-2")
@register_vision("CPTMP-Approach-Phase-Transient-Masking")
def vis_approach_phase_transient_masking(images, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    max_episode_steps = int(runtime.get("max_episode_steps", cfg.get("max_episode_steps", 400)))
    control_freq = float(runtime.get("control_freq", cfg.get("control_freq", 20.0)))

    start_fraction = float(cfg.get("start_fraction", 0.10))
    end_fraction = float(cfg.get("end_fraction", 0.30))
    duration_seconds = float(cfg.get("duration_seconds", 1.0))
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }

    window_start = max(0, int(round(max_episode_steps * start_fraction)))
    window_end = max(window_start, int(round(max_episode_steps * end_fraction)))
    mask_steps = max(1, int(round(control_freq * duration_seconds)))

    if step_index < window_start or step_index > window_end:
        return images

    mask_start_step = cfg.get("_transient_mask_start_step", None)
    if mask_start_step is None:
        cfg["_transient_mask_start_step"] = step_index
        mask_start_step = step_index

    if step_index >= int(mask_start_step) + mask_steps:
        return images

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
