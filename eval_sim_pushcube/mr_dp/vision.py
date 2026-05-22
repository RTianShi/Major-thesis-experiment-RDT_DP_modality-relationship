"""PushCube vision mutations."""

import numpy as np
from PIL import Image, ImageEnhance

from eval_sim.mr.vision import *  # noqa: F401,F403

from .registry import register_vision


def _black_color_for_image(im, fill_value):
    if im.mode in {"1", "L", "I", "F", "P"}:
        return fill_value
    return tuple([fill_value] * len(im.getbands()))


def _apply_light_perturbation(im, brightness_factor, noise_std):
    out = im
    if abs(float(brightness_factor) - 1.0) > 1e-8:
        out = ImageEnhance.Brightness(out).enhance(float(brightness_factor))

    if float(noise_std) > 0.0:
        arr = np.asarray(out).astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=float(noise_std), size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        out = Image.fromarray(arr, mode=out.mode)
    return out


@register_vision("MR3")
@register_vision("MR-3")
@register_vision("Visual-Perturbation")
def vis_mr3_visual_perturbation(images, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    apply_mode = str(cfg.get("apply_mode", "always")).strip().lower()
    if apply_mode == "initial" and step_index != 0:
        return images

    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0, 1, 2])
    }
    brightness_factor = float(cfg.get("brightness_factor", 0.7))
    noise_std = float(cfg.get("noise_std", 10.0))

    out = []
    for idx, im in enumerate(images):
        if im is None:
            out.append(None)
            continue
        if idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue
        out.append(_apply_light_perturbation(im, brightness_factor, noise_std))
    return out


@register_vision("MR-GDIP1")
@register_vision("GDIP-Full-Temporal-Vision-Blackout")
def vis_global_visual_deprivation(images, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    apply_mode = str(cfg.get("apply_mode", "always")).strip().lower()
    if apply_mode == "initial" and step_index != 0:
        return images

    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    blackout_strength = float(cfg.get("blackout_strength", 1.0))
    blackout_strength = float(np.clip(blackout_strength, 0.0, 1.0))
    if blackout_strength <= 0.0:
        return images

    out = []
    for idx, im in enumerate(images):
        if im is None:
            out.append(None)
            continue
        if idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue
        if blackout_strength >= 1.0:
            out.append(Image.new(im.mode, im.size, color=_black_color_for_image(im, fill_value)))
            continue
        black = Image.new(im.mode, im.size, color=_black_color_for_image(im, fill_value))
        out.append(Image.blend(im, black, alpha=blackout_strength))
    return out


@register_vision("MR-FPDP1")
@register_vision("MR-FPDP-1")
@register_vision("FPDP-High-Frequency-Edge-Deprivation")
def vis_high_frequency_edge_deprivation(images, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    apply_mode = str(cfg.get("apply_mode", "always")).strip().lower()
    if apply_mode == "initial" and step_index != 0:
        return images

    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }
    radius = float(cfg.get("radius", 5.0))
    blur_strength = float(np.clip(cfg.get("blur_strength", 1.0), 0.0, 1.0))
    if radius <= 0.0 or blur_strength <= 0.0:
        return images

    out = []
    for idx, im in enumerate(images):
        if im is None:
            out.append(None)
            continue
        if idx % camera_group_size not in target_camera_indices:
            out.append(im)
            continue
        blurred = im.filter(ImageFilter.GaussianBlur(radius=radius))
        if blur_strength >= 1.0:
            out.append(blurred)
            continue
        out.append(Image.blend(im, blurred, alpha=blur_strength))
    return out


@register_vision("MR-CPTMP1")
@register_vision("MR-CPTMP-1")
@register_vision("CPTMP-Post-Contact-Blindfold")
def vis_post_contact_blindfold(images, cfg):
    runtime = cfg.get("runtime", {})
    step_index = int(runtime.get("step_index", 0))
    tcp_to_obj_distance = runtime.get("tcp_to_obj_distance", None)
    contact_distance = float(cfg.get("contact_distance", 0.035))
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }
    apply_mode = str(cfg.get("apply_mode", "post_contact")).strip().lower()
    duration_steps = max(1, int(cfg.get("duration_steps", 1)))
    retrigger = bool(cfg.get("retrigger", False))
    fire_after_step = int(cfg.get("fire_after_step", 0))

    if step_index == 0:
        cfg.pop("_cptmp1_fired_step", None)

    if apply_mode == "initial" and step_index != 0:
        return images
    if step_index < fire_after_step:
        return images

    fired_step = cfg.get("_cptmp1_fired_step", None)
    if fired_step is not None:
        if not retrigger:
            pass
        elif step_index < int(fired_step) + duration_steps:
            pass
        else:
            fired_step = None
            cfg.pop("_cptmp1_fired_step", None)

    if fired_step is None:
        if tcp_to_obj_distance is None or not np.isfinite(float(tcp_to_obj_distance)):
            return images
        if float(tcp_to_obj_distance) > contact_distance:
            return images
        cfg["_cptmp1_fired_step"] = step_index
        fired_step = step_index

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
    apply_mode = str(cfg.get("apply_mode", "windowed")).strip().lower()

    start_fraction = float(cfg.get("start_fraction", 0.10))
    end_fraction = float(cfg.get("end_fraction", 0.30))
    duration_seconds = float(cfg.get("duration_seconds", 1.0))
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    blackout_strength = float(np.clip(cfg.get("blackout_strength", 1.0), 0.0, 1.0))
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = {
        int(camera_idx) % camera_group_size
        for camera_idx in cfg.get("target_camera_indices", [0])
    }

    if step_index == 0:
        cfg.pop("_transient_mask_start_step", None)

    if apply_mode == "initial" and step_index != 0:
        return images
    if blackout_strength <= 0.0:
        return images

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
        if blackout_strength >= 1.0:
            out.append(Image.new(im.mode, im.size, color=_black_color_for_image(im, fill_value)))
            continue
        black = Image.new(im.mode, im.size, color=_black_color_for_image(im, fill_value))
        out.append(Image.blend(im, black, alpha=blackout_strength))
    return out
