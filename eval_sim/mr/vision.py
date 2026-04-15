from PIL import Image, ImageFilter
import numpy as np
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


@register_vision("identity")
def vis_identity(images, cfg):
    return images


@register_vision("MR-LTSEP5")
def vis_delay_passthrough(images, cfg):
    return images


@register_vision("MR-GDIP1")
def vis_global_visual_deprivation(images, cfg):
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
def vis_high_frequency_feature_degradation(images, cfg):
    camera_group_size = max(1, int(cfg.get("camera_group_size", 3)))
    target_camera_indices = cfg.get("target_camera_indices", [0])
    mode = str(cfg.get("mode", "gaussian_blur")).strip().lower()
    if not target_camera_indices:
        return images

    target_camera_indices = _target_camera_index_set(
        camera_group_size, target_camera_indices
    )
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
            radius = float(cfg.get("radius", 7.5))
            degraded = im.filter(ImageFilter.GaussianBlur(radius=radius))
        out.append(degraded)
    return out


@register_vision("MR-CPTMP1")
def vis_terminal_alignment_deprivation(images, cfg):
    runtime = cfg.get("runtime", {})
    cube_goal_distance = runtime.get("cube_goal_distance", None)
    trigger_distance = float(cfg.get("trigger_distance", 0.05))

    if cube_goal_distance is None or not np.isfinite(cube_goal_distance):
        return images
    if float(cube_goal_distance) >= trigger_distance:
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


@register_vision("blur")
def vis_blur(images, cfg):
    radius = float(cfg.get("radius", 1.5))
    out = []
    for im in images:
        if im is None:
            out.append(None)
        else:
            out.append(im.filter(ImageFilter.GaussianBlur(radius=radius)))
    return out


@register_vision("noise")
def vis_noise(images, cfg):
    sigma = float(cfg.get("sigma", 5.0))
    out = []
    for im in images:
        if im is None:
            out.append(None)
            continue
        arr = np.array(im).astype(np.float32)
        arr += np.random.randn(*arr.shape) * sigma
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out.append(im.fromarray(arr))
    return out

