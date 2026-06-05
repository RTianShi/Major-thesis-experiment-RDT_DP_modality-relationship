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


def _blackout_all_images(images, fill_value):
    out = []
    for im in images:
        if im is None:
            out.append(None)
            continue
        out.append(Image.new(im.mode, im.size, color=_black_color_for_image(im, fill_value)))
    return out


def _apply_brightness_delta(im, delta_ratio):
    arr = np.asarray(im).astype(np.float32)
    arr = arr * (1.0 + float(delta_ratio))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _apply_gaussian_noise(im, sigma):
    arr = np.asarray(im).astype(np.float32)
    arr += np.random.randn(*arr.shape) * float(sigma)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@register_vision("identity")
def vis_identity(images, cfg):
    return images


@register_vision("MR-LTSEP5")
def vis_delay_passthrough(images, cfg):
    # 延迟由 eval 脚本里的历史帧缓存实现，这里保持透传。
    return images


@register_vision("MR-GDIP1")
def vis_global_visual_deprivation(images, cfg):
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    # DP 当前仅向 policy 提供单路视觉输入 head_cam。
    # MR-GDIP1 在该链路下等价于将本次传入的全部有效图像直接置黑。
    return _blackout_all_images(images, fill_value=fill_value)


@register_vision("MR-FPDP1")
def vis_high_frequency_feature_degradation(images, cfg):
    mode = str(cfg.get("mode", "gaussian_blur")).strip().lower()
    out = []
    for idx, im in enumerate(images):
        if im is None:
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


@register_vision("MR3")
@register_vision("MR-3")
@register_vision("Visual-Perturbation")
def vis_mr3_visual_perturbation(images, cfg):
    mode = str(cfg.get("mode", "brightness")).strip().lower()
    brightness_delta = float(cfg.get("brightness_delta", 0.2))
    sigma = float(cfg.get("sigma", 6.0))
    out = []
    for im in images:
        if im is None:
            out.append(im)
            continue
        if mode in {"noise", "gaussian_noise", "gaussian-noise"}:
            out.append(_apply_gaussian_noise(im, sigma=sigma))
        else:
            out.append(_apply_brightness_delta(im, delta_ratio=brightness_delta))
    return out


@register_vision("MR-CPTMP1")
def vis_terminal_alignment_deprivation(images, cfg):
    runtime = cfg.get("runtime", {})
    cube_goal_distance = runtime.get("cube_goal_distance", None)
    trigger_distance = float(cfg.get("trigger_distance", 0.05))
    fill_value = int(np.clip(cfg.get("fill_value", 0), 0, 255))
    is_triggered = bool(runtime.get("terminal_alignment_cutoff_triggered", False))

    if not is_triggered:
        if cube_goal_distance is None or not np.isfinite(cube_goal_distance):
            return images
        if float(cube_goal_distance) < trigger_distance:
            runtime["terminal_alignment_cutoff_triggered"] = True
            runtime["terminal_alignment_trigger_distance"] = float(cube_goal_distance)
            is_triggered = True
        else:
            return images

    # DP 当前仅向 policy 提供单路视觉输入 head_cam。
    # 这里使用锁存触发：首次进入末端精调区后，本 episode 后续对全部有效图像保持黑屏。
    return _blackout_all_images(images, fill_value=fill_value)


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
        out.append(_apply_gaussian_noise(im, sigma=sigma))
    return out
