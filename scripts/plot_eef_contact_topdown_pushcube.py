#!/usr/bin/env python3
import argparse
import html
import json
import math
from pathlib import Path


def _xyz(point):
    while isinstance(point, list) and len(point) == 1 and isinstance(point[0], list):
        point = point[0]
    if not isinstance(point, list) or len(point) < 3:
        return None
    return [float(point[0]), float(point[1]), float(point[2])]


def _xyz_array(seq):
    points = []
    for item in seq or []:
        point = _xyz(item)
        if point is not None:
            points.append(point)
    return points


def _first_position(positions, names):
    for name in names:
        point = _xyz(positions.get(name))
        if point is not None:
            return [point], name
    return None, None


def _first_point(mapping, names):
    for name in names:
        point = _xyz(mapping.get(name))
        if point is not None:
            return point, name
    return None, None


def _first_traj_point(trajectory, names):
    for name in names:
        points = _xyz_array(trajectory.get(name))
        if points:
            return points[-1], f"trajectory.{name}"
    return None, None


def _object_color(name):
    lower_name = name.lower()
    if "blue" in lower_name:
        return "#1f77b4"
    if "cube" in lower_name:
        return "#1f77b4"
    if "green" in lower_name:
        return "#2ca02c"
    if "yellow" in lower_name:
        return "#bcbd22"
    return "#7f7f7f"


def _object_positions(positions):
    objects = []
    for name, raw_point in positions.items():
        if raw_point is None:
            continue
        point = _xyz(raw_point)
        if point is None:
            continue
        if name.endswith("_initial"):
            phase = "initial"
            object_name = name[: -len("_initial")]
        elif name.endswith("_final"):
            phase = "final"
            object_name = name[: -len("_final")]
        else:
            phase = "position"
            object_name = name
        lower_name = object_name.lower()
        if "red" in lower_name or "target" in lower_name or "goal" in lower_name:
            continue
        objects.append(
            {
                "name": object_name,
                "phase": phase,
                "point": point,
                "color": _object_color(object_name),
            }
        )
    return objects


def _pushcube_object_positions(record):
    positions = record.get("positions", {})
    trajectory = record.get("trajectory", {})
    mr_eval = record.get("mr_eval", {})
    objects = _object_positions(positions)

    has_initial_cube = any(item["phase"] == "initial" and "cube" in item["name"].lower() for item in objects)
    if not has_initial_cube:
        point, _ = _first_point(trajectory, ["initial_cube_pos"])
        if point is None:
            point, _ = _first_point(mr_eval, ["initial_cube_pos"])
        if point is not None:
            objects.append(
                {
                    "name": "blue_cube",
                    "phase": "initial",
                    "point": point,
                    "color": "#1f77b4",
                }
            )

    has_final_cube = any(item["phase"] == "final" and "cube" in item["name"].lower() for item in objects)
    if not has_final_cube:
        point, _ = _first_traj_point(trajectory, ["cube_pos", "src_cube_pos"])
        if point is not None:
            objects.append(
                {
                    "name": "blue_cube",
                    "phase": "final",
                    "point": point,
                    "color": "#1f77b4",
                }
            )
    return objects


def _target_position(record):
    positions = record.get("positions", {})
    trajectory = record.get("trajectory", {})
    mr_eval = record.get("mr_eval", {})
    initial_cube, _ = _first_point(trajectory, ["initial_cube_pos"])
    if initial_cube is None:
        initial_cube, _ = _first_point(mr_eval, ["initial_cube_pos"])

    point, source = _first_point(
        positions,
        ["initial_target_pos", "target_pos", "dst_cube_pos", "goal_pos", "goal"],
    )
    if point is not None:
        if _target_matches_initial_cube(record, point, initial_cube):
            return _infer_pushcube_target_from_initial_cube(initial_cube), "inferred.initial_cube_pos+0.20x"
        return point, f"positions.{source}"
    point, source = _first_point(mr_eval, ["initial_target_pos", "target_pos", "dst_cube_pos"])
    if point is not None:
        if _target_matches_initial_cube(record, point, initial_cube):
            return _infer_pushcube_target_from_initial_cube(initial_cube), "inferred.initial_cube_pos+0.20x"
        return point, f"mr_eval.{source}"
    point, source = _first_traj_point(trajectory, ["target_pos", "dst_cube_pos"])
    if point is not None and _target_matches_initial_cube(record, point, initial_cube):
        return _infer_pushcube_target_from_initial_cube(initial_cube), "inferred.initial_cube_pos+0.20x"
    return point, source


def _target_matches_initial_cube(record, target_point, initial_cube):
    if target_point is None or initial_cube is None:
        return False
    mr_type = str(record.get("mr_eval", {}).get("mr_type") or "").lower()
    if any(token in mr_type for token in ("semp", "sadp", "jsap")):
        return False
    dist = math.hypot(float(target_point[0]) - float(initial_cube[0]), float(target_point[1]) - float(initial_cube[1]))
    return dist < 1e-4


def _infer_pushcube_target_from_initial_cube(initial_cube):
    if initial_cube is None:
        return None
    return [float(initial_cube[0]) + 0.20, float(initial_cube[1]), float(initial_cube[2])]


def _merge_duplicate_objects(objects, tolerance=1e-6):
    grouped = {}
    for item in objects:
        point = item["point"]
        key = (
            item["phase"],
            round(point[0] / tolerance),
            round(point[1] / tolerance),
            round(point[2] / tolerance),
        )
        grouped.setdefault(key, []).append(item)

    merged = []
    for items in grouped.values():
        if len(items) == 1:
            merged.append(items[0])
            continue
        names = sorted({item["name"] for item in items})
        colors = {item["color"] for item in items}
        merged.append(
            {
                "name": " / ".join(names),
                "phase": items[0]["phase"],
                "point": items[0]["point"],
                "color": items[0]["color"] if len(colors) == 1 else "#7f7f7f",
                "merged_count": len(items),
            }
        )
    return merged


def _compact_object_label(name, phase):
    compact_name = (
        name.replace("_cube", "")
        .replace("_sphere", "")
        .replace("_block", "")
        .replace("_cup", "")
        .replace("_", " ")
    )
    compact_phase = {"initial": "init", "final": "final", "position": "pos"}.get(phase, phase)
    return f"{compact_name} {compact_phase}"


def _load_record(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _nearest_index(eef_xy, cube_xy):
    count = min(len(eef_xy), len(cube_xy))
    if count == 0:
        return None, None
    distances = [
        math.hypot(eef_xy[index][0] - cube_xy[index][0], eef_xy[index][1] - cube_xy[index][1])
        for index in range(count)
    ]
    index = min(range(count), key=distances.__getitem__)
    return index, distances[index]


def _moving_average_xy(points, window):
    if window <= 1 or len(points) < 3:
        return points
    radius = window // 2
    smoothed = []
    for index in range(len(points)):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        count = end - start
        smoothed.append(
            (
                sum(point[0] for point in points[start:end]) / count,
                sum(point[1] for point in points[start:end]) / count,
            )
        )
    return smoothed


def _rotate_xy(points, rotate_deg):
    normalized = rotate_deg % 360
    if normalized == 0:
        return points
    if normalized == 90:
        return [(-point[1], point[0]) for point in points]
    if normalized == 180:
        return [(-point[0], -point[1]) for point in points]
    if normalized == 270:
        return [(point[1], -point[0]) for point in points]
    raise ValueError("--rotate-deg must be one of: 0, 90, 180, 270")


def plot_topdown(
    json_path,
    output_path=None,
    contact_threshold=0.03,
    target_radius=0.05,
    paper_mode=False,
    smooth_window=1,
    rotate_deg=180,
    compact=True,
    fixed_scale=True,
    fixed_extent=0.30,
    fixed_center="data",
    show_nearest=True,
):
    if paper_mode and smooth_window <= 1:
        smooth_window = 7
    record = _load_record(json_path)
    trajectory = record.get("trajectory", {})
    positions = record.get("positions", {})

    eef = _xyz_array(trajectory.get("eef_path"))
    if len(eef) == 0:
        raise ValueError("missing trajectory.eef_path")

    cube = _xyz_array(trajectory.get("cube_pos"))
    cube_source = "trajectory.cube_pos"
    if len(cube) == 0:
        cube, cube_source = _first_position(
            positions,
            [
                "blue_cube_initial",
                "initial_cube_pos",
                "blue_cube_final",
                "cube_final",
            ],
        )
    if len(cube) == 0:
        cube_point, cube_source = _first_point(
            trajectory,
            ["initial_cube_pos"],
        )
        cube = [cube_point] if cube_point is not None else []

    eef_xy = _rotate_xy([(point[0], point[1]) for point in eef], rotate_deg)
    display_eef_xy = _moving_average_xy(eef_xy, smooth_window)
    cube_xy = _rotate_xy([(point[0], point[1]) for point in cube], rotate_deg) if cube is not None else []
    target_point, target_source = _target_position(record)
    target_xy = _rotate_xy([(target_point[0], target_point[1])], rotate_deg)[0] if target_point else None
    object_points = _merge_duplicate_objects(_pushcube_object_positions(record))
    for item in object_points:
        item["plot_point"] = _rotate_xy([(item["point"][0], item["point"][1])], rotate_deg)[0]
    object_xy = [item["plot_point"] for item in object_points]

    nearest_text = "nearest: unavailable"
    nearest_idx = None
    nearest_dist = None
    contact_indices = []
    if cube_xy:
        nearest_idx, nearest_dist = _nearest_index(eef_xy, cube_xy)
        if nearest_idx is not None:
            nearest_text = f"nearest frame={nearest_idx}, xy_distance_m={nearest_dist:.6f}"
        if len(cube_xy) > 1:
            count = min(len(eef_xy), len(cube_xy))
            contact_indices = [
                index
                for index in range(count)
                if math.hypot(
                    eef_xy[index][0] - cube_xy[index][0],
                    eef_xy[index][1] - cube_xy[index][1],
                )
                <= contact_threshold
            ]

    if output_path is None:
        path = Path(json_path)
        output_path = path.with_name(f"{path.stem}_eef_contact_topdown.svg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_xy = list(eef_xy) + list(cube_xy) + list(object_xy)
    if target_xy is not None:
        all_xy.append(target_xy)
    if fixed_scale:
        if fixed_center == "origin":
            center_x = 0.0
            center_y = 0.0
        else:
            raw_min_x = min(point[0] for point in all_xy)
            raw_max_x = max(point[0] for point in all_xy)
            raw_min_y = min(point[1] for point in all_xy)
            raw_max_y = max(point[1] for point in all_xy)
            center_x = (raw_min_x + raw_max_x) / 2.0
            center_y = (raw_min_y + raw_max_y) / 2.0
        min_x = center_x - fixed_extent
        max_x = center_x + fixed_extent
        min_y = center_y - fixed_extent
        max_y = center_y + fixed_extent
    else:
        min_x = min(point[0] for point in all_xy)
        max_x = max(point[0] for point in all_xy)
        min_y = min(point[1] for point in all_xy)
        max_y = max(point[1] for point in all_xy)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        pad = max(span_x, span_y) * (0.04 if compact else 0.08)
        min_x -= pad
        max_x += pad
        min_y -= pad
        max_y += pad
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    legend_width = 230
    plot_width = 700
    plot_height = int(plot_width * span_y / span_x) if fixed_scale else max(520, min(900, int(plot_width * span_y / span_x))) if compact else 720
    width = plot_width + legend_width + 95
    height = plot_height + 115
    margin_left = 58
    margin_top = 52
    margin_right = legend_width + 32
    margin_bottom = 48
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    scale = min((plot_right - margin_left) / span_x, (plot_bottom - margin_top) / span_y)

    def screen(point):
        x = margin_left + (point[0] - min_x) * scale
        y = plot_bottom - (point[1] - min_y) * scale
        return x, y

    def polyline(points):
        return " ".join(f"{screen(point)[0]:.2f},{screen(point)[1]:.2f}" for point in points)

    def circle(point, radius, color, stroke="none", extra=""):
        x, y = screen(point)
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" '
            f'fill="{color}" stroke="{stroke}" {extra}/>'
        )

    def metric_circle(point, radius_m, color, stroke="none", extra=""):
        x, y = screen(point)
        radius_px = max(1.0, float(radius_m) * scale)
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius_px:.2f}" '
            f'fill="{color}" stroke="{stroke}" {extra}/>'
        )

    def square(point, size, color, stroke="none", extra=""):
        x, y = screen(point)
        half = size / 2
        return (
            f'<rect x="{x - half:.2f}" y="{y - half:.2f}" width="{size}" height="{size}" '
            f'fill="{color}" stroke="{stroke}" {extra}/>'
        )

    title = html.escape(f"Top-down EEF trajectory: {Path(json_path).name}")
    nearest_label = html.escape(nearest_text)
    scale_text = f"fixed +/-{fixed_extent:.2f} m around {fixed_center}" if fixed_scale else "auto-scaled"
    footer = html.escape(
        f"Rotated {rotate_deg}°, units: meters, scale: {scale_text}. EEF is recorded TCP / gripper-end center."
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="25" font-family="Arial" font-size="15">{title}</text>',
        f'<text x="{margin_left}" y="43" font-family="Arial" font-size="11">{nearest_label}</text>',
        f'<text x="{margin_left}" y="{height - 16}" font-family="Arial" font-size="10">{footer}</text>',
        f'<line x1="{margin_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#888" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_bottom}" stroke="#888" stroke-width="1"/>',
        f'<text x="{plot_right - 10}" y="{plot_bottom + 20}" font-family="Arial" font-size="10">X</text>',
        f'<text x="{margin_left - 20}" y="{margin_top + 4}" font-family="Arial" font-size="10">Y</text>',
        (
            f'<polyline points="{polyline(eef_xy)}" fill="none" stroke="#777777" '
            'stroke-width="1.0" opacity="0.25"/>'
            if paper_mode
            else f'<polyline points="{polyline(eef_xy)}" fill="none" stroke="#222222" stroke-width="2.2"/>'
        ),
        (
            f'<polyline points="{polyline(display_eef_xy)}" fill="none" stroke="#222222" '
            'stroke-width="2.4"/>'
            if paper_mode and display_eef_xy != eef_xy
            else ""
        ),
        circle(eef_xy[0], 5, "#222222"),
        circle(eef_xy[-1], 6, "white", "#222222", 'stroke-width="2"'),
    ]

    if cube_xy:
        if len(cube_xy) == 1:
            svg.append(circle(cube_xy[0], 7, "#ff7f0e"))
        else:
            svg.append(
                f'<polyline points="{polyline(cube_xy)}" fill="none" stroke="#ff7f0e" '
                'stroke-width="1.8" stroke-dasharray="6 4"/>'
            )
            svg.append(circle(cube_xy[0], 6, "#ff7f0e"))
            svg.append(circle(cube_xy[-1], 6, "white", "#ff7f0e", 'stroke-width="2"'))
    for item in object_points:
        xy = item["plot_point"]
        if item["phase"] == "initial":
            svg.append(square(xy, 20, item["color"], "black", 'stroke-width="0.8"'))
        elif item["phase"] == "final":
            svg.append(square(xy, 20, "white", item["color"], 'stroke-width="2.6"'))
        else:
            svg.append(circle(xy, 9, item["color"], "black", 'stroke-width="0.8"'))
    if target_xy is not None:
        svg.append(metric_circle(target_xy, target_radius, "#d62728", "#d62728", 'stroke-width="1.8" opacity="0.18"'))
        svg.append(circle(target_xy, 4, "#d62728", "none", 'opacity="0.85"'))
    if not paper_mode:
        for index in contact_indices:
            svg.append(circle(eef_xy[index], 3, "#2ca02c", extra='opacity="0.65"'))
    if show_nearest and nearest_idx is not None:
        svg.append(circle(eef_xy[nearest_idx], 9, "#e377c2", "black", 'stroke-width="1.5"'))

    legend_x = plot_right + 18
    legend_y = 62
    svg.append(f'<g font-family="Arial" font-size="11">')
    svg.append(f'<text x="{legend_x}" y="{legend_y - 20}" font-size="13">Legend</text>')

    def legend_text(y, label):
        svg.append(f'<text x="{legend_x + 32}" y="{y + 4}">{html.escape(label)}</text>')

    y = legend_y
    svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="#222222" stroke-width="2.4"/>')
    legend_text(y, "EEF/TCP trajectory")
    y += 22
    if paper_mode:
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="#777777" stroke-width="1.2" opacity="0.35"/>')
        legend_text(y, "raw EEF trajectory")
        y += 22
    svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="#ff7f0e" stroke-width="1.8" stroke-dasharray="6 4"/>')
    legend_text(y, f"cube trajectory ({cube_source})")
    y += 22
    if not paper_mode:
        svg.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="4" fill="#2ca02c" opacity="0.65"/>')
        legend_text(y, f"near cube <= {contact_threshold:.3f} m")
        y += 22
    if show_nearest:
        svg.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="7" fill="#e377c2" stroke="black" stroke-width="1.2"/>')
        legend_text(y, "nearest point")
        y += 26
    svg.append(f'<rect x="{legend_x + 2}" y="{y - 10}" width="20" height="20" fill="#1f77b4" stroke="black" stroke-width="0.8"/>')
    legend_text(y, "blue object")
    if target_xy is not None:
        y += 24
        svg.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="10" fill="#d62728" stroke="#d62728" stroke-width="1.2" opacity="0.18"/>')
        svg.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="3" fill="#d62728" opacity="0.85"/>')
        legend_text(y, f"target disk r={target_radius:.3f}m ({target_source})")
    svg.append("</g>")
    svg.append("</svg>")

    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path, nearest_text


def main():
    parser = argparse.ArgumentParser(
        description="Plot a top-down trajectory of the recorded TCP/gripper-end point near the cube."
    )
    parser.add_argument("json_path", help="Episode JSON containing trajectory.eef_path")
    parser.add_argument("-o", "--output", help="Output SVG path")
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.03,
        help="XY distance threshold in meters for highlighting near-contact frames",
    )
    parser.add_argument(
        "--target-radius",
        type=float,
        default=0.05,
        help="Target disk radius in meters; default is 0.05.",
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Draw a cleaner figure: light raw path, smoothed main path, and fewer contact markers.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Odd moving-average window for the displayed EEF path; paper mode defaults to 7.",
    )
    parser.add_argument(
        "--rotate-deg",
        type=int,
        default=180,
        choices=[0, 90, 180, 270],
        help="Rotate the top-down XY plot in 90-degree increments; default is 180.",
    )
    parser.add_argument(
        "--loose-layout",
        action="store_true",
        help="Use larger margins and a less compact canvas.",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Scale axes to the current trajectory instead of using a fixed PushCube workspace range.",
    )
    parser.add_argument(
        "--fixed-extent",
        type=float,
        default=0.30,
        help="Half-width/height in meters for the fixed top-down workspace; default is +/-0.30 m.",
    )
    parser.add_argument(
        "--fixed-center",
        choices=["data", "origin"],
        default="data",
        help="Center fixed-scale plots on the data bounds or world origin; default is data.",
    )
    parser.add_argument(
        "--hide-nearest",
        action="store_true",
        help="Do not draw the pink nearest-point marker or its legend entry.",
    )
    args = parser.parse_args()

    output_path, nearest_text = plot_topdown(
        args.json_path,
        output_path=args.output,
        contact_threshold=args.contact_threshold,
        target_radius=args.target_radius,
        paper_mode=args.paper_mode,
        smooth_window=args.smooth_window,
        rotate_deg=args.rotate_deg,
        compact=not args.loose_layout,
        fixed_scale=not args.auto_scale,
        fixed_extent=args.fixed_extent,
        fixed_center=args.fixed_center,
        show_nearest=not args.hide_nearest,
    )
    print(f"saved: {output_path}")
    print(nearest_text)


if __name__ == "__main__":
    main()
