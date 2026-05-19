#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional

import numpy as np

# Ensure repo root on sys.path when running directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.grasp_event import derive_grasp_and_yaw_from_raw


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


@dataclass
class EpisodeRecord:
    file: str
    episode_id: Optional[int]
    seed: Optional[int]
    success: Optional[bool]
    total_steps: Optional[int]
    path_len: Optional[float]
    gripper_width: List[Optional[float]]
    gripper_finger_qpos: List[List[float]]
    cube_pos: List[List[float]]
    src_cube_pos: List[List[float]]
    dst_cube_pos: List[List[float]]
    goal_point: Optional[List[float]]
    mr_eval: Dict[str, Optional[List[float]]]
    trajectory: Dict[str, Optional[List[float]]]
    eef_path: List[List[float]]
    gripper_action_cmd: List[Optional[float]]
    eef_yaw_deg: List[Optional[float]]
    mr_eval_grasp_frame_index: Optional[int]
    mr_eval_eef_yaw_at_grasp: Optional[float]
    mr_eval_expected_delta_yaw_deg: Optional[float]

    # derived
    cube_path_len: Optional[float]
    cube_net_disp: Optional[float]
    gripper_mean: Optional[float]
    gripper_max: Optional[float]
    gripper_min: Optional[float]
    derived_grasp_frame_index: Optional[int]
    derived_eef_yaw_at_grasp: Optional[float]


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _dist_path(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 2:
        return 0.0 if points else None
    try:
        arr = np.asarray(points, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim >= 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    d = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def _net_disp(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 2:
        return 0.0 if points else None
    try:
        arr = np.asarray(points, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim >= 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return float(np.linalg.norm(arr[-1, :3] - arr[0, :3]))


def _normalize_xyz_sequence(seq):
    if not seq:
        return seq
    out = []
    changed = False
    for item in seq:
        if item is None:
            out.append(None)
            continue
        try:
            arr = np.asarray(item, dtype=np.float64).reshape(-1)
        except Exception:
            out.append(item)
            continue
        if arr.size >= 3:
            out.append([float(arr[0]), float(arr[1]), float(arr[2])])
            changed = True
        else:
            out.append(item)
    return out if changed else seq


def _safe_xyz_list(x):
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def load_records(traj_dir: str) -> List[EpisodeRecord]:
    files = sorted(glob.glob(os.path.join(traj_dir, "*.json")))
    # 过滤掉 run_config.json
    files = [f for f in files if os.path.basename(f) != "run_config.json"]

    records: List[EpisodeRecord] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        traj = obj.get("trajectory", {}) if isinstance(obj, dict) else {}
        metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
        mr_eval = obj.get("mr_eval", {}) if isinstance(obj, dict) else {}

        gripper_width = traj.get("gripper_width", []) or []
        gripper_finger_qpos = traj.get("gripper_finger_qpos", []) or []
        cube_pos = _normalize_xyz_sequence(traj.get("cube_pos", []) or [])
        src_cube_pos = _normalize_xyz_sequence(traj.get("src_cube_pos", []) or [])
        dst_cube_pos = _normalize_xyz_sequence(traj.get("dst_cube_pos", []) or [])
        goal_point = _first_not_none(
            traj.get("goal_point"),
            mr_eval.get("goal_point"),
            obj.get("goal_point"),
        )
        eef_path = _normalize_xyz_sequence(traj.get("eef_path", []) or [])
        gripper_action_cmd = traj.get("gripper_action_cmd", []) or []
        eef_yaw_deg = traj.get("eef_yaw_deg", []) or []

        rec = EpisodeRecord(
            file=fp,
            episode_id=_safe_int(obj.get("episode_id")),
            seed=_safe_int(obj.get("seed")),
            success=bool(metrics.get("env_success")) if "env_success" in metrics else None,
            total_steps=_safe_int(traj.get("total_steps")),
            path_len=_safe_float(traj.get("total_path_length_meters")),
            gripper_width=[None if x is None else float(x) for x in gripper_width],
            gripper_finger_qpos=gripper_finger_qpos,
            cube_pos=cube_pos,
            src_cube_pos=src_cube_pos,
            dst_cube_pos=dst_cube_pos,
            goal_point=_safe_xyz_list(goal_point),
            mr_eval={
                "goal_point": _safe_xyz_list(mr_eval.get("goal_point")),
                "expected_delta_yaw_deg": _safe_float(mr_eval.get("expected_delta_yaw_deg")),
                "grasp_frame_index": _safe_int(mr_eval.get("grasp_frame_index")),
            },
            trajectory={
                "goal_point": _safe_xyz_list(traj.get("goal_point")),
                "eef_path": eef_path,
                "gripper_width": gripper_width,
                "gripper_action_cmd": gripper_action_cmd,
                "gripper_finger_qpos": gripper_finger_qpos,
            },
            eef_path=eef_path,
            gripper_action_cmd=[None if x is None else float(x) for x in gripper_action_cmd],
            eef_yaw_deg=[None if x is None else float(x) for x in eef_yaw_deg],
            mr_eval_grasp_frame_index=_safe_int(mr_eval.get("grasp_frame_index")),
            mr_eval_eef_yaw_at_grasp=_safe_float(mr_eval.get("eef_yaw_at_grasp")),
            mr_eval_expected_delta_yaw_deg=_safe_float(mr_eval.get("expected_delta_yaw_deg")),
            cube_path_len=None,
            cube_net_disp=None,
            gripper_mean=None,
            gripper_max=None,
            gripper_min=None,
            derived_grasp_frame_index=None,
            derived_eef_yaw_at_grasp=None,
        )

        # 说明：抓取判定由多序列（指令/宽度/末端/物体/ yaw）综合推导，并非单一宽度阈值
        rec.derived_grasp_frame_index, rec.derived_eef_yaw_at_grasp = derive_grasp_and_yaw_from_raw(
            rec.gripper_action_cmd,
            rec.gripper_width,
            rec.eef_path,
            rec.cube_pos,
            rec.eef_yaw_deg,
        )

        rec.cube_path_len = _dist_path(rec.cube_pos)
        rec.cube_net_disp = _net_disp(rec.cube_pos)
        valid_gripper_width = [float(x) for x in rec.gripper_width if x is not None]
        if len(valid_gripper_width) > 0:
            rec.gripper_mean = float(mean(valid_gripper_width))
            rec.gripper_max = float(max(valid_gripper_width))
            rec.gripper_min = float(min(valid_gripper_width))

        records.append(rec)
    return records


def _summary_numeric(vals: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in vals if v is not None and not math.isnan(v)]
    if len(vals) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize(records: List[EpisodeRecord]) -> Dict:
    n = len(records)
    success_vals = [1.0 if r.success else 0.0 for r in records if r.success is not None]
    success_rate = (sum(success_vals) / len(success_vals) * 100.0) if success_vals else None

    steps = [r.total_steps for r in records if r.total_steps is not None]
    eef_path_len = [r.path_len for r in records if r.path_len is not None]
    cube_path_len = [r.cube_path_len for r in records if r.cube_path_len is not None]
    cube_net_disp = [r.cube_net_disp for r in records if r.cube_net_disp is not None]
    g_mean = [r.gripper_mean for r in records if r.gripper_mean is not None]
    g_max = [r.gripper_max for r in records if r.gripper_max is not None]
    g_min = [r.gripper_min for r in records if r.gripper_min is not None]

    # 新增：收集成功的 episode_id
    success_episode_ids = [r.episode_id for r in records if r.success and r.episode_id is not None]

    return {
        "episodes": n,
        "success_rate_percent": success_rate,
        "success_count": int(sum(success_vals)) if success_vals else 0,
        "success_episode_ids": success_episode_ids,  # 新增字段
        "steps": _summary_numeric(steps),
        "eef_path_length_m": _summary_numeric(eef_path_len),
        "cube_path_length_m": _summary_numeric(cube_path_len),
        "cube_net_displacement_m": _summary_numeric(cube_net_disp),
        "gripper_width_mean": _summary_numeric(g_mean),
        "gripper_width_max": _summary_numeric(g_max),
        "gripper_width_min": _summary_numeric(g_min),
    }


try:
    from eval_sim_stackcube.analysis_RDT.mr_rules import MR_RULE_REGISTRY
except Exception:
    # 兼容直接运行该脚本（python eval_sim_stackcube/analysis_RDT/mr_eval_analyzer_rdt.py）
    from eval_sim_stackcube.analysis_RDT.mr_rules.registry import MR_RULE_REGISTRY


def _normalize_mr_id(mr_id: str) -> str:
    if mr_id is None:
        return "default"
    raw = str(mr_id).strip()
    aliases = {
        "MR-SESP-1": "MR-SESP1",
        "mr-sesp-1": "mr_sesp_1",
        "mr-sesp1": "mr_sesp_1",
        "mr_sesp1": "mr_sesp_1",
    }
    if raw in aliases:
        return aliases[raw]
    return raw


def _default_output_path(mr_id: str) -> str:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_outputs")
    safe_mr_id = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in (mr_id or "single"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"StackCube_dp_{safe_mr_id}_{ts}.json")


def main():
    parser = argparse.ArgumentParser(description="Analyze eval_dp trajectory JSONs for MR evaluation.")
    parser.add_argument("--traj-dir", type=str, default=None, help="single-set analysis directory")
    parser.add_argument("--base-traj-dir", type=str, default=None, help="baseline directory for MR compare")
    parser.add_argument("--mr-traj-dir", type=str, default=None, help="MR directory for MR compare")
    parser.add_argument("--mr-id", type=str, default="default", help="MR rule id in registry")
    parser.add_argument("--list-mr-rules", action="store_true", help="List all registered MR rules and exit")

    # 通用/默认规则参数
    parser.add_argument("--path-len-ratio-tol", type=float, default=1.5, help="MR path length ratio tolerance")

    # MR-DRP-1 参数
    parser.add_argument("--expected-delta-z", type=float, default=0.23, help="Expected |Zrelease_f - Zrelease_s|")
    parser.add_argument("--delta-z-tol", type=float, default=0.05, help="Tolerance for expected delta z")
    parser.add_argument("--low-release-min", type=float, default=0.05, help="Low release band min z")
    parser.add_argument("--low-release-max", type=float, default=0.15, help="Low release band max z")
    parser.add_argument("--translation-dx", type=float, default=0.04, help="Expected translation delta x for translation-equivariance MRs")
    parser.add_argument("--translation-dy", type=float, default=-0.04, help="Expected translation delta y for translation-equivariance MRs")
    parser.add_argument("--translation-dz", type=float, default=0.0, help="Expected translation delta z for translation-equivariance MRs")
    parser.add_argument("--position-tol", type=float, default=0.025, help="Position tolerance for translation-equivariance MRs")
    parser.add_argument("--sadp-grasp-tol", type=float, default=0.03, help="Position tolerance at grasp point for SADP invariance MRs")
    parser.add_argument("--sadp-final-tol", type=float, default=0.03, help="Position tolerance at final point for SADP invariance MRs")

    parser.add_argument("--out", type=str, default=None, help="output json path")
    args = parser.parse_args()
    args.mr_id = _normalize_mr_id(args.mr_id)
    if args.list_mr_rules:
      print(json.dumps({"mr_rules": sorted(MR_RULE_REGISTRY.keys())}, ensure_ascii=False, indent=2))
      return

    single_mode = args.traj_dir is not None
    compare_mode = args.base_traj_dir is not None and args.mr_traj_dir is not None

    if not single_mode and not compare_mode:
        raise ValueError("Provide either --traj-dir, or both --base-traj-dir and --mr-traj-dir")

    # 增加：若为 compare 模式，先校验目录与 JSON 文件是否存在（排除 run_config.json）
    if compare_mode:
        if not os.path.isdir(args.base_traj_dir):
            raise ValueError(f"--base-traj-dir 不存在: {args.base_traj_dir}")
        if not os.path.isdir(args.mr_traj_dir):
            raise ValueError(f"--mr-traj-dir 不存在: {args.mr_traj_dir}")

        base_files = sorted(glob.glob(os.path.join(args.base_traj_dir, "*.json")))
        base_files = [f for f in base_files if os.path.basename(f) != "run_config.json"]
        mr_files = sorted(glob.glob(os.path.join(args.mr_traj_dir, "*.json")))
        mr_files = [f for f in mr_files if os.path.basename(f) != "run_config.json"]

        if len(base_files) == 0:
            raise ValueError(f"--base-traj-dir 中未找到任何 traj JSON 文件: {args.base_traj_dir}")
        if len(mr_files) == 0:
            raise ValueError(f"--mr-traj-dir 中未找到任何 traj JSON 文件: {args.mr_traj_dir}")

    result = {}

    if single_mode:
        recs = load_records(args.traj_dir)
        result["mode"] = "single"
        result["traj_dir"] = args.traj_dir
        result["summary"] = summarize(recs)

    if compare_mode:
        brecs = load_records(args.base_traj_dir)
        mrecs = load_records(args.mr_traj_dir)
        result["mode"] = "compare" if not single_mode else "single+compare"
        result["base_traj_dir"] = args.base_traj_dir
        result["mr_traj_dir"] = args.mr_traj_dir
        result["base_summary"] = summarize(brecs)
        result["mr_summary"] = summarize(mrecs)
        if args.mr_id not in MR_RULE_REGISTRY:
            raise ValueError(f"Unknown --mr-id={args.mr_id}. Available: {sorted(MR_RULE_REGISTRY.keys())}")

        rule_fn = MR_RULE_REGISTRY[args.mr_id]
        result["mr_rule"] = args.mr_id
        result["mr_compare"] = rule_fn(
            brecs,
            mrecs,
            path_len_ratio_tol=args.path_len_ratio_tol,
            expected_delta_z=args.expected_delta_z,
            delta_z_tol=args.delta_z_tol,
            low_release_min=args.low_release_min,
            low_release_max=args.low_release_max,
            translation_delta=[args.translation_dx, args.translation_dy, args.translation_dz],
            position_tol=args.position_tol,
            sadp_grasp_tol=args.sadp_grasp_tol,
            sadp_final_tol=args.sadp_final_tol,
        )
        mr_compare = result["mr_compare"]
        analyzable_count = int(mr_compare.get("analyzable_episodes", 0))
        unavailable_count = int(mr_compare.get("unavailable_count", 0))
        violation_count = mr_compare.get("violations")
        violation_rate = mr_compare.get("violation_rate_percent")

        # 兼容“通过式”MR规则：若规则只返回通过数，则把可判定且未通过的样本视为违反。
        if violation_count is None:
            passed_count = mr_compare.get("passed_count", mr_compare.get("pass_count"))
            if passed_count is not None:
                violation_count = analyzable_count - int(passed_count)

        if violation_rate is None and violation_count is not None and analyzable_count > 0:
            violation_rate = float(violation_count) / analyzable_count * 100.0

        result["followup_mr_violation_count"] = int(violation_count or 0)
        result["followup_mr_violation_rate_percent"] = violation_rate
        result["followup_unavailable_count"] = unavailable_count
        result["followup_analyzable_count"] = analyzable_count

    if compare_mode:
        vc = result.get("followup_mr_violation_count", 0)
        vr = result.get("followup_mr_violation_rate_percent")
        uc = result.get("followup_unavailable_count", 0)
        ac = result.get("followup_analyzable_count", 0)
        vr_text = "N/A" if vr is None else f"{vr:.2f}%"
        print(
            f"[MR] 违法用例个数: {vc}, 衍生用例MR违反率: {vr_text}, 不可判定: {uc}, 可判定: {ac}",
            file=sys.stderr,
        )

    # 统一保存：--out 优先；否则自动保存到 analysis/analysis_outputs
    out_path = args.out
    if not out_path:
        auto_mr_id = result.get("mr_rule") or ("single" if result.get("mode") == "single" else args.mr_id)
        out_path = _default_output_path(auto_mr_id)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[MR] 分析结果已保存: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
