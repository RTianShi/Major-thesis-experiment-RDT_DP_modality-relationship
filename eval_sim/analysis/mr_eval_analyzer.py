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


@dataclass
class EpisodeRecord:
    file: str
    episode_id: Optional[int]
    seed: Optional[int]
    success: Optional[bool]
    total_steps: Optional[int]
    path_len: Optional[float]
    gripper_width: List[float]
    cube_pos: List[List[float]]
    eef_path: List[List[float]]

    # derived
    cube_path_len: Optional[float]
    cube_net_disp: Optional[float]
    gripper_mean: Optional[float]
    gripper_max: Optional[float]
    gripper_min: Optional[float]


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
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    d = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def _net_disp(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 2:
        return 0.0 if points else None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return float(np.linalg.norm(arr[-1, :3] - arr[0, :3]))


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

        gripper_width = traj.get("gripper_width", []) or []
        cube_pos = traj.get("cube_pos", []) or []
        eef_path = traj.get("eef_path", []) or []

        rec = EpisodeRecord(
            file=fp,
            episode_id=_safe_int(obj.get("episode_id")),
            seed=_safe_int(obj.get("seed")),
            success=bool(metrics.get("env_success")) if "env_success" in metrics else None,
            total_steps=_safe_int(traj.get("total_steps")),
            path_len=_safe_float(traj.get("total_path_length_meters")),
            gripper_width=[float(x) for x in gripper_width if x is not None],
            cube_pos=cube_pos,
            eef_path=eef_path,
            cube_path_len=None,
            cube_net_disp=None,
            gripper_mean=None,
            gripper_max=None,
            gripper_min=None,
        )

        rec.cube_path_len = _dist_path(rec.cube_pos)
        rec.cube_net_disp = _net_disp(rec.cube_pos)
        if len(rec.gripper_width) > 0:
            rec.gripper_mean = float(mean(rec.gripper_width))
            rec.gripper_max = float(max(rec.gripper_width))
            rec.gripper_min = float(min(rec.gripper_width))

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

    return {
        "episodes": n,
        "success_rate_percent": success_rate,
        "success_count": int(sum(success_vals)) if success_vals else 0,
        "steps": _summary_numeric(steps),
        "eef_path_length_m": _summary_numeric(eef_path_len),
        "cube_path_length_m": _summary_numeric(cube_path_len),
        "cube_net_displacement_m": _summary_numeric(cube_net_disp),
        "gripper_width_mean": _summary_numeric(g_mean),
        "gripper_width_max": _summary_numeric(g_max),
        "gripper_width_min": _summary_numeric(g_min),
    }


try:
    from eval_sim.analysis.mr_rules import MR_RULE_REGISTRY
except Exception:
    # 兼容直接运行该脚本（python eval_sim/analysis/mr_eval_analyzer.py）
    from mr_rules import MR_RULE_REGISTRY


def _default_output_path(mr_id: str) -> str:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_outputs")
    safe_mr_id = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in (mr_id or "single"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{safe_mr_id}_{ts}.json")


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

    parser.add_argument("--out", type=str, default=None, help="output json path")
    args = parser.parse_args()
    if args.list_mr_rules:
      print(json.dumps({"mr_rules": sorted(MR_RULE_REGISTRY.keys())}, ensure_ascii=False, indent=2))
      return

    single_mode = args.traj_dir is not None
    compare_mode = args.base_traj_dir is not None and args.mr_traj_dir is not None

    if not single_mode and not compare_mode:
        raise ValueError("Provide either --traj-dir, or both --base-traj-dir and --mr-traj-dir")

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
        )
        result["followup_mr_violation_count"] = int(result["mr_compare"].get("violations", 0))
        result["followup_mr_violation_rate_percent"] = result["mr_compare"].get("violation_rate_percent")
        result["followup_unavailable_count"] = int(result["mr_compare"].get("unavailable_count", 0))
        result["followup_analyzable_count"] = int(result["mr_compare"].get("analyzable_episodes", 0))

    print(json.dumps(result, ensure_ascii=False, indent=2))

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