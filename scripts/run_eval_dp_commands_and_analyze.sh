#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 直接修改下面这些命令即可。
# 约定：
# 1. 默认取第 1 条命令对应的轨迹目录作为 base_traj_dir
# 2. 默认取第 2 条命令对应的轨迹目录作为 mr_traj_dir
# 3. 如果你写了第 3 条命令，它只会继续执行，不会自动参与分析，除非你改 BASE_CMD_INDEX / MR_CMD_INDEX
# 4. 命令里可以不写 --traj-dir。脚本会按 eval_dp.py 的默认命名规则：
#    eef_traj_dp/PickCube/<env-id>_<mr-type>_<timestamp>
#    自动找到本次运行后最新生成的目录。

COMMANDS=(
  "python -m eval_sim.eval_dp --pretrained_path ./700.ckpt -e PickCube-v1 --show --mr-type MR-SADP-1-原"
  "python -m eval_sim.eval_dp --pretrained_path ./700.ckpt -e PickCube-v1 --show --mr-type MR-SADP-1-0.05"
  # "python -m eval_sim.eval_dp --pretrained_path ./700.ckpt -e PickCube-v1 --show --mr-type MR-SADP-1-0.05"
)

# 可选：为每条命令单独指定 YAML 文件路径。
# 留空("") 表示不额外传 --mr-config，沿用 eval_dp.py 默认值。
MR_CONFIGS=(
  ""
  ""
  # ""
)

# 可选：也可以直接在这里写 YAML 内容。
# 如果某一项非空，脚本会优先把它写入临时 yaml 文件，并自动追加 --mr-config <tmpfile>。
# MR_CONFIGS_INLINE 和 COMMANDS 必须一一对应；不用时留空字符串即可。
MR_CONFIGS_INLINE=(
  $'mr:\n  language:\n    type: identity\n  vision:\n    type: identity\n  proprio:\n    type: identity\n  env:\n    type: MR-SADP-1-translate_cube_xy'
  $'mr:\n  language:\n    type: identity\n  vision:\n    type: identity\n  proprio:\n    type: identity\n  env:\n    type: MR-SADP-1\n    scale: 0.05'
  # ""
)

MR_ID="MR-SADP-1"
BASE_CMD_INDEX=0
MR_CMD_INDEX=1
ANALYSIS_OUT=""
DEFAULT_TRAJ_ROOT="$ROOT_DIR/eef_traj_dp/PickCube"
declare -a TMP_YAML_FILES=()

cleanup() {
  if [[ "${#TMP_YAML_FILES[@]}" -gt 0 ]]; then
    rm -f "${TMP_YAML_FILES[@]}"
  fi
}
trap cleanup EXIT

extract_flag_value() {
  local cmd="$1"
  local target_flag="$2"
  local next_is_value=0
  for token in $cmd; do
    if [[ "$next_is_value" == "1" ]]; then
      printf '%s\n' "$token"
      return 0
    fi
    if [[ "$token" == "$target_flag" ]]; then
      next_is_value=1
    fi
  done
  return 1
}

extract_env_id() {
  local cmd="$1"
  extract_flag_value "$cmd" "--env-id" || extract_flag_value "$cmd" "-e"
}

extract_mr_type() {
  local cmd="$1"
  extract_flag_value "$cmd" "--mr-type"
}

slugify() {
  python - "$1" <<'PY'
import sys
s = sys.argv[1].strip().replace(" ", "_")
out = []
for ch in s:
    if ch.isalnum() or ch in "-_.":
        out.append(ch)
    else:
        out.append("_")
print("".join(out))
PY
}

guess_latest_traj_dir() {
  local cmd="$1"
  local explicit_traj_dir=""
  explicit_traj_dir="$(extract_flag_value "$cmd" "--traj-dir" || true)"
  if [[ -n "$explicit_traj_dir" ]]; then
    printf '%s\n' "$explicit_traj_dir"
    return 0
  fi

  local env_id mr_type env_tag mr_tag prefix latest
  env_id="$(extract_env_id "$cmd")" || return 1
  mr_type="$(extract_mr_type "$cmd" || true)"
  env_tag="$(slugify "$env_id")"

  if [[ -n "$mr_type" ]]; then
    mr_tag="$(slugify "$mr_type")"
    prefix="${env_tag}_${mr_tag}_"
  else
    prefix="${env_tag}_"
  fi

  latest="$(find "$DEFAULT_TRAJ_ROOT" -maxdepth 1 -mindepth 1 -type d -name "${prefix}*" -printf '%f\n' | sort | tail -n 1)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf '%s\n' "$DEFAULT_TRAJ_ROOT/$latest"
}

if [[ "${#COMMANDS[@]}" -lt 2 ]]; then
  echo "Need at least 2 commands in COMMANDS." >&2
  exit 1
fi

if [[ "${#MR_CONFIGS[@]}" -gt 0 && "${#MR_CONFIGS[@]}" -ne "${#COMMANDS[@]}" ]]; then
  echo "MR_CONFIGS must either be empty or have the same length as COMMANDS." >&2
  exit 1
fi

if [[ "${#MR_CONFIGS_INLINE[@]}" -gt 0 && "${#MR_CONFIGS_INLINE[@]}" -ne "${#COMMANDS[@]}" ]]; then
  echo "MR_CONFIGS_INLINE must either be empty or have the same length as COMMANDS." >&2
  exit 1
fi

declare -a RESOLVED_TRAJ_DIRS=()

for i in "${!COMMANDS[@]}"; do
  run_cmd="${COMMANDS[$i]}"
  if [[ "${#MR_CONFIGS_INLINE[@]}" -gt 0 ]]; then
    mr_cfg_inline="${MR_CONFIGS_INLINE[$i]}"
    if [[ -n "$mr_cfg_inline" ]]; then
      tmp_yaml="$(mktemp /tmp/eval_dp_inline_mr_XXXXXX.yaml)"
      printf '%s\n' "$mr_cfg_inline" > "$tmp_yaml"
      TMP_YAML_FILES+=("$tmp_yaml")
      run_cmd="$run_cmd --mr-config $tmp_yaml"
    fi
  fi
  if [[ "$run_cmd" == "${COMMANDS[$i]}" && "${#MR_CONFIGS[@]}" -gt 0 ]]; then
    mr_cfg="${MR_CONFIGS[$i]}"
    if [[ -n "$mr_cfg" ]]; then
      run_cmd="$run_cmd --mr-config $mr_cfg"
    fi
  fi
  echo
  echo "[RUN $((i + 1))/${#COMMANDS[@]}] $run_cmd"
  eval "$run_cmd"
  resolved_dir="$(guess_latest_traj_dir "${COMMANDS[$i]}")" || {
    echo "Failed to resolve trajectory directory for command index $i." >&2
    exit 1
  }
  RESOLVED_TRAJ_DIRS+=("$resolved_dir")
  echo "[RUN $((i + 1))/${#COMMANDS[@]}] traj-dir => $resolved_dir"
done

BASE_TRAJ_DIR="${RESOLVED_TRAJ_DIRS[$BASE_CMD_INDEX]}"
MR_TRAJ_DIR="${RESOLVED_TRAJ_DIRS[$MR_CMD_INDEX]}"

if [[ -z "$ANALYSIS_OUT" ]]; then
  echo
  echo "[ANALYZE] base=$BASE_TRAJ_DIR mr=$MR_TRAJ_DIR mr_id=$MR_ID"
  python "$ROOT_DIR/eval_sim/analysis_dp/mr_eval_analyzer_dp.py" \
    --base-traj-dir "$BASE_TRAJ_DIR" \
    --mr-traj-dir "$MR_TRAJ_DIR" \
    --mr-id "$MR_ID"
else
  echo
  echo "[ANALYZE] base=$BASE_TRAJ_DIR mr=$MR_TRAJ_DIR mr_id=$MR_ID out=$ANALYSIS_OUT"
  python "$ROOT_DIR/eval_sim/analysis_dp/mr_eval_analyzer_dp.py" \
    --base-traj-dir "$BASE_TRAJ_DIR" \
    --mr-traj-dir "$MR_TRAJ_DIR" \
    --mr-id "$MR_ID" \
    --out "$ANALYSIS_OUT"
fi
