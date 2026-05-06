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

#第一处要修改的位置，执行评估命令，修改为当前MR所需的场景，--mr-type是保存的MR文件名
COMMANDS=(
  #"python -m eval_sim.eval_dp --pretrained_path ./700.ckpt -e PickCube-v1 --show --mr-type MR-SADP-1-原"
  "python -m eval_sim.eval_dp --pretrained_path ./700.ckpt -e PickCube-v1 --show  --mr-type MR-FPDP2"
  # "python -m eval_sim.eval_dp --pretrained_path ./700.ckpt -e PickCube-v1 --show --mr-type MR-SADP-1-0.05"
)

# 直接指定基线（原用例）或衍生（MR）轨迹目录/文件路径的快捷变量。
# 若不为空则优先使用，脚本不会为该角色强制运行命令来生成轨迹。
BASE_TRAJ_OVERRIDE="eef_traj_dp/PickCube/PickCube-v1_20260421_153203"
MR_TRAJ_OVERRIDE="eef_traj_dp/PickCube/PickCube-v1_MR-FPDP2_20260506_215704"

# 可选：直接指定某条命令对应的已存在轨迹目录（或单个 json 文件路径）。
# 若对应项非空，则脚本不会运行该命令，直接使用该路径作为 traj-dir。
# 数量若非空，必须与 COMMANDS 等长。
TRAJ_DIR_OVERRIDES=(
  ""   # 可写: "eef_traj_dp/PickCube/...."
  ""   # 第二条命令的覆盖路径
  # ""
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

#第二处要修改的位置，决定了当前任务是否配置正确的MR
MR_CONFIGS_INLINE=(
  #$'mr:\n  language:\n    type: identity\n  vision:\n    type: identity\n  proprio:\n    type: identity\n  env:\n    type: MR-SADP-1-translate_cube_xy'
  $'mr:\n  language:\n    type: identity\n  vision:\n    type: identity\n  proprio:\n    type: MR-FPDP2\n  env:\n    type: identity\n '
  # ""
)

#第三处要修改的位置，MR_ID决定了分析文件时使用哪个MR评测指标，通常与 --mr-type 保持一致（但不强制）
MR_ID="MR-FPDP-2"
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

# 检查：只在没有用 override 填充对应角色时，才要求足够的 COMMANDS
needed_commands=0
if [[ -z "${BASE_TRAJ_OVERRIDE:-}" ]]; then
  needed_commands=$((needed_commands + 1))
fi
if [[ -z "${MR_TRAJ_OVERRIDE:-}" ]]; then
  needed_commands=$((needed_commands + 1))
fi
if [[ "${#COMMANDS[@]}" -lt "$needed_commands" ]]; then
  echo "Need at least $needed_commands commands in COMMANDS (or provide BASE_TRAJ_OVERRIDE / MR_TRAJ_OVERRIDE)." >&2
  exit 1
fi

if [[ "${#MR_CONFIGS[@]}" -gt 0 && "${#MR_CONFIGS[@]}" -lt "${#COMMANDS[@]}" ]]; then
  echo "MR_CONFIGS must be empty or have at least the same length as COMMANDS." >&2
  exit 1
fi

if [[ "${#MR_CONFIGS_INLINE[@]}" -gt 0 && "${#MR_CONFIGS_INLINE[@]}" -lt "${#COMMANDS[@]}" ]]; then
  echo "MR_CONFIGS_INLINE must be empty or have at least the same length as COMMANDS." >&2
  exit 1
fi

if [[ "${#TRAJ_DIR_OVERRIDES[@]}" -gt 0 && "${#TRAJ_DIR_OVERRIDES[@]}" -lt "${#COMMANDS[@]}" ]]; then
  echo "TRAJ_DIR_OVERRIDES must be empty or have at least the same length as COMMANDS." >&2
  exit 1
fi

declare -a RESOLVED_TRAJ_DIRS=()

# 新增：如果两条轨迹都已经存在，则不执行 COMMANDS，直接分析
if [[ -n "${BASE_TRAJ_OVERRIDE:-}" && -n "${MR_TRAJ_OVERRIDE:-}" && -e "$BASE_TRAJ_OVERRIDE" && -e "$MR_TRAJ_OVERRIDE" ]]; then
  BASE_TRAJ_DIR="$BASE_TRAJ_OVERRIDE"
  MR_TRAJ_DIR="$MR_TRAJ_OVERRIDE"

  echo
  echo "[ANALYZE] base=$BASE_TRAJ_DIR mr=$MR_TRAJ_DIR mr_id=$MR_ID"
  python -m eval_sim.analysis_dp.mr_eval_analyzer_dp \
    --base-traj-dir "$BASE_TRAJ_DIR" \
    --mr-traj-dir "$MR_TRAJ_DIR" \
    --mr-id "$MR_ID"
  exit 0
fi

for i in "${!COMMANDS[@]}"; do
  # ...existing code before building run_cmd...
  run_cmd="${COMMANDS[$i]}"

  # apply inline MR config temp file if any
  if [[ "${#MR_CONFIGS_INLINE[@]}" -gt 0 ]]; then
    mr_cfg_inline="${MR_CONFIGS_INLINE[$i]:-}"
    if [[ -n "$mr_cfg_inline" ]]; then
      tmp_yaml="$(mktemp /tmp/eval_dp_inline_mr_XXXXXX.yaml)"
      printf '%s\n' "$mr_cfg_inline" > "$tmp_yaml"
      TMP_YAML_FILES+=("$tmp_yaml")
      run_cmd="$run_cmd --mr-config $tmp_yaml"
    fi
  fi
  if [[ "$run_cmd" == "${COMMANDS[$i]}" && "${#MR_CONFIGS[@]}" -gt 0 ]]; then
    mr_cfg="${MR_CONFIGS[$i]:-}"
    if [[ -n "$mr_cfg" ]]; then
      run_cmd="$run_cmd --mr-config $mr_cfg"
    fi
  fi

  # 新增：优先使用 TRAJ_DIR_OVERRIDES（如果指定则跳过 eval 运行）
  override_dir=""
  if [[ "${#TRAJ_DIR_OVERRIDES[@]}" -gt 0 ]]; then
    override_dir="${TRAJ_DIR_OVERRIDES[$i]:-}"
  fi

  if [[ -n "$override_dir" ]]; then
    echo
    echo "[SKIP RUN $((i + 1))/${#COMMANDS[@]}] Using override traj dir: $override_dir"
    if [[ ! -e "$override_dir" ]]; then
      echo "TRAJ_DIR_OVERRIDES[$i] 指定的路径不存在: $override_dir" >&2
      exit 1
    fi
    RESOLVED_TRAJ_DIRS+=("$override_dir")
    echo "[SKIP RUN $((i + 1))/${#COMMANDS[@]}] traj-dir => $override_dir"
    continue
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

resolve_dir_by_role() {
  local role_name="$1"
  local override_value="$2"
  local default_index="$3"

  if [[ -n "$override_value" ]]; then
    if [[ ! -e "$override_value" ]]; then
      echo "${role_name}_TRAJ_OVERRIDE 指定路径不存在: $override_value" >&2
      exit 1
    fi
    printf '%s\n' "$override_value"
    return 0
  fi

  if [[ "$default_index" =~ ^[0-9]+$ && "$default_index" -lt "${#RESOLVED_TRAJ_DIRS[@]}" ]]; then
    printf '%s\n' "${RESOLVED_TRAJ_DIRS[$default_index]}"
    return 0
  fi

  # 只有一条命令时，允许把唯一结果用于未显式指定的另一侧
  if [[ "${#RESOLVED_TRAJ_DIRS[@]}" -eq 1 ]]; then
    printf '%s\n' "${RESOLVED_TRAJ_DIRS[0]}"
    return 0
  fi

  echo "无法解析 ${role_name} 轨迹目录：请检查 ${role_name}_TRAJ_OVERRIDE 或 COMMANDS / ${role_name}_CMD_INDEX" >&2
  exit 1
}

# 决定最终用作分析的 base/mr 路径
BASE_TRAJ_DIR="$(resolve_dir_by_role "BASE" "${BASE_TRAJ_OVERRIDE:-}" "${BASE_CMD_INDEX}")"
MR_TRAJ_DIR="$(resolve_dir_by_role "MR" "${MR_TRAJ_OVERRIDE:-}" "${MR_CMD_INDEX}")"

if [[ -z "$ANALYSIS_OUT" ]]; then
  echo
  echo "[ANALYZE] base=$BASE_TRAJ_DIR mr=$MR_TRAJ_DIR mr_id=$MR_ID"
  python -m eval_sim.analysis_dp.mr_eval_analyzer_dp \
    --base-traj-dir "$BASE_TRAJ_DIR" \
    --mr-traj-dir "$MR_TRAJ_DIR" \
    --mr-id "$MR_ID"
else
  echo
  echo "[ANALYZE] base=$BASE_TRAJ_DIR mr=$MR_TRAJ_DIR mr_id=$MR_ID out=$ANALYSIS_OUT"
  python -m eval_sim.analysis_dp.mr_eval_analyzer_dp \
    --base-traj-dir "$BASE_TRAJ_DIR" \
    --mr-traj-dir "$MR_TRAJ_DIR" \
    --mr-id "$MR_ID" \
    --out "$ANALYSIS_OUT"
fi
