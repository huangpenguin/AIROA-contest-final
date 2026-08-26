#!/usr/bin/env bash
# Rebuild Axis Pi0.5 inputs from the tracked lock and HF credentials.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
LOCK="$ROOT/configs/axis_pi05/data_sources.lock.yaml"
ENV_ROOT="${AXIS_PI05_ENV_ROOT:-$ROOT/.venv_axis_pi05}"
if [[ -n "${AXIS_PI05_PYTHON:-}" ]]; then
    PYTHON_BIN="$AXIS_PI05_PYTHON"
elif command -v python3.12 >/dev/null; then
    PYTHON_BIN="$(command -v python3.12)"
elif command -v python3.11 >/dev/null; then
    PYTHON_BIN="$(command -v python3.11)"
else
    PYTHON_BIN="python3"
fi

if [[ "${1:-}" == "--lock" && -n "${2:-}" ]]; then
    LOCK="$ROOT/${2#./}"
else
    echo "用法: $0 --lock configs/axis_pi05/data_sources.lock.yaml" >&2
    exit 2
fi

command -v git >/dev/null || { echo "缺少 git" >&2; exit 1; }
command -v uv >/dev/null || { echo "缺少 uv；不会修改正赛 evaluator 环境" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "缺少 ffmpeg" >&2; exit 1; }
[[ -f "$LOCK" ]] || { echo "lock 不存在: $LOCK" >&2; exit 1; }

GPU_STATUS="unavailable"
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null; then
    GPU_STATUS="available"
else
    echo "WARN: nvidia-smi/GPU 不可用；bootstrap 可继续，训练入口会拒绝。" >&2
fi
echo "GPU=$GPU_STATUS HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && echo present || echo absent)"
echo "disk=$(df -Pk "$ROOT" | tail -1 | awk '{print $4 " KB free"}')"

if [[ -x "$ENV_ROOT/bin/python" ]] && ! "$ENV_ROOT/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    uv venv --clear --python "$PYTHON_BIN" "$ENV_ROOT"
elif [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    uv venv --python "$PYTHON_BIN" "$ENV_ROOT"
fi
BOOT_PY="$ENV_ROOT/bin/python"
uv pip install --python "$BOOT_PY" "PyYAML>=6" "huggingface_hub>=0.30"
PYTHONPATH="$ROOT" "$BOOT_PY" -m tools.axis_pi05.bootstrap --lock "$LOCK" --root "$ROOT" --env "$ENV_ROOT"
if [[ -z "${MUJOCO_PLUGIN_PATH:-}" ]] && [[ -d "${MUJOCO_PATH:-}/bin/mujoco_plugin" ]]; then
    export MUJOCO_PLUGIN_PATH="$MUJOCO_PATH/bin/mujoco_plugin"
fi
UV_PROJECT_ENVIRONMENT="$ENV_ROOT" uv sync --project "$ROOT/third_party/axis_openpi" --python "$PYTHON_BIN" --no-dev
