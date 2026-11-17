#!/usr/bin/env bash

set -euo pipefail

cd /root/work/yexuemei/verl

PYTHON_BIN=/usr/local/python3.11.14/bin/python3
export PATH=/usr/local/python3.11.14/bin:${PATH}

source /usr/local/Ascend/cann-8.5.1/set_env.sh

export PYTHONPATH="/root/work/yexuemei/verl:${PYTHONPATH:-}"

export VERL_PLATFORM=huawei
export DEVICE=npu
export VERL_USE_UV=0
export RAY_ADDRESS=auto
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export VERL_LOGGING_LEVEL=INFO
export MPLBACKEND=Agg

export E2B_API_URL="${E2B_API_URL:-https://dev-e2b.xiaobei.top}"
export E2B_DOMAIN="${E2B_DOMAIN:-dev-e2b.xiaobei.top}"
export E2B_PROXY="${E2B_PROXY:-http://10.202.9.50:3928}"

: "${E2B_API_KEY:?请先通过九天 Secret 或 export E2B_API_KEY=... 设置密钥}"

MODEL_PATH=/root/work/models/Qwen3-8B
TRAIN_FILE=/root/work/yexuemei/data/e2b-agent-gsm8k-full/train-full-fixed.parquet 
VAL_FILE=/root/work/yexuemei/data/e2b-agent-gsm8k-full/test-full-fixed.parquet
TOOL_CONFIG=/root/work/yexuemei/verl/examples/e2b_agent/tool_config.yaml

TARGET_STEPS=2
VALIDATION_FREQ=-1
SAVE_FREQ=-1

# 每个 AgentLoopWorker 的 tool_config.yaml 中 max_concurrency=1。
# 因此这里设置为 4 时，全局最多约有 4 个存活中的 sandbox。
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-50}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="/root/work/yexuemei/training-runs/e2b-gspo-8b-50steps-${RUN_ID}"
RUN_START_EPOCH="$(date +%s)"

export RUN_ROOT TARGET_STEPS RUN_START_EPOCH

mkdir -p \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/validation" \
  "${RUN_ROOT}/rollouts" \
  "${RUN_ROOT}/checkpoints" \
  "${RUN_ROOT}/metrics" \
  "${RUN_ROOT}/plots" \
  "${RUN_ROOT}/environment"

echo "===== Configuration ====="

cat <<EOF
RUN_ROOT=${RUN_ROOT}
MODEL_PATH=${MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE}
VAL_FILE=${VAL_FILE}
TOOL_CONFIG=${TOOL_CONFIG}
TARGET_STEPS=${TARGET_STEPS}
VALIDATION_FREQ=${VALIDATION_FREQ}
SAVE_FREQ=${SAVE_FREQ}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS}
ROLLOUT_N=8
EOF

echo "===== Save environment ====="

"${PYTHON_BIN}" -m pip freeze --all \
  > "${RUN_ROOT}/environment/pip-freeze.txt" || true

git rev-parse HEAD \
  > "${RUN_ROOT}/environment/verl-git-commit.txt" || true

{
  date
  uname -a
  "${PYTHON_BIN}" --version
  echo "VERL_PLATFORM=${VERL_PLATFORM}"
  echo "E2B_API_URL=${E2B_API_URL}"
  echo "E2B_DOMAIN=${E2B_DOMAIN}"
  echo "E2B_PROXY=${E2B_PROXY}"
  echo "E2B_API_KEY_SET=yes"
} > "${RUN_ROOT}/environment/runtime-info.txt"

echo "===== Ray cluster ====="
ray status

echo "===== NPU status ====="
npu-smi info

echo "===== Input files ====="

ls -lh \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_FILE}" \
  "${VAL_FILE}" \
  "${TOOL_CONFIG}"

echo "===== E2B preflight ====="

"${PYTHON_BIN}" -m verl.tools.e2b_preflight \
  --tool-config "${TOOL_CONFIG}" \
  2>&1 | tee "${RUN_ROOT}/logs/e2b-preflight.log"

echo "===== Start 50-step Agentic RL training ====="

set +e
set -o pipefail

(
  NNODES=2 \
  NDEVICES_PER_NODE=4 \
  MODEL_PATH="${MODEL_PATH}" \
  TRAIN_FILE="${TRAIN_FILE}" \
  TEST_FILE="${VAL_FILE}" \
  TRAIN_BATCH_SIZE=8 \
  PPO_MINI_BATCH_SIZE=8 \
  ROLLOUT_N=2 \
  SP_SIZE=2 \
  ROLLOUT_TP=2 \
  ROLLOUT_GPU_MEM_UTIL=0.40 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_RESPONSE_LENGTH=1536 \
  PPO_MAX_TOKEN_LEN_PER_GPU=3072 \
  TOTAL_EPOCHS=1 \
  SAVE_FREQ="${SAVE_FREQ}" \
  TEST_FREQ="${VALIDATION_FREQ}" \
  PROJECT_NAME=E2B-GSPO-QWEN3-8B \
  EXPERIMENT_NAME=Qwen3-8B-E2B-50Steps \
  bash examples/gspo_trainer/run_qwen3_8b_fsdp.sh \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.return_raw_chat=True \
    data.validation_shuffle=False \
    trainer.logger='["console"]' \
    trainer.device=npu \
    trainer.total_training_steps="${TARGET_STEPS}" \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.log_val_generations=8 \
    trainer.test_freq="${VALIDATION_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.validation_data_dir="${RUN_ROOT}/validation" \
    trainer.rollout_data_dir="${RUN_ROOT}/rollouts" \
    trainer.default_local_dir="${RUN_ROOT}/checkpoints" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO
) 2>&1 | tee "${RUN_ROOT}/logs/train.log"

TRAIN_RC=${PIPESTATUS[0]}
set -e

echo "TRAIN_EXIT_CODE=${TRAIN_RC}" \
  | tee -a "${RUN_ROOT}/logs/train.log"

echo "===== Collect E2B lifecycle logs ====="

LIFECYCLE_LOG="${RUN_ROOT}/logs/e2b-lifecycle.log"
: > "${LIFECYCLE_LOG}"

# 优先使用训练日志，避免与 Ray worker 日志重复统计。
grep -aE \
  'sandbox created backend=e2b|sandbox execute tool_call_id=|sandbox closed backend=e2b|E2B create failed|E2B execute failed|sandbox close failed' \
  "${RUN_ROOT}/logs/train.log" \
  > "${LIFECYCLE_LOG}" || true

# 如果训练日志没有转发 INFO 生命周期日志，再从 Ray worker 日志提取。
if [ ! -s "${LIFECYCLE_LOG}" ]; then
  while IFS= read -r -d '' ray_log; do
    grep -haE \
      'sandbox created backend=e2b|sandbox execute tool_call_id=|sandbox closed backend=e2b|E2B create failed|E2B execute failed|sandbox close failed' \
      "${ray_log}" >> "${LIFECYCLE_LOG}" || true
  done < <(
    find /root/work/ray_sessions \
      -type f \
      -newermt "@${RUN_START_EPOCH}" \
      \( -name 'worker-*.out' -o -name 'worker-*.err' \) \
      -print0 2>/dev/null
  )
fi

echo "===== Parse E2B lifecycle statistics ====="

"${PYTHON_BIN}" - <<'PY'
import json
import os
import re
from collections import Counter
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
log_path = root / "logs" / "e2b-lifecycle.log"

text = log_path.read_text(errors="replace") if log_path.exists() else ""
lines = text.splitlines()

exit_codes = Counter()
for match in re.finditer(r"exit_code=(None|-?\d+)", text):
    exit_codes[match.group(1)] += 1

summary = {
    "create_success": sum(
        "sandbox created backend=e2b" in line for line in lines
    ),
    "create_failure": sum(
        "E2B create failed" in line for line in lines
    ),
    "execute_returned": sum(
        "sandbox execute tool_call_id=" in line for line in lines
    ),
    "execute_failure": sum(
        "E2B execute failed" in line for line in lines
    ),
    "execute_exit_0": exit_codes.get("0", 0),
    "execute_exit_nonzero": sum(
        count
        for code, count in exit_codes.items()
        if code not in {"0", "None"}
    ),
    "execute_exit_none": exit_codes.get("None", 0),
    "timeout_count": sum(
        re.search(r"timed_out=(True|1)", line) is not None
        for line in lines
    ),
    "close_success": sum(
        "sandbox closed backend=e2b" in line
        and re.search(r"success=(1|True)", line) is not None
        for line in lines
    ),
    "close_failure": sum(
        "sandbox close failed" in line
        or (
            "sandbox closed backend=e2b" in line
            and re.search(r"success=(0|False)", line) is not None
        )
        for line in lines
    ),
    "exit_codes": dict(exit_codes),
}

create_total = summary["create_success"] + summary["create_failure"]
execute_total = summary["execute_returned"] + summary["execute_failure"]

summary["create_success_rate"] = (
    summary["create_success"] / create_total if create_total else None
)
summary["execute_success_rate"] = (
    summary["execute_exit_0"] / execute_total if execute_total else None
)
summary["close_completion_rate"] = (
    summary["close_success"] / summary["create_success"]
    if summary["create_success"]
    else None
)

json_path = root / "metrics" / "e2b-lifecycle-summary.json"
json_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)

env_path = root / "metrics" / "e2b-lifecycle-summary.env"
env_path.write_text(
    "\n".join(
        [
            f"E2B_CREATE_SUCCESS={summary['create_success']}",
            f"E2B_CREATE_FAILURE={summary['create_failure']}",
            f"E2B_EXECUTE_RETURNED={summary['execute_returned']}",
            f"E2B_EXECUTE_FAILURE={summary['execute_failure']}",
            f"E2B_EXECUTE_EXIT_0={summary['execute_exit_0']}",
            f"E2B_EXECUTE_EXIT_NONZERO={summary['execute_exit_nonzero']}",
            f"E2B_TIMEOUT_COUNT={summary['timeout_count']}",
            f"E2B_CLOSE_SUCCESS={summary['close_success']}",
            f"E2B_CLOSE_FAILURE={summary['close_failure']}",
        ]
    )
    + "\n"
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== Extract training metrics and generate plots ====="

"${PYTHON_BIN}" - <<'PY'
import csv
import json
import math
import os
import re
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
log_path = root / "logs" / "train.log"
text = log_path.read_text(errors="replace")
lines = text.splitlines()

number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

wanted = [
    "actor/loss",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/entropy",
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "response_length/mean",
    "training/num_turns/mean",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/step",
    "perf/throughput",
    "perf/mfu/actor",
]

rows = {}

def row_for(step):
    return rows.setdefault(step, {"step": step})

for line in lines:
    step_match = re.search(r"(?:^|\s)step:(\d+)", line)
    if not step_match:
        step_match = re.search(r"training/global_step:(\d+)", line)

    if step_match:
        step = int(step_match.group(1))
        row = row_for(step)

        for key in wanted:
            match = re.search(
                re.escape(key) + rf":({number})",
                line,
            )
            if match:
                row[key] = float(match.group(1))

        for match in re.finditer(
            rf"([A-Za-z0-9_./@-]*val[A-Za-z0-9_./@-]*"
            rf"(?:acc|reward)/mean@1):({number})",
            line,
        ):
            row[match.group(1)] = float(match.group(2))

# trainer.val_before_train=True 的 step 0 基线通常打印成字典。
for line in lines:
    if "Initial validation metrics" not in line:
        continue

    row = row_for(0)
    for match in re.finditer(
        rf"""['"]([^'"]*(?:acc|reward)/mean@1)['"]\s*:\s*({number})""",
        line,
    ):
        row[match.group(1)] = float(match.group(2))

ordered = [rows[step] for step in sorted(rows)]

columns = ["step"]
for row in ordered:
    for key in row:
        if key not in columns:
            columns.append(key)

csv_path = root / "metrics" / "training-metrics.csv"
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(ordered)

grad_values = [
    row.get("actor/grad_norm")
    for row in ordered
    if row.get("actor/grad_norm") is not None
]
max_grad_norm = max(grad_values) if grad_values else None
max_step = max(rows) if rows else 0

validation_keys = [
    key
    for key in columns
    if "val" in key and (
        "acc/mean@1" in key or "reward/mean@1" in key
    )
]

validation_summary = {}
for key in validation_keys:
    points = [
        (row["step"], row[key])
        for row in ordered
        if key in row
    ]
    if points:
        validation_summary[key] = {
            "baseline_step": points[0][0],
            "baseline": points[0][1],
            "final_step": points[-1][0],
            "final": points[-1][1],
            "delta": points[-1][1] - points[0][1],
        }

summary = {
    "max_completed_step": max_step,
    "max_grad_norm": max_grad_norm,
    "nonzero_gradient_observed": bool(
        max_grad_norm is not None and max_grad_norm > 0
    ),
    "validation": validation_summary,
}

(root / "metrics" / "training-summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)

(root / "metrics" / "training-summary.env").write_text(
    "\n".join(
        [
            f"MAX_COMPLETED_STEP={max_step}",
            "NONZERO_GRADIENT_OBSERVED="
            + (
                "1"
                if summary["nonzero_gradient_observed"]
                else "0"
            ),
        ]
    )
    + "\n"
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"metrics CSV: {csv_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"Metrics saved, but plot failed: {exc}")
    raise SystemExit(0)

def points(key):
    return [
        (row["step"], row[key])
        for row in ordered
        if key in row
        and isinstance(row[key], (int, float))
        and math.isfinite(row[key])
    ]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for key in [
    "actor/loss",
    "actor/pg_loss",
    "actor/grad_norm",
]:
    values = points(key)
    if values:
        axes[0][0].plot(
            [x for x, _ in values],
            [y for _, y in values],
            marker="o",
            label=key,
        )

axes[0][0].set_title("Actor training")
axes[0][0].set_xlabel("step")
axes[0][0].grid(True, alpha=0.3)
axes[0][0].legend()

for key in [
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
]:
    values = points(key)
    if values:
        axes[0][1].plot(
            [x for x, _ in values],
            [y for _, y in values],
            marker="o",
            label=key,
        )

axes[0][1].set_title("Reward and advantage")
axes[0][1].set_xlabel("step")
axes[0][1].grid(True, alpha=0.3)
axes[0][1].legend()

for key in validation_keys:
    values = points(key)
    if values:
        axes[1][0].plot(
            [x for x, _ in values],
            [y for _, y in values],
            marker="o",
            label=key,
        )

axes[1][0].set_title("Validation")
axes[1][0].set_xlabel("step")
axes[1][0].grid(True, alpha=0.3)
axes[1][0].legend()

for key in [
    "perf/throughput",
    "timing_s/step",
    "timing_s/gen",
]:
    values = points(key)
    if values:
        axes[1][1].plot(
            [x for x, _ in values],
            [y for _, y in values],
            marker="o",
            label=key,
        )

axes[1][1].set_title("Performance")
axes[1][1].set_xlabel("step")
axes[1][1].grid(True, alpha=0.3)
axes[1][1].legend()

fig.tight_layout()

plot_path = root / "plots" / "training-curves.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"plot: {plot_path}")
PY

echo "===== Generated validation and rollout files ====="

find "${RUN_ROOT}/validation" \
  -type f \
  -name '*.jsonl' \
  -ls || true

find "${RUN_ROOT}/rollouts" \
  -type f \
  -name '*.jsonl' \
  -ls || true

echo "===== Checkpoints ====="

find "${RUN_ROOT}/checkpoints" \
  -maxdepth 1 \
  -type d \
  -name 'global_step_*' \
  -print \
  | sort -V || true

echo "===== Training summary ====="

grep -aE \
  'step:[0-9]+|Initial validation metrics|Final validation metrics|TRAIN_EXIT_CODE|Traceback|RayTaskError|OutOfMemory' \
  "${RUN_ROOT}/logs/train.log" \
  | tail -50 || true

source "${RUN_ROOT}/metrics/training-summary.env"
source "${RUN_ROOT}/metrics/e2b-lifecycle-summary.env"

FINAL_STATUS=0

if [ "${TRAIN_RC}" -ne 0 ]; then
  echo "FAILED: training process exit=${TRAIN_RC}"
  FINAL_STATUS=1
fi

if [ "${MAX_COMPLETED_STEP}" -lt "${TARGET_STEPS}" ]; then
  echo "FAILED: only completed step ${MAX_COMPLETED_STEP}/${TARGET_STEPS}"
  FINAL_STATUS=1
fi

if [ "${NONZERO_GRADIENT_OBSERVED}" -ne 1 ]; then
  echo "FAILED: no non-zero actor gradient was observed"
  FINAL_STATUS=1
fi

if [ "${E2B_CREATE_SUCCESS}" -le 0 ]; then
  echo "FAILED: no successful E2B sandbox creation was recorded"
  FINAL_STATUS=1
fi

if [ "${E2B_EXECUTE_RETURNED}" -le 0 ]; then
  echo "FAILED: no E2B tool execution was recorded"
  FINAL_STATUS=1
fi

if [ "${E2B_CREATE_FAILURE}" -gt 0 ]; then
  echo "WARNING: E2B create failures=${E2B_CREATE_FAILURE}"
fi

if [ "${E2B_EXECUTE_FAILURE}" -gt 0 ]; then
  echo "WARNING: E2B execute failures=${E2B_EXECUTE_FAILURE}"
fi

if [ "${E2B_CLOSE_FAILURE}" -gt 0 ]; then
  echo "WARNING: E2B close failures=${E2B_CLOSE_FAILURE}"
fi

echo "===== Final result ====="
echo "TRAIN_EXIT_CODE=${TRAIN_RC}"
echo "MAX_COMPLETED_STEP=${MAX_COMPLETED_STEP}"
echo "NONZERO_GRADIENT_OBSERVED=${NONZERO_GRADIENT_OBSERVED}"
echo "E2B_CREATE_SUCCESS=${E2B_CREATE_SUCCESS}"
echo "E2B_CREATE_FAILURE=${E2B_CREATE_FAILURE}"
echo "E2B_EXECUTE_RETURNED=${E2B_EXECUTE_RETURNED}"
echo "E2B_EXECUTE_EXIT_0=${E2B_EXECUTE_EXIT_0}"
echo "E2B_EXECUTE_EXIT_NONZERO=${E2B_EXECUTE_EXIT_NONZERO}"
echo "E2B_EXECUTE_FAILURE=${E2B_EXECUTE_FAILURE}"
echo "E2B_TIMEOUT_COUNT=${E2B_TIMEOUT_COUNT}"
echo "E2B_CLOSE_SUCCESS=${E2B_CLOSE_SUCCESS}"
echo "E2B_CLOSE_FAILURE=${E2B_CLOSE_FAILURE}"
echo "RUN_ROOT=${RUN_ROOT}"

echo "Metrics:"
echo "  ${RUN_ROOT}/metrics/training-metrics.csv"
echo "  ${RUN_ROOT}/metrics/training-summary.json"
echo "  ${RUN_ROOT}/metrics/e2b-lifecycle-summary.json"

echo "Plot:"
echo "  ${RUN_ROOT}/plots/training-curves.png"

if [ "${FINAL_STATUS}" -eq 0 ]; then
  echo "AGENTIC_E2B_TRAINING_SUCCESS"
else
  echo "AGENTIC_E2B_TRAINING_FAILED"
fi

exit "${FINAL_STATUS}"