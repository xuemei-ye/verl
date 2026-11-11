#!/usr/bin/env bash
# P0 link check: Qwen3-8B + ToolAgentLoop + E2BPythonTool on a single node.
# Required env: E2B_API_KEY, E2B_DOMAIN (or E2B_API_URL), MODEL_PATH, TRAIN_FILES, VAL_FILES.
# Multi-node: run `python -m verl.tools.e2b_preflight --tool-config ...` on every node before `ray start`.
set -euo pipefail

: "${E2B_API_KEY:?E2B_API_KEY is not set}"
: "${MODEL_PATH:?MODEL_PATH is not set}"
: "${TRAIN_FILES:?TRAIN_FILES is not set, e.g. \"['/data/train.parquet']\"}"
: "${VAL_FILES:?VAL_FILES is not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-$SCRIPT_DIR/tool_config.yaml}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# 1 worker x max_concurrency 1 => at most one alive sandbox globally. Raise only after the link is stable.
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-1}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2048}

python3 -m verl.tools.e2b_preflight --tool-config "$TOOL_CONFIG_PATH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.return_raw_chat=True \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-16} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-2048} \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-8192} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME:-vllm} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT:-4} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP:-2} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_TOOL_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
    trainer.logger=['console'] \
    trainer.project_name=${PROJECT_NAME:-verl_e2b_agent} \
    trainer.experiment_name=${EXPERIMENT_NAME:-qwen3_8b_e2b_p0} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    "$@"
