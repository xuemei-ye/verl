# E2B Python tool for ToolAgentLoop (P0)

`verl.tools.e2b_python_tool.E2BPythonTool` runs model-written Python in a remote E2B sandbox.
The generic interface lives in `verl/sandbox/`; `verl/sandbox/e2b_backend.py` is the only module that imports the E2B SDK.

## Install

```bash
pip install -e ".[e2b]"   # pins e2b>=2.50.0,<2.51
```

Do not install `e2b-code-interpreter`; P0 uses plain command execution.

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `E2B_API_KEY` | yes | API key. Only from env / secret manager, never in YAML or scripts. |
| `E2B_DOMAIN` | one of the two | Domain of the E2B service; the SDK derives `api.<domain>` and sandbox URLs from it. |
| `E2B_API_URL` | one of the two | Explicit control-plane URL for self-hosted deployments that need it. |

The tool refuses to start if neither `E2B_DOMAIN` nor `E2B_API_URL` is set, so it never falls back to the public `e2b.app` silently.
The sandbox access token is returned by the create call and handled by the SDK; there is no `E2B_ACCESS_TOKEN` setting.

## Fail-fast in three stages

1. `E2BPythonTool.__init__` (every AgentLoopWorker): SDK version, env vars and config keys/values. No network calls.
2. Preflight, **once per physical node before `ray start` / training**:

   ```bash
   python -m verl.tools.e2b_preflight --tool-config examples/e2b_agent/tool_config.yaml
   ```

   It runs create -> `python3` under coreutils `timeout` -> kill, and exits non-zero on any failure.
   The template must provide `bash`, coreutils (`timeout`, `base64`, `mktemp`) and `python3`.
3. At runtime `_call_tool` re-raises `FatalToolError` instead of turning it into an observation:

| Situation | What happens |
|---|---|
| Python exception / non-zero exit / syntax error | Normal observation with stdout, stderr, `[exit_code] N` |
| Code exceeds `command_timeout_s` | Normal observation `[status] timed out after N s` |
| Missing or empty `code`, code too long | Normal observation with an error message the model can fix |
| Auth failure, invalid template/config | `FatalToolError`, the rollout aborts |
| 429 / 503 / control-plane or network timeout | `TransientToolError` (a `FatalToolError`), the rollout aborts in P0 |

P0 does not drop and resample trajectories on transient failures; that is P1 work.

## Concurrency

`max_concurrency` limits **alive sandboxes per AgentLoopWorker process**. Each worker loads its own tool instance, so the global upper bound is about `agent.num_workers x max_concurrency` (default `8 x 1`).
For the first link check use `actor_rollout_ref.rollout.agent.num_workers=1` with `max_concurrency: 1`, which allows at most one alive sandbox.
Even then, LLM generation for different samples in the same worker is still concurrent.

## Output length: two truncation layers

1. The backend keeps the last `max_output_chars` characters of stdout and of stderr.
2. `ToolAgentLoop` truncates the whole observation to `multi_turn.max_tool_response_length` characters (default **256**).

Always set `max_tool_response_length` explicitly (1024-4096 is a reasonable range for math/Python tasks).
The status line is placed at the end of the observation, so it survives the default left-side truncation.

## Lifecycle and cleanup

P0 is per tool call: every call creates a sandbox, runs once and kills it in `release()`. State does not carry over between calls.
`close` is idempotent, runs in a shielded task with its own timeout and always releases the local capacity permit.
`sandbox_ttl_s` is set on create, so sandboxes that a crashed process never closed are reclaimed by the server.
`sandbox_ttl_s` must be at least `create_request_timeout_s + command_timeout_s + close_request_timeout_s`.

## Troubleshooting

- `Body of the request is None` / unexpected HTTP 200 on create: the self-hosted service returns a response shape the pinned SDK does not accept. This is a protocol mismatch between server and SDK version; fix the server or pick an SDK version validated against it. Do not monkey-patch the SDK.
- Preflight fails with exit code 127: the template is missing `timeout`, `base64`, `mktemp` or `python3`.
- Observations end in `(truncated)` at 256 chars: `max_tool_response_length` was not set.

## Run

```bash
export E2B_API_KEY=...        # from your secret manager
export E2B_DOMAIN=...
export MODEL_PATH=/path/to/Qwen3-8B
export TRAIN_FILES="['/path/train.parquet']" VAL_FILES="['/path/val.parquet']"
bash examples/e2b_agent/run_qwen3_8b_fsdp.sh
```

The dataset must contain chat prompts that instruct the model to call the `python` tool, and a reward function suitable for the task.
