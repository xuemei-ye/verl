# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2B connectivity preflight. Run once per physical node before starting Ray / training:

    python -m verl.tools.e2b_preflight --tool-config examples/e2b_agent/tool_config.yaml

Performs create -> run a Python command under coreutils ``timeout`` -> kill. Exits non-zero on any failure.
"""

import argparse
import asyncio
import sys

from omegaconf import OmegaConf

from verl.sandbox.e2b_backend import E2BBackend, E2BConnectionSettings, sandbox_id_hash
from verl.sandbox.errors import SandboxError
from verl.sandbox.models import SandboxCreateRequest, SandboxExecuteRequest
from verl.tools.e2b_python_tool import E2BPythonToolConfig, build_python_command

TOOL_CLASS_NAME = "verl.tools.e2b_python_tool.E2BPythonTool"
_PROBE_TOKEN = "verl-e2b-preflight-ok"


def load_tool_config(path: str) -> E2BPythonToolConfig:
    tools = OmegaConf.load(path).get("tools") or []
    for tool in tools:
        if tool.get("class_name") == TOOL_CLASS_NAME:
            return E2BPythonToolConfig.from_dict(OmegaConf.to_container(tool.config, resolve=True))
    raise SandboxError(f"No tool with class_name {TOOL_CLASS_NAME} in {path}")


async def run_preflight(cfg: E2BPythonToolConfig, backend: E2BBackend) -> None:
    handle = await backend.create(
        SandboxCreateRequest(
            template=cfg.template,
            sandbox_ttl_s=cfg.sandbox_ttl_s,
            request_timeout_s=cfg.create_request_timeout_s,
            metadata={"purpose": "verl-preflight"},
        )
    )
    sid = sandbox_id_hash(handle.sandbox_id)
    print(f"[preflight] create ok sandbox_id_hash={sid} create_ms={handle.create_ms:.0f}")
    try:
        code = f"import sys\nprint({_PROBE_TOKEN!r}, sys.version_info[:2])"
        result = await backend.execute(
            handle, SandboxExecuteRequest(command=build_python_command(code), timeout_s=cfg.command_timeout_s)
        )
        if result.exit_code != 0 or _PROBE_TOKEN not in result.stdout:
            raise SandboxError(
                f"probe command failed: exit_code={result.exit_code} timed_out={result.timed_out} "
                f"stderr={result.stderr[-500:]!r}. The template needs bash, coreutils (timeout, base64, mktemp) "
                "and python3."
            )
        print(f"[preflight] execute ok elapsed_ms={result.elapsed_ms:.0f} stdout={result.stdout.strip()!r}")
    finally:
        await backend.close(handle)
    print("[preflight] close ok")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check E2B sandbox connectivity before training.")
    parser.add_argument("--tool-config", required=True, help="Path to the tool YAML containing E2BPythonTool.")
    args = parser.parse_args(argv)
    try:
        cfg = load_tool_config(args.tool_config)
        backend = E2BBackend(
            settings=E2BConnectionSettings.from_env(),
            max_concurrency=1,
            close_request_timeout_s=cfg.close_request_timeout_s,
            max_output_chars=cfg.max_output_chars,
        )
        asyncio.run(run_preflight(cfg, backend))
    except SandboxError as exc:
        print(f"[preflight] FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("[preflight] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
