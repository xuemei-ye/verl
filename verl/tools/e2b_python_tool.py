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
"""Python code tool backed by a remote E2B sandbox.

P0 lifecycle is per tool call: ``ToolAgentLoop._call_tool`` runs create -> execute -> release for every
call, so no file or process state survives between calls of the same trajectory.
"""

import base64
import dataclasses
import logging
import os
import shlex
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from verl.sandbox.e2b_backend import BACKEND_NAME, E2BBackend, E2BConnectionSettings, sandbox_id_hash
from verl.sandbox.errors import (
    SandboxAuthenticationError,
    SandboxConfigurationError,
    SandboxError,
)
from verl.sandbox.models import SandboxCreateRequest, SandboxExecuteRequest, SandboxHandle, SandboxResult
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool, FatalToolError, TransientToolError
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The whole command is one argv string for `bash -c` inside the sandbox; Linux caps a single argument at
# 128 KiB. Worst case 4-byte UTF-8 * 4/3 base64 overhead keeps this limit under ~128 KB.
_MAX_CODE_CHARS_LIMIT = 24000


@dataclass(frozen=True)
class E2BPythonToolConfig:
    template: Optional[str] = None
    max_concurrency: int = 1
    sandbox_ttl_s: float = 600.0
    create_request_timeout_s: float = 60.0
    command_timeout_s: float = 30.0
    close_request_timeout_s: float = 30.0
    max_code_chars: int = 20000
    max_output_chars: int = 16000

    @classmethod
    def from_dict(cls, config: dict) -> "E2BPythonToolConfig":
        values = dict(config)
        if values.pop("type", "native") != "native":
            raise SandboxConfigurationError("E2BPythonTool requires config.type == 'native'.")
        backend = values.pop("backend", BACKEND_NAME)
        if backend != BACKEND_NAME:
            raise SandboxConfigurationError(f"Unsupported sandbox backend {backend!r}; only 'e2b' is available.")

        allowed = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise SandboxConfigurationError(f"Unknown E2BPythonTool config keys: {unknown}")

        for name in ("max_concurrency", "max_code_chars", "max_output_chars"):
            if name in values and (not isinstance(values[name], int) or isinstance(values[name], bool)):
                raise SandboxConfigurationError(f"{name} must be an integer, got {values[name]!r}")
        for name in ("sandbox_ttl_s", "create_request_timeout_s", "command_timeout_s", "close_request_timeout_s"):
            if name in values:
                if isinstance(values[name], bool) or not isinstance(values[name], (int, float)):
                    raise SandboxConfigurationError(f"{name} must be a number, got {values[name]!r}")
                values[name] = float(values[name])
        if values.get("template") is not None and not isinstance(values["template"], str):
            raise SandboxConfigurationError("template must be a string or null")

        cfg = cls(**values)
        for f in dataclasses.fields(cls):
            value = getattr(cfg, f.name)
            if f.name != "template" and value <= 0:
                raise SandboxConfigurationError(f"{f.name} must be > 0, got {value}")
        if cfg.max_code_chars > _MAX_CODE_CHARS_LIMIT:
            raise SandboxConfigurationError(f"max_code_chars must be <= {_MAX_CODE_CHARS_LIMIT}")
        lifecycle_budget = cfg.create_request_timeout_s + cfg.command_timeout_s + cfg.close_request_timeout_s
        if cfg.sandbox_ttl_s < lifecycle_budget:
            raise SandboxConfigurationError(
                f"sandbox_ttl_s ({cfg.sandbox_ttl_s}) must cover create + command + close timeouts "
                f"({lifecycle_budget}); otherwise the server may kill the sandbox mid-call."
            )
        return cfg


def build_python_command(code: str) -> str:
    """Write code to a temp file via base64 so no model-generated text is interpreted by the shell."""
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return f'f="$(mktemp)" && printf %s {shlex.quote(encoded)} | base64 -d > "$f" && python3 "$f"'


def format_result(result: SandboxResult, timeout_s: float) -> str:
    # Status goes last: ToolAgentLoop truncates from the left by default, keeping the tail.
    parts = []
    if result.stdout:
        parts.append(f"[stdout]\n{result.stdout}")
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr}")
    if not parts:
        parts.append("(no output)")
    if result.truncated:
        parts.append("[note] output was truncated")
    if result.timed_out:
        parts.append(f"[status] timed out after {timeout_s:g} s")
    else:
        parts.append(f"[exit_code] {result.exit_code}")
    return "\n".join(parts)


def _to_tool_error(exc: SandboxError) -> FatalToolError:
    fatal = isinstance(exc, (SandboxConfigurationError, SandboxAuthenticationError))
    cls = FatalToolError if fatal else TransientToolError
    error = cls(f"{type(exc).__name__}: {exc}")
    error.__cause__ = exc
    return error


class E2BPythonTool(BaseTool):
    """Executes model-written Python in a remote E2B sandbox.

    ``__init__`` only does local checks (SDK version, env vars, config). Remote reachability and
    authentication are verified by ``python -m verl.tools.e2b_preflight`` before training starts.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.tool_config = E2BPythonToolConfig.from_dict(config)
        super().__init__(config, tool_schema)
        self.backend = E2BBackend(
            settings=E2BConnectionSettings.from_env(),
            max_concurrency=self.tool_config.max_concurrency,
            close_request_timeout_s=self.tool_config.close_request_timeout_s,
            max_output_chars=self.tool_config.max_output_chars,
        )
        self._handles: dict[str, SandboxHandle] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        request = SandboxCreateRequest(
            template=self.tool_config.template,
            sandbox_ttl_s=self.tool_config.sandbox_ttl_s,
            request_timeout_s=self.tool_config.create_request_timeout_s,
        )
        try:
            handle = await self.backend.create(request)
        except SandboxError as exc:
            raise _to_tool_error(exc) from exc
        instance_id = instance_id or str(uuid4())
        self._handles[instance_id] = handle
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        handle = self._handles.get(instance_id)
        if handle is None:
            raise FatalToolError(f"E2BPythonTool.execute called with unknown instance_id {instance_id}")
        metrics = {
            "sandbox/backend": BACKEND_NAME,
            "sandbox/queue_ms": handle.queue_ms,
            "sandbox/create_ms": handle.create_ms,
            "sandbox/execute_ms": 0.0,
            "sandbox/success": 0,
            "sandbox/timeout": 0,
        }

        code = parameters.get("code") if isinstance(parameters, dict) else None
        if not isinstance(code, str) or not code.strip():
            return ToolResponse(text="Error: 'code' must be a non-empty string."), 0.0, metrics
        if len(code) > self.tool_config.max_code_chars:
            return (
                ToolResponse(
                    text=f"Error: code is {len(code)} characters; the limit is {self.tool_config.max_code_chars}."
                ),
                0.0,
                metrics,
            )

        timeout_s = self.tool_config.command_timeout_s
        request = SandboxExecuteRequest(command=build_python_command(code), timeout_s=timeout_s, request_id=instance_id)
        try:
            result = await self.backend.execute(handle, request)
        except SandboxError as exc:
            raise _to_tool_error(exc) from exc

        metrics["sandbox/execute_ms"] = result.elapsed_ms
        metrics["sandbox/success"] = int(result.exit_code == 0)
        metrics["sandbox/timeout"] = int(result.timed_out)
        logger.info(
            "sandbox execute tool_call_id=%s sandbox_id_hash=%s exit_code=%s timed_out=%s execute_ms=%.1f",
            instance_id,
            sandbox_id_hash(handle.sandbox_id),
            result.exit_code,
            result.timed_out,
            result.elapsed_ms,
        )
        return ToolResponse(text=format_result(result, timeout_s)), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        handle = self._handles.pop(instance_id, None)
        if handle is None:
            return
        try:
            await self.backend.close(handle)
        except SandboxError as exc:
            # The local permit is already released and the server-side TTL reclaims the sandbox.
            logger.warning("E2B release failed for tool_call_id=%s: %s", instance_id, exc)
