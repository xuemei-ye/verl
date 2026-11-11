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
"""CPU tests for E2BPythonTool and the E2B preflight against a fake E2B SDK."""

import asyncio
import base64
import os
import re
import shlex
import tempfile
import unittest
from unittest.mock import patch

from tests.sandbox.fake_e2b_sdk import (
    AuthenticationException,
    RateLimitException,
    TemplateException,
    TimeoutException,
    make_fake_sdk,
)
from verl.sandbox.e2b_backend import E2BBackend, E2BConnectionSettings
from verl.sandbox.errors import SandboxConfigurationError, SandboxError
from verl.sandbox.models import SandboxResult
from verl.tools.base_tool import FatalToolError, TransientToolError
from verl.tools.e2b_preflight import main as preflight_main
from verl.tools.e2b_preflight import run_preflight
from verl.tools.e2b_python_tool import E2BPythonTool, E2BPythonToolConfig, build_python_command, format_result
from verl.tools.schemas import OpenAIFunctionToolSchema

ENV = {"E2B_API_KEY": "test-key-123", "E2B_DOMAIN": "e2b.example.internal"}
SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "code"}},
                "required": ["code"],
            },
        },
    }
)


def _make_tool(sdk, **config):
    full = {"type": "native", "backend": "e2b", "template": "base", **config}
    with patch.dict(os.environ, ENV, clear=True), patch("verl.sandbox.e2b_backend.load_e2b_sdk", return_value=sdk):
        return E2BPythonTool(config=full, tool_schema=SCHEMA)


def _decode_code(command: str) -> str:
    """Recover the Python source embedded by build_python_command."""
    token = re.search(r"printf %s (\S+) \|", command).group(1)
    return base64.b64decode(shlex.split(token)[0]).decode("utf-8")


class TestE2BPythonToolConfig(unittest.TestCase):
    def test_defaults_are_valid(self):
        cfg = E2BPythonToolConfig.from_dict({"type": "native", "backend": "e2b", "sandbox_ttl_s": 600})
        assert cfg.max_concurrency == 1
        assert cfg.sandbox_ttl_s == 600.0

    def test_invalid_configs_rejected(self):
        bad_configs = [
            {"unknown_key": 1},
            {"backend": "docker"},
            {"type": "mcp"},
            {"max_concurrency": 0},
            {"max_concurrency": True},
            {"max_concurrency": 1.5},
            {"command_timeout_s": 0},
            {"command_timeout_s": "30"},
            {"sandbox_ttl_s": 100},  # < 60 + 30 + 30
            {"max_code_chars": 100000},
            {"template": 123},
        ]
        for bad in bad_configs:
            with self.subTest(config=bad), self.assertRaises(SandboxConfigurationError):
                E2BPythonToolConfig.from_dict({"type": "native", **bad})

    def test_init_requires_env_and_makes_no_remote_call(self):
        sdk = make_fake_sdk()
        with patch.dict(os.environ, {}, clear=True), patch(
            "verl.sandbox.e2b_backend.load_e2b_sdk", return_value=sdk
        ):
            with self.assertRaises(SandboxConfigurationError):
                E2BPythonTool(config={"type": "native"}, tool_schema=SCHEMA)
        _make_tool(sdk)
        assert sdk.create_calls == []


class TestE2BPythonTool(unittest.IsolatedAsyncioTestCase):
    async def test_full_lifecycle(self):
        sdk = make_fake_sdk(run=lambda cmd, **kw: ("42\n", "", 0))
        tool = _make_tool(sdk)
        instance_id, _ = await tool.create(create_kwargs={})
        response, reward, metrics = await tool.execute(instance_id, {"code": "print(6 * 7)"})
        assert "[stdout]\n42" in response.text
        assert response.text.endswith("[exit_code] 0")
        assert reward == 0.0
        assert metrics["sandbox/backend"] == "e2b"
        assert metrics["sandbox/success"] == 1
        assert metrics["sandbox/timeout"] == 0
        assert set(metrics) == {
            "sandbox/backend",
            "sandbox/queue_ms",
            "sandbox/create_ms",
            "sandbox/execute_ms",
            "sandbox/success",
            "sandbox/timeout",
        }
        cmd, _ = sdk.created[0].commands.calls[0]
        assert cmd.startswith("timeout -k 5s 30s bash -c ")
        await tool.release(instance_id)
        assert sdk.created[0].kill_calls == 1
        assert tool.backend.active_sessions == 0

    async def test_invalid_code_is_observation_not_error(self):
        sdk = make_fake_sdk()
        tool = _make_tool(sdk, max_code_chars=10)
        instance_id, _ = await tool.create()
        for params in ({}, {"code": ""}, {"code": "   "}, {"code": 123}, {"code": "x" * 11}):
            response, _, metrics = await tool.execute(instance_id, params)
            assert response.text.startswith("Error:")
            assert metrics["sandbox/success"] == 0
        assert sdk.created[0].commands.calls == []
        await tool.release(instance_id)

    async def test_non_zero_exit_is_observation(self):
        sdk = make_fake_sdk(run=lambda cmd, **kw: ("", "ZeroDivisionError: division by zero", 1))
        tool = _make_tool(sdk)
        instance_id, _ = await tool.create()
        response, _, metrics = await tool.execute(instance_id, {"code": "1/0"})
        assert "[stderr]\nZeroDivisionError" in response.text
        assert response.text.endswith("[exit_code] 1")
        assert metrics["sandbox/success"] == 0
        await tool.release(instance_id)

    async def test_command_timeout_is_observation(self):
        async def run(cmd, **kwargs):
            await asyncio.sleep(0.05)
            return "", "", 124

        tool = _make_tool(make_fake_sdk(run=run), command_timeout_s=0.01)
        instance_id, _ = await tool.create()
        response, _, metrics = await tool.execute(instance_id, {"code": "while True: pass"})
        assert response.text.endswith("[status] timed out after 0.01 s")
        assert metrics["sandbox/timeout"] == 1
        await tool.release(instance_id)

    async def test_create_errors_are_classified(self):
        cases = [
            (AuthenticationException("401"), FatalToolError, False),
            (TemplateException("no such template"), FatalToolError, False),
            (RateLimitException("429"), TransientToolError, True),
            (TimeoutException("deadline"), TransientToolError, True),
        ]
        for sdk_error, expected, transient in cases:
            tool = _make_tool(make_fake_sdk(create_error=sdk_error))
            with self.assertRaises(expected) as ctx:
                await tool.create()
            assert isinstance(ctx.exception, TransientToolError) == transient
            assert isinstance(ctx.exception.__cause__, SandboxError)
            assert "test-key-123" not in str(ctx.exception)
            assert tool.backend.active_sessions == 0

    async def test_execute_infrastructure_error_raises_and_release_cleans_up(self):
        def run(cmd, **kwargs):
            raise TimeoutException("stream deadline exceeded")

        sdk = make_fake_sdk(run=run)
        tool = _make_tool(sdk)
        instance_id, _ = await tool.create()
        with self.assertRaises(TransientToolError):
            await tool.execute(instance_id, {"code": "print(1)"})
        await tool.release(instance_id)
        assert sdk.created[0].kill_calls == 1
        assert tool.backend.active_sessions == 0

    async def test_release_swallows_close_error_and_is_idempotent(self):
        sdk = make_fake_sdk(kill_error=TimeoutException("kill timed out"))
        tool = _make_tool(sdk)
        instance_id, _ = await tool.create()
        await tool.release(instance_id)
        await tool.release(instance_id)
        await tool.release("never-created")
        assert sdk.created[0].kill_calls == 1
        await asyncio.wait_for(tool.create(), timeout=1)


class TestCommandAndFormatting(unittest.TestCase):
    def test_build_python_command_round_trips_untrusted_code(self):
        code = "print('it\\'s'); import os\nprint(\"$(rm -rf /) `id` ; | && 中文\")\n"
        command = build_python_command(code)
        assert _decode_code(command) == code
        assert "rm -rf" not in command
        assert command.endswith('python3 "$f"')

    def test_format_result_puts_status_last(self):
        text = format_result(SandboxResult(stdout="out", stderr="err", exit_code=2, elapsed_ms=1.0), 30)
        assert text == "[stdout]\nout\n[stderr]\nerr\n[exit_code] 2"
        text = format_result(
            SandboxResult(stdout="", stderr="", exit_code=None, elapsed_ms=1.0, timed_out=True, truncated=True), 30
        )
        assert text == "(no output)\n[note] output was truncated\n[status] timed out after 30 s"


class TestPreflight(unittest.IsolatedAsyncioTestCase):
    async def test_run_preflight_create_execute_close(self):
        def run(cmd, **kwargs):
            inner = shlex.split(cmd)[-1]
            assert "verl-e2b-preflight-ok" in _decode_code(inner)
            return "verl-e2b-preflight-ok (3, 11)\n", "", 0

        sdk = make_fake_sdk(run=run)
        backend = E2BBackend(settings=E2BConnectionSettings(api_key="k", domain="d"), sdk=sdk)
        await run_preflight(E2BPythonToolConfig(), backend)
        assert sdk.created[0].kill_calls == 1
        assert backend.active_sessions == 0

    async def test_run_preflight_fails_on_missing_python_and_still_closes(self):
        sdk = make_fake_sdk(run=lambda cmd, **kw: ("", "python3: command not found", 127))
        backend = E2BBackend(settings=E2BConnectionSettings(api_key="k", domain="d"), sdk=sdk)
        with self.assertRaises(SandboxError):
            await run_preflight(E2BPythonToolConfig(), backend)
        assert sdk.created[0].kill_calls == 1

    def test_main_exits_non_zero_without_env(self):
        yaml_text = (
            "tools:\n"
            "  - class_name: verl.tools.e2b_python_tool.E2BPythonTool\n"
            "    config:\n"
            "      type: native\n"
            "      backend: e2b\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
        try:
            with patch.dict(os.environ, {}, clear=True), patch(
                "verl.sandbox.e2b_backend.load_e2b_sdk", return_value=make_fake_sdk()
            ):
                assert preflight_main(["--tool-config", f.name]) == 1
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
