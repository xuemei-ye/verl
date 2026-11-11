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
"""CPU tests for E2BBackend against a fake E2B SDK (no network, e2b not required)."""

import asyncio
import importlib.metadata
import os
import unittest
from unittest.mock import patch

from tests.sandbox.fake_e2b_sdk import (
    AuthenticationException,
    InvalidArgumentException,
    NotFoundException,
    RateLimitException,
    ServiceBusyException,
    TemplateException,
    TimeoutException,
    make_fake_sdk,
)
from verl.sandbox.e2b_backend import E2BBackend, E2BConnectionSettings, load_e2b_sdk
from verl.sandbox.errors import (
    SandboxAuthenticationError,
    SandboxCapacityError,
    SandboxCloseError,
    SandboxConfigurationError,
    SandboxCreateError,
    SandboxNotFoundError,
    SandboxTimeoutError,
)
from verl.sandbox.models import SandboxCreateRequest, SandboxExecuteRequest

SETTINGS = E2BConnectionSettings(api_key="test-key-123", domain="e2b.example.internal")


def _backend(sdk, **kwargs):
    return E2BBackend(settings=SETTINGS, sdk=sdk, **kwargs)


def _create_request(**kwargs):
    defaults = dict(template="base", sandbox_ttl_s=600.0, request_timeout_s=60.0)
    defaults.update(kwargs)
    return SandboxCreateRequest(**defaults)


class TestE2BBackendCreate(unittest.IsolatedAsyncioTestCase):
    async def test_create_passes_ttl_and_request_timeout_separately(self):
        sdk = make_fake_sdk()
        backend = _backend(sdk)
        handle = await backend.create(_create_request(sandbox_ttl_s=599.2, request_timeout_s=45.0))
        call = sdk.create_calls[0]
        assert call["timeout"] == 600
        assert call["request_timeout"] == 45.0
        assert call["template"] == "base"
        assert call["api_key"] == "test-key-123"
        assert call["domain"] == "e2b.example.internal"
        assert "api_url" not in call
        assert handle.backend == "e2b"
        assert backend.active_sessions == 1

    async def test_capacity_blocks_until_close(self):
        sdk = make_fake_sdk()
        backend = _backend(sdk, max_concurrency=1)
        first = await backend.create(_create_request())
        second_task = asyncio.create_task(backend.create(_create_request()))
        await asyncio.sleep(0.05)
        assert not second_task.done()
        await backend.close(first)
        second = await asyncio.wait_for(second_task, timeout=1)
        assert second.queue_ms > 0
        assert backend.active_sessions == 1

    async def test_failed_create_releases_permit(self):
        sdk = make_fake_sdk(create_error=RateLimitException("429"))
        backend = _backend(sdk, max_concurrency=1)
        with self.assertRaises(SandboxCapacityError):
            await backend.create(_create_request())
        sdk.create_error = None
        handle = await asyncio.wait_for(backend.create(_create_request()), timeout=1)
        assert handle.sandbox_id

    async def test_sdk_error_mapping_keeps_cause(self):
        cases = [
            (AuthenticationException("bad key test-key-123"), SandboxAuthenticationError),
            (RateLimitException("429"), SandboxCapacityError),
            (ServiceBusyException("503"), SandboxCapacityError),
            (TimeoutException("deadline"), SandboxTimeoutError),
            (InvalidArgumentException("bad"), SandboxConfigurationError),
            (TemplateException("no template"), SandboxConfigurationError),
            (NotFoundException("gone"), SandboxNotFoundError),
            (RuntimeError("protocol mismatch"), SandboxCreateError),
        ]
        for sdk_error, expected in cases:
            backend = _backend(make_fake_sdk(create_error=sdk_error), max_concurrency=1)
            with self.assertRaises(expected) as ctx:
                await backend.create(_create_request())
            assert ctx.exception.__cause__ is sdk_error
            assert "test-key-123" not in str(ctx.exception)

    async def test_cancelled_create_releases_permit(self):
        sdk = make_fake_sdk()
        backend = _backend(sdk, max_concurrency=1)

        async def slow_create(**kwargs):
            await asyncio.sleep(10)

        with patch.object(sdk.AsyncSandbox, "create", slow_create):
            task = asyncio.create_task(backend.create(_create_request()))
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        await asyncio.wait_for(backend.create(_create_request()), timeout=1)


class TestE2BBackendExecute(unittest.IsolatedAsyncioTestCase):
    async def test_success_and_timeout_wrapper(self):
        sdk = make_fake_sdk(run=lambda cmd, **kw: ("hello\n", "", 0))
        backend = _backend(sdk)
        handle = await backend.create(_create_request())
        result = await backend.execute(handle, SandboxExecuteRequest(command="echo hello", timeout_s=30))
        assert result.stdout == "hello\n"
        assert result.exit_code == 0
        assert not result.timed_out
        cmd, kwargs = sdk.created[0].commands.calls[0]
        assert cmd == "timeout -k 5s 30s bash -c 'echo hello'"
        assert kwargs["timeout"] > 30

    async def test_non_zero_exit_is_a_normal_result(self):
        sdk = make_fake_sdk(run=lambda cmd, **kw: ("", "Traceback\nZeroDivisionError", 1))
        backend = _backend(sdk)
        handle = await backend.create(_create_request())
        result = await backend.execute(handle, SandboxExecuteRequest(command="python3 x.py"))
        assert result.exit_code == 1
        assert "ZeroDivisionError" in result.stderr
        assert not result.timed_out

    async def test_command_timeout_maps_to_timed_out_result(self):
        async def run(cmd, **kwargs):
            await asyncio.sleep(0.05)
            return "partial", "", 124

        backend = _backend(make_fake_sdk(run=run))
        handle = await backend.create(_create_request())
        result = await backend.execute(handle, SandboxExecuteRequest(command="sleep 100", timeout_s=0.01))
        assert result.timed_out
        assert result.exit_code is None
        assert result.stdout == "partial"

    async def test_fast_exit_124_is_not_a_timeout(self):
        backend = _backend(make_fake_sdk(run=lambda cmd, **kw: ("", "", 124)))
        handle = await backend.create(_create_request())
        result = await backend.execute(handle, SandboxExecuteRequest(command="exit 124", timeout_s=30))
        assert not result.timed_out
        assert result.exit_code == 124

    async def test_output_truncated_to_tail(self):
        backend = _backend(make_fake_sdk(run=lambda cmd, **kw: ("a" * 50 + "END", "", 0)), max_output_chars=10)
        handle = await backend.create(_create_request())
        result = await backend.execute(handle, SandboxExecuteRequest(command="x"))
        assert result.truncated
        assert len(result.stdout) == 10
        assert result.stdout.endswith("END")

    async def test_sdk_timeout_is_infrastructure_error_and_close_still_works(self):
        def run(cmd, **kwargs):
            raise TimeoutException("stream deadline exceeded")

        backend = _backend(make_fake_sdk(run=run), max_concurrency=1)
        handle = await backend.create(_create_request())
        with self.assertRaises(SandboxTimeoutError):
            await backend.execute(handle, SandboxExecuteRequest(command="x"))
        await backend.close(handle)
        assert backend.active_sessions == 0
        await asyncio.wait_for(backend.create(_create_request()), timeout=1)

    async def test_cancel_during_execute_propagates_and_allows_close(self):
        async def run(cmd, **kwargs):
            await asyncio.sleep(10)

        sdk = make_fake_sdk(run=run)
        backend = _backend(sdk, max_concurrency=1)
        handle = await backend.create(_create_request())
        task = asyncio.create_task(backend.execute(handle, SandboxExecuteRequest(command="x")))
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await backend.close(handle)
        assert sdk.created[0].kill_calls == 1
        await asyncio.wait_for(backend.create(_create_request()), timeout=1)

    async def test_unknown_handle(self):
        backend = _backend(make_fake_sdk())
        handle = await backend.create(_create_request())
        await backend.close(handle)
        with self.assertRaises(SandboxNotFoundError):
            await backend.execute(handle, SandboxExecuteRequest(command="x"))


class TestE2BBackendClose(unittest.IsolatedAsyncioTestCase):
    async def test_close_is_idempotent(self):
        sdk = make_fake_sdk()
        backend = _backend(sdk, max_concurrency=1)
        handle = await backend.create(_create_request())
        await backend.close(handle)
        await backend.close(handle)
        assert sdk.created[0].kill_calls == 1
        # BoundedSemaphore raises ValueError on over-release, so a double release would have failed above.
        await asyncio.wait_for(backend.create(_create_request()), timeout=1)

    async def test_concurrent_close_kills_once(self):
        sdk = make_fake_sdk(kill_delay_s=0.02)
        backend = _backend(sdk)
        handle = await backend.create(_create_request())
        await asyncio.gather(backend.close(handle), backend.close(handle))
        assert sdk.created[0].kill_calls == 1

    async def test_close_failure_still_releases_permit(self):
        sdk = make_fake_sdk(kill_error=TimeoutException("kill timed out"))
        backend = _backend(sdk, max_concurrency=1)
        handle = await backend.create(_create_request())
        with self.assertRaises(SandboxCloseError):
            await backend.close(handle)
        assert backend.active_sessions == 0
        await backend.close(handle)
        await asyncio.wait_for(backend.create(_create_request()), timeout=1)

    async def test_cancelled_close_finishes_in_background(self):
        sdk = make_fake_sdk(kill_delay_s=0.05)
        backend = _backend(sdk, max_concurrency=1)
        handle = await backend.create(_create_request())
        closer = asyncio.create_task(backend.close(handle))
        await asyncio.sleep(0.01)
        closer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closer
        await asyncio.sleep(0.1)
        assert backend.active_sessions == 0
        assert sdk.created[0].kill_calls == 1
        await asyncio.wait_for(backend.create(_create_request()), timeout=1)

    async def test_reset_returns_new_sandbox_with_same_request(self):
        sdk = make_fake_sdk()
        backend = _backend(sdk, max_concurrency=1)
        request = _create_request(template="custom", sandbox_ttl_s=900.0)
        old = await backend.create(request)
        new = await backend.reset(old)
        assert new.sandbox_id != old.sandbox_id
        assert sdk.created[0].kill_calls == 1
        assert sdk.create_calls[1]["template"] == "custom"
        assert sdk.create_calls[1]["timeout"] == 900
        assert backend.active_sessions == 1


class TestE2BLocalChecks(unittest.TestCase):
    def test_settings_from_env(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SandboxConfigurationError):
                E2BConnectionSettings.from_env()
        with patch.dict(os.environ, {"E2B_API_KEY": "k"}, clear=True):
            with self.assertRaises(SandboxConfigurationError):
                E2BConnectionSettings.from_env()
        with patch.dict(os.environ, {"E2B_API_KEY": "k", "E2B_API_URL": "http://e2b-api.internal"}, clear=True):
            settings = E2BConnectionSettings.from_env()
            assert settings.api_params() == {"api_key": "k", "api_url": "http://e2b-api.internal"}

    def test_settings_repr_hides_key(self):
        assert "test-key-123" not in repr(SETTINGS)

    def test_sdk_not_installed(self):
        with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaises(SandboxConfigurationError) as ctx:
                load_e2b_sdk()
        assert "pip install" in str(ctx.exception)

    def test_sdk_version_out_of_range(self):
        for bad in ("2.49.9", "2.51.0", "3.0.0"):
            with patch.object(importlib.metadata, "version", return_value=bad):
                with self.assertRaises(SandboxConfigurationError):
                    load_e2b_sdk()

    def test_sdk_version_in_range(self):
        sentinel = object()
        with (
            patch.object(importlib.metadata, "version", return_value="2.50.3"),
            patch("verl.sandbox.e2b_backend.importlib.import_module", return_value=sentinel),
        ):
            assert load_e2b_sdk() is sentinel


if __name__ == "__main__":
    unittest.main()
