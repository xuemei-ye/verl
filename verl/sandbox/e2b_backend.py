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
"""E2B implementation of ``SandboxBackend``. The only module in verl that imports the E2B SDK."""

import asyncio
import hashlib
import importlib
import logging
import math
import os
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .errors import (
    SandboxAuthenticationError,
    SandboxCapacityError,
    SandboxCloseError,
    SandboxConfigurationError,
    SandboxCreateError,
    SandboxError,
    SandboxExecutionError,
    SandboxNotFoundError,
    SandboxTimeoutError,
)
from .models import SandboxCreateRequest, SandboxExecuteRequest, SandboxHandle, SandboxResult

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BACKEND_NAME = "e2b"
E2B_SDK_REQUIREMENT = ">=2.50.0,<2.51"

# Commands run under coreutils `timeout`, so a user-code timeout is a normal exit status (124, or 137
# after the kill-after SIGKILL) and any SDK-level timeout can be treated as an infrastructure failure.
_TIMEOUT_EXIT_CODES = (124, 137)
_KILL_AFTER_S = 5.0
_SDK_GRACE_S = 10.0
_MAX_ERROR_CHARS = 500


def load_e2b_sdk() -> Any:
    """Import the E2B SDK and check the pinned version. Raises ``SandboxConfigurationError``."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.specifiers import SpecifierSet

    try:
        installed = version("e2b")
    except PackageNotFoundError as exc:
        raise SandboxConfigurationError(
            f"E2B SDK is not installed. Install it with: pip install 'e2b{E2B_SDK_REQUIREMENT}'"
        ) from exc
    if installed not in SpecifierSet(E2B_SDK_REQUIREMENT):
        raise SandboxConfigurationError(
            f"E2B SDK version {installed} is not supported; required e2b{E2B_SDK_REQUIREMENT}."
        )
    try:
        return importlib.import_module("e2b")
    except ImportError as exc:
        raise SandboxConfigurationError(f"Failed to import E2B SDK {installed}: {exc}") from exc


@dataclass(frozen=True)
class E2BConnectionSettings:
    api_key: str = field(repr=False)
    domain: Optional[str] = None
    api_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "E2BConnectionSettings":
        api_key = os.environ.get("E2B_API_KEY", "").strip()
        domain = os.environ.get("E2B_DOMAIN", "").strip() or None
        api_url = os.environ.get("E2B_API_URL", "").strip() or None
        if not api_key:
            raise SandboxConfigurationError("Environment variable E2B_API_KEY is not set.")
        # Refuse to silently fall back to the public e2b.app endpoint.
        if domain is None and api_url is None:
            raise SandboxConfigurationError("Set E2B_DOMAIN (or E2B_API_URL for self-hosted deployments).")
        return cls(api_key=api_key, domain=domain, api_url=api_url)

    def api_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"api_key": self.api_key}
        if self.domain is not None:
            params["domain"] = self.domain
        if self.api_url is not None:
            params["api_url"] = self.api_url
        return params


@dataclass
class _SessionRecord:
    sandbox: Any
    create_request: SandboxCreateRequest
    created_at: float
    last_used_at: float
    state: str = "ready"  # ready / executing / closing / closed / failed
    holds_permit: bool = True
    close_task: Optional["asyncio.Task[Optional[SandboxError]]"] = None
    trajectory_id: Optional[str] = None
    trace_id: Optional[str] = None


def sandbox_id_hash(sandbox_id: str) -> str:
    return hashlib.sha256(sandbox_id.encode()).hexdigest()[:12]


def _truncate_tail(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[-limit:], True


def _sdk_types(sdk: Any, *names: str) -> tuple[type, ...]:
    return tuple(t for t in (getattr(sdk, n, None) for n in names) if isinstance(t, type))


class E2BBackend:
    """E2B ``SandboxBackend`` with a per-process limit on alive sandboxes.

    A capacity permit is acquired before ``create`` and held until ``close``. The semaphore is local to the
    current process (one AgentLoopWorker); it is not a global limit.
    """

    def __init__(
        self,
        settings: E2BConnectionSettings,
        max_concurrency: int = 1,
        close_request_timeout_s: float = 30.0,
        max_output_chars: int = 16000,
        sdk: Any = None,
    ):
        if max_concurrency < 1:
            raise SandboxConfigurationError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._settings = settings
        self._sdk = sdk if sdk is not None else load_e2b_sdk()
        self._capacity = asyncio.BoundedSemaphore(max_concurrency)
        self._close_request_timeout_s = close_request_timeout_s
        self._max_output_chars = max_output_chars
        self._sessions: dict[str, _SessionRecord] = {}
        # Strong references so in-flight close tasks are not garbage collected.
        self._close_tasks: set[asyncio.Task] = set()

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    async def create(self, request: SandboxCreateRequest) -> SandboxHandle:
        queue_start = time.monotonic()
        await self._capacity.acquire()
        queue_ms = (time.monotonic() - queue_start) * 1000

        create_start = time.monotonic()
        try:
            sandbox = await asyncio.wait_for(
                self._sdk.AsyncSandbox.create(
                    template=request.template,
                    timeout=max(1, math.ceil(request.sandbox_ttl_s)),
                    metadata=dict(request.metadata) or None,
                    request_timeout=request.request_timeout_s,
                    **self._settings.api_params(),
                ),
                timeout=request.request_timeout_s + _SDK_GRACE_S,
            )
        except BaseException as exc:
            self._capacity.release()
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, asyncio.TimeoutError):
                logger.warning("E2B create timed out locally; a remote sandbox may linger until its TTL expires.")
            raise self._map_error(exc, "create", SandboxCreateError) from exc

        now = time.monotonic()
        create_ms = (now - create_start) * 1000
        sandbox_id = sandbox.sandbox_id
        self._sessions[sandbox_id] = _SessionRecord(
            sandbox=sandbox, create_request=request, created_at=now, last_used_at=now
        )
        logger.info(
            "sandbox created backend=%s sandbox_id_hash=%s queue_ms=%.1f create_ms=%.1f",
            BACKEND_NAME,
            sandbox_id_hash(sandbox_id),
            queue_ms,
            create_ms,
        )
        return SandboxHandle(sandbox_id=sandbox_id, backend=BACKEND_NAME, queue_ms=queue_ms, create_ms=create_ms)

    async def execute(self, handle: SandboxHandle, request: SandboxExecuteRequest) -> SandboxResult:
        record = self._sessions.get(handle.sandbox_id)
        if record is None:
            raise SandboxNotFoundError(f"Unknown sandbox handle {sandbox_id_hash(handle.sandbox_id)}")
        if record.state != "ready":
            raise SandboxExecutionError(f"Sandbox is not ready (state={record.state})")

        record.state = "executing"
        record.last_used_at = time.monotonic()
        command = self._wrap_with_timeout(request.command, request.timeout_s)
        start = time.monotonic()
        try:
            try:
                result = await asyncio.wait_for(
                    record.sandbox.commands.run(
                        command,
                        envs=dict(request.environment) or None,
                        cwd=request.working_directory,
                        timeout=request.timeout_s + _SDK_GRACE_S,
                    ),
                    timeout=request.timeout_s + 2 * _SDK_GRACE_S,
                )
                stdout, stderr, exit_code = result.stdout, result.stderr, result.exit_code
            except tuple(_sdk_types(self._sdk, "CommandExitException")) as exc:
                stdout, stderr, exit_code = exc.stdout, exc.stderr, exc.exit_code
        except BaseException as exc:
            record.state = "failed"
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise self._map_error(exc, "execute", SandboxExecutionError) from exc

        record.state = "ready"
        elapsed_ms = (time.monotonic() - start) * 1000
        # Client-side elapsed time is always >= remote elapsed time, so this cannot misfire on fast exits.
        timed_out = exit_code in _TIMEOUT_EXIT_CODES and elapsed_ms >= request.timeout_s * 1000
        stdout, stdout_truncated = _truncate_tail(stdout or "", self._max_output_chars)
        stderr, stderr_truncated = _truncate_tail(stderr or "", self._max_output_chars)
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=None if timed_out else exit_code,
            elapsed_ms=elapsed_ms,
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
        )

    async def reset(self, handle: SandboxHandle) -> SandboxHandle:
        record = self._sessions.get(handle.sandbox_id)
        if record is None:
            raise SandboxNotFoundError(f"Unknown sandbox handle {sandbox_id_hash(handle.sandbox_id)}")
        request = record.create_request
        await self.close(handle)
        return await self.create(request)

    async def close(self, handle: SandboxHandle) -> None:
        """Idempotent. Local state and the capacity permit are always released, even if this call is
        cancelled or the remote kill fails; the server-side TTL covers what the remote kill misses."""
        record = self._sessions.get(handle.sandbox_id)
        if record is None:
            return
        if record.close_task is None:
            record.state = "closing"
            task = asyncio.get_running_loop().create_task(self._close_remote(handle.sandbox_id, record))
            self._close_tasks.add(task)
            task.add_done_callback(self._close_tasks.discard)
            record.close_task = task
        error = await asyncio.shield(record.close_task)
        if error is not None:
            raise error

    async def health_check(self) -> bool:
        raise NotImplementedError("health_check is deferred to P1; use verl.tools.e2b_preflight before training.")

    async def _close_remote(self, sandbox_id: str, record: _SessionRecord) -> Optional[SandboxError]:
        start = time.monotonic()
        error: Optional[SandboxError] = None
        try:
            await asyncio.wait_for(
                record.sandbox.kill(request_timeout=self._close_request_timeout_s),
                timeout=self._close_request_timeout_s + _SDK_GRACE_S,
            )
        except Exception as exc:
            error = SandboxCloseError(f"E2B close failed: {self._safe_message(exc)}")
            error.__cause__ = exc
            logger.warning(
                "sandbox close failed backend=%s sandbox_id_hash=%s error_type=%s; relying on TTL",
                BACKEND_NAME,
                sandbox_id_hash(sandbox_id),
                type(exc).__name__,
            )
        finally:
            record.state = "closed"
            self._sessions.pop(sandbox_id, None)
            if record.holds_permit:
                record.holds_permit = False
                self._capacity.release()
        logger.info(
            "sandbox closed backend=%s sandbox_id_hash=%s close_ms=%.1f success=%d",
            BACKEND_NAME,
            sandbox_id_hash(sandbox_id),
            (time.monotonic() - start) * 1000,
            int(error is None),
        )
        return error

    @staticmethod
    def _wrap_with_timeout(command: str, timeout_s: float) -> str:
        return f"timeout -k {_KILL_AFTER_S:g}s {timeout_s:g}s bash -c {shlex.quote(command)}"

    def _safe_message(self, exc: BaseException) -> str:
        message = str(exc) or type(exc).__name__
        if self._settings.api_key:
            message = message.replace(self._settings.api_key, "***")
        return message[:_MAX_ERROR_CHARS]

    def _map_error(self, exc: BaseException, operation: str, default: type[SandboxError]) -> SandboxError:
        sdk = self._sdk
        message = f"E2B {operation} failed ({type(exc).__name__}): {self._safe_message(exc)}"
        if isinstance(exc, _sdk_types(sdk, "AuthenticationException")):
            cls: type[SandboxError] = SandboxAuthenticationError
        elif isinstance(exc, _sdk_types(sdk, "RateLimitException", "ServiceBusyException")):
            cls = SandboxCapacityError
        elif isinstance(exc, (asyncio.TimeoutError, *_sdk_types(sdk, "TimeoutException"))):
            cls = SandboxTimeoutError
        elif isinstance(exc, _sdk_types(sdk, "NotFoundException")):
            cls = SandboxNotFoundError
        elif isinstance(exc, _sdk_types(sdk, "InvalidArgumentException", "TemplateException")):
            cls = SandboxConfigurationError
        else:
            cls = default
        return cls(message)
