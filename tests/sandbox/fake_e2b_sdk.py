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
"""In-memory stand-in for the subset of the e2b 2.50 SDK that E2BBackend uses.

Class names and the exception hierarchy mirror e2b 2.50.0: AuthenticationException and ServiceBusyException
are not SandboxException subclasses, and CommandExitException carries stdout/stderr/exit_code.
"""

import asyncio
import itertools
from types import SimpleNamespace
from typing import Any, Callable, Optional


class SandboxException(Exception):
    pass


class TimeoutException(SandboxException):
    pass


class InvalidArgumentException(SandboxException):
    pass


class NotFoundException(SandboxException):
    pass


class TemplateException(SandboxException):
    pass


class RateLimitException(SandboxException):
    pass


class AuthenticationException(Exception):
    pass


class ServiceBusyException(Exception):
    pass


class CommandExitException(SandboxException):
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 1):
        super().__init__(f"Command exited with code {exit_code}")
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


def make_fake_sdk(
    run: Optional[Callable[..., Any]] = None,
    create_error: Optional[BaseException] = None,
    kill_error: Optional[BaseException] = None,
    kill_delay_s: float = 0.0,
) -> SimpleNamespace:
    """Build a fake ``e2b`` module.

    ``run(cmd, **kwargs)`` may be sync or async and returns ``(stdout, stderr, exit_code)``; a non-zero exit
    code raises ``CommandExitException`` like the real SDK. Inspect ``sdk.created`` / ``sdk.create_calls``.
    """
    sdk = SimpleNamespace(created=[], create_calls=[], create_error=create_error)
    ids = itertools.count()

    class FakeCommands:
        def __init__(self):
            self.calls = []

        async def run(self, cmd, **kwargs):
            self.calls.append((cmd, kwargs))
            out = run(cmd, **kwargs) if run is not None else ("", "", 0)
            if asyncio.iscoroutine(out):
                out = await out
            stdout, stderr, exit_code = out
            if exit_code != 0:
                raise CommandExitException(stdout, stderr, exit_code)
            return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=0, error=None)

    class FakeAsyncSandbox:
        def __init__(self, sandbox_id: str):
            self.sandbox_id = sandbox_id
            self.commands = FakeCommands()
            self.kill_calls = 0

        @classmethod
        async def create(cls, **kwargs):
            sdk.create_calls.append(kwargs)
            if sdk.create_error is not None:
                raise sdk.create_error
            sandbox = cls(f"sbx-{next(ids)}")
            sdk.created.append(sandbox)
            return sandbox

        async def kill(self, **kwargs):
            self.kill_calls += 1
            if kill_delay_s:
                await asyncio.sleep(kill_delay_s)
            if kill_error is not None:
                raise kill_error
            return True

    sdk.AsyncSandbox = FakeAsyncSandbox
    for exc in (
        SandboxException,
        TimeoutException,
        InvalidArgumentException,
        NotFoundException,
        TemplateException,
        RateLimitException,
        AuthenticationException,
        ServiceBusyException,
        CommandExitException,
    ):
        setattr(sdk, exc.__name__, exc)
    return sdk
