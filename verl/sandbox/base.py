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
from typing import Protocol

from .models import SandboxCreateRequest, SandboxExecuteRequest, SandboxHandle, SandboxResult


class SandboxBackend(Protocol):
    """Vendor-neutral async sandbox protocol.

    Contract:
    - ``close`` is idempotent and always frees local capacity, even if the remote close fails.
    - ``execute`` never retries; a non-zero exit code is a normal ``SandboxResult``.
    - ``asyncio.CancelledError`` is always propagated.
    - ``health_check`` is part of the protocol but has no caller until P1.
    """

    async def create(self, request: SandboxCreateRequest) -> SandboxHandle: ...

    async def execute(self, handle: SandboxHandle, request: SandboxExecuteRequest) -> SandboxResult: ...

    async def reset(self, handle: SandboxHandle) -> SandboxHandle: ...

    async def close(self, handle: SandboxHandle) -> None: ...

    async def health_check(self) -> bool: ...
