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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SandboxHandle:
    sandbox_id: str
    backend: str
    queue_ms: float = field(default=0.0, compare=False)
    create_ms: float = field(default=0.0, compare=False)


@dataclass(frozen=True)
class SandboxCreateRequest:
    template: Optional[str] = None
    # Sandbox lifetime enforced by the server; the orphan safety net when local close never happens.
    sandbox_ttl_s: float = 600.0
    # Control-plane request timeout. Must never be passed as the TTL.
    request_timeout_s: float = 60.0
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxExecuteRequest:
    command: str
    timeout_s: float = 30.0
    working_directory: Optional[str] = None
    environment: Mapping[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    # None when the command timed out or never produced an exit status.
    exit_code: Optional[int]
    elapsed_ms: float
    timed_out: bool = False
    truncated: bool = False
    error_type: Optional[str] = None
