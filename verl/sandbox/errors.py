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
"""Sandbox error hierarchy.

Messages must never contain credentials. Raise with ``raise X(...) from exc`` so the SDK error stays in
``__cause__``.
"""


class SandboxError(RuntimeError):
    pass


class SandboxConfigurationError(SandboxError):
    pass


class SandboxAuthenticationError(SandboxError):
    pass


class SandboxCapacityError(SandboxError):
    pass


class SandboxCreateError(SandboxError):
    pass


class SandboxExecutionError(SandboxError):
    pass


class SandboxTimeoutError(SandboxError):
    pass


class SandboxNotFoundError(SandboxError):
    pass


class SandboxCloseError(SandboxError):
    pass
