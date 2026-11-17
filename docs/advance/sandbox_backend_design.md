# 沙箱通用后端接口设计

## 1. 文档目的

本文定义 verl 与外部沙箱系统之间的厂商无关接口。目标是让 Agent、ToolAgentLoop 和训练流程只依赖统一的生命周期与数据模型，而不直接依赖 E2B、Kubernetes、Docker 或其他沙箱 SDK。

当前 P0 已实现异步接口、统一数据模型、异常分层和 E2B 后端。会话复用、全局并发配额、健康检查和沙箱指标进入训练指标系统属于后续工作。

## 2. 代码位置

| 文件 | 作用 |
|---|---|
| `verl/sandbox/base.py` | 定义 `SandboxBackend` 协议 |
| `verl/sandbox/models.py` | 定义请求、句柄和执行结果 |
| `verl/sandbox/errors.py` | 定义统一异常层次 |
| `verl/sandbox/e2b_backend.py` | E2B 的具体实现 |
| `verl/tools/e2b_python_tool.py` | verl Tool 与沙箱后端之间的适配层 |

依赖方向为：

```text
ToolAgentLoop
    -> E2BPythonTool
    -> SandboxBackend
    -> E2BBackend
    -> E2B SDK / E2B 服务
```

通用层不导入 E2B SDK。只有 `e2b_backend.py` 依赖 E2B。

## 3. 接口定义

`SandboxBackend` 是异步 `Protocol`：

```python
class SandboxBackend(Protocol):
    async def create(self, request: SandboxCreateRequest) -> SandboxHandle: ...

    async def execute(
        self,
        handle: SandboxHandle,
        request: SandboxExecuteRequest,
    ) -> SandboxResult: ...

    async def reset(self, handle: SandboxHandle) -> SandboxHandle: ...

    async def close(self, handle: SandboxHandle) -> None: ...

    async def health_check(self) -> bool: ...
```

接口语义：

- `create`：创建一个远程沙箱并返回厂商无关句柄。
- `execute`：在指定沙箱中执行命令；非零退出码属于正常结果，不是基础设施异常。
- `reset`：释放旧实例并按原创建参数重新创建实例。
- `close`：幂等关闭沙箱；无论远端关闭是否成功，都必须释放本地容量。
- `health_check`：后端健康检查扩展点。当前 P0 未接入调用链。

## 4. 统一数据模型

### 4.1 SandboxCreateRequest

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `template` | `str \| None` | `None` | 沙箱模板标识 |
| `sandbox_ttl_s` | `float` | `600` | 服务端强制存活时间；用于孤儿实例回收 |
| `request_timeout_s` | `float` | `60` | 创建请求超时，与 TTL 分离 |
| `metadata` | `Mapping[str, str]` | 空 | 创建时附带的追踪或业务元数据 |

`sandbox_ttl_s` 不能被当作 HTTP 请求超时使用。TTL 是远端资源生命周期，`request_timeout_s` 是控制面请求等待时间。

### 4.2 SandboxHandle

| 字段 | 类型 | 含义 |
|---|---|---|
| `sandbox_id` | `str` | 后端返回的实例 ID |
| `backend` | `str` | 后端名称，例如 `e2b` |
| `queue_ms` | `float` | 等待本地容量许可的时间 |
| `create_ms` | `float` | 远程创建耗时 |

`SandboxHandle` 是不可变对象。日志中不得直接输出完整 `sandbox_id`，应输出哈希后的短标识。

### 4.3 SandboxExecuteRequest

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `command` | `str` | 必填 | 要执行的命令 |
| `timeout_s` | `float` | `30` | 用户命令执行时限 |
| `working_directory` | `str \| None` | `None` | 工作目录 |
| `environment` | `Mapping[str, str]` | 空 | 传入命令的环境变量 |
| `request_id` | `str \| None` | `None` | 工具调用或轨迹追踪 ID |

### 4.4 SandboxResult

| 字段 | 类型 | 含义 |
|---|---|---|
| `stdout` | `str` | 标准输出 |
| `stderr` | `str` | 标准错误 |
| `exit_code` | `int \| None` | 命令退出码；超时或未产生状态时为 `None` |
| `elapsed_ms` | `float` | 执行耗时 |
| `timed_out` | `bool` | 是否命令超时 |
| `truncated` | `bool` | 输出是否被后端截断 |
| `error_type` | `str \| None` | 可选的结构化错误类型 |

代码语法错误、运行时异常和非零退出码应返回 `SandboxResult`。认证失败、网络故障、容量不足等基础设施错误才抛出异常。

## 5. 异常模型

统一异常均继承 `SandboxError`：

| 异常 | 含义 |
|---|---|
| `SandboxConfigurationError` | 缺少配置、模板无效、SDK 不兼容 |
| `SandboxAuthenticationError` | 认证失败 |
| `SandboxCapacityError` | 限流、服务繁忙或容量不足 |
| `SandboxCreateError` | 创建失败 |
| `SandboxExecutionError` | 基础设施层执行失败 |
| `SandboxTimeoutError` | 控制面或 SDK 请求超时 |
| `SandboxNotFoundError` | 句柄不存在或实例已消失 |
| `SandboxCloseError` | 远程关闭失败 |

异常信息不得包含 API Key、Token 或其他凭据。底层异常通过 `raise ... from exc` 保存在 `__cause__` 中，便于排查而不污染面向模型的文本。

`asyncio.CancelledError` 必须原样传播，不能转换为业务错误。

## 6. 生命周期契约

P0 在每次工具调用时执行独立生命周期：

```text
ToolAgentLoop._call_tool
    -> tool.create()
    -> tool.execute()
    -> tool.release()
       -> backend.close()
```

因此当前行为是：

- 每次 Python 工具调用创建一个新沙箱。
- 命令完成后立即关闭。
- 同一轨迹中的多次工具调用不共享文件和进程状态。
- `close` 幂等；重复关闭未知句柄直接返回。
- 远程关闭失败时释放本地容量，依赖服务端 TTL 最终回收。
- 取消期间的远程关闭使用 shielded task，减少资源泄漏。

## 7. 并发语义

`max_concurrency` 表示单个 AgentLoopWorker 进程内同时存活的沙箱数量，不是集群全局限制。

近似全局上限为：

```text
agent.num_workers * max_concurrency
```

例如 8 个 AgentLoopWorker、每个后端 `max_concurrency=1`，理论上最多同时存在约 8 个沙箱。

信号量许可在 `create` 前获取，直到 `close` 完成后释放。因此它控制的是存活实例数，而不是正在执行命令的数量。

当前没有跨 Ray 进程的全局配额。256 Agent 并发场景需要服务端配额，或增加集中式 Ray Actor/容量服务进行全局控制。

## 8. 可观测性

统一句柄和结果为后端提供以下基础指标：

- `queue_ms`
- `create_ms`
- `execute_ms`
- 创建成功/失败数
- 执行成功/失败数
- 非零退出码数
- 超时数
- 关闭成功/失败数
- 当前存活实例数

当前 E2B Tool 会返回 `sandbox/*` 指标，但现有 `ToolAgentLoop` 对工具返回值的第三项尚未接入训练指标聚合，因此 P0 主要通过生命周期日志进行统计。

日志只能记录哈希后的沙箱 ID，不记录密钥、访问 Token 或完整远端异常正文。

## 9. 安全要求

- 模型生成的代码不得直接拼接为宿主 Shell 指令。
- 凭据只能由环境变量或 Secret Manager 注入。
- YAML、训练脚本、日志、轨迹和 checkpoint 中不得保存密钥。
- 命令必须有执行超时，沙箱必须有服务端 TTL。
- 输出必须限制长度，防止超大 stdout/stderr 占用训练内存。
- 基础设施错误不能伪装成正常 observation 混入训练数据。

## 10. 实现状态

| 能力 | 状态 |
|---|---|
| 异步通用接口 | 已实现 |
| 统一请求、句柄、结果 | 已实现 |
| 统一异常层次 | 已实现 |
| create/execute/reset/close | 已实现 |
| 幂等关闭与 TTL 兜底 | 已实现 |
| 单进程并发限制 | 已实现 |
| E2B 具体后端 | 已实现 |
| `health_check` 调用链 | P1 待实现 |
| 同轨迹沙箱复用 | P1 待实现 |
| 跨 Ray 进程全局配额 | P1 待实现 |
| 工具 metrics 接入训练聚合 | P1 待实现 |
| 失败轨迹丢弃和重采样 | P1 待实现 |

## 11. 新后端接入要求

新增 Docker、Kubernetes 或其他远端后端时，应：

1. 实现 `SandboxBackend` 的全部方法。
2. 将厂商 SDK 异常映射为统一异常。
3. 保证非零退出码返回正常 `SandboxResult`。
4. 保证 `close` 幂等并释放本地容量。
5. 提供请求超时与服务端孤儿回收机制。
6. 不让上层 Tool 或 Agent 依赖厂商 SDK 类型。
7. 提供 create/execute/close 的连通性检查和生命周期日志。

