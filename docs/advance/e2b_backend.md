# E2B 沙箱后端实现与接入

## 1. 文档目的

本文说明 `E2BBackend` 如何实现通用 `SandboxBackend`，以及它在 verl ToolAgentLoop 中的配置、运行、验证和运维方式。

当前 P0 面向 Python 工具调用：模型生成 Python 代码，verl 为每次工具调用创建远程 E2B 沙箱，执行代码、返回观察结果并销毁实例。

## 2. 实现组成

| 文件 | 关键对象 | 作用 |
|---|---|---|
| `verl/sandbox/e2b_backend.py` | `E2BBackend` | E2B SDK 封装和通用接口实现 |
| `verl/sandbox/e2b_backend.py` | `E2BConnectionSettings` | 从环境变量加载连接参数 |
| `verl/tools/e2b_python_tool.py` | `E2BPythonTool` | verl BaseTool 适配器 |
| `verl/tools/e2b_python_tool.py` | `E2BPythonToolConfig` | 工具配置校验 |
| `verl/tools/e2b_preflight.py` | preflight CLI | 训练前连通性和模板检查 |
| `examples/e2b_agent/tool_config.yaml` | 工具配置 | 注册 `python` 工具及后端参数 |
| `examples/e2b_agent/setup_and_preflight.sh` | 部署辅助脚本 | 安装 SDK、设置环境并执行 preflight |

`e2b_backend.py` 是 verl 中唯一直接导入 E2B SDK 的模块。

## 3. 依赖版本

当前固定依赖范围：

```text
e2b>=2.50.0,<2.51
```

安装方式：

```bash
pip install -e ".[e2b]"
```

P0 使用 E2B 的普通命令执行接口，不依赖 `e2b-code-interpreter`。

SDK 版本不在上述范围内时，工具初始化直接抛出 `SandboxConfigurationError`，避免在训练期间出现协议不兼容。

## 4. 连接配置

| 环境变量 | 必需 | 含义 |
|---|---|---|
| `E2B_API_KEY` | 是 | API Key，只能从环境或 Secret 注入 |
| `E2B_DOMAIN` | 二选一 | E2B 服务域名 |
| `E2B_API_URL` | 二选一 | 自建 E2B 控制面地址 |
| `E2B_PROXY` | 部署相关 | 九天训练容器访问 E2B 时使用的代理；由运行环境或 SDK transport 配置处理 |
| `VERL_LOGGING_LEVEL` | 否 | 设为 `INFO` 时输出生命周期日志 |

如果 `E2B_DOMAIN` 和 `E2B_API_URL` 都未设置，初始化直接失败，不会静默访问公共 `e2b.app`。

`E2B_ACCESS_TOKEN` 不是当前后端的输入配置。沙箱访问 Token 由 create 响应返回并由 SDK 管理。

密钥禁止写入：

- `tool_config.yaml`
- Git 仓库
- 训练启动脚本
- Ray 日志
- rollout 和 validation JSONL

## 5. 工具配置

示例：

```yaml
tools:
  - class_name: verl.tools.e2b_python_tool.E2BPythonTool
    config:
      type: native
      backend: e2b
      template: base
      max_concurrency: 1
      sandbox_ttl_s: 600
      create_request_timeout_s: 60
      command_timeout_s: 30
      close_request_timeout_s: 30
      max_code_chars: 20000
      max_output_chars: 16000
    tool_schema:
      type: function
      function:
        name: python
        description: Execute Python code in an isolated sandbox and return stdout, stderr and exit status.
        parameters:
          type: object
          properties:
            code:
              type: string
          required: [code]
```

配置语义：

| 参数 | 含义 |
|---|---|
| `template` | E2B 模板；必须包含 bash、python3、timeout、base64、mktemp |
| `max_concurrency` | 单个 AgentLoopWorker 中允许存活的沙箱数 |
| `sandbox_ttl_s` | 服务端实例存活上限和孤儿回收兜底 |
| `create_request_timeout_s` | 创建控制面请求超时 |
| `command_timeout_s` | 模型代码执行时限 |
| `close_request_timeout_s` | kill 请求超时 |
| `max_code_chars` | 模型代码最大字符数，不能超过 24000 |
| `max_output_chars` | stdout 和 stderr 各自的后端截断上限 |

要求：

```text
sandbox_ttl_s >=
create_request_timeout_s + command_timeout_s + close_request_timeout_s
```

否则配置校验失败，避免沙箱在一次工具调用中途被服务端回收。

## 6. 创建流程

`E2BBackend.create()` 执行：

1. 等待进程内 `BoundedSemaphore` 容量许可。
2. 调用 `AsyncSandbox.create()`。
3. 将 `sandbox_ttl_s` 传给 E2B `timeout` 参数。
4. 将 `request_timeout_s` 传给控制面请求超时。
5. 保存 SDK sandbox 对象和创建参数到进程内 session 表。
6. 返回厂商无关的 `SandboxHandle`。

创建日志示例：

```text
sandbox created backend=e2b sandbox_id_hash=... queue_ms=... create_ms=...
```

日志只输出 `sandbox_id` 的 SHA-256 短哈希。

## 7. Python 代码执行

### 7.1 安全封装

模型代码不会直接作为 Shell 片段插值。`E2BPythonTool` 执行以下转换：

1. UTF-8 代码编码为 Base64。
2. 沙箱内使用 `mktemp` 创建临时文件。
3. 使用 `base64 -d` 写入代码。
4. 使用 `python3` 执行临时文件。

这避免模型代码在写文件阶段被宿主 Shell 解释。

### 7.2 超时控制

实际命令由 coreutils `timeout` 包装：

```text
timeout -k 5s <command_timeout_s>s bash -c '<command>'
```

退出码 124 或超时强杀产生的 137，在达到执行时限时转换为：

```python
SandboxResult(exit_code=None, timed_out=True)
```

SDK 调用外层还有 `asyncio.wait_for` 兜底。SDK 层超时视为基础设施错误，用户代码超时视为正常工具 observation。

### 7.3 非零退出码

E2B SDK 对非零退出码可能抛出 `CommandExitException`。后端捕获该异常并提取：

- `stdout`
- `stderr`
- `exit_code`

然后返回正常 `SandboxResult`。因此 Python 语法错误和运行时异常可以作为反馈返回模型，而不会被误判为 E2B 服务故障。

### 7.4 输出格式

模型可见的 observation 示例：

```text
[stdout]
2499
[exit_code] 0
```

或：

```text
[stderr]
Traceback ...
[exit_code] 1
```

状态行放在末尾，以便 ToolAgentLoop 左侧截断时仍保留退出状态。

## 8. 输出截断

存在两层截断：

1. E2B Backend 对 stdout 和 stderr 分别保留最后 `max_output_chars` 个字符。
2. ToolAgentLoop 对整个 observation 应用 `multi_turn.max_tool_response_length`，默认值可能只有 256。

训练配置必须显式设置：

```text
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048
```

模型最终看到的长度上限是两层配置共同作用的结果。

## 9. 关闭与资源回收

`E2BBackend.close()` 的性质：

- 幂等。
- 创建独立的关闭 task，并使用 `asyncio.shield` 等待。
- 调用 E2B `sandbox.kill()`。
- 即使 kill 失败，也删除本地 session 并释放信号量许可。
- 远端关闭失败时依赖 `sandbox_ttl_s` 最终回收。

关闭日志：

```text
sandbox closed backend=e2b sandbox_id_hash=... close_ms=... success=1
```

当前 P0 每次工具调用都创建和关闭沙箱。同一轨迹中的下一次工具调用会获得新实例，不保留文件或 Python 进程状态。

## 10. 异常转换

| E2B/SDK 情况 | 统一异常 | Tool 层行为 |
|---|---|---|
| 认证失败 | `SandboxAuthenticationError` | `FatalToolError`，终止 rollout |
| 模板或参数错误 | `SandboxConfigurationError` | `FatalToolError`，终止 rollout |
| 429、服务繁忙 | `SandboxCapacityError` | `TransientToolError`，P0 终止 rollout |
| 控制面超时 | `SandboxTimeoutError` | `TransientToolError` |
| 实例不存在 | `SandboxNotFoundError` | `TransientToolError` |
| 创建失败 | `SandboxCreateError` | `TransientToolError` |
| 执行基础设施失败 | `SandboxExecutionError` | `TransientToolError` |
| Python 非零退出 | 不抛异常 | 返回 observation |
| Python 执行超时 | 不抛异常 | 返回超时 observation |

异常消息会移除 API Key，并限制最大长度。

## 11. Fail-fast 与 preflight

工具初始化执行本地检查：

- SDK 是否安装。
- SDK 版本是否满足要求。
- API Key、Domain/API URL 是否存在。
- YAML 参数类型、范围和未知字段。

远程连通性通过 preflight 检查：

```bash
python -m verl.tools.e2b_preflight \
  --tool-config examples/e2b_agent/tool_config.yaml
```

preflight 执行：

```text
create
  -> 执行 Python 探针
  -> 校验 stdout 和 exit_code
  -> close
```

探针同时验证模板中是否存在：

- `bash`
- `timeout`
- `base64`
- `mktemp`
- `python3`

多机训练前应在每个物理节点执行一次 preflight，因为网络、代理和 Python 环境可能不同。

## 12. verl 接入配置

关键训练参数：

```text
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.multi_turn.tool_config_path=<tool_config.yaml>
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
actor_rollout_ref.rollout.agent.num_workers=1
```

初次链路验证建议 `agent.num_workers=1`、`max_concurrency=1`。并发测试再逐步提高 AgentLoopWorker 数量。

需要把 E2B 配置显式传入 Ray runtime env，保证所有 Worker 可见。密钥应通过平台 Secret 注入，不能直接写进命令或配置文件。

## 13. 生命周期日志与指标

设置：

```bash
export VERL_LOGGING_LEVEL=INFO
```

日志事件：

```text
sandbox created backend=e2b ...
sandbox execute tool_call_id=... exit_code=... timed_out=... execute_ms=...
sandbox closed backend=e2b ... success=...
```

Tool 返回的指标：

```text
sandbox/backend
sandbox/queue_ms
sandbox/create_ms
sandbox/execute_ms
sandbox/success
sandbox/timeout
```

当前 ToolAgentLoop 会丢弃工具返回值的第三项，因此这些指标尚未自动汇总到训练 step metrics。现阶段以日志统计 create/execute/close 成功率和耗时；后续需将第三项接入 `agent_data.metrics`。

## 14. 已完成验证

当前验证分三层：

1. preflight：验证 SDK、网络、模板、create/execute/close。
2. Tool Adapter：通过 verl Tool Registry 加载 `E2BPythonTool`，执行 Python 并返回标准 ToolResponse。
3. 模型驱动链路：验证模型生成 tool call，ToolAgentLoop 调用 E2B，并将 tool response 返回模型。

已完成的模型驱动样本结果：

```text
样本数：8
产生工具调用：8/8
收到工具响应：8/8
实际工具执行：9
成功退出：7
非零退出：2
最终回答正确：7/8
```

该结果证明 P0 链路可用，不代表高并发、24 小时稳定性或训练收益已经完成验证。

## 15. 当前限制与后续工作

| 能力 | 状态 |
|---|---|
| E2B create/execute/close | 已实现并验证 |
| Python Tool 适配 | 已实现并验证 |
| 模型驱动工具调用 | 已验证 |
| 服务端 TTL 回收 | 已实现 |
| 单 Worker 并发限制 | 已实现 |
| 同轨迹沙箱复用 | 待开发 |
| 跨 Worker 全局并发控制 | 待开发 |
| 256 Agent 并发压测 | 待验证 |
| 24 小时稳定性 | 待验证 |
| metrics 接入训练聚合 | 待开发 |
| transient failure 重试/重采样 | 待开发 |
| GPU bubble 收益对比 | 待实验 |

## 16. 验收建议

P0 功能验收应至少保留以下证据：

- 两个训练节点的 preflight 输出。
- Tool Adapter 的 stdout、stderr、exit_code 和生命周期指标。
- validation JSONL 中的 tool call 和 tool response。
- create/execute/close 日志及成功率汇总。
- 无未关闭实例或由 TTL 正常回收的记录。

性能阶段再逐步进行 1、4、8、32、64、256 并发测试，并分别记录沙箱吞吐、创建延迟、执行延迟、错误率、轨迹吞吐和端到端训练吞吐。

