# Claude Code Router 配置指南

## 简介

Claude Code Router (CCR) 是一个代理工具，可以让 Claude Code 使用第三方模型提供商，而不仅限于 Anthropic 官方 API。

## 安装

```bash
# 安装 Claude Code
npm install -g @anthropic-ai/claude-code

# 安装 Claude Code Router
npm install -g @musistudio/claude-code-router
```

## 配置文件位置

配置文件路径：`~/.claude-code-router/config.json`

## 配置文件结构

### 基本配置

```json
{
  "APIKEY": "your-secret-key",
  "PROXY_URL": "http://127.0.0.1:7890",
  "LOG": true,
  "LOG_LEVEL": "info",
  "API_TIMEOUT_MS": 600000,
  "Providers": [...],
  "Router": {...}
}
```

### 配置项说明

| 配置项 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `APIKEY` | string | CCR 服务的认证密钥（可选） | 无 |
| `PROXY_URL` | string | HTTP 代理地址（可选） | 无 |
| `LOG` | boolean | 是否启用日志 | true |
| `LOG_LEVEL` | string | 日志级别：fatal/error/warn/info/debug/trace | debug |
| `API_TIMEOUT_MS` | number | API 请求超时时间（毫秒） | 600000 |
| `NON_INTERACTIVE_MODE` | boolean | 非交互模式（用于 CI/CD） | false |

## Providers 配置

### 基本结构

```json
{
  "name": "provider-name",
  "api_base_url": "https://api.example.com/v1/chat/completions",
  "api_key": "your-api-key",
  "models": ["model-1", "model-2"],
  "transformer": {
    "use": ["transformer-name"]
  }
}
```

### 配置项说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 提供商唯一标识 |
| `api_base_url` | 是 | API 端点地址 |
| `api_key` | 是 | API 密钥 |
| `models` | 是 | 支持的模型列表 |
| `transformer` | 否 | 请求/响应转换器配置 |

## Transformer 说明

Transformer 用于在不同 API 格式之间转换请求和响应。

### 内置 Transformer

| Transformer | 用途 | 端点格式 |
|-------------|------|----------|
| `Anthropic` | 保留原始 Anthropic 格式 | `/v1/messages` |
| `OpenAI` | OpenAI 格式 | `/v1/chat/completions` |
| `deepseek` | DeepSeek API 适配 | - |
| `gemini` | Google Gemini API 适配 | - |
| `openrouter` | OpenRouter API 适配 | - |
| `tooluse` | 增强工具调用支持 | - |
| `enhancetool` | 增强工具定义 | - |
| `reasoning` | 推理模型支持 | - |
| `maxtoken` | 设置最大 token 数 | - |

### Transformer 使用示例

#### 1. 单个 Transformer

```json
"transformer": {
  "use": ["openrouter"]
}
```

#### 2. 多个 Transformer（按顺序执行）

```json
"transformer": {
  "use": [
    ["maxtoken", {"max_tokens": 65536}],
    "enhancetool"
  ]
}
```

#### 3. 针对特定模型的 Transformer

```json
"transformer": {
  "use": ["deepseek"],
  "deepseek-chat": {
    "use": ["tooluse"]
  },
  "deepseek-reasoner": {
    "use": ["reasoning"]
  }
}
```

## Router 配置

Router 定义不同场景下使用的模型。

### 配置格式

```json
"Router": {
  "default": "provider-name,model-name",
  "background": "provider-name,model-name",
  "think": "provider-name,model-name",
  "longContext": "provider-name,model-name",
  "longContextThreshold": 60000,
  "webSearch": "provider-name,model-name"
}
```

### 路由类型说明

| 路由类型 | 说明 | 使用场景 |
|---------|------|----------|
| `default` | 默认模型 | 所有未匹配其他规则的请求 |
| `background` | 后台任务模型 | 快速、简单的后台任务 |
| `think` | 思考模型 | 需要深度推理的任务 |
| `longContext` | 长上下文模型 | 超过阈值的长对话 |
| `longContextThreshold` | 长上下文阈值 | Token 数量阈值 |
| `webSearch` | 网络搜索模型 | 需要联网搜索的任务 |

## 完整配置示例

### 示例 1：连接 Toolify

```json
{
  "APIKEY": "ronghuaxueleng",
  "LOG": true,
  "LOG_LEVEL": "info",
  "API_TIMEOUT_MS": 600000,
  "Providers": [
    {
      "name": "toolify",
      "api_base_url": "http://localhost:8000/v1/chat/completions",
      "api_key": "ronghuaxueleng",
      "models": [
        "qwen3-coder-plus",
        "qwen3-max-2025-10-30",
        "qwq-32b-thinking"
      ],
      "transformer": {
        "use": ["openrouter"]
      }
    }
  ],
  "Router": {
    "default": "toolify,qwen3-coder-plus",
    "think": "toolify,qwq-32b-thinking"
  }
}
```

### 示例 2：多提供商配置

```json
{
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "sk-xxx",
      "models": ["anthropic/claude-sonnet-4"],
      "transformer": {
        "use": ["openrouter"]
      }
    },
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "sk-xxx",
      "models": ["deepseek-chat", "deepseek-reasoner"],
      "transformer": {
        "use": ["deepseek"],
        "deepseek-chat": {
          "use": ["tooluse"]
        }
      }
    }
  ],
  "Router": {
    "default": "deepseek,deepseek-chat",
    "think": "deepseek,deepseek-reasoner"
  }
}
```

## 使用方法

### 启动 CCR

```bash
# 启动 CCR 服务
ccr start

# 或直接运行 Claude Code（会自动启动 CCR）
ccr code
```

### 激活环境变量

```bash
# 激活 CCR 环境变量
eval "$(ccr activate)"

# 之后可以直接使用 claude 命令
claude
```

### 动态切换模型

在 Claude Code 中使用 `/model` 命令：

```
/model toolify,qwen3-coder-plus
/model toolify,qwq-32b-thinking
```

### 管理模型

```bash
# 列出所有可用模型
ccr model list

# 查看当前模型
ccr model current

# 切换模型
ccr model set provider-name,model-name
```

## 日志文件

CCR 使用两个日志系统：

1. **服务器日志**：`~/.claude-code-router/logs/ccr-*.log`
   - HTTP 请求、API 调用、服务器事件

2. **应用日志**：`~/.claude-code-router/claude-code-router.log`
   - 路由决策、业务逻辑事件

## 环境变量插值

配置文件支持环境变量引用：

```json
{
  "OPENAI_API_KEY": "$OPENAI_API_KEY",
  "Providers": [
    {
      "api_key": "${OPENAI_API_KEY}"
    }
  ]
}
```

## 常见问题

### 1. 工具调用不工作

**问题**：Claude Code 不调用工具，只输出文本。

**原因**：
- CCR 的 transformer 配置不正确
- 目标 API 不支持 OpenAI 格式的工具调用
- Claude Code 检测到非官方 API 而禁用工具

**解决方案**：
- 使用正确的 transformer（如 `openrouter`）
- 确保目标 API 支持 `tools` 参数
- 考虑直接连接支持工具的 API

### 2. 401 认证错误

**问题**：请求返回 401 Unauthorized。

**解决方案**：
- 检查 `api_key` 是否正确
- 确保目标服务的 `allowed_keys` 包含该密钥
- 重启 CCR 和目标服务

### 3. 配置修改不生效

**问题**：修改 config.json 后没有效果。

**解决方案**：
- CCR 启动时加载配置，需要重启：`ccr restart` 或重新运行 `ccr code`

## 参考资源

- [Claude Code Router GitHub](https://github.com/musistudio/claude-code-router)
- [Claude Code 官方文档](https://docs.anthropic.com/en/docs/claude-code/quickstart)
