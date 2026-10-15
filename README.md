# 环保监测智能运维 AI 助手（ai-helper）

面向「电子证据鉴定仪 + 污染源在线监测」场景的 AI 助手后端：
一边给**安装运维**查设备、排故障，一边给**环保局领导**查数据、出报告，
中间靠**多智能体 + RAG 混合检索 + 流式输出**把链路串起来。

> 设计与取舍的完整说明见 [docs/技术方案.md](docs/技术方案.md)；
> 真实踩坑与修复记录见 [docs/代码审查记录.md](docs/代码审查记录.md)。

## 技术栈

Python 3.11 · FastAPI · LangChain · LangGraph · 百炼（DashScope）·
RAG（向量 + BM25 + RRF + rerank）· SQLite/MySQL · SSE + WebSocket

## 目录结构

```
app/
  config.py          # .env 配置（模型名、地址、分块参数、时区）
  llm.py             # 对话模型工厂（OpenAI 兼容模式 + 关闭思考模式）
  embeddings.py      # 向量化（qwen3.7-text-embedding，1024 维）
  vectorstore.py     # 向量库：本地 numpy 实现；文件末尾附 Milvus 生产写法
  hybrid.py          # 中文分词 + BM25L + RRF 融合
  rerank.py          # qwen3.7-text-rerank 精排
  rag.py             # 解析 → 分块 → 入库 → 混合检索
  database.py        # 业务数据层（演示用 SQLite，接口对齐 MySQL）
  timeutil.py        # 统一时区口径（踩过坑，务必用它取时间）
  console.py         # Windows 控制台 UTF-8 修复
  tools/             # 5 个 LangChain 工具 + 角色权限白名单
  agents/            # LangGraph 多智能体：主管 + 知识/运维/报告三个员工
scripts/
  check_api.py       # 接口自检（不需要装依赖就能跑）
  ingest.py          # 文档批量入库
  query.py           # 检索单测
  seed.py            # 造演示数据（幂等，可重复执行）
  agent_chat.py      # 命令行跑完整多智能体对话
evaluation/
  eval_set.json      # 59 条评测问题（问题 → 期望来源文档）
  evaluate.py        # recall@k
data/
  docs/              # 知识库语料（法规/标准/手册/SOP/案例库）
  skills/            # 技能手册（按需加载，不占系统提示词）
```

## 快速开始

```bash
# 1. 依赖（推荐 uv；没有 uv 用 venv + pip 也行）
uv sync

# 2. 配置：复制 .env.example 为 .env，填入百炼 API Key（sk- 开头）
cp .env.example .env

# 3. 自检：确认 Key、模型名、向量维度都对
python scripts/check_api.py

# 4. 造演示数据 + 导入知识库
python scripts/seed.py
python scripts/ingest.py

# 5. 检索单测 / 评测
python scripts/query.py "二氧化硫排放限值是多少" 5
python evaluation/evaluate.py

# 6. 跑一次完整的多智能体对话
python scripts/agent_chat.py "CEMS-003 这台设备离线了，帮我看看什么原因" --role ops
python scripts/agent_chat.py "分析 3 号排口最近 24 小时二氧化硫并出报告" --role leader
```

## 两个角色能看到什么

| 角色 | 可用工具 | 典型问题 |
|---|---|---|
| `ops`（安装运维） | 全部（含设备状态、技能手册） | 「CEMS-003 离线了什么原因」「波特率怎么配」 |
| `leader`（环保局领导） | 只读：查数据、查标准、出报告 | 「3 号排口最近 24 小时达标吗，出一份报告」 |
| `service`（客服） | 仅知识检索 | 「企业要装监测设备有什么要求」 |

权限在**提示词层和工具层同时隔离**（`app/tools/__init__.py` 的 `ROLE_TOOLS`），
白名单里没写的一律不给——默认拒绝。

## 注意事项

- `.env` 已在 `.gitignore` 中，**不要把 Key 提交到仓库**；
- 数据文件（`data/app.db`、`data/vectors.json`、`data/reports/`）同样不入库；
- 换 embedding 模型必须重新全量入库，并同步修改 `EMBED_DIM`；
- 云端账户欠费时，接口会返回 `Arrearage`，代码会给出中文提示（详见审查记录）。
