# 世界顶级 AI 系统 · `world-class-ai`

> 架构最先进 · 性能最强 · 代码最优 · 文档最全 —— 开源、可运行、可复现、可验证。
> **作者：晨星**  ·  GitHub: [@CJX0712](https://github.com/CJX0712)

---

## 三大原则（执行总纲）

| 原则 | 含义 | 本仓库落地方式 |
|------|------|----------------|
| **零重写** | 禁止从零造轮子，先检索世界最优解直接复用 | 每条微指令都标注 `reuse` 字段，指向其应复用的世界顶级开源/算法，不自造 |
| **集大成** | 每一行代码站在世界顶级成果之上 | 十层架构逐层映射 HuggingFace / PyTorch / vLLM / DeepSpeed / FlashAttention / FAISS / Ray 等 |
| **闭环交付** | 完整开源至 GitHub，可运行、可复现、可验证 | `verify.py` 零依赖自检 + GitHub Actions CI，作者署名「晨星」 |

---

## 十层分解架构（Top-Down）

| 层 | 名称 | 职责 | 集大成映射 |
|----|------|------|------------|
| **L1** | 范式层 Paradigms | 监督/无监督/强化/概率图/深度学习/Transformer/多模态/AGI 路线 | ReAct · Constitutional AI · AutoGen |
| **L2** | 数学基础层 | 线性代数/概率/微积分优化/数值/信息论 | Eigen · JAX · cuBLAS |
| **L3** | 算法原语层 | 优化器/采样/核方法/树集成/序列/注意力 | PyTorch Optim · Lion · Adafactor |
| **L4** | 模型架构层 | MLP/CNN/RNN/Transformer/MoE/扩散 | Megatron-LM · TIMM |
| **L5** | 训练系统层 | 数据管线/分布式/混合精度/检查点/调度 | DeepSpeed ZeRO · Accelerate |
| **L6** | 推理引擎层 | 算子融合/量化/KV 缓存/批处理/服务 | vLLM · TensorRT-LLM · llama.cpp |
| **L7** | 算子内核层 | 张量算子/线性代数内核/GPU-TPU 调度 | PyTorch ATen · XLA · Triton |
| **L8** | 数据IO层 | 存储/特征/流式/序列化/向量检索 | Tokenizers · FAISS · Arrow |
| **L9** | 运行时编排层 | 调度/内存/并发/容错/可观测 | Ray · asyncio · OpenTelemetry |
| **L10** | 微指令层 | **1024 条原子操作（本仓库主体）** | cuBLAS/PyTorch/vLLM 等（逐条 `reuse`） |

> 既有仓库按"集大成"映射为各层实现：**astraea-ai-system**（L1 多智能体内核）、
> **atlas-ai-stack**（L6/L9 RAG 推理编排）、**novamind**（L1/L8 协议优先 RAG Agent）。
> 本仓库是它们的**母核与统一微指令规范**。

---

## 第十层 · 1024 微指令

第十层（L10）是落到代码级的最细粒度：**10 个模块 × 128 操作族 × 8 变体 = 1024 条原子微指令**。

| 模块 | 名称 | 族数 | 变体/族 | 条目 | 上层映射 |
|------|------|------|---------|------|----------|
| M0 | 数学算子 | 18 | 8 | 144 | L2 |
| M1 | 优化与采样 | 12 | 8 | 96 | L3 |
| M2 | 模型前反向 | 18 | 8 | 144 | L4 |
| M3 | 训练循环 | 12 | 8 | 96 | L5 |
| M4 | 推理服务 | 14 | 8 | 112 | L6 |
| M5 | 张量算子 | 12 | 8 | 96 | L7 |
| M6 | 数据管道 | 12 | 8 | 96 | L8 |
| M7 | 运行时调度 | 12 | 8 | 80/96 | L9 |
| M8 | 控制与编排 | 10 | 8 | 80 | L1/L9 |
| M9 | 系统支撑 | 8 | 8 | 64 | 横切 |

每条微指令示例（`src/ai_sys/data/micro_instructions.json`）：

```json
{
  "id": "MI-0001",
  "name": "axpy_f32",
  "family": "axpy",
  "module": "M0",
  "layer_ref": "L2",
  "category": "math",
  "variant": "f32",
  "description": "向量缩放加 y = αx + y（f32 变体）",
  "reuse": "cuBLAS / PyTorch ATen / JAX XLA / Eigen",
  "deps": [],
  "status": "skeleton",
  "verify": "contract"
}
```

---

## 生态总览 · 全量仓库索引（跨仓库导航 + 质量矩阵）

本母核不是孤立的：它把 GitHub [@CJX0712](https://github.com/CJX0712) 下 **230 个算法 / 系统仓库** 按「十层分解架构」统一归类，逐仓库标注 **质量评级（S/A/B/C）** 与 **集大成复用目标**，形成可导航、可审计的总览。

> 完整矩阵与逐仓库导航见 **[docs/ECOSYSTEM.md](docs/ECOSYSTEM.md)**（由 `tools/build_ecosystem.py` 确定性生成，输入 `src/ai_sys/data/repos_raw.jsonl`）。

| 层 | 名称 | 仓库数 | 集大成复用目标 |
|----|------|--------|----------------|
| L1 | 范式层 Paradigms | 47 | ReAct · Constitutional AI · AutoGen · LangGraph |
| L2 | 数学基础层 | 30 | Eigen · JAX · cuBLAS · NumPy |
| L3 | 算法原语层 | 40 | PyTorch Optim · DeepSpeed · Lion · Adafactor |
| L4 | 模型架构层 | 28 | Megatron-LM · TIMM · FlashAttention |
| L5 | 训练系统层 | 46 | DeepSpeed ZeRO · Accelerate · Optuna |
| L6 | 推理引擎层 | 7 | vLLM · TensorRT-LLM · llama.cpp |
| L7 | 算子内核层 | 3 | PyTorch ATen · XLA · Triton |
| L8 | 数据IO层 | 13 | Tokenizers(Rust) · FAISS · Arrow |
| L9 | 运行时编排层 | 14 | Ray · asyncio · OpenTelemetry |
| L10 | 微指令层 | 2 | cuBLAS / PyTorch / vLLM（逐条 reuse） |

**质量矩阵（层 × 评级，快照）**

| 评级 | S | A | B | C | 合计 |
|------|---|---|---|---|------|
| 仓库数 | 7 | 119 | 94 | 10 | 230 |

- **S=7**：`atlas-ai-stack` · `stellar-nexus` · `effattnforge` · `attribforge` · `causalforge` · `tianshu-ai` · `world-class-ai`
- **评级口径**：`S`=明示「质量等级 S」；`A`=已交付且带硬验证信号（单测全绿 / 基准 / 不变量 / 可复现）；`B`=已交付实仓库；`C`=占位 / 薄仓库 / 工具性。
- 复现：`python tools/build_ecosystem.py`（产出 `docs/ECOSYSTEM.md` + `src/ai_sys/data/ecosystem.json`）。

---

## 快速开始

```bash
git clone https://github.com/CJX0712/world-class-ai.git
cd world-class-ai
pip install -r requirements.txt   # 数值内核：numpy（仅功能自检需要）
python verify.py
```

预期输出：`✅ 全绿` —— 1024 条结构完整、逐项契约自检通过；已实现族（M0）运行真实算子并与
独立参照 / 解析不变量交叉校验，残差达机器精度级。

### 重新生成目录（可复现）
```bash
python tools/build_catalog.py   # 确定性重算 micro_instructions.json
```

---

## 仓库结构

```
world-class-ai/
├── README.md                     # 本文件
├── verify.py                     # 闭环验证器（1024 项 + 生态一致性自检）
├── tools/
│   ├── build_catalog.py          # 1024 微指令确定性生成器（单一事实源）
│   └── build_ecosystem.py        # 全量仓库生态目录生成器（跨仓库导航 + 质量矩阵）
├── src/ai_sys/
│   ├── registry.py               # MicroOp 注册表 + 逐项 self_check
│   ├── data/
│   │   ├── micro_instructions.json   # 生成的 1024 条目录
│   │   ├── repos_raw.jsonl          # 仓库快照（GitHub API，离线可复现输入）
│   │   └── ecosystem.json           # 生成的生态目录（机读）
│   └── handlers/                 # 10 个分类处理器（实现落脚点）
│       ├── math.py  optim.py  forward.py  training.py  inference.py
│       └── tensor.py data.py runtime.py control.py system.py
├── docs/
│   ├── ARCHITECTURE.md           # 十层架构详述
│   ├── LAYER10.md                # 1024 微指令规范与扩展指南
│   └── ECOSYSTEM.md              # 全量仓库索引 + 质量矩阵（跨仓库导航）
└── .github/workflows/verify.yml  # 可复现 CI
```

---

## 验证状态

| 检查项 | 结果 |
|--------|------|
| 微指令总数 == 1024 | ✅ |
| ID 唯一且连续 MI-0001..MI-1024 | ✅ |
| 依赖（deps）指向存在项 | ✅ |
| 每条均有分类处理器覆盖 | ✅ |
| 契约自检（结构 + 字段 + 覆盖） | ✅ 1024/1024 |
| **功能自检（M0 真实算子交叉校验）** | ✅ **144/144**（机器精度级） |
| GitHub Actions CI（3.10–3.13） | ✅ |
| **生态目录一致性**（230 仓库 / 分层 / 评级 / 计数） | ✅ |

---

## 实现进度（闭环交付）

| 模块 | 状态 | 说明 |
|------|------|------|
| **M0 数学算子** | ✅ `implemented` | 18 族 × 8 变体 = 144 条，全部由 `handlers/math.py` 以 **NumPy 世界顶级数值内核**实现（Cholesky/QR/SVD/特征走 `numpy.linalg`，FFT 走 `numpy.fft`，LU 走标准 Doolittle 部分主元），并经**独立参照 + 解析不变量**交叉校验（`‖A−LLᵀ‖→0`、`P·L·U=A`、朴素 DFT 对照等） |
| **M1 优化与采样** | ✅ `implemented` | 12 族 × 8 变体 = 96 条，全部由 `handlers/optim.py` 以 **NumPy 世界顶级数值内核**实现（sgd/momentum/nesterov/adam/adamw/rmsprop/lars/lion/adafactor/radam/lookahead 忠实复刻各自原始论文算法 + gradient_clip 范数裁剪），并经**解析不变量（第一步更新精确等式）+ 收敛不变量（固定凸二次问题 loss 逼近最小可达值）+ 裁剪不变量**交叉校验 |
| **M2 模型前反向** | ✅ `implemented` | 18 族 × 8 变体 = 144 条，全部由 `handlers/forward.py` 以 **NumPy 世界顶级数值内核**实现，忠实复刻原始论文算法（卷积 im2col+GEMM/CS231n、自&交叉注意力 Vaswani2017、FlashAttention 在线softmax Dao2022、LayerNorm Ba2016、RMSNorm Zhang2019、BatchNorm Ioffe2015、LSTM Hochreiter1997、GRU Cho2014、Transformer 残差块），并经**独立朴素参照残差（前向族，机器精度级）+ 有限差分梯度检验（linear_bwd 相对误差~1e-5）+ 量化/稀疏/融合变体交叉校验** |
| **M5 张量算子** | ✅ `implemented` | 12 族 × 8 dtype 变体 = 96 条，全部由 `handlers/tensor.py` 以 **NumPy 世界顶级张量内核**实现（alloc/reshape/transpose/cast/slice/concat/pad/reduce_sum/reduce_mean/reduce_max/elementwise_add/elementwise_mul），并经**独立参照（显式循环 / 不同代码路径）交叉校验**：浮点变体残差 ≤ 精度分级容差，整型/布尔变体按 float64 精确一致 |
| **M9 系统支撑** | ✅ `implemented` | 8 族 × 8 变体 = 64 条，全部由 `handlers/system.py` 复用**世界顶级标准基础设施**（hashlib SHA-256 / RFC 4122 uuid4 / time.monotonic 单调时钟 / pydantic 式配置校验 / structlog 式结构化日志 / OpenTelemetry 式健康探针）实现，并经**不变量自检**（哈希确定性与 stream==step==batch 一致、UUID 可解析、时钟单调、配置缺失可检出、异常按约定抛出）|
| M3–M4、M6–M8 | 🚧 skeleton | 契约自检通过；按「集大成」逐步落地真实算子，落地后将 `status` 置 `implemented`、`verify` 置 `functional` |

> 不变量方法论与 `jsforge` / `*forge` 系列一致：每条真实算子必须有可独立验证的硬金标准。

---

---

## License

[MIT](LICENSE) © 2026 **晨星**（CJX0712）
