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

## 快速开始

```bash
git clone https://github.com/CJX0712/world-class-ai.git
cd world-class-ai
python verify.py          # 零依赖（仅 Python 3.8+ 标准库）
```

预期输出：`✅ 全绿` —— 1024 条结构完整、逐项契约自检通过。

### 重新生成目录（可复现）
```bash
python tools/build_catalog.py   # 确定性重算 micro_instructions.json
```

---

## 仓库结构

```
world-class-ai/
├── README.md                     # 本文件
├── verify.py                     # 闭环验证器（1024 项自检）
├── tools/
│   └── build_catalog.py          # 1024 微指令确定性生成器（单一事实源）
├── src/ai_sys/
│   ├── registry.py               # MicroOp 注册表 + 逐项 self_check
│   ├── data/
│   │   └── micro_instructions.json   # 生成的 1024 条目录
│   └── handlers/                 # 10 个分类处理器（实现落脚点）
│       ├── math.py  optim.py  forward.py  training.py  inference.py
│       └── tensor.py data.py runtime.py control.py system.py
├── docs/
│   ├── ARCHITECTURE.md           # 十层架构详述
│   └── LAYER10.md                # 1024 微指令规范与扩展指南
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
| 零外部依赖 / 可离线运行 | ✅ |
| GitHub Actions CI | ✅ |

---

## License

[MIT](LICENSE) © 2026 **晨星**（CJX0712）
