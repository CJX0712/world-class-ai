# 十层分解架构（详述）

> 设计立场：**站在世界顶级成果之上**。每一层都不是从零设计，而是对现代 AI 系统
> 解剖学的提炼与复用编排。本文件给出各层的职责边界、关键组件与世界顶级实现映射。

---

## L1 · 范式层 Paradigms
**职责**：定义系统可承载的学习/推理范式。
- 监督 / 无监督 / 自监督 / 强化学习 / 概率图模型
- Transformer 统一 backbone、多模态融合、迈向 AGI 的路线选择
- **集大成**：ReAct（推理-行动循环）、Constitutional AI（宪法式对齐）、AutoGen（多智能体协作）
- 落地仓库：**astraea-ai-system-qhui1**（自演化多智能体内核）

## L2 · 数学基础层 Mathematical Foundations
**职责**：一切计算的数学底座。
- 线性代数（SVD / 特征值 / Cholesky）、概率与统计（贝叶斯 / KL）、微积分优化（链式法则 / 凸性）、数值方法（logsumexp / 稳定性）、信息论（熵 / 互信息）
- **集大成**：Eigen、JAX（自动微分 + XLA）、cuBLAS
- 对应本仓库 **M0 数学算子**

## L3 · 算法原语层 Algorithmic Primitives
**职责**：可复用的优化与采样算法原语。
- SGD / Adam(W) / LARS / Lion / Adafactor / RAdam、梯度裁剪、学习率调度
- **集大成**：PyTorch `torch.optim`、DeepSpeed、Lion(Google)、Adafactor
- 对应本仓库 **M1 优化与采样**

## L4 · 模型架构层 Model Architectures
**职责**：可组合的模型积木。
- MLP / CNN / RNN / Transformer / MoE / 扩散模型、注意力变体
- **集大成**：Megatron-LM（大规模Transformer）、TIMM（视觉模型库）
- 对应本仓库 **M2 模型前反向**

## L5 · 训练系统层 Training System
**职责**：把模型跑起来并跑得快。
- 数据管线、分布式训练（DDP/ZeRO）、混合精度、检查点、调度
- **集大成**：DeepSpeed ZeRO、Accelerate、Megatron-LM
- 对应本仓库 **M3 训练循环**

## L6 · 推理引擎层 Inference Engine
**职责**：把模型服务出去并服务得省。
- 算子融合、量化（W8A8 / W4）、PagedAttention、连续批处理、投机解码、采样
- **集大成**：vLLM、TensorRT-LLM、llama.cpp、SGLang
- 对应本仓库 **M4 推理服务**

## L7 · 算子内核层 Compute Kernels
**职责**：张量级原子计算与硬件调度。
- 张量视图/转换/归约/逐元素、BLAS 内核、GPU/TPU 调度
- **集大成**：PyTorch ATen、XLA、Triton
- 对应本仓库 **M5 张量算子**

## L8 · 数据IO层 Data & IO
**职责**：数据的流入、编码与检索。
- 分词（BPE/WordPiece）、序列化（JSON/Parquet）、特征编码、向量索引与近邻检索、流式
- **集大成**：HuggingFace Tokenizers(Rust)、FAISS、Apache Arrow / Pandas
- 对应本仓库 **M6 数据管道**

## L9 · 运行时编排层 Runtime & Orchestration
**职责**：系统的"操作系统"——调度、内存、并发、容错、可观测。
- 任务调度、线程池、异步、内存管理、Barrier、流水线、故障重启、日志/指标/健康
- **集大成**：Ray、asyncio、tcmalloc/mimalloc、OpenTelemetry
- 对应本仓库 **M7 运行时调度 / M9 系统支撑 / M8 控制与编排**

## L10 · 微指令层 Micro-Instructions（本仓库主体）
**职责**：落到代码级的最细粒度——1024 条原子操作。
- 10 个模块（M0–M9），128 操作族，每族 8 变体（dtype 或 mode）
- 每条均标注 `reuse`（应复用的世界顶级实现）与 `deps`（依赖的其它微指令）
- 由 `tools/build_catalog.py` 确定性生成，单一事实源 `micro_instructions.json`
- 由 `verify.py` 逐项契约自检，确保可复现、可验证

> 设计纪律：L10 不重新发明任何算子，只**规范化**其接口、依赖与复用目标；真实实现
> 一律"集大成"指向 L2–L9 列出的世界顶级开源。
