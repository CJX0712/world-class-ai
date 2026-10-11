# 全量生态目录 · 跨仓库导航 + 质量矩阵

> 世界顶级 AI 系统（`world-class-ai`）母核的统一生态索引。

> 把 GitHub [@CJX0712](https://github.com/CJX0712) 下的全部算法仓库按「十层分解架构」归类，
> 逐仓库标注 **质量评级（S/A/B/C）** 与 **集大成复用目标**，形成可导航、可审计的总览。

> **作者：晨星** · 由 `tools/build_ecosystem.py` 确定性生成（输入 `src/ai_sys/data/repos_raw.jsonl`）。

## 总览

- **收录仓库**：230 个（已排除主页/Gist/探针类非算法仓库）
- **质量评级分布**：**S**=7  **A**=119  **B**=94  **C**=10
- **评级口径**：`S`=描述明示「质量等级 S / S-grade」；`A`=已交付且带硬验证信号
  （单测全绿 / 基准 / 不变量 / 可复现 / 零依赖自检 等）；`B`=已交付实仓库（标注作者晨星或实质算法）；
  `C`=占位/薄仓库（「Delivered by 晨星」无实质内容）/ 工具性仓库 / 无描述。

## 质量矩阵（层 × 评级）

| 层 | 名称 | S | A | B | C | 合计 |
|---|---|---|---|---|---|---|
| L1 | 范式层 Paradigms | 2 | 3 | 37 | 5 | 47 |
| L2 | 数学基础层 Foundations | 0 | 21 | 9 | 0 | 30 |
| L3 | 算法原语层 Primitives | 0 | 31 | 9 | 0 | 40 |
| L4 | 模型架构层 Architectures | 1 | 18 | 9 | 0 | 28 |
| L5 | 训练系统层 Training | 2 | 34 | 9 | 1 | 46 |
| L6 | 推理引擎层 Inference | 0 | 1 | 6 | 0 | 7 |
| L7 | 算子内核层 Kernels | 0 | 3 | 0 | 0 | 3 |
| L8 | 数据IO层 Data & IO | 0 | 7 | 6 | 0 | 13 |
| L9 | 运行时编排层 Runtime | 0 | 1 | 9 | 4 | 14 |
| L10 | 微指令层 Micro-Instructions | 2 | 0 | 0 | 0 | 2 |
| — | **总计** | 7 | 119 | 94 | 10 | 230 |

## 分层索引（跨仓库导航）

### L1 · 范式层 Paradigms

> 集大成复用目标：`ReAct · Constitutional AI · AutoGen · LangGraph`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| S | [atlas-ai-stack](https://github.com/CJX0712/atlas-ai-stack) | ATLAS - a modular, end-to-end runnable AI system: hybrid retrieval, guarded cross-encoder reranking, determin… | Python | 0 | 2026-10-07 |
| S | [stellar-nexus](https://github.com/CJX0712/stellar-nexus) | 可自证的复合智能系统：检索增强 + 自适应链路路由 + 断言级归因校验 + 智能体闭环 + 评测门禁。 | Python | 0 | 2026-10-07 |
| A | [diff-scope](https://github.com/CJX0712/diff-scope) | 差分镜 —— 两段文本差在哪，几乎所有工具都能告诉你；这个工具想说的是它是怎么找到那个答案的。Myers 差分算法的每一轮波前、每条对角线的推进、最后那条最短编辑路径，全部画出来。单文件 HTML，零依赖，完全离线。 | HTML | 0 | 2026-09-24 |
| A | [mojibake-er](https://github.com/CJX0712/mojibake-er) | 乱码急诊室 —— 把一段已经烂掉的乱码，反推回它本来应该是什么字。单文件 HTML，零依赖，完全离线。 | HTML | 0 | 2026-09-24 |
| A | [prompt-terrain](https://github.com/CJX0712/prompt-terrain) | Prompt redundancy tomography - single-file HTML, zero deps, fully offline | HTML | 0 | 2026-09-24 |
| B | [aether-ai-core](https://github.com/CJX0712/aether-ai-core) | 把「文档摄取 → 向量召回 → 生成 → 工具调用」切成 10 个职责单一的模块，每块可单独跑单测，也能拼成完整链路。 | Python | 0 | 2026-10-07 |
| B | [aether-ai-core-v2](https://github.com/CJX0712/aether-ai-core-v2) | 晨星智核 v2：默认栈零重依赖（哈希 TF-IDF + 内存索引 + 抽取式生成），Chroma / FAISS / vLLM·Ollama 以适配器即插即用。 | Python | 0 | 2026-10-07 |
| B | [aether-research](https://github.com/CJX0712/aether-research) | Deep research you can audit — a verifiable research agent with verbatim-quote validation, built on AetherFlow… | TypeScript | 0 | 2026-09-24 |
| B | [ai-platform-agent-rag](https://github.com/CJX0712/ai-platform-agent-rag) | LangGraph 编排 Agent + Chroma 知识库 + FastAPI 网关 + 单文件 WebUI，12 个可独立验证的模块。 | Python | 0 | 2026-10-07 |
| B | [ai-rag-platform](https://github.com/CJX0712/ai-rag-platform) | End-to-end RAG knowledge QA platform: FastAPI + Qdrant + Ollama + React. | Python | 0 | 2026-10-07 |
| B | [aletheia-cognition](https://github.com/CJX0712/aletheia-cognition) | Aletheia 澄明 v1.0 — 推理时计算 Scaling 引擎 (local RAG + inference-time compute, CPU-ready) | Python | 0 | 2026-10-07 |
| B | [astraea-ai-system-qhui1](https://github.com/CJX0712/astraea-ai-system-qhui1) | Astraea - 自演化多智能体 AI 系统内核（晨星） | Python | 0 | 2026-09-24 |
| B | [aurora](https://github.com/CJX0712/aurora) | 本地自主智能体系统 —— 纯本机 CPU 推理（llama.cpp + Qwen2.5-7B + nomic-embed），内置 RAG 检索与 ReAct 工具调用循环，零云端密钥 | Python | 0 | 2026-09-24 |
| B | [aurora-rag](https://github.com/CJX0712/aurora-rag) | 模块化、可插拔、CPU/离线可运行的检索增强生成（RAG）系统 | Python | 0 | 2026-10-04 |
| B | [chenxing-ai-stack](https://github.com/CJX0712/chenxing-ai-stack) | 覆盖「摄入 → 向量化 → 召回 → 重排 → 编排 → 推理 → 服务 → 观测」全链路，每模块单一职责、可独立验证。 | Python | 0 | 2026-10-07 |
| B | [cxrag](https://github.com/CJX0712/cxrag) | CxRAG: 混合检索增强问答引擎（离线/CPU/模块化/可一键复现）by 晨星 | Python | 0 | 2026-10-04 |
| B | [dialectica-ai](https://github.com/CJX0712/dialectica-ai) | Deliberative multi-agent reasoning with evidence anchoring and provable convergence. Local-first, CPU-only, o… | Python | 0 | 2026-09-30 |
| B | [glassbox](https://github.com/CJX0712/glassbox) | Offline, glass-box RAG engine: BM25 + dense + RRF + rerank with a fully visible evidence chain. Pure CPU, no … | Python | 0 | 2026-09-24 |
| B | [helios-ai-stack](https://github.com/CJX0712/helios-ai-stack) | helios-ai-stack: 端到端 RAG 系统。分层架构、Protocol 可注入、三档 profile、混合检索(RRF)+rerank、可量化评测基线。。 | Python | 0 | 2026-09-24 |
| B | [helix-ai-engine](https://github.com/CJX0712/helix-ai-engine) | Provider-agnostic AI orchestration engine: RAG + tool-using agents wired through interface contracts, offline… | Python | 0 | 2026-10-07 |
| B | [hyperion-ai](https://github.com/CJX0712/hyperion-ai) | 双引擎：Python/numpy 训练内核 + TypeScript 零运行期依赖推理内核；13 个模块，含重排护栏与单文件离线控制台。 | TypeScript | 0 | 2026-10-07 |
| B | [lumen-ai](https://github.com/CJX0712/lumen-ai) | RAG + Agent + Tools 应用平台：把 Ollama / FAISS / FastAPI / Gradio 用抽象接口编排起来，每模块带 Mock，后端可插拔。 | Python | 0 | 2026-10-07 |
| B | [morningstar-ai](https://github.com/CJX0712/morningstar-ai) | 晨星 AI · MorningStar 系统原型（模块化 RAG + Agent，Python 3.13） | Python | 0 | 2026-10-07 |
| B | [morningstar-ai-collab](https://github.com/CJX0712/morningstar-ai-collab) | 开源可自托管的 AI 原生协作平台：看板 / 知识库 / 会议纪要 / 代码助手共享同一项目上下文，全栈 TypeScript。 | TypeScript | 0 | 2026-10-07 |
| B | [nebula-ai-core](https://github.com/CJX0712/nebula-ai-core) | 11 个 AI 功能模块，每块 = Protocol 接口 + 默认实现；默认后端零依赖离线可跑，生产实现一键切 FAISS / fastembed / llama.cpp。 | Python | 0 | 2026-10-07 |
| B | [nebula-ai-stack](https://github.com/CJX0712/nebula-ai-stack) | 在无 GPU、15GB 内存的 Windows 上跑通「文档 → 检索 → 重排 → 引用生成 → 工具 Agent → 评测」全链路。 | Python | 0 | 2026-10-07 |
| B | [novaai-stack](https://github.com/CJX0712/novaai-stack) | 零依赖默认的 RAG + ReAct Agent：ingest / embed / vectorstore / lexical / rerank / llm / agent 等十个职责单一的模块。 | Python | 0 | 2026-09-30 |
| B | [novamind](https://github.com/CJX0712/novamind) | Protocol-first RAG agent system, offline-verifiable through zero-dependency mock backends. | Python | 0 | 2026-10-07 |
| B | [phosphor-ai-stack](https://github.com/CJX0712/phosphor-ai-stack) | Phosphor AI Stack — 端到端检索增强智能体平台：摄入→分块→向量化→混合检索→重排→工具推理→多智能体编排→评估与服务，零依赖默认实现、离线一键验证。 | Python | 0 | 2026-09-24 |
| B | [quasar-ai-runtime](https://github.com/CJX0712/quasar-ai-runtime) | 混合检索 + 工具调用智能体 + 证据绑定回答；离线档零模型零网络全绿，在线档接本机 Ollama，依赖全锁定。 | Python | 0 | 2026-10-07 |
| B | [ragnext](https://github.com/CJX0712/ragnext) | RAGNext v0.1.0 — 模块化检索增强生成（RAG）系统。CPU 可跑、一键复现、模块可独立验证；复用 FAISS/sentence-transformers/ragas/pypdf，含零下载离线兜底链路。 | Python | 0 | 2026-10-04 |
| B | [starforge-ai](https://github.com/CJX0712/starforge-ai) | 接口驱动的 AI 工程骨架：模块只依赖 ABC 抽象接口，内置确定性 Mock 后端，零网络零密钥即可跑通端到端。 | Python | 0 | 2026-10-07 |
| B | [starlight-ai-stack](https://github.com/CJX0712/starlight-ai-stack) | 本地优先 RAG：摄取 → 混合检索 → 引用生成，全链路可在无 GPU 的 Windows 机器上跑通。 | Python | 0 | 2026-10-07 |
| B | [stellar-ai-workbench](https://github.com/CJX0712/stellar-ai-workbench) | 桌面 AI 工作台：统一模型网关（故障转移 + Mock）、可插拔 RAG、智能体编排、工具注册与记忆，前端 React。 | Python | 0 | 2026-10-07 |
| B | [tekmor](https://github.com/CJX0712/tekmor) | 本地优先、端到端可验证的 RAG 知识库与证据锚定平台（Python 3.13 + 单文件 HTML 控制台）。aetheros 同源品牌家族。 | Python | 0 | 2026-10-07 |
| B | [worldai](https://github.com/CJX0712/worldai) | 本地优先、CPU 可跑、一键可复现的 RAG + Agent 平台 | Local-first CPU RAG + Agent platform, offline-verifiable | Python | 0 | 2026-10-07 |
| B | [worldai-nexus](https://github.com/CJX0712/worldai-nexus) | 本地优先的 RAG 链路：入库 → 混合检索 → 工具路由 → 生成；默认零依赖 Mock，可切 OpenAI 兼容 / GGUF / FAISS。 | Python | 0 | 2026-10-07 |
| B | [worldai-rag](https://github.com/CJX0712/worldai-rag) | 模块化 RAG + Agent 系统：faiss + BM25 混合检索 / ONNX 重排 / GGUF 本地推理，离线可验证 | Python | 0 | 2026-10-07 |
| B | [worldai-stack](https://github.com/CJX0712/worldai-stack) | RAG 知识库 + ReAct 工具智能体：FastEmbed(ONNX) / FAISS / Rank-BM25 / llama.cpp / FastAPI，模块只依赖 Protocol。 | Python | 0 | 2026-10-07 |
| B | [worldforge-ai](https://github.com/CJX0712/worldforge-ai) | 零依赖核心、可插拔后端的端到端 AI 系统：算法内核 + 混合检索 + Agent + 网关 + 评估门禁 | Python | 0 | 2026-10-07 |
| B | [zhi-rag](https://github.com/CJX0712/zhi-rag) | ZhiDa: Multimodal RAG engine — hybrid retrieval (BM25+dense+RRF), pluggable LLM, offline-default, one-click r… | Python | 0 | 2026-09-29 |
| B | [zhishu](https://github.com/CJX0712/zhishu) | 智枢 ZhiShu：本地优先、混合推理的企业级 AI 知识中枢与智能体平台（RAG + Agent，Web/API/CLI 三形态交付，Docker 一键复现）。。 | Python | 0 | 2026-10-04 |
| C | [chronoforge](https://github.com/CJX0712/chronoforge) | Delivered by 晨星 | Python | 0 | 2026-10-10 |
| C | [distilforge](https://github.com/CJX0712/distilforge) | Delivered by 晨星 | Python | 0 | 2026-10-05 |
| C | [pgmforge](https://github.com/CJX0712/pgmforge) | Delivered by 晨星 | Python | 0 | 2026-10-04 |
| C | [topoforge](https://github.com/CJX0712/topoforge) | Delivered by 晨星 | Python | 0 | 2026-10-04 |
| C | [valuaforge](https://github.com/CJX0712/valuaforge) | Delivered by 晨星 | Python | 0 | 2026-10-05 |

### L2 · 数学基础层 Foundations

> 集大成复用目标：`Eigen · JAX · cuBLAS · NumPy(SciPy/LAPACK)`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| A | [chainforge](https://github.com/CJX0712/chainforge) | 手写贝叶斯后验推断工具箱：NUTS / ADVI / Laplace + 旗舰 VarioNUTS（变分预条件 NUTS + 诊断门控 + 诚实回退）· 作者晨星 | Python | 0 | 2026-10-05 |
| A | [fourier-forge](https://github.com/CJX0712/fourier-forge) | Single-file offline Fourier epicycles path drawing machine | HTML | 0 | 2026-09-24 |
| A | [gaussforge](https://github.com/CJX0712/gaussforge) | GaussForge: modular Gaussian Process system with AutoKernel kernel-structure search (Optuna TPE over RBF/RQ/M… | Python | 0 | 2026-10-05 |
| A | [gp-forge](https://github.com/CJX0712/gp-forge) | 高斯过程回归实验室 · RBF 核 / Cholesky 推断 / 边际似然解析梯度 / 多起点超参优化 · 零依赖单文件 HTML · 8 条不变量自检 | HTML | 0 | 2026-09-24 |
| A | [gpforge](https://github.com/CJX0712/gpforge) | World-class deterministic Gaussian Process toolkit: exact GP regression with hyperparameter optimization and … | Python | 0 | 2026-10-09 |
| A | [gravity-forge](https://github.com/CJX0712/gravity-forge) | Barnes-Hut 四叉树 N 体引力实验室：三体 8 字轨道 / 星系对撞实时演示，Verlet 辛积分，42 项无头自检全绿，单文件 HTML 零依赖离线。 | JavaScript | 0 | 2026-10-04 |
| A | [hmm-forge](https://github.com/CJX0712/hmm-forge) | 隐马尔可夫模型实验室 · 前向-后向 / Viterbi / Baum-Welch · 零依赖单文件 HTML · 8 条不变量自检 | HTML | 0 | 2026-09-28 |
| A | [ica-forge](https://github.com/CJX0712/ica-forge) | 手写独立成分分析 (ICA) / FastICA：中心化→Jacobi特征分解→白化→定点迭代（kurtosis/tanh, deflation/symmetric）。零依赖单文件 HTML，16/16 无头不变量 +… | HTML | 0 | 2026-09-24 |
| A | [jsforge](https://github.com/CJX0712/jsforge) | 纯 JS 单文件世界级 ML/数值算法库 · 零依赖零构建 · 浏览器一键自检 + 无头 Node 8/8 全绿 · 零重写集大成 · 作者 晨星 | JavaScript | 0 | 2026-10-11 |
| A | [kalman-forge](https://github.com/CJX0712/kalman-forge) | Kalman 滤波与平滑实验室 · KF / 信息滤波 / RTS / 批处理块三对角 / EM / 粒子滤波 · 零依赖单文件 HTML · 8 条不变量自检 | HTML | 0 | 2026-09-24 |
| A | [mcmc-forge](https://github.com/CJX0712/mcmc-forge) | MCMC 实验室 · Metropolis-Hastings / HMC leapfrog / Ising Gibbs · 零依赖单文件 HTML · 8 条不变量自检（含 2^16 精确配分函数对拍） | HTML | 0 | 2026-09-24 |
| A | [nmf-forge](https://github.com/CJX0712/nmf-forge) | 手写非负矩阵分解 (NMF, Lee & Seung 1999)：Lee-Seung 乘法更新 (Frobenius + KL 双散度) + ALS 交替非负最小二乘 + 幂迭代截断 SVD 对照。零依赖单文件 HTM… | HTML | 0 | 2026-09-24 |
| A | [odeforge](https://github.com/CJX0712/odeforge) | ODEForge: 连续时间动力学求解 + 可微分积分。Yoshida 辛复合 + 复步长哈密顿检测 + SympFuse 路由。G1 能量守恒 ratio 4.65e-04 (门槛0.10) / 梯度检验 2.4e-… | Python | 0 | 2026-10-03 |
| A | [orbit-forge](https://github.com/CJX0712/orbit-forge) | 引力锻造炉 — 单文件离线 2D n-body 引力模拟，速度 Verlet 积分，动量严格守恒/能量近似守恒/二体圆轨道，内置守恒律自检 | HTML | 0 | 2026-09-24 |
| A | [ot-forge](https://github.com/CJX0712/ot-forge) | 手写熵正则化最优传输 / Sinkhorn 算法 (Wasserstein, Cuturi 2013). 零依赖单文件 HTML，16/16 无头不变量全绿. | HTML | 0 | 2026-09-24 |
| A | [otforge](https://github.com/CJX0712/otforge) | World-class deterministic optimal transport: exact Kantorovich LP, entropic Sinkhorn, Wasserstein barycenters… | Python | 0 | 2026-10-09 |
| A | [pca-forge](https://github.com/CJX0712/pca-forge) | 主成分分析锻造炉：纯 JS 手写中心化 + 协方差 + 雅可比特征分解，幂迭代主成分与解析特征分解交叉验证，旋转等变重建。单文件零依赖。 | HTML | 0 | 2026-09-24 |
| A | [sindyforge](https://github.com/CJX0712/sindyforge) | SindyForge · 稀疏非线性动力学系统辨识 (SINDy / StabSINDy)，纯 numpy 离线可跑 | Python | 0 | 2026-10-04 |
| A | [svm-forge](https://github.com/CJX0712/svm-forge) | SVM 与核方法实验室 · SMO 解析子问题 / KKT 五条件 / 核技巧 == 显式 φ / 2D 硬间隔精确最优性证书 · 零依赖单文件 HTML · 8 条不变量自检 | JavaScript | 0 | 2026-09-24 |
| A | [vi-forge](https://github.com/CJX0712/vi-forge) | 手写变分推断 (Mean-Field CAVI)：贝叶斯高斯混合 + 坐标上升变分推断，ELBO 闭合形式(含 digamma/lgamma)。零依赖单文件 HTML，18/18 无头不变量 + 17/17 UI 全绿… | HTML | 0 | 2026-09-24 |
| A | [wave-forge](https://github.com/CJX0712/wave-forge) | Single-file, zero-dependency 1D time-dependent Schrodinger workbench - hand-written Crank-Nicolson solver, ch… | HTML | 0 | 2026-10-04 |
| B | [convexforge](https://github.com/CJX0712/convexforge) | World-class KKT-certified convex optimization suite: penalty / ADMM / Chambolle-Pock / Mehrotra interior poin… | Python | 0 | 2026-10-10 |
| B | [dictforge](https://github.com/CJX0712/dictforge) | World-class dictionary learning / sparse coding AI (K-SVD, MOD, Online, fused). Author: 晨星. | Python | 0 | 2026-10-09 |
| B | [gmm-forge](https://github.com/CJX0712/gmm-forge) | GMM-EM forge: hand-written Gaussian Mixture Model + Expectation-Maximization with 1D exact parameter recovery… | HTML | 0 | 2026-09-24 |
| B | [hawkesforge](https://github.com/CJX0712/hawkesforge) | HawkesForge — 纯 numpy 多元 Hawkes 自激发过程建模系统 (author 晨星) | Python | 0 | 2026-10-04 |
| B | [rs-forge](https://github.com/CJX0712/rs-forge) | Single-file offline Reed-Solomon FEC forge over GF(256) with self-verifying decode | HTML | 0 | 2026-09-24 |
| B | [sparseforge](https://github.com/CJX0712/sparseforge) | SparseForge · world-class compressed sensing / sparse recovery system (author 晨星) | Python | 0 | 2026-10-05 |
| B | [svmforge](https://github.com/CJX0712/svmforge) | svmforge — world-class kernel-SVM system (author 晨星) | Python | 0 | 2026-10-04 |
| B | [tensorforge](https://github.com/CJX0712/tensorforge) | TensorForge v0.1.0 — 世界顶级张量分解系统 (作者: 晨星) | Python | 0 | 2026-10-05 |
| B | [topicforge](https://github.com/CJX0712/topicforge) | TopicForge · 世界级主题建模系统（LDA/NMF/TopicFuse），纯 NumPy 离线兜底，作者 晨星 | Python | 0 | 2026-10-06 |

### L3 · 算法原语层 Primitives

> 集大成复用目标：`PyTorch Optim · DeepSpeed · Lion(Google) · Adafactor`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| A | [SubmodForge](https://github.com/CJX0712/SubmodForge) | 世界顶级子模优化库 · 数学保证 (1-1/e) · 精确 SFM · 零依赖 · 作者：晨星 | Python | 0 | 2026-10-10 |
| A | [activeforge](https://github.com/CJX0712/activeforge) | ActiveForge: 模块化主动学习系统 — 5 查询策略(不确定性/QBC-BALD/核心集/批感知混合) x 5 数据集, 随机采样强基线对照 Δacc=+0.0396 (5/5 显著), 12 单测全绿, 确… | Python | 0 | 2026-09-27 |
| A | [alignforge](https://github.com/CJX0712/alignforge) | AlignForge: 偏好优化与对齐系统（DPO/IPO/KTO/AlignFuse，纯 numpy 可复现） | Python | 0 | 2026-10-07 |
| A | [ant-forge](https://github.com/CJX0712/ant-forge) | Single-file offline Langton's ant cellular automaton forge | HTML | 0 | 2026-09-24 |
| A | [astar-forge](https://github.com/CJX0712/astar-forge) | Single-file offline A* pathfinding forge with heap invariant and BFS cross-validation | HTML | 0 | 2026-09-24 |
| A | [avl-forge](https://github.com/CJX0712/avl-forge) | Single-file offline AVL tree forge with full invariant self-validation | HTML | 0 | 2026-09-24 |
| A | [az-forge](https://github.com/CJX0712/az-forge) | 从零手写的 AlphaZero 自我对弈 + MCTS — 浏览器内训练，纯自我博弈学会下 Connect-4，零依赖单文件 HTML (forge #16) | HTML | 0 | 2026-09-24 |
| A | [banditforge](https://github.com/CJX0712/banditforge) | BanditForge · 上下文老虎机工具箱：在线去偏校准 BanditFuse（LinUCB/LinTS/UCB1 + 去偏方差校准旗舰）+ 离策略评估 OPE（CF-DR-AC）。纯 NumPy，确定性可复现。作… | Python | 0 | 2026-10-09 |
| A | [bayesforge](https://github.com/CJX0712/bayesforge) | bayesforge: deterministic Bayesian Optimization forge — pure-numpy GP (RBF/Matérn) with analytic NLL gradient… | Python | 0 | 2026-10-02 |
| A | [bitcrc](https://github.com/CJX0712/bitcrc) | 校验和锻造炉 — 单文件离线 CRC-32/CRC-16/Adler-32/FNV-1a 计算，对照公开标准向量 + 残差闭环自检 | HTML | 0 | 2026-09-24 |
| A | [cartpole-forge](https://github.com/CJX0712/cartpole-forge) | CartPole Forge - 单文件零依赖 PPO 强化学习实验室：手写 CartPole 物理+策略/价值网络+GAE+裁剪代理目标+熵正则，贪心策略10x200步全顶格，GAE双路径对拍+梯度检验，Node无头… | HTML | 0 | 2026-09-24 |
| A | [cellular-lab](https://github.com/CJX0712/cellular-lab) | 元胞自动机实验室：2D 生命游戏 + 1D 初等元胞自动机（Wolfram Rule 0–255），零依赖单文件 HTML。 | HTML | 0 | 2026-10-04 |
| A | [ctrlforge](https://github.com/CJX0712/ctrlforge) | CtrlForge: world-class model-based RL & optimal control suite (iLQR/DDP, LQR, CEM, ridge MBRL). Pure numpy, d… | Python | 0 | 2026-10-07 |
| A | [dagforge](https://github.com/CJX0712/dagforge) | DagForge: 确定性因果发现系统（NOTEARS 旗舰 + PC + 离线兜底）。。 | Python | 0 | 2026-10-04 |
| A | [ifs-atlas](https://github.com/CJX0712/ifs-atlas) | IFS Atlas - iterated function system chaos-game fractal explorer (single-file offline) | HTML | 0 | 2026-09-24 |
| A | [koch-forge](https://github.com/CJX0712/koch-forge) | Single-file offline Koch snowflake fractal forge | HTML | 0 | 2026-09-24 |
| A | [life-forge](https://github.com/CJX0712/life-forge) | 生命游戏锻造炉 — 单文件离线康威生命游戏，B3/S23 规则 + 确定性种子 + 经典图案库 + 一键自检 | HTML | 0 | 2026-09-24 |
| A | [maze-forge](https://github.com/CJX0712/maze-forge) | 单文件迷宫生成与寻路可视化工具 (Maze Forge) | HTML | 0 | 2026-09-24 |
| A | [mcts-forge](https://github.com/CJX0712/mcts-forge) | Single-file, zero-dependency MCTS (UCT) workbench - hand-written engine, cross-checked against exact Minimax,… | HTML | 0 | 2026-10-04 |
| A | [nashforge](https://github.com/CJX0712/nashforge) | NashForge：不完美信息博弈精确均衡求解（序列式 LP oracle + CFR/CFR+ + 四条交付门禁）。作者：晨星 | Python | 0 | 2026-10-06 |
| A | [notearsforge](https://github.com/CJX0712/notearsforge) | World-class causal-discovery toolkit (GES/PC/NOTEARS): deterministic, offline, CPU-only. Author 晨星. | Python | 0 | 2026-10-08 |
| A | [offlineforge](https://github.com/CJX0712/offlineforge) | OfflineForge — 离线强化学习（Offline RL + OPE）系统。零依赖 numpy 内核，BC/CQL-lite/FQE 学习器，六种 OPE 估计器对照 DP 真值。 | Python | 0 | 2026-10-04 |
| A | [optiforge](https://github.com/CJX0712/optiforge) | OptiForge: modular combinatorial optimization suite (TSP/knapsack/assignment) - OR-Tools CP-SAT/Routing + sci… | Python | 0 | 2026-10-04 |
| A | [perclass-mia](https://github.com/CJX0712/perclass-mia) | PerClass-MIA: 免影子模型的逐类参考归一化成员推断攻击 — AUC 0.727 vs 强基线 0.680 (+0.047, t=29.0)。纯 NumPy、确定性可复现。作者: 晨星 | Python | 0 | 2026-10-03 |
| A | [rl-forge](https://github.com/CJX0712/rl-forge) | Single-file offline Q-learning forge cross-validated against Value Iteration | HTML | 0 | 2026-09-24 |
| A | [rlforge](https://github.com/CJX0712/rlforge) | RlForge · 世界顶级强化学习系统（纯 NumPy·确定性·可验证）。天枢十层架构谱系，补全 AI 第三范式。Author 晨星。 | Python | 0 | 2026-10-11 |
| A | [rx-forge](https://github.com/CJX0712/rx-forge) | Single-file offline regex railway-diagram forge with Thompson NFA | HTML | 0 | 2026-09-24 |
| A | [seqmineforge](https://github.com/CJX0712/seqmineforge) | Verified deterministic sequential pattern mining — SeqFuse flagship 1.84x faster than the strongest SOTA base… | Python | 0 | 2026-10-05 |
| A | [sudo-forge](https://github.com/CJX0712/sudo-forge) | 种子数独锻造炉 — 单文件离线数独生成器/求解器，确定性种子 + 唯一解保证 | HTML | 0 | 2026-09-24 |
| A | [sudoku-lab](https://github.com/CJX0712/sudoku-lab) | Zero-dependency single-file Sudoku generator and solver with guaranteed unique solutions | HTML | 0 | 2026-10-04 |
| A | [wave-loom](https://github.com/CJX0712/wave-loom) | 坍缩织机 —— Wave Function Collapse 可视化织机。不靠随机堆砌图案，靠约束让图案自己长出来，并把熵、约束传播、冲突、回溯整套过程摊开来看。单文件 HTML，零依赖，完全离线。 | HTML | 0 | 2026-09-24 |
| B | [bitweave](https://github.com/CJX0712/bitweave) | Huffman 无损压缩工坊：频率统计 + 最优前缀码树可视化 + 无损往返 (Bit Weave) | HTML | 0 | 2026-09-24 |
| B | [eternalforge](https://github.com/CJX0712/eternalforge) | EternalForge · eternal-run / non-convergence analysis for iterative optimizers (author 晨星) | Python | 0 | 2026-10-04 |
| B | [featforge](https://github.com/CJX0712/featforge) | FeatForge — AutoML 特征工程系统。复用 scikit-learn/numpy/scipy，旗舰 FeatForgeSynth 相关性引导合成 + FeatForgeSelect mRMR/前向选择，聚… | Python | 0 | 2026-10-04 |
| B | [gridworld-rl-lab](https://github.com/CJX0712/gridworld-rl-lab) | GridWorld RL Lab: value iteration vs Q-learning in one zero-dependency HTML file, headless-verified | HTML | 0 | 2026-09-24 |
| B | [matchforge](https://github.com/CJX0712/matchforge) | MatchForge: world-class graph matching system (QAP/FAQ/spectral/GRAMPA/seeded). Author: 晨星 | Python | 0 | 2026-10-10 |
| B | [miaforge](https://github.com/CJX0712/miaforge) | MiaForge: 成员推理攻击(MIA) 端到端系统 — Shokri 影子模型攻击显著超越 loss 阈值基线。作者: 晨星 | Python | 0 | 2026-10-03 |
| B | [neuroforge](https://github.com/CJX0712/neuroforge) | NeuroForge v0.1.0 — 深度强化学习训练系统（Gymnasium+SB3+Optuna，纯 numpy DQN 兜底）。 | Python | 0 | 2026-10-04 |
| B | [portforge](https://github.com/CJX0712/portforge) | PortForge · 世界级投资组合优化系统（作者：晨星） | Python | 0 | 2026-10-07 |
| B | [tabularofflineforge](https://github.com/CJX0712/tabularofflineforge) | TabularOfflineForge: 离线强化学习（Offline RL + OPE），以表格 MDP 精确真值为金标准的 FQI/CFQI 控制 + FQE/WIS 评估。 | Python | 0 | 2026-10-04 |

### L4 · 模型架构层 Architectures

> 集大成复用目标：`Megatron-LM · TIMM · FlashAttention · Torch nn`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| S | [effattnforge](https://github.com/CJX0712/effattnforge) | S-grade efficient / sparse attention benchmark (6 variants, G1-G4 gates) | Python | 0 | 2026-10-06 |
| A | [attn-forge](https://github.com/CJX0712/attn-forge) | 注意力锻造炉 — 纯 JS 手写 scaled dot-product self-attention + 因果掩码 + 多头 + 正弦位置编码，解析反向传播梯度检验 + 软 argmax 学习演示（零依赖单文件） | HTML | 0 | 2026-09-24 |
| A | [ddpm-forge](https://github.com/CJX0712/ddpm-forge) | DDPM 扩散模型单文件实验室 v2 - 零依赖可无头自检, 31+16 项不变量全绿, loss 地板 E_t[abBar]/MSE 分解/eps 配对调试实录 (single-file self-verifying… | HTML | 0 | 2026-09-24 |
| A | [diffusion-forge](https://github.com/CJX0712/diffusion-forge) | Diffusion Forge - 单文件零依赖 DDPM 扩散模型实验室：手写前向闭式解/后验公式/MLP噪声预测/Adam/DDPM+DDIM采样，后验代数恒等式+梯度检验+四簇生成全覆盖，Node无头18项不变量… | HTML | 0 | 2026-09-24 |
| A | [diffusionforge](https://github.com/CJX0712/diffusionforge) | 世界顶级扩散模型系统：纯 numpy 离线兜底引擎 + 可选 torch/diffusers SOTA 后端，DDPM/DDIM 采样 + 跨调度×采样器自适应选优，31 单测全绿。作者：晨星 | Python | 0 | 2026-10-04 |
| A | [flowforge](https://github.com/CJX0712/flowforge) | World-class normalizing-flow density estimation in pure NumPy: RealNVP + MAF with exact Jacobians, FlowFuse E… | Python | 0 | 2026-10-11 |
| A | [forest-forge-v2](https://github.com/CJX0712/forest-forge-v2) | Random Forest 集成树实验室 · 纯JS零依赖单文件 · Breiman 2001 · CART+Bagging+OOB+特征重要性 | HTML | 0 | 2026-09-29 |
| A | [graphrepforge](https://github.com/CJX0712/graphrepforge) | GraphRepForge: graph representation learning benchmark — node classification / link prediction / 4 embedders … | Python | 0 | 2026-10-04 |
| A | [metricforge](https://github.com/CJX0712/metricforge) | MetricForge · 度量学习/对比表征学习框架（自研纯 numpy 实现，零第三方 DML 依赖，CPU-only 可复现）。作者：晨星 | Python | 0 | 2026-10-04 |
| A | [nasforge](https://github.com/CJX0712/nasforge) | NasForge: world-class Neural Architecture Search (DarFuse DARTS 2nd-order + GRL + one-shot weight inheritance… | Python | 0 | 2026-10-08 |
| A | [nn-forge](https://github.com/CJX0712/nn-forge) | Single-file offline neural network forge: hand-written MLP backprop with live decision boundary | HTML | 0 | 2026-09-24 |
| A | [nodeforge](https://github.com/CJX0712/nodeforge) | NodeForge: reproducible GNN node-classification benchmark with a validation-tuned multi-channel propagation m… | Python | 0 | 2026-09-27 |
| A | [operatorforge](https://github.com/CJX0712/operatorforge) | OperatorForge · 世界级神经算子/科学机器学习系统（FNO / DeepONet / OpFuse），纯 NumPy 离线可跑、确定性可复现。作者：晨星 | Python | 0 | 2026-10-07 |
| A | [seqforge](https://github.com/CJX0712/seqforge) | SeqForge · 序贯贝叶斯推断与状态空间估计实验台（KF/EKF/IEKF/UKF + RTS · bootstrap SIR / Auxiliary PF · FFBS）| 22 条确定性不变量门禁 · APF… | Python | 0 | 2026-10-03 |
| A | [ssmforge](https://github.com/CJX0712/ssmforge) | SSMForge · Structured State Space Models (S4/S4D) toolkit — S4DFuse flagship solves the adding problem to ~1e… | Python | 0 | 2026-10-05 |
| A | [transformer-forge](https://github.com/CJX0712/transformer-forge) | 从零手写的 mini-Transformer/GPT 实验室 — 零依赖、浏览器内训练、Node 无头自检 (forge #14) | HTML | 0 | 2026-09-24 |
| A | [transformer-forge-v2](https://github.com/CJX0712/transformer-forge-v2) | Transformer Forge v2 - 单文件零依赖手写 decoder-only 因果 Transformer（多头自注意力+正弦位置编码+FFN+LN+交叉熵+Adam+梯度裁剪），合成复制/回显任务自回归学… | HTML | 0 | 2026-09-24 |
| A | [tree-forge](https://github.com/CJX0712/tree-forge) | TreeForge 决策树锻造炉 — 纯 JS 手写决策树（熵/Gini）+ Bootstrap 随机森林，内置 8 项不变量自检：根分裂增益与暴力枚举交叉验证 1e-9。单文件 HTML，零依赖。 | HTML | 0 | 2026-09-24 |
| A | [vaeforge](https://github.com/CJX0712/vaeforge) | 世界级变分自编码器(VAE/IWAE)系统 —— 纯 numpy、离线、确定性可复现、CPU-only，7 条可证数学不变式门禁。作者：晨星 | Python | 0 | 2026-10-09 |
| B | [diffuforge](https://github.com/CJX0712/diffuforge) | NFE-budgeted diffusion / score-based generative modeling with the DiffuFuse flagship sampler | Python | 0 | 2026-10-08 |
| B | [graphforge-linkpred](https://github.com/CJX0712/graphforge-linkpred) | GraphForge: faithful node2vec link prediction (graph representation learning). Pure numpy/sklearn/networkx/op… | Python | 0 | 2026-09-28 |
| B | [heteroforge](https://github.com/CJX0712/heteroforge) | HeteroForge: homophily-aware adaptive routing (HAAR) for graph representation learning - dual-channel + budge… | Python | 0 | 2026-10-05 |
| B | [kgforge](https://github.com/CJX0712/kgforge) | KGForge: world-class knowledge-graph embedding & link prediction — KGFuse reciprocal-rank-fusion flagship. Fi… | Python | 0 | 2026-10-10 |
| B | [mmforge](https://github.com/CJX0712/mmforge) | MMForge · 跨模态对比学习系统（手写双编码器对比学习 + CCA 融合旗舰），作者晨星 | Python | 0 | 2026-10-05 |
| B | [modalforge](https://github.com/CJX0712/modalforge) | ModalForge · 多模态跨模态检索系统：CCA/PLS + 对比学习 + MMFuse 融合。纯 numpy 离线兜底，sklearn/CLIP 可选后端。作者：晨星 | Python | 0 | 2026-10-04 |
| B | [mooforge](https://github.com/CJX0712/mooforge) | MOOForge: multi-objective EMO toolkit with the novel AHVA-MOEA adaptive flagship. Author: 晨星. | Python | 0 | 2026-10-05 |
| B | [mvforge](https://github.com/CJX0712/mvforge) | MVForge · 多模态/多视图表征学习与融合系统 (纯 numpy 离线兜底 + 可选 mvlearn/torch SOTA). 作者: 晨星 | Python | 0 | 2026-10-04 |
| B | [tinyml-lab](https://github.com/CJX0712/tinyml-lab) | TinyML Lab: from-scratch MLP with momentum, live decision-boundary visualization, zero dependencies | HTML | 0 | 2026-09-24 |

### L5 · 训练系统层 Training

> 集大成复用目标：`DeepSpeed ZeRO · Accelerate · Megatron · Optuna`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| S | [attribforge](https://github.com/CJX0712/attribforge) | AttribForge — 可解释性 XAI 特征归因系统（纯 numpy 离线 + FGE 门控融合，质量等级 S） | Python | 0 | 2026-10-02 |
| S | [causalforge](https://github.com/CJX0712/causalforge) | CausalForge v0.1.0 — 世界顶级因果推断系统。交叉拟合 DML (LightGBM/XGBoost/sklearn) 估计 ATE，含 OLS/PSM/朴素差分基线、可选 EconML 森林后端、Mo… | Python | 0 | 2026-10-05 |
| A | [FeynmanForge](https://github.com/CJX0712/FeynmanForge) | FeynmanForge · 世界级符号回归系统（Age-Fitness Pareto 遗传编程）· 纯 numpy 离线可跑 · 确定性可复现 · 作者 晨星 | Python | 0 | 2026-10-05 |
| A | [anoforge](https://github.com/CJX0712/anoforge) | AnoForge: world-class unsupervised anomaly detection with the max-consensus AnoFuse ensemble. ROC-AUC 0.806 (… | Python | 0 | 2026-10-10 |
| A | [anomalyforge](https://github.com/CJX0712/anomalyforge) | AnomalyForge: 模块化异常检测系统 (PyOD + scikit-learn + numpy 兜底) | Python | 0 | 2026-10-04 |
| A | [balanceforge](https://github.com/CJX0712/balanceforge) | Geometry-routed rebalancing (MOSAIC) for imbalanced and long-tailed learning. 16 methods x 5 imbalance geomet… | Python | 0 | 2026-10-02 |
| A | [causaldiscforge](https://github.com/CJX0712/causaldiscforge) | ConsensusCD: cross-family stability-guided ensemble for causal structure learning (PC/HillClimbing-BIC/NOTEAR… | Python | 0 | 2026-10-04 |
| A | [chrono-forge](https://github.com/CJX0712/chrono-forge) | ChronoForge: 随机创新的顶级时间序列预测系统。复用 statsmodels/pmdarima/scikit-learn + 残差堆叠集成(Residual-Stacking)。零自研SOTA，离线兜底，可复… | Python | 0 | 2026-10-04 |
| A | [clusterforge](https://github.com/CJX0712/clusterforge) | ClusterForge — 聚类与表征学习系统：共识集成 EAC 旗舰，纯 numpy 离线兜底，确定性可复现基准。 | Python | 0 | 2026-10-04 |
| A | [confforge](https://github.com/CJX0712/confforge) | Confforge - distribution-free predictive inference (conformal prediction) with falsifiable gates. 8 methods, … | Python | 0 | 2026-10-03 |
| A | [conformalforge](https://github.com/CJX0712/conformalforge) | ConformalForge v0.2.0 — 分布无关不确定性量化（分类/回归融合旗舰，作者 晨星） | Python | 0 | 2026-10-07 |
| A | [conformforge](https://github.com/CJX0712/conformforge) | ConformForge — 全栈共形预测基准实验室: 11 方法 × 10 数据集 × 双后端(numpy/MAPIE), CAFuse 旗舰, 覆盖率硬保证 | Author: 晨星 | Python | 0 | 2026-09-29 |
| A | [domainforge](https://github.com/CJX0712/domainforge) | DomainForge: modular domain adaptation system (CORAL/TCA/JDA/KLIEP pure-numpy + SAFuse flagship: HPO + gain-g… | Python | 0 | 2026-10-04 |
| A | [ensembleforge](https://github.com/CJX0712/ensembleforge) | EnsembleForge：自动化集成学习系统（Ensemble Selection + Stacked Generalization）。纯 numpy/scikit-learn 离线可跑、确定性可复现。作者：晨星 (… | Python | 0 | 2026-10-06 |
| A | [equiforge](https://github.com/CJX0712/equiforge) | World-class deterministic algorithmic-fairness toolkit: Reweighing + Equalized-Odds postprocessing. Author 晨星. | Python | 0 | 2026-10-09 |
| A | [explainforge](https://github.com/CJX0712/explainforge) | ExplainForge: model-agnostic feature attribution (XAI) — pure-numpy KernelSHAP/LIME offline fallback, Faithfu… | Python | 0 | 2026-10-04 |
| A | [fairforge](https://github.com/CJX0712/fairforge) | FairForge - reproducible fairness audit & mitigation toolkit with backend fallback (作者: 晨星) | Python | 0 | 2026-10-04 |
| A | [featureforge](https://github.com/CJX0712/featureforge) | FeatureForge — 自动化特征工程×选择器融合(FEFFuse)。零重型依赖、可审计、可复现、诚实。 | Python | 0 | 2026-10-04 |
| A | [gacsforge](https://github.com/CJX0712/gacsforge) | GACSForge - 半监督学习系统: 旗舰 GACS 图-树混合类平衡自训练, aggregate macro-F1 显著优于最强基线 +0.0536 (3 seeds) | Python | 0 | 2026-09-27 |
| A | [graphforge](https://github.com/CJX0712/graphforge) | graphForge: 纯 NumPy 世界级谱图学习与社区发现系统 — 谱聚类 + 旗舰 CommunityFuse(多起点Louvain+模块度增益归并+粗化谱归并+局部精炼,修复分辨率极限) + WL图核 + P… | Python | 0 | 2026-10-10 |
| A | [imitationforge](https://github.com/CJX0712/imitationforge) | ImitationForge · 世界顶级模仿学习基准（BC/DAGGER/ImiteFuse），作者晨星 | Python | 0 | 2026-10-08 |
| A | [intentforge](https://github.com/CJX0712/intentforge) | IntentForge — confidence-calibrated cascaded routing (C3R) for text classification. CPU-only, modular, reprod… | Python | 0 | 2026-10-04 |
| A | [labelforge](https://github.com/CJX0712/labelforge) | LabelForge · 弱监督 / 程序化标注标签模型系统：纯 numpy Dawid-Skene EM 无金标估计 LF 准确率，以 EM 似然单调 / 参数可恢复 / 标签翻转识别为可验证金标准 | Python | 0 | 2026-10-05 |
| A | [metaforge](https://github.com/CJX0712/metaforge) | MetaForge v1.0.0 — 少样本元学习系统 (作者: 晨星). ProtoNet/Reptile/MAML/MatchingNet + 旗舰 MetaFuse(SMF). 纯 numpy 手写 MLP, 零… | Python | 0 | 2026-10-04 |
| A | [neurosymforge](https://github.com/CJX0712/neurosymforge) | NeuroSymForge: 神经符号可微分 ILP / 张量化逻辑网络（LogicFuse 旗舰，系统性组合泛化 benchmark 全 pass） | Python | 0 | 2026-10-06 |
| A | [oodforge](https://github.com/CJX0712/oodforge) | OODForge v0.1.0 · 分布外检测(OOD)与置信度校准系统 · 置信度门控路由CGOR + Optuna多信号融合 + 选择性预测弃权带 | Python | 0 | 2026-10-04 |
| A | [rankforge](https://github.com/CJX0712/rankforge) | RankForge · 世界顶级 Learning to Rank 系统（LightGBM lambdamart + XGBoost rank:ndcg + 纯 numpy RankNet 离线旗舰） | Python | 0 | 2026-09-27 |
| A | [recforge](https://github.com/CJX0712/recforge) | RecForge: 模块化隐式反馈推荐系统基准框架（） | Python | 0 | 2026-10-04 |
| A | [robustforge](https://github.com/CJX0712/robustforge) | RobustForge: 对抗鲁棒性 / 对抗机器学习系统 (作者: 晨星) — 纯 numpy 离线兜底 MLP + FGSM/PGD/CW 攻击 + 对抗训练 + RobustFuse-AT 旗舰(非劣守护). M… | Python | 0 | 2026-10-04 |
| A | [searchforge](https://github.com/CJX0712/searchforge) | SearchForge · 世界级神经/混合检索与学习重排系统（BM25+FAISS/LSI 双路召回 → RRF → LightGBM LambdaMART 重排，旗舰 RankFuse nDCG@10 相对最强基线… | Python | 0 | 2026-10-06 |
| A | [semiforge](https://github.com/CJX0712/semiforge) | SemiForge - 半监督学习系统: 旗舰 GACS 图-树混合类平衡自训练, aggregate macro-F1 显著优于最强基线 +0.0536 (3 seeds) | Python | 0 | 2026-10-10 |
| A | [spectral-forge](https://github.com/CJX0712/spectral-forge) | 谱聚类实验室 - Spectral Clustering Lab: Ng-Jordan-Weiss spectral clustering + Lanczos tridiagonalization, zero-depe… | HTML | 0 | 2026-09-29 |
| A | [survforge](https://github.com/CJX0712/survforge) | SurvForge: modular survival analysis system (Cox/spline-Cox/XGBoost AFT+Cox + C-index-aware rank fusion flags… | Python | 0 | 2026-09-27 |
| A | [transforge](https://github.com/CJX0712/transforge) | TransForge: a verified, deterministic optimal-transport (distribution-alignment) system. TF-epsKnee / TF-SPR … | Python | 0 | 2026-10-04 |
| A | [tscforge](https://github.com/CJX0712/tscforge) | TscForge - deterministic time series classification: TinyRocket + K-medoid DTW prototype fusion (PRISM). Pure… | Python | 0 | 2026-10-04 |
| A | [tsforge](https://github.com/CJX0712/tsforge) | tsforge: 时间序列预测系统（）— 复用 statsmodels/sklearn/pmdarima，M4 指标 + 滚动原点基准，零下载可跑 demo | Python | 0 | 2026-09-29 |
| B | [changeforge](https://github.com/CJX0712/changeforge) | ChangeForge: 变化点检测（Changepoint Detection）工程框架 — 纯 numpy 离线兜底 + ruptures 后端 + ConStab-CPD 旗舰共识集成。 | Python | 0 | 2026-10-04 |
| B | [fedforge](https://github.com/CJX0712/fedforge) | FedForge: modular federated learning system - FedAvg/FedMedian/FedTrimmedMean, Dirichlet non-IID partitioning… | Python | 0 | 2026-10-04 |
| B | [kernelforge](https://github.com/CJX0712/kernelforge) | KernelForge · world-class random-feature kernel approximation system (KernelFuse flagship). Author: 晨星 | Python | 0 | 2026-10-08 |
| B | [kmeans-forge](https://github.com/CJX0712/kmeans-forge) | K-Means clustering forge: hand-written k-means++ and Lloyd with 1D exact DP oracle | HTML | 0 | 2026-09-24 |
| B | [privforge](https://github.com/CJX0712/privforge) | PrivForge · 世界顶级差分隐私机器学习系统 (DP-SGD + RDP 矩会计 + 自适应裁剪 DPFuse)，作者：晨星 | Python | 0 | 2026-10-09 |
| B | [recoforge](https://github.com/CJX0712/recoforge) | RecoForge: 混合协同过滤推荐系统 (手写WRMF-ALS + implicit SOTA后端) | Python | 0 | 2026-10-04 |
| B | [spectralforge](https://github.com/CJX0712/spectralforge) | SpectralForge v0.1.0 谱聚类系统（作者：晨星） | Python | 0 | 2026-10-05 |
| B | [tabulaforge](https://github.com/CJX0712/tabulaforge) | TabulaForge: 模块化自动化表格机器学习 (XGBoost + LightGBM + Optuna + scikit-learn 兜底) | Python | 0 | 2026-10-04 |
| B | [vecforge](https://github.com/CJX0712/vecforge) | VecForge: 文档向量检索与聚类系统 (FAISS + scikit-learn 兜底) | Python | 0 | 2026-10-04 |
| C | [densforge](https://github.com/CJX0712/densforge) | DensForge — density estimation + manifold learning. C-grade blocked record (density median -1.05% FAIL, manif… | Python | 0 | 2026-10-04 |

### L6 · 推理引擎层 Inference

> 集大成复用目标：`vLLM · TensorRT-LLM · llama.cpp · SGLang`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| A | [watermarkforge](https://github.com/CJX0712/watermarkforge) | watermarkforge: channel-diverse & distortion-free LLM text watermarking (equal-distortion calibrated, multi-s… | Python | 0 | 2026-10-10 |
| B | [ai-llm-api](https://github.com/CJX0712/ai-llm-api) | 本地 LLM 推理服务：FastAPI + llama-cpp-python，CPU 推理，镜像自动推送到 ghcr.io。 | Python | 0 | 2026-10-07 |
| B | [novamind-rag](https://github.com/CJX0712/novamind-rag) | FAISS + BM25 双路召回，ONNX cross-encoder 按 0.3 加权融合；ReAct Agent 带 AST 安全计算器，/health 暴露后端回退原因。 | Python | 0 | 2026-10-07 |
| B | [ptqforge](https://github.com/CJX0712/ptqforge) | PTQForge: post-training quantization (per-channel asym + KL clip + mixed precision + adaptive rounding). Auth… | Python | 0 | 2026-10-02 |
| B | [quantforge](https://github.com/CJX0712/quantforge) | Post-training quantization research toolkit: RTN, AWQ and GPTQ-based weight compression in pure NumPy | Python | 0 | 2026-10-02 |
| B | [starlight-rag](https://github.com/CJX0712/starlight-rag) | BM25 + 向量双路召回 RRF 融合；查询覆盖率重排权重封顶 50%，防止弱重排器压掉正确答案。免 GPU、无 torch。 | Python | 0 | 2026-10-07 |
| B | [worldrag](https://github.com/CJX0712/worldrag) | CJK 单字+双字 BM25（零依赖 tokenizer）+ 向量双路召回；独立索引评测 recall@k / MRR，配 API + CLI + React 控制台。 | Python | 0 | 2026-10-07 |

### L7 · 算子内核层 Kernels

> 集大成复用目标：`PyTorch ATen · XLA · Triton · cuBLAS`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| A | [ray-forge](https://github.com/CJX0712/ray-forge) | 光线锻造炉 — 单文件离线光线追踪，多球+地面+漫反射/高光/阴影，相机环绕旋转，内置相交几何与光照自检 | HTML | 0 | 2026-09-24 |
| A | [ray-forge-v2](https://github.com/CJX0712/ray-forge-v2) | 纯 JS CPU 光线追踪 · Whitted 风格 · 反射/折射/Fresnel/多重采样 AA · 零依赖单文件。 | JavaScript | 0 | 2026-10-04 |
| A | [sort-forge](https://github.com/CJX0712/sort-forge) | 排序锻造炉 — 单文件离线排序算法动画，6 种算法逐帧可视化 + 比较/写入次数实测 + 确定性种子 + 一键自检 | HTML | 0 | 2026-09-24 |

### L8 · 数据IO层 Data & IO

> 集大成复用目标：`HuggingFace Tokenizers(Rust) · FAISS · Arrow · Pandas`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| A | [bpe-forge](https://github.com/CJX0712/bpe-forge) | 字符级 BPE 分词器实验室 · 从零训练/可视化/数学不变量验证 Byte Pair Encoding，零依赖单文件 HTML · Char-level BPE tokenizer lab | HTML | 0 | 2026-09-29 |
| A | [denoise-forge](https://github.com/CJX0712/denoise-forge) | 从零手写的图像 DDPM 实验室 — U-Net + FiLM 时间条件 + DDIM，零依赖单文件 HTML，Node 无头自检 (forge #15) | HTML | 0 | 2026-09-24 |
| A | [forgestack](https://github.com/CJX0712/forgestack) | forgestack: 本地可运行 AI 系统 — 复用 forest/spectral/bpe 三引擎 + ml-distance 向量检索 + 自研标准度量，一键复现，41 单测全绿。 | JavaScript | 0 | 2026-09-29 |
| A | [schema-dowser](https://github.com/CJX0712/schema-dowser) | 表格探矿仪 - 从 CSV 里探出隐藏的键、函数依赖、脏数据和本该拆出去的表，单文件 HTML 零依赖离线运行 | HTML | 0 | 2026-09-24 |
| A | [synth-forge](https://github.com/CJX0712/synth-forge) | 合成锻造炉 — 单文件离线合成器，4 种波形+ADSR 包络+确定性旋律种子，WebAudio 播放+波形可视化，内置采样数学自检 | HTML | 0 | 2026-09-24 |
| A | [voiceforge](https://github.com/CJX0712/voiceforge) | VoiceForge: 语音/音频机器学习系统 — 复用 librosa/whisper/speechbrain/optuna 顶级开源，纯 numpy 离线兜底，21 单测全绿。 | Python | 0 | 2026-10-04 |
| A | [word2vec-forge](https://github.com/CJX0712/word2vec-forge) | Word2Vec Forge - 单文件零依赖词向量实验室：手写 skip-gram+负采样，合成语料学出 king-man+woman=queen 类比，PCA投影/热力图/loss曲线，Node无头26项不变量全绿 | HTML | 0 | 2026-09-24 |
| B | [SketchForge](https://github.com/CJX0712/SketchForge) | Fixed-budget data-stream sketching with data-driven budget allocation | Python | 0 | 2026-10-05 |
| B | [cvforge](https://github.com/CJX0712/cvforge) | cvforge: 基于特征的图像分类系统（）— 复用 scikit-image/OpenCV/scikit-learn，accuracy/macro-F1+NMI/ARI 评测，可选后端离线降级零下载可跑 demo | Python | 0 | 2026-09-29 |
| B | [imputeforge](https://github.com/CJX0712/imputeforge) | ImputeForge: tabular missing-value imputation (MCAR/MAR/MNAR), flagship ImputeFuse | Python | 0 | 2026-10-11 |
| B | [onlineforge](https://github.com/CJX0712/onlineforge) | OnlineForge · 流式/增量学习系统（river + scikit-learn + 纯 numpy 离线兜底） | Python | 0 | 2026-10-04 |
| B | [riverforge](https://github.com/CJX0712/riverforge) | RiverForge — 在线学习（流式 / 概念漂移）系统。复用 scikit-learn/river/numpy，旗舰 DriftForge 漂移感知在线集成。 | Python | 0 | 2026-10-04 |
| B | [visionforge](https://github.com/CJX0712/visionforge) | VisionForge: CPU-only content-based image retrieval (CBIR) with AMDF adaptive multi-descriptor fusion. 零下载可跑的… | Python | 0 | 2026-10-04 |

### L9 · 运行时编排层 Runtime

> 集大成复用目标：`Ray · asyncio · tcmalloc/mimalloc · OpenTelemetry`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| A | [synthmind-forge](https://github.com/CJX0712/synthmind-forge) | SynthMind-Forge · RCG-NAS — 递归批判引导的神经网络架构搜索引擎。随机抽签整合 LangGraph/Haystack/ONNX Runtime/Diffusers，零依赖闭环可跑。Author… | Python | 0 | 2026-09-29 |
| B | [aetherflow](https://github.com/CJX0712/aetherflow) | A universal, durable and observable runtime for AI agents. TypeScript · Node 22 · zero vendor lock-in · MCP-n… | TypeScript | 0 | 2026-10-07 |
| B | [aetheros](https://github.com/CJX0712/aetheros) | Local-first agent runtime that refuses unsourced claims by default and turns every failure into the next skil… | HTML | 0 | 2026-09-24 |
| B | [evolver](https://github.com/CJX0712/evolver) | Self-evolving layer for AI agents — skill/pitfall distillation, control-group benchmark, portable skill packs | Python | 0 | 2026-09-24 |
| B | [loom](https://github.com/CJX0712/loom) | 本地优先、MCP 原生的智能体运行时 —— 把 Ollama、MCP 和 crawl4ai 组装成一个真正会干活的本地智能体 | Python | 0 | 2026-09-24 |
| B | [nexus-ai-core](https://github.com/CJX0712/nexus-ai-core) | AI 能力中台：模型网关 / 知识检索 / 智能体 / 工作流 / 记忆；无 Key、无网络、无 GPU 时自动降级到离线抽取式应答。 | Python | 0 | 2026-10-07 |
| B | [nexus-ai-platform](https://github.com/CJX0712/nexus-ai-platform) | NexusAI 平台 · 模块化 AI 能力中台：模型网关 / 知识检索 / 智能体（Python 3.13） | Python | 0 | 2026-10-07 |
| B | [stellar-ai-core](https://github.com/CJX0712/stellar-ai-core) | 以「单一职责 + 清晰接口 + 能力注册表」组织 AI 能力，模块间只经抽象契约通信，可独立验证。 | Python | 0 | 2026-10-07 |
| B | [stellar-ai-stack](https://github.com/CJX0712/stellar-ai-stack) | 默认零依赖、离线可跑、确定性可验证：mock LLM + BLAKE2b 哈希嵌入 + 内存余弦索引 + IDF 词法重排。 | Python | 0 | 2026-10-07 |
| B | [veritas-ai](https://github.com/CJX0712/veritas-ai) | 不变量优先、可自进化的 Agent 运行时 —— 把「正确性」从测试环节提升到架构层。 | Python | 0 | 2026-10-07 |
| C | [ai-dev-env-report](https://github.com/CJX0712/ai-dev-env-report) | Windows 本机 AI 开发环境安装清单 —— 单文件 HTML 报告，表格化呈现每一步的环境状态。 | HTML | 0 | 2026-09-24 |
| C | [ai-library-dashboard](https://github.com/CJX0712/ai-library-dashboard) | AI 资料库总览看板 —— 单文件 HTML，零依赖零构建，双击即用。 | HTML | 0 | 2026-09-24 |
| C | [repo-hygiene-gate](https://github.com/CJX0712/repo-hygiene-gate) | 最小仓库卫生门禁 · CI 冒烟验证模板 | — | 0 | 2026-09-24 |
| C | [starforge](https://github.com/CJX0712/starforge) | starforge ★ · GitHub 高星仓库榜单浏览器 — 247 真实仓库 / 12.25M 星 / 10 赛道筛选 / 收藏导出 · 零依赖单文件 HTML | HTML | 0 | 2026-09-29 |

### L10 · 微指令层 Micro-Instructions

> 集大成复用目标：`cuBLAS / PyTorch / vLLM 等（逐条 reuse 字段）`

| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |
|------|------|--------|------|---|----------|
| S | [tianshu-ai](https://github.com/CJX0712/tianshu-ai) | 天枢 TianShu — 世界顶级 AI 系统：十层架构 · 元路由 · 融合旗舰 · 可证门禁。零重写集大成：32 算法家族 + Caruana 集成选择 + 显著性门控。作者：晨星 | Python | 0 | 2026-10-11 |
| S | [world-class-ai](https://github.com/CJX0712/world-class-ai) | 世界顶级 AI 系统 · 十层分解架构 · 第十层 1024 微指令（零重写·集大成·闭环交付·可验证）。作者：晨星 | Python | 0 | 2026-10-11 |

## 与第十层（1024 微指令）的关系

- 本目录是 **L1–L9 的真实仓库落地**；`world-class-ai` 自身的 `src/ai_sys/data/micro_instructions.json`
  是 **L10 微指令规范**（1024 条原子操作，由 `verify.py` 闭环自检）。
- 每层仓库的算法能力，对应 L10 同名操作族：例如 L2 的 `gp-forge`/`bayesforge` 对应 M0 数学算子，
  L3 的 `ctrlforge`/`banditforge` 对应 M1 优化与采样，依此类推。
- 复现：`python tools/build_ecosystem.py` 重新生成本文档与 `ecosystem.json`。

---

*本目录由脚本确定性生成，分类关键词透明可审计；人工策展覆盖见 `tools/build_ecosystem.py` 的 `OVERRIDES`。*
