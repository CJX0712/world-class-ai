#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ecosystem.py — 世界顶级 AI 系统 · 全量生态目录生成器（跨仓库导航 + 质量矩阵）
====================================================================================

设计原则（与项目总纲一致）：
  · 零重写   —— 不重建任何算法仓库，仅把既有 233 个仓库按「十层分解架构」归类索引。
  · 集大成   —— 每层标注其应复用的世界顶级实现（取自 docs/ARCHITECTURE.md）。
  · 闭环交付 —— 确定性生成 docs/ECOSYSTEM.md（人读）与 src/ai_sys/data/ecosystem.json
               （机读），可复现、可验证（verify.py 消费 ecosystem.json）。

分类方法：
  · 单一事实源：src/ai_sys/data/repos_raw.jsonl（由 `gh api` 抓取的仓库快照，离线可复现）。
  · 主分类器：关键词打分（LAYER_KEYWORDS），按 LAYER_PRIORITY 顺序取首个命中层。
  · 人工策展层：OVERRIDES（仓库名 -> 层/评级/备注），优先于关键词，可审可扩。
  · 质量评级：从描述抽取 S/A/B/C（详见 grade_of）。

作者：晨星
"""
import json
import os
from collections import defaultdict, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "src", "ai_sys", "data")
RAW = os.path.join(DATA, "repos_raw.jsonl")
OUT_JSON = os.path.join(DATA, "ecosystem.json")
OUT_MD = os.path.join(ROOT, "docs", "ECOSYSTEM.md")

# --------------------------------------------------------------------------- #
# 十层架构（与 docs/ARCHITECTURE.md 严格一致）
# --------------------------------------------------------------------------- #
LAYERS = OrderedDict([
    ("L1",  "范式层 Paradigms"),
    ("L2",  "数学基础层 Foundations"),
    ("L3",  "算法原语层 Primitives"),
    ("L4",  "模型架构层 Architectures"),
    ("L5",  "训练系统层 Training"),
    ("L6",  "推理引擎层 Inference"),
    ("L7",  "算子内核层 Kernels"),
    ("L8",  "数据IO层 Data & IO"),
    ("L9",  "运行时编排层 Runtime"),
    ("L10", "微指令层 Micro-Instructions"),
])

LAYER_REUSE = {
    "L1":  "ReAct · Constitutional AI · AutoGen · LangGraph",
    "L2":  "Eigen · JAX · cuBLAS · NumPy(SciPy/LAPACK)",
    "L3":  "PyTorch Optim · DeepSpeed · Lion(Google) · Adafactor",
    "L4":  "Megatron-LM · TIMM · FlashAttention · Torch nn",
    "L5":  "DeepSpeed ZeRO · Accelerate · Megatron · Optuna",
    "L6":  "vLLM · TensorRT-LLM · llama.cpp · SGLang",
    "L7":  "PyTorch ATen · XLA · Triton · cuBLAS",
    "L8":  "HuggingFace Tokenizers(Rust) · FAISS · Arrow · Pandas",
    "L9":  "Ray · asyncio · tcmalloc/mimalloc · OpenTelemetry",
    "L10": "cuBLAS / PyTorch / vLLM 等（逐条 reuse 字段）",
}

# --------------------------------------------------------------------------- #
# 关键词 -> 层（按优先级解析，首个命中层胜出）
# --------------------------------------------------------------------------- #
LAYER_KEYWORDS = {
    "L7": ["ray trace", "光线追", "光线", "render", "渲染", "sort", "排序", "ray-forge"],
    "L8": ["bpe", "word2vec", "tokeniz", "分词", "vision", "图像", "image", "cbir",
           "音频", "audio", "语音", "voice", "speech", "向量索引", "serialize", "序列化",
           "json", "csv", "parquet", "schema", "草图", "sketch", "缺失", "impute",
           "缺失值", "data stream", "数据流", "流式"],
    "L4": ["transformer", "注意力", "attention", "attn", "扩散", "diffusion", "ddpm",
           "变分自编码", "vae", "归一化流", "normalizing flow", "flowforge", "gnn",
           "graph neural", "node2vec", "graph representation", "图卷积", "图嵌入",
           "knowledge graph", "知识图谱", "kgforge", "cnn", "卷积", "rnn", "lstm", "gru",
           "moe", "专家混合", "度量学习", "metric learn", "对比学习", "contrastive",
           "多模态", "multimodal", "状态空间", "state space", "ssm", "神经算子",
           "neural operator", "科学机器学习", "scientific ml", "embedding", "mlp",
           "backprop", "决策树", "decision tree", "随机森林", "random forest",
           "tree-forge", "nn-forge"],
    "L2": ["高斯过程", "gaussian process", "gp 回归", "gp回归", "bayesian inference",
           "贝叶斯推断", "posterior", "变分推断", "variational inference", "最优传输",
           "optimal transport", "wasserstein", "sinkhorn", "凸优化", "convex",
           "kernel method", "核方法", "矩阵分解", "matrix factorization", "nmf", "独立成分",
           "fastica", "pca", "主成分", "kalman", "卡尔曼", "hmm", "隐马尔可夫", "hawkes",
           "微分方程", "schrodinger", "薛定谔", "傅里叶", "fourier", "引力", "gravity",
           "n-body", "n体", "压缩感知", "compressed sensing", "稀疏恢复", "sparse recovery",
           "张量分解", "tensor decomposition", "支持向量", "support vector", "svm",
           "数值", "numerical", "滤波", "filtering", "状态估计", "mcmc", "gaussian mixture",
           "gmm", "crank-nicolson", "辛积分", "symplectic"],
    "L3": ["优化", "optimiz", "贝叶斯优化", "bayesian optim", "bandit", "老虎机", "强化",
           "reinforcement", "ppo", "mcts", "控制", "control", "最优控制",
           "optimal control", "nash", "cfr", "组合优化", "combinatorial", "submodular",
           "子模", "采样", "sampl", "蒙特卡洛", "monte carlo", "因果发现", "causal discovery",
           "notevers", "ges", "qap", "图匹配", "graph matching", "序列式", "q-learning",
           "q learning", "value iteration", "a*", "astar", "pathfinding", "寻路", "maze",
           "迷宫", "sudoku", "数独", "regex", "正则", "huffman", "crc", "checksum",
           "fractal", "分形", "koch", "生命游戏", "cellular", "avl", "ant-forge", "langton",
           "wfc", "wave function collapse"],
    "L5": ["异常", "anomaly", "聚类", "cluster", "变化点", "changepoint", "共形", "conformal",
           "因果推断", "causal inference", "因果", "causal", "不均衡", "imbalance", "长尾",
           "long-tailed", "域适应", "domain adaptation", "半监督", "semi-supervised",
           "主动学习", "active learning", "对齐", "align", "dpo", "ipo", "kto", "偏好",
           "preference", "元学习", "meta-learn", "少样本", "few-shot", "联邦", "federated",
           "集成", "ensemble", "蒸馏", "distill", "生存", "survival", "主题", "topic",
           "推荐", "recommend", "表格", "tabular", "特征", "feature", "automat", "automl",
           "公平", "fair", "差分隐私", "differential privacy", "模仿", "imitation",
           "神经符号", "neuro-symbol", "密度估计", "density estimation", "流形", "manifold",
           "符号回归", "symbolic regression", "博弈", "game"],
    "L6": ["推理服务", "推理引擎", "llm api", "serving", "ptq", "量化", "int8", "w4", "w8a8",
           "quantiz"],
    "L9": ["runtime", "运行时", "编排", "orchestrat", "调度", "scheduler", "内存", "memory",
           "mcp", "可观测", "observability", "工作流", "workflow", "事件循环", "event loop",
           "异步", "async", "演化", "evolv", "veritas", "loom", "中台", "registry",
           "能力注册", "工程骨架", "agent runtime", "智能体运行时", "skill", "技能蒸馏"],
}

LAYER_PRIORITY = ["L10", "L7", "L8", "L4", "L2", "L3", "L5", "L6", "L9", "L1"]

# 人工策展层（优先于关键词，可审可扩）。备注可解释分类理由。
OVERRIDES = {
    "world-class-ai": ("L10", "S", "本仓库：十层分解架构母核 + 1024 微指令规范"),
    "tianshu-ai":     ("L10", "S", "天枢：十层架构 · 元路由 · 融合旗舰（架构级实现）"),
    "astraea-ai-system-qhui1": ("L1", "B", "自演化多智能体内核"),
    "atlas-ai-stack": ("L1", "S", "端到端可运行 AI 系统（混合检索+重排+ReAct+归因）"),
    "aether-ai-core": ("L1", "B", "文档摄取→向量召回→生成→工具调用 10 模块"),
    "aether-ai-core-v2": ("L1", "B", "零重依赖智核（哈希 TF-IDF + 内存索引）"),
    "aether-research": ("L1", "B", "可审计 deep-research agent"),
    "aetherflow": ("L9", "B", "通用可持久可观测 agent 运行时"),
    "aetheros": ("L9", "B", "本地优先 agent 运行时（拒无源声明）"),
    "aurora": ("L1", "B", "本地自主智能体系统（CPU 推理 + RAG + ReAct）"),
    "aurora-rag": ("L1", "B", "模块化可插拔 RAG"),
    "chenxing-ai-stack": ("L1", "B", "全链路 RAG 栈"),
    "dialectica-ai": ("L1", "B", "审慎多智能体推理（证据锚定）"),
    "evolver": ("L9", "B", "自演化层：技能/陷阱蒸馏"),
    "helio-ai-stack": ("L1", "B", "端到端 RAG 系统"),
    "helios-ai-stack": ("L1", "B", "端到端 RAG 系统"),
    "helix-ai-engine": ("L1", "B", "Provider-agnostic 编排引擎"),
    "hyperion-ai": ("L1", "B", "双引擎（numpy 训练 + TS 推理）"),
    "loom": ("L9", "B", "本地优先 MCP 原生 agent 运行时"),
    "lumen-ai": ("L1", "B", "RAG+Agent+Tools 平台"),
    "morningstar-ai": ("L1", "B", "MorningStar RAG+Agent 原型"),
    "morningstar-ai-collab": ("L1", "B", "开源可自托管 AI 协作平台"),
    "nebula-ai-core": ("L1", "B", "11 模块 AI 能力中台"),
    "nebula-ai-stack": ("L1", "B", "无 GPU Windows 全链路"),
    "nexus-ai-core": ("L9", "B", "AI 能力中台（网关/检索/智能体）"),
    "nexus-ai-platform": ("L9", "B", "模块化 AI 能力中台"),
    "novaai-stack": ("L1", "B", "零依赖 RAG + ReAct Agent"),
    "novamind": ("L1", "B", "协议优先 RAG agent"),
    "novamind-rag": ("L6", "B", "FAISS+BM25 双路召回 + ReAct"),
    "phosphor-ai-stack": ("L1", "B", "端到端检索增强智能体平台"),
    "quasar-ai-runtime": ("L1", "B", "混合检索 + 工具调用智能体"),
    "ragnext": ("L1", "B", "模块化 RAG 系统"),
    "starlight-ai-stack": ("L1", "B", "本地优先 RAG"),
    "starlight-rag": ("L6", "B", "BM25+向量双路召回"),
    "stellar-ai-core": ("L9", "B", "单一职责 + 能力注册表"),
    "stellar-ai-stack": ("L9", "B", "零依赖离线可验证栈"),
    "stellar-ai-workbench": ("L1", "B", "桌面 AI 工作台（React）"),
    "stellar-nexus": ("L1", "S", "可自证复合智能系统"),
    "tekmor": ("L1", "B", "可验证 RAG 知识库"),
    "veritas-ai": ("L9", "B", "不变量优先 agent 运行时"),
    "worldai": ("L1", "B", "本地优先 RAG+Agent 平台"),
    "worldai-nexus": ("L1", "B", "本地优先 RAG 链路"),
    "worldai-rag": ("L1", "B", "模块化 RAG+Agent"),
    "worldai-stack": ("L1", "B", "RAG+ReAct 工具智能体"),
    "worldforge-ai": ("L1", "B", "端到端 AI 系统（可插拔后端）"),
    "worldrag": ("L6", "B", "CJK BM25 + 向量双路召回"),
    "zhi-rag": ("L1", "B", "多模态 RAG 引擎"),
    "zhishu": ("L1", "B", "智枢：企业级 AI 知识中枢"),
    "ai-rag-platform": ("L1", "B", "端到端 RAG QA 平台"),
    "ai-platform-agent-rag": ("L1", "B", "LangGraph 编排 Agent + Chroma"),
    "ai-llm-api": ("L6", "B", "本地 LLM 推理服务（FastAPI+llama.cpp）"),
    # 学习/算法类精准归位（关键词易误分，人工策展优先）
    "anomalyforge": ("L5", "A", "模块化异常检测系统"),
    "anoforge": ("L5", "A", "无监督异常检测(AnoFuse)"),
    "metaforge": ("L5", "A", "少样本元学习系统"),
    "robustforge": ("L5", "A", "对抗鲁棒性系统"),
    "labelforge": ("L5", "A", "弱监督/程序化标注"),
    "rankforge": ("L5", "A", "Learning to Rank"),
    "searchforge": ("L5", "A", "神经/混合检索与重排"),
    "oodforge": ("L5", "A", "分布外检测与校准"),
    "sindyforge": ("L2", "A", "稀疏动力学辨识(SINDy)"),
    "chainforge": ("L2", "A", "贝叶斯后验推断(NUTS/ADVI)"),
    "odeforge": ("L2", "A", "连续时间动力学求解(ODE/Symplectic)"),
    "tsforge": ("L5", "A", "时间序列预测系统"),
    "tscforge": ("L5", "A", "确定性时间序列分类(TinyRocket)"),
    "rs-forge": ("L2", "B", "Reed-Solomon 编码(FEC/GF(256))"),
    "intentforge": ("L5", "A", "置信度校准级联路由文本分类"),
    "nasforge": ("L4", "A", "神经架构搜索(NAS)"),
    "dictforge": ("L2", "B", "字典学习/稀疏编码"),
    "watermarkforge": ("L6", "A", "LLM 文本水印"),
    "starforge-ai": ("L9", "B", "接口驱动 AI 工程骨架"),
    "ai-dev-env-report": ("L9", "C", "环境清单报告（工具性，非算法）"),
    "ai-library-dashboard": ("L9", "C", "资料库看板（工具性）"),
    "starforge": ("L9", "C", "GitHub 高星榜单浏览器（工具性）"),
    "starforge-ai": ("L1", "B", "接口驱动 AI 工程骨架"),
    "repo-hygiene-gate": ("L9", "C", "仓库卫生门禁模板"),
    "CJX0712": (None, None, "GitHub 个人主页仓库（排除）"),
    "-": (None, None, "Gist 式空仓库（排除）"),
    "__probe_none": (None, None, "探针/重复仓库（排除）"),
}

GRADE_ORDER = ["S", "A", "B", "C"]

# A 级硬验证信号：描述中出现以下任一，视为「已交付且经测试/基准/不变量验证」。
_A_SIGNALS = ["全绿", "green", "pytest", "test", "测试", "benchmark", "基准", "不变量",
             "invariant", "可复现", "reproducible", "确定性", "deterministic", "门禁",
             "gate", "verify", "验证", "self-check", "自检", "cross-valid", "对拍",
             "single-file", "单文件", "零依赖"]
# S 级显式信号
_S_SIGNALS = ["质量等级 s", "quality grade s", "s-grade", " grade s", "质量 s", "s 级",
             "质量等级s"]


def grade_of(name, desc):
    d = (desc or "").lower()
    if any(s in d for s in _S_SIGNALS):
        return "S"
    if "fail" in d or "blocked" in d or "c-grade" in d or "c 级" in d:
        return "C"
    if d.strip().startswith("delivered by 晨星") or "delivered by 晨星" in d:
        return "C"  # 占位/薄仓库
    if not desc or desc.strip() == "?" or desc.strip() == "":
        return "C"
    has_author = ("作者 晨星" in d or "作者：晨星" in d or "author: 晨星" in d
                  or "author 晨星" in d or "by 晨星" in d or "晨星" in d)
    has_a = any(s in d for s in _A_SIGNALS)
    if has_author and has_a:
        return "A"
    if has_a:
        return "A"
    if has_author:
        return "B"
    return "B"


def classify(name, desc):
    if name in OVERRIDES:
        layer, grade, _ = OVERRIDES[name]
        if layer is None:
            return None, grade, "override-excluded"
        return layer, grade, "override"
    text = (name + " " + (desc or "")).lower()
    for layer in LAYER_PRIORITY:
        for kw in LAYER_KEYWORDS.get(layer, []):
            if kw in text:
                return layer, None, kw
    return "L1", None, "default"


def short_desc(desc, n=110):
    if not desc:
        return "—"
    d = desc.strip()
    return d if len(d) <= n else d[:n - 1] + "…"


def main():
    # 读取快照
    repos = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                repos.append(json.loads(line))

    entries = []
    for r in repos:
        name = r["name"]
        desc = r.get("description") or ""
        layer, ov_grade, how = classify(name, desc)
        if layer is None:
            continue  # 排除项
        grade = ov_grade if ov_grade else grade_of(name, desc)
        entries.append({
            "name": name,
            "description": short_desc(desc),
            "description_full": desc,
            "layer": layer,
            "layer_name": LAYERS[layer],
            "grade": grade,
            "stars": r.get("stars", 0),
            "lang": r.get("lang", ""),
            "pushed": (r.get("pushed") or "")[:10],
            "created": (r.get("created") or "")[:10],
            "reuse": LAYER_REUSE[layer],
            "classify_hint": how,
        })

    # 排序：层序 -> 评级 -> 名称
    grade_rank = {g: i for i, g in enumerate(GRADE_ORDER)}
    entries.sort(key=lambda e: (list(LAYERS).index(e["layer"]),
                                grade_rank.get(e["grade"], 9), e["name"]))

    # 统计
    total = len(entries)
    by_layer = OrderedDict((k, 0) for k in LAYERS)
    by_grade = defaultdict(int)
    matrix = {k: {g: 0 for g in GRADE_ORDER} for k in LAYERS}
    for e in entries:
        by_layer[e["layer"]] += 1
        by_grade[e["grade"]] += 1
        matrix[e["layer"]][e["grade"]] += 1

    catalog = {
        "meta": {
            "project": "world-class-ai",
            "title": "世界顶级 AI 系统 · 全量生态目录（跨仓库导航 + 质量矩阵）",
            "author": "晨星",
            "total_repos": total,
            "layers": LAYERS,
            "grade_order": GRADE_ORDER,
            "generated_by": "tools/build_ecosystem.py",
        },
        "summary": {
            "by_layer": by_layer,
            "by_grade": dict(by_grade),
            "matrix": matrix,
        },
        "repos": entries,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    # 生成 markdown
    md = render_markdown(catalog)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ 生成 {total} 个仓库目录 -> {os.path.normpath(OUT_JSON)}")
    print(f"✅ 生成文档 -> {os.path.normpath(OUT_MD)}")
    print(f"   评级分布: " + "  ".join(f"{g}={by_grade.get(g,0)}" for g in GRADE_ORDER))
    print(f"   分层计数: " + "  ".join(f"{k}={by_layer[k]}" for k in LAYERS))
    return catalog


def render_markdown(cat):
    L = []
    L.append("# 全量生态目录 · 跨仓库导航 + 质量矩阵\n")
    L.append("> 世界顶级 AI 系统（`world-class-ai`）母核的统一生态索引。\n")
    L.append("> 把 GitHub [@CJX0712](https://github.com/CJX0712) 下的全部算法仓库按「十层分解架构」归类，")
    L.append("> 逐仓库标注 **质量评级（S/A/B/C）** 与 **集大成复用目标**，形成可导航、可审计的总览。\n")
    L.append("> **作者：晨星** · 由 `tools/build_ecosystem.py` 确定性生成（输入 `src/ai_sys/data/repos_raw.jsonl`）。\n")

    total = cat["meta"]["total_repos"]
    by_grade = cat["summary"]["by_grade"]
    L.append(f"## 总览\n")
    L.append(f"- **收录仓库**：{total} 个（已排除主页/Gist/探针类非算法仓库）")
    L.append(f"- **质量评级分布**：" + "  ".join(
        f"**{g}**={by_grade.get(g,0)}" for g in GRADE_ORDER))
    L.append(f"- **评级口径**：`S`=描述明示「质量等级 S / S-grade」；`A`=已交付且带硬验证信号")
    L.append(f"  （单测全绿 / 基准 / 不变量 / 可复现 / 零依赖自检 等）；`B`=已交付实仓库（标注作者晨星或实质算法）；")
    L.append(f"  `C`=占位/薄仓库（「Delivered by 晨星」无实质内容）/ 工具性仓库 / 无描述。")
    L.append("")

    # 质量矩阵
    L.append("## 质量矩阵（层 × 评级）\n")
    L.append("| 层 | 名称 | " + " | ".join(GRADE_ORDER) + " | 合计 |")
    L.append("|" + "---|" * (len(GRADE_ORDER) + 3))
    matrix = cat["summary"]["matrix"]
    for k, name in LAYERS.items():
        row = matrix[k]
        cells = " | ".join(str(row[g]) for g in GRADE_ORDER)
        L.append(f"| {k} | {name} | {cells} | {sum(row.values())} |")
    tot_row = [sum(matrix[k][g] for k in LAYERS) for g in GRADE_ORDER]
    L.append(f"| — | **总计** | " + " | ".join(str(x) for x in tot_row) +
             f" | {sum(tot_row)} |")
    L.append("")

    # 逐层明细
    L.append("## 分层索引（跨仓库导航）\n")
    repos = cat["repos"]
    for k, lname in LAYERS.items():
        items = [e for e in repos if e["layer"] == k]
        if not items:
            continue
        L.append(f"### {k} · {lname}\n")
        L.append(f"> 集大成复用目标：`{LAYER_REUSE[k]}`\n")
        L.append(f"| 评级 | 仓库 | 一句话 | 语言 | ★ | 最近推送 |")
        L.append(f"|------|------|--------|------|---|----------|")
        for e in items:
            L.append(f"| {e['grade']} | [{e['name']}](https://github.com/CJX0712/{e['name']}) "
                     f"| {e['description']} | {e['lang'] or '—'} | {e['stars']} | {e['pushed'] or '—'} |")
        L.append("")

    L.append("## 与第十层（1024 微指令）的关系\n")
    L.append("- 本目录是 **L1–L9 的真实仓库落地**；`world-class-ai` 自身的 `src/ai_sys/data/micro_instructions.json`")
    L.append("  是 **L10 微指令规范**（1024 条原子操作，由 `verify.py` 闭环自检）。")
    L.append("- 每层仓库的算法能力，对应 L10 同名操作族：例如 L2 的 `gp-forge`/`bayesforge` 对应 M0 数学算子，")
    L.append("  L3 的 `ctrlforge`/`banditforge` 对应 M1 优化与采样，依此类推。")
    L.append("- 复现：`python tools/build_ecosystem.py` 重新生成本文档与 `ecosystem.json`。\n")

    L.append("---\n")
    L.append("*本目录由脚本确定性生成，分类关键词透明可审计；人工策展覆盖见 `tools/build_ecosystem.py` 的 `OVERRIDES`。*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
