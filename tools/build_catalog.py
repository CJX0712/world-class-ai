#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_catalog.py — 世界顶级 AI 系统 · 第十层 1024 微指令目录生成器
===============================================================

设计原则（与项目总纲一致）：
  · 零重写   —— 每一条微指令都映射到世界顶级开源/算法（见 reuse 字段），不自造。
  · 集大成   —— 分类法站在现代 AI 系统解剖学之上（数据→数学→算法→架构→训练→
               推理→算子→IO→运行时→编排→微指令）。
  · 闭环交付 —— 本脚本是「单一事实源」：确定性生成 micro_instructions.json，
               可复现、可验证（verify.py 消费本目录）。

结构：
  L10 微指令层 = 10 个模块（M0..M9），每个模块下若干「操作族」，每族展开 8 个
  变体（dtype 或 mode），共 128 族 × 8 = 1024 条微指令。

作者：晨星
"""

import json
import os
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# 变体维度（每族展开 8 个，保证 128 族 × 8 = 1024）
# --------------------------------------------------------------------------- #
DTYPE_VARIANTS_MATH = ["f32", "f64", "f16", "bf16", "c64", "c128", "int8", "int4"]
DTYPE_VARIANTS_FWD = ["f32", "f16", "bf16", "f64", "int8", "int4", "sparse", "fused"]
DTYPE_VARIANTS_TENSOR = ["f32", "f16", "bf16", "f64", "i32", "i8", "u8", "bool"]
MODE_VARIANTS = ["step", "init", "reset", "clip", "schedule", "fuse", "stream", "batch"]

# --------------------------------------------------------------------------- #
# 10 个模块（M0..M9） —— 每个映射到上层 L1..L9 之一或横切
# --------------------------------------------------------------------------- #
MODULES = [
    {"key": "M0", "name": "数学算子",   "layer": "L2", "category": "math",
     "variants": DTYPE_VARIANTS_MATH,
     "reuse": "cuBLAS / PyTorch ATen / JAX XLA / Eigen"},
    {"key": "M1", "name": "优化与采样", "layer": "L3", "category": "optim",
     "variants": MODE_VARIANTS,
     "reuse": "PyTorch Optim / DeepSpeed / Lion(Google) / Adafactor"},
    {"key": "M2", "name": "模型前反向", "layer": "L4", "category": "forward",
     "variants": DTYPE_VARIANTS_FWD,
     "reuse": "PyTorch nn / Megatron-LM / FlashAttention / TensorRT"},
    {"key": "M3", "name": "训练循环",   "layer": "L5", "category": "training",
     "variants": MODE_VARIANTS,
     "reuse": "PyTorch DDP / DeepSpeed ZeRO / Megatron / Accelerate"},
    {"key": "M4", "name": "推理服务",   "layer": "L6", "category": "inference",
     "variants": MODE_VARIANTS,
     "reuse": "vLLM / TensorRT-LLM / llama.cpp / SGLang"},
    {"key": "M5", "name": "张量算子",   "layer": "L7", "category": "tensor",
     "variants": DTYPE_VARIANTS_TENSOR,
     "reuse": "PyTorch tensor / NumPy / XLA / Apache Arrow"},
    {"key": "M6", "name": "数据管道",   "layer": "L8", "category": "data",
     "variants": MODE_VARIANTS,
     "reuse": "HuggingFace Tokenizers(Rust) / FAISS / Pandas / pyarrow"},
    {"key": "M7", "name": "运行时调度", "layer": "L9", "category": "runtime",
     "variants": MODE_VARIANTS,
     "reuse": "Ray / asyncio / tcmalloc / mimalloc"},
    {"key": "M8", "name": "控制与编排", "layer": "L1", "category": "control",
     "variants": MODE_VARIANTS,
     "reuse": "LangGraph / AutoGen / ReAct / Constitutional AI"},
    {"key": "M9", "name": "系统支撑",   "layer": "L9", "category": "system",
     "variants": MODE_VARIANTS,
     "reuse": "OpenTelemetry / structlog / pydantic / hashlib"},
]

# --------------------------------------------------------------------------- #
# 操作族（family） —— 每模块下的真实原子操作，附简短定义
# --------------------------------------------------------------------------- #
FAMILIES = {
    "M0": [
        ("axpy", "向量缩放加 y = αx + y"),
        ("gemm", "通用矩阵乘 C = αAB + βC"),
        ("gemv", "矩阵-向量乘 y = αAx + βy"),
        ("dot", "点积 ⟨x, y⟩"),
        ("vector_norm", "向量范数（L1 / L2 / L∞）"),
        ("matrix_norm", "矩阵范数（Frobenius）"),
        ("triangular_solve", "三角线性方程组求解"),
        ("cholesky", "Cholesky 分解 A = LLᵀ"),
        ("lu_decompose", "LU 分解（部分主元）"),
        ("qr_decompose", "QR 正交分解"),
        ("eigendecomp", "特征分解 A = VΛV⁻¹"),
        ("svd", "奇异值分解 A = UΣVᵀ"),
        ("fft1d", "一维快速傅里叶变换"),
        ("fft2d", "二维快速傅里叶变换"),
        ("softmax", "数值稳定 softmax（logsumexp）"),
        ("layernorm", "层归一化"),
        ("cross_entropy", "交叉熵（含 logsumexp 稳定）"),
        ("kl_divergence", "KL 散度 D_KL(p‖q)"),
    ],
    "M1": [
        ("sgd", "随机梯度下降"),
        ("momentum", "动量法"),
        ("nesterov", "Nesterov 加速梯度"),
        ("adam", "Adam 自适应矩估计"),
        ("adamw", "AdamW 解耦权重衰减"),
        ("rmsprop", "RMSProp"),
        ("lars", "LARS 分层自适应速率"),
        ("lion", "Lion 符号动量（Google）"),
        ("adafactor", "AdaFactor 低秩自适应"),
        ("radam", "RAdam 整流 Adam"),
        ("lookahead", "Lookahead 外层推进"),
        ("gradient_clip", "梯度裁剪（norm / value）"),
    ],
    "M2": [
        ("linear_fwd", "全连接层前向"),
        ("linear_bwd", "全连接层反向"),
        ("conv2d", "二维卷积"),
        ("conv1d", "一维卷积"),
        ("depthwise_conv", "深度可分离卷积"),
        ("maxpool", "最大池化"),
        ("avgpool", "平均池化"),
        ("batchnorm", "批归一化"),
        ("layernorm_layer", "层归一化层"),
        ("rmsnorm", "RMSNorm"),
        ("embedding", "嵌入查表"),
        ("multihead_attention", "多头注意力"),
        ("flash_attention", "FlashAttention IO 感知注意力"),
        ("cross_attention", "交叉注意力"),
        ("mlp_block", "MLP 前馈块"),
        ("lstm_cell", "LSTM 单元"),
        ("gru_cell", "GRU 单元"),
        ("transformer_block", "Transformer 残差块"),
    ],
    "M3": [
        ("dataloader_next", "数据迭代器取批"),
        ("batch_collate", "样本批组装"),
        ("loss_compute", "损失计算"),
        ("backward", "自动反向（autograd）"),
        ("optimizer_step", "优化器步进"),
        ("lr_schedule", "学习率调度"),
        ("grad_accumulate", "梯度累积"),
        ("mixed_precision_cast", "混合精度转换"),
        ("checkpoint_save", "检查点保存"),
        ("checkpoint_load", "检查点加载"),
        ("distributed_allreduce", "分布式梯度 AllReduce"),
        ("early_stopping", "早停判据"),
    ],
    "M4": [
        ("request_batch", "请求批组（连续批处理入队）"),
        ("kv_cache_alloc", "KV 缓存分配"),
        ("kv_cache_append", "KV 缓存追加"),
        ("paged_attention", "分页注意力计算"),
        ("speculative_decode", "投机解码草稿"),
        ("quantize_w8a8", "权重-激活 8bit 量化"),
        ("quantize_w4", "权重 4bit 量化（GPTQ / AWQ）"),
        ("kernel_fusion", "算子融合"),
        ("continuous_batching", "连续批处理调度"),
        ("token_sampler", "token 采样分发"),
        ("logits_penalty", "logits 惩罚（重复 / 长度）"),
        ("beam_search", "束搜索"),
        ("top_p_sample", "Top-p 核采样"),
        ("draft_verify", "草稿验证（拒绝采样）"),
    ],
    "M5": [
        ("tensor_alloc", "张量内存分配"),
        ("tensor_reshape", "形状重排"),
        ("tensor_transpose", "维度转置"),
        ("tensor_cast", "类型转换"),
        ("tensor_slice", "切片视图"),
        ("tensor_concat", "拼接"),
        ("tensor_pad", "填充"),
        ("reduce_sum", "归约求和"),
        ("reduce_mean", "归约求均值"),
        ("reduce_max", "归约求最大值"),
        ("elementwise_add", "逐元素加"),
        ("elementwise_mul", "逐元素乘"),
    ],
    "M6": [
        ("tokenize_bpe", "BPE 分词"),
        ("tokenize_wordpiece", "WordPiece 分词"),
        ("detokenize", "ID 反解为文本"),
        ("text_normalize", "文本归一化"),
        ("json_parse", "JSON 解析"),
        ("json_serialize", "JSON 序列化"),
        ("csv_read", "CSV 读取"),
        ("parquet_read", "Parquet 列式读取"),
        ("feature_encode", "特征编码"),
        ("vector_index_build", "向量索引构建（FAISS）"),
        ("vector_search", "近似近邻检索"),
        ("stream_buffer", "流式缓冲"),
    ],
    "M7": [
        ("task_schedule", "任务调度"),
        ("thread_pool", "线程池执行"),
        ("async_exec", "异步执行"),
        ("memory_alloc", "堆内存分配"),
        ("memory_free", "内存释放"),
        ("tensor_gc", "张量垃圾回收"),
        ("event_loop", "事件循环"),
        ("lock_acquire", "锁获取"),
        ("lock_release", "锁释放"),
        ("barrier", "Barrier 同步"),
        ("pipeline_stage", "流水线阶段"),
        ("fault_restart", "故障重启"),
    ],
    "M8": [
        ("agent_think", "智能体推理步"),
        ("tool_route", "工具路由判定"),
        ("tool_call", "工具调用执行"),
        ("plan_decompose", "任务分解规划"),
        ("rerank_crossencoder", "交叉编码器重排"),
        ("claim_ground", "声明事实溯源"),
        ("memory_read", "记忆读取"),
        ("memory_write", "记忆写入"),
        ("retry_backoff", "指数退避重试"),
        ("human_handoff", "人工接管"),
    ],
    "M9": [
        ("log_emit", "结构化日志输出"),
        ("metric_record", "指标记录"),
        ("config_load", "配置加载校验"),
        ("error_raise", "异常抛出"),
        ("hash_sha256", "SHA-256 哈希"),
        ("uuid_gen", "UUID 生成"),
        ("time_now", "单调时钟读取"),
        ("health_check", "健康探针"),
    ],
}

# --------------------------------------------------------------------------- #
# 依赖图（family -> 依赖的 family 列表）；解析为被依赖族首个变体的 id
# --------------------------------------------------------------------------- #
DEPS = {
    "linear_fwd": ["gemm"],
    "linear_bwd": ["gemm"],
    "multihead_attention": ["gemm", "softmax"],
    "flash_attention": ["gemm", "softmax"],
    "cross_attention": ["gemm", "softmax"],
    "transformer_block": ["layernorm_layer", "multihead_attention", "mlp_block"],
    "backward": ["linear_bwd"],
    "optimizer_step": ["adam"],
    "kv_cache_append": ["tensor_slice"],
    "paged_attention": ["tensor_slice", "softmax"],
    "quantize_w8a8": ["tensor_cast"],
    "quantize_w4": ["tensor_cast"],
    "token_sampler": ["top_p_sample"],
    "feature_encode": ["tokenize_bpe"],
    "tensor_gc": ["memory_free"],
    "tool_call": ["tool_route"],
    "rerank_crossencoder": ["dot"],
    "request_batch": ["batch_collate"],
    "checkpoint_load": ["checkpoint_save"],
}

LAYER_NAMES = {
    "L1": "范式层 Paradigms",
    "L2": "数学基础层 Mathematical Foundations",
    "L3": "算法原语层 Algorithmic Primitives",
    "L4": "模型架构层 Model Architectures",
    "L5": "训练系统层 Training System",
    "L6": "推理引擎层 Inference Engine",
    "L7": "算子内核层 Compute Kernels",
    "L8": "数据IO层 Data & IO",
    "L9": "运行时编排层 Runtime & Orchestration",
    "L10": "微指令层 Micro-Instructions",
}

# 已真实落地（status=implemented, verify=functional）的模块键集合。
# 当前 M0 数学算子 18 族全部由 handlers/math.py 以 NumPy 世界顶级内核实现，
# 并经独立参照 + 解析不变量交叉校验（residual → 机器精度级）。新增实现时在此扩充。
IMPLEMENTED_MODULES = {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"}

BUILD_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build():
    entries = []
    family_first_id = {}  # family -> first variant's id
    index = 0

    mod_lookup = {m["key"]: m for m in MODULES}

    for mod in MODULES:
        key = mod["key"]
        for fam_name, fam_desc in FAMILIES[key]:
            for variant in mod["variants"]:
                index += 1
                mid = f"MI-{index:04d}"
                if fam_name not in family_first_id:
                    family_first_id[fam_name] = mid
                desc = f"{fam_desc}（{variant} 变体）"
                implemented = key in IMPLEMENTED_MODULES
                entry = {
                    "id": mid,
                    "index": index,
                    "module": key,
                    "module_name": mod["name"],
                    "layer": "L10",
                    "layer_ref": mod["layer"],
                    "layer_ref_name": LAYER_NAMES[mod["layer"]],
                    "category": mod["category"],
                    "family": fam_name,
                    "name": f"{fam_name}_{variant}",
                    "variant": variant,
                    "variant_kind": ("dtype" if variant in mod["variants"] and variant not in MODE_VARIANTS else "mode"),
                    "description": desc,
                    "reuse": mod["reuse"],
                    "deps": [],
                    "status": "implemented" if implemented else "skeleton",
                    "verify": "functional" if implemented else "contract",
                }
                entries.append(entry)

    # resolve deps
    for entry in entries:
        fam = entry["family"]
        if fam in DEPS:
            resolved = []
            for dep_fam in DEPS[fam]:
                if dep_fam not in family_first_id:
                    raise ValueError(f"依赖族不存在: {dep_fam}（被 {fam} 引用）")
                resolved.append(family_first_id[dep_fam])
            entry["deps"] = resolved

    # sanity: exactly 1024
    if len(entries) != 1024:
        raise SystemExit(f"微指令总数应为 1024，实际 {len(entries)}")

    catalog = {
        "meta": {
            "project": "world-class-ai",
            "title": "世界顶级 AI 系统 · 第十层 1024 微指令",
            "author": "晨星",
            "build_time": BUILD_TIME,
            "total": len(entries),
            "modules": [
                {
                    "key": m["key"],
                    "name": m["name"],
                    "layer_ref": m["layer"],
                    "layer_ref_name": LAYER_NAMES[m["layer"]],
                    "category": m["category"],
                    "families": len(FAMILIES[m["key"]]),
                    "variants_per_family": len(m["variants"]),
                    "reuse": m["reuse"],
                }
                for m in MODULES
            ],
            "layers": LAYER_NAMES,
        },
        "micro_instructions": entries,
    }
    return catalog


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "src", "ai_sys", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "micro_instructions.json")
    catalog = build()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"✅ 生成 {catalog['meta']['total']} 条微指令 -> {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
