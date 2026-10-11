# 第十层 · 1024 微指令规范与扩展指南

---

## 1. 为什么是 1024

`1024 = 2^10`，是二进制世界最自然的"整层细分"粒度。第十层作为十层架构的最底层，
以 **10 个模块 × 128 操作族 × 8 变体 = 1024** 的形式，把上层（L1–L9）抽象逐层
降解为可编码、可测试、可复用的原子操作。

- **10 个模块（M0–M9）**：对应上层功能域（数学/优化/前反向/训练/推理/张量/数据/运行时/控制/系统）
- **128 操作族**：每个模块下的真实原子操作（如 `gemm`、`adam`、`flash_attention`）
- **8 变体/族**：`dtype` 类（f32/f16/bf16/f64/int8/int4/sparse/fused 等）或 `mode` 类（step/init/reset/clip/schedule/fuse/stream/batch）

## 2. 目录结构（单一事实源）

`src/ai_sys/data/micro_instructions.json` 由 `tools/build_catalog.py` 确定性生成，
包含：
- `meta`：项目信息、10 模块统计、10 层命名
- `micro_instructions[]`：1024 条，字段如下

| 字段 | 含义 |
|------|------|
| `id` | `MI-0001`..`MI-1024`，唯一且连续 |
| `name` | `族_变体`，如 `gemm_f16` |
| `module` / `module_name` | 所属模块 |
| `layer_ref` | 向上映射到的层（L1–L9） |
| `category` | 分类（math/optim/forward/.../system） |
| `family` | 操作族名 |
| `variant` / `variant_kind` | 变体名与类型（dtype / mode） |
| `description` | 简短定义 |
| `reuse` | **集大成**：应复用的世界顶级开源/算法 |
| `deps` | 依赖的其它微指令 id 列表 |
| `status` | `skeleton`（骨架）/ `implemented`（已实现） |
| `verify` | 自检类型（`contract` 当前；未来 `functional`） |

## 3. 操作族清单（128 族）

- **M0 数学算子（18）**: axpy, gemm, gemv, dot, vector_norm, matrix_norm, triangular_solve, cholesky, lu_decompose, qr_decompose, eigendecomp, svd, fft1d, fft2d, softmax, layernorm, cross_entropy, kl_divergence
- **M1 优化与采样（12）**: sgd, momentum, nesterov, adam, adamw, rmsprop, lars, lion, adafactor, radam, lookahead, gradient_clip
- **M2 模型前反向（18）**: linear_fwd, linear_bwd, conv2d, conv1d, depthwise_conv, maxpool, avgpool, batchnorm, layernorm_layer, rmsnorm, embedding, multihead_attention, flash_attention, cross_attention, mlp_block, lstm_cell, gru_cell, transformer_block
- **M3 训练循环（12）**: dataloader_next, batch_collate, loss_compute, backward, optimizer_step, lr_schedule, grad_accumulate, mixed_precision_cast, checkpoint_save, checkpoint_load, distributed_allreduce, early_stopping
- **M4 推理服务（14）**: request_batch, kv_cache_alloc, kv_cache_append, paged_attention, speculative_decode, quantize_w8a8, quantize_w4, kernel_fusion, continuous_batching, token_sampler, logits_penalty, beam_search, top_p_sample, draft_verify
- **M5 张量算子（12）**: tensor_alloc, tensor_reshape, tensor_transpose, tensor_cast, tensor_slice, tensor_concat, tensor_pad, reduce_sum, reduce_mean, reduce_max, elementwise_add, elementwise_mul
- **M6 数据管道（12）**: tokenize_bpe, tokenize_wordpiece, detokenize, text_normalize, json_parse, json_serialize, csv_read, parquet_read, feature_encode, vector_index_build, vector_search, stream_buffer
- **M7 运行时调度（12）**: task_schedule, thread_pool, async_exec, memory_alloc, memory_free, tensor_gc, event_loop, lock_acquire, lock_release, barrier, pipeline_stage, fault_restart
- **M8 控制与编排（10）**: agent_think, tool_route, tool_call, plan_decompose, rerank_crossencoder, claim_ground, memory_read, memory_write, retry_backoff, human_handoff
- **M9 系统支撑（8）**: log_emit, metric_record, config_load, error_raise, hash_sha256, uuid_gen, time_now, health_check

## 4. 依赖图（示例）

部分微指令声明 `deps`，形成可验证的依赖链，例如：
- `linear_fwd / linear_bwd` → `gemm`
- `multihead_attention / flash_attention / cross_attention` → `gemm`, `softmax`
- `transformer_block` → `layernorm_layer`, `multihead_attention`, `mlp_block`
- `paged_attention` → `tensor_slice`, `softmax`
- `quantize_w8a8 / quantize_w4` → `tensor_cast`
- `token_sampler` → `top_p_sample`
- `tool_call` → `tool_route`
- `rerank_crossencoder` → `dot`

`verify.py` 会校验每条 `deps` 都指向存在的微指令 id。

## 5. 自检机制

`verify.py` 做三类校验：
1. **结构完整性**：总数 == 1024、ID 唯一连续、依赖有效。
2. **逐项契约自检**：每条 `MicroOp.self_check()` 校验字段齐全、分类处理器已覆盖该族。
3. **功能正确性**：已实现族（`status=implemented`）运行真实算子，并与**独立参照实现**
   （朴素 O(n³)/O(n²) 或直接公式）及**解析不变量**（如 `‖A−LLᵀ‖→0`、`P·L·U=A`、朴素 DFT
   对照）交叉校验，残差达机器精度级。

当前 **M0 数学算子**（144/144 条 `functional` 全绿）与 **M1 优化与采样**（96/96 条 `functional` 全绿）均已落地；
其余模块仍为 skeleton 阶段（契约自检通过 ✅）。新增某族真实算子后，将其 `status` 改为 `implemented`、在对应
`handlers/<category>.py` 中落地 `execute`/`check`，并把 `verify` 升级为 `functional`，
自检仍保持全绿。

## 6. 如何扩展

1. 在 `tools/build_catalog.py` 的 `FAMILIES` / `MODULES` 中增删族或模块
   （注意维持 128 族 × 8 = 1024，或同步调整 `verify.py` 的期望值）。
2. 重新生成：`python tools/build_catalog.py`。
3. 运行校验：`python verify.py`，确认仍全绿。
4. 在 `src/ai_sys/handlers/<category>.py` 落地真实实现，更新 `status`。

> 纪律：**新增任何算子前，先查世界是否有更强现成方案**；有则直接复用并在 `reuse`
> 中标注，禁止从零重写。
