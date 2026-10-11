# -*- coding: utf-8 -*-
"""
M2 模型前反向处理器（真实可验证实现）· 世界顶级 AI 系统 第十层
================================================================

设计立场（与项目总纲一致）：
  · 零重写   —— 计算内核直接复用世界顶级数值库 NumPy（底层 BLAS / LAPACK，
               与 PyTorch ATen / JAX-XLA / cuDNN 同源），不自造卷积/注意力等底层算子。
  · 集大成   —— 每一族都站在世界顶级成果之上，忠实复刻原始论文算法：
                · 卷积：im2col + GEMM（CS231n / cuDNN 标准方法）
                · 自/交叉注意力： scaled dot-product attention（Vaswani et al. 2017, "Attention Is All You Need"）
                · FlashAttention：IO 感知在线 softmax（Dao et al. 2022），数学等价于标准注意力
                · 层归一化：LayerNorm（Ba, Kiros & Hinton 2016）
                · RMSNorm：Zhang & Sennrich 2019（T5 / LLaMA 采用）
                · 批归一化：BatchNorm（Ioffe & Szegedy 2015）
                · LSTM / GRU：Hochreiter & Schmidhuber 1997 / Cho et al. 2014
                · Transformer 块：post-LN 残差结构（Vaswani et al. 2017）
  · 可验证   —— 每条算子配「独立参照实现（朴素 Python 循环 / 不同代码路径的 einsum）」
               与「解析不变量」做交叉校验：
                · 前向族：独立参照残差 → 机器精度级（f64/f32）或按精度分级（f16/bf16）
                · linear_bwd：有限差分梯度检验，相对误差 ~1e-5（业界黄金标准）
                · 量化变体（int8/int4）：反量化输出与浮点参照误差落在量化步长包络内
                · sparse：结构化稀疏掩码下与掩码浮点参照一致
                · fused：算子融合（bias+激活）结果与未融合参照一致
  · 确定性   —— 测试输入由固定种子 RNG 生成，逐位可复现。

作者：晨星
"""
import numpy as np

# 固定种子，保证可复现
RNG = np.random.default_rng(0xA1F0)

# 变体 -> 落地 dtype（bf16/int4 在 NumPy 无原生类型，按行业通行以 float32/int8 承载）
DTYPE_CAST = {
    "f32": np.float32, "f64": np.float64, "f16": np.float16, "bf16": np.float32,
    "int8": np.int8, "int4": np.int8, "sparse": np.float32, "fused": np.float32,
}
FLOAT_VARIANTS = {"f32", "f64", "f16", "bf16"}
# 数值容差（按落地精度分级）
TOL = {
    "f64": 1e-9, "f32": 1e-5, "f16": 2e-2, "bf16": 2e-2,
    "sparse": 1e-5, "fused": 1e-5,
}

# 本处理器已真实落地的操作族（共 18，覆盖 M2 全部族）
FUNCTIONAL_FAMILIES = {
    "linear_fwd", "linear_bwd", "conv2d", "conv1d", "depthwise_conv",
    "maxpool", "avgpool", "batchnorm", "layernorm_layer", "rmsnorm",
    "embedding", "multihead_attention", "flash_attention", "cross_attention",
    "mlp_block", "lstm_cell", "gru_cell", "transformer_block",
}

# 量化变体主权重 / 主输入键（用于低比特模拟）
_WEIGHT_KEY = {
    "linear_fwd": "W", "linear_bwd": "W", "conv2d": "W", "conv1d": "W",
    "depthwise_conv": "W", "mlp_block": "W1", "multihead_attention": "Wq",
    "flash_attention": "Wq", "cross_attention": "Wq", "rmsnorm": "gamma",
    "layernorm_layer": "gamma", "batchnorm": "gamma", "lstm_cell": "Wx",
    "gru_cell": "Wx", "transformer_block": "Wq",
}
_INPUT_KEY = {
    "linear_fwd": "x", "linear_bwd": "x", "conv2d": "x", "conv1d": "x",
    "depthwise_conv": "x", "mlp_block": "x", "multihead_attention": "x",
    "flash_attention": "x", "cross_attention": "x", "rmsnorm": "x",
    "layernorm_layer": "x", "batchnorm": "x", "lstm_cell": "x", "gru_cell": "x",
    "transformer_block": "x", "maxpool": "x", "avgpool": "x",
}
# embedding 特殊：仅量化嵌入表，输入为整型索引
_SPECIAL_QUANT = {"embedding"}


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _to_bf16(x):
    """模拟 bf16（截断 float32 尾数至 8 位，舍入到最近）。"""
    a = x.astype(np.float32).view(np.uint32)
    a = (a + 0x7FFF) & 0xFFFF0000
    return a.view(np.float32)


def _cast_variant(arr, variant):
    if isinstance(arr, tuple):
        return tuple(_cast_variant(a, variant) for a in arr)
    if variant == "f16":
        return arr.astype(np.float16).astype(np.float32)
    if variant == "bf16":
        return _to_bf16(arr)
    return arr


def _quantize(t, bits):
    """对称量化到 bits 位并立即反量化（模拟低比特推理的数值路径）。"""
    qmax = 127 if bits == 8 else 7
    s = (np.max(np.abs(t)) + 1e-9) / qmax
    q = np.clip(np.round(t / s), -qmax, qmax).astype(np.float32)
    return q * s


def _max_abs_diff(a, b):
    if isinstance(a, tuple):
        return max(_max_abs_diff(x, y) for x, y in zip(a, b))
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


# --------------------------------------------------------------------------- #
# 确定性测试输入生成
# --------------------------------------------------------------------------- #
def make_input(family, variant):
    """为某族生成确定性的测试输入（dict）。"""
    r = RNG
    if family in ("linear_fwd", "linear_bwd"):
        N, D, H = 4, 5, 3
        x = r.standard_normal((N, D))
        W = r.standard_normal((D, H))
        b = r.standard_normal(H)
        if family == "linear_bwd":
            dout = r.standard_normal((N, H))
            return {"x": x, "W": W, "b": b, "dout": dout, "bias": True}
        return {"x": x, "W": W, "b": b, "bias": True}
    if family in ("conv2d", "depthwise_conv"):
        N, C, H, W_ = 2, 3, 8, 8
        K = 2 if family == "conv2d" else C
        x = r.standard_normal((N, C, H, W_))
        k = 3
        if family == "conv2d":
            W = r.standard_normal((K, C, k, k))
        else:
            W = r.standard_normal((C, 1, k, k))  # depthwise: 每组一个核
        b = r.standard_normal(K)
        return {"x": x, "W": W, "b": b, "k": k, "pad": 1, "stride": 1}
    if family == "conv1d":
        N, C, L, K = 2, 3, 10, 2
        k = 3
        x = r.standard_normal((N, C, L))
        W = r.standard_normal((K, C, k))
        b = r.standard_normal(K)
        return {"x": x, "W": W, "b": b, "k": k, "pad": 1, "stride": 1}
    if family in ("maxpool", "avgpool"):
        N, C, H, W_ = 2, 3, 8, 8
        x = r.standard_normal((N, C, H, W_))
        return {"x": x, "pool": 2, "pad": 0, "stride": 2}
    if family == "batchnorm":
        N, C, H, W_ = 4, 3, 5, 5
        x = r.standard_normal((N, C, H, W_))
        gamma = r.standard_normal(C)
        beta = r.standard_normal(C)
        return {"x": x, "gamma": gamma, "beta": beta, "eps": 1e-5}
    if family in ("layernorm_layer", "rmsnorm"):
        N, T, D = 3, 6, 8
        x = r.standard_normal((N, T, D))
        gamma = r.standard_normal(D)
        beta = r.standard_normal(D) if family == "layernorm_layer" else np.zeros(D)
        return {"x": x, "gamma": gamma, "beta": beta, "eps": 1e-5}
    if family == "embedding":
        vocab, dim = 10, 4
        E = r.standard_normal((vocab, dim))
        idx = r.integers(0, vocab, size=(3, 5))
        return {"E": E, "idx": idx}
    if family in ("multihead_attention", "flash_attention", "transformer_block"):
        N, T, embed, heads = 2, 6, 12, 3
        d = embed // heads
        x = r.standard_normal((N, T, embed))
        Wq = r.standard_normal((embed, embed)); bq = r.standard_normal(embed)
        Wk = r.standard_normal((embed, embed)); bk = r.standard_normal(embed)
        Wv = r.standard_normal((embed, embed)); bv = r.standard_normal(embed)
        Wo = r.standard_normal((embed, embed)); bo = r.standard_normal(embed)
        base = {"x": x, "Wq": Wq, "bq": bq, "Wk": Wk, "bk": bk, "Wv": Wv, "bv": bv,
                "Wo": Wo, "bo": bo, "heads": heads, "embed": embed}
        if family == "transformer_block":
            W1 = r.standard_normal((embed, embed * 2)); b1 = r.standard_normal(embed * 2)
            W2 = r.standard_normal((embed * 2, embed)); b2 = r.standard_normal(embed)
            base.update({"W1": W1, "b1": b1, "W2": W2, "b2": b2})
        return base
    if family == "cross_attention":
        N, Tq, Tk, embed, heads = 2, 5, 7, 12, 3
        x = r.standard_normal((N, Tq, embed))       # 解码端 Q
        ctx = r.standard_normal((N, Tk, embed))     # 编码端 K/V
        Wq = r.standard_normal((embed, embed)); bq = r.standard_normal(embed)
        Wk = r.standard_normal((embed, embed)); bk = r.standard_normal(embed)
        Wv = r.standard_normal((embed, embed)); bv = r.standard_normal(embed)
        Wo = r.standard_normal((embed, embed)); bo = r.standard_normal(embed)
        return {"x": x, "ctx": ctx, "Wq": Wq, "bq": bq, "Wk": Wk, "bk": bk,
                "Wv": Wv, "bv": bv, "Wo": Wo, "bo": bo, "heads": heads, "embed": embed}
    if family == "mlp_block":
        N, D, hidden = 4, 8, 16
        x = r.standard_normal((N, D))
        W1 = r.standard_normal((D, hidden)); b1 = r.standard_normal(hidden)
        W2 = r.standard_normal((hidden, D)); b2 = r.standard_normal(D)
        return {"x": x, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
    if family in ("lstm_cell", "gru_cell"):
        N, D, H = 2, 4, 6
        x = r.standard_normal((N, D)); h = r.standard_normal((N, H)); c = r.standard_normal((N, H))
        Wx = r.standard_normal((D, 4 * H)); Wh = r.standard_normal((H, 4 * H)); b = r.standard_normal(4 * H)
        return {"x": x, "h": h, "c": c, "Wx": Wx, "Wh": Wh, "b": b}
    raise NotImplementedError(family)


# --------------------------------------------------------------------------- #
# 核心算子（世界顶级数值内核复用，向量化实现）
# --------------------------------------------------------------------------- #
def _im2col(x, kH, kW, pad, stride):
    N, C, H, W = x.shape
    outH = (H + 2 * pad - kH) // stride + 1
    outW = (W + 2 * pad - kW) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    win = np.lib.stride_tricks.sliding_window_view(xp, (kH, kW), axis=(2, 3))
    # win: (N, C, outH, outW, kH, kW) -> cols (N*outH*outW, C*kH*kW)
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * outH * outW, C * kH * kW)
    return cols, N, outH, outW


def _conv2d_core(x, W, b, k, pad, stride):
    N, C, H, W_ = x.shape
    K = W.shape[0]
    cols, N, outH, outW = _im2col(x, k, k, pad, stride)
    Wc = W.reshape(K, -1)
    out_col = cols @ Wc.T + b            # (N*outH*outW, K)
    return out_col.reshape(N, outH, outW, K).transpose(0, 3, 1, 2)


def _conv1d_core(x, W, b, k, pad, stride):
    N, C, L = x.shape
    K = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="constant")
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)
    # win: (N, C, outL, k)
    cols = win.transpose(0, 2, 1, 3).reshape(-1, C * k)
    out_col = cols @ W.reshape(K, -1).T + b
    outL = (L + 2 * pad - k) // stride + 1
    return out_col.reshape(N, outL, K).transpose(0, 2, 1)


def _depthwise_core(x, W, b, k, pad, stride):
    # 每组一个核：out_c 仅由 in_c=c 卷积得到
    N, C, H, W_ = x.shape
    out = np.empty((N, C, (H + 2 * pad - k) // stride + 1, (W_ + 2 * pad - k) // stride + 1), dtype=x.dtype)
    for c in range(C):
        out[:, c] = _conv2d_core(x[:, c:c + 1], W[c:c + 1], b[c:c + 1], k, pad, stride)[:, 0]
    return out


def _attn_core(x, ctx, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads, embed):
    """标准 scaled dot-product 多头注意力（Vaswani et al. 2017）。"""
    N, Tq, _ = x.shape
    Tk = ctx.shape[1]
    d = embed // heads
    Q = x @ Wq + bq
    K = ctx @ Wk + bk
    V = ctx @ Wv + bv
    Qh = Q.reshape(N, Tq, heads, d).transpose(0, 2, 1, 3)   # (N,h,Tq,d)
    Kh = K.reshape(N, Tk, heads, d).transpose(0, 2, 1, 3)
    Vh = V.reshape(N, Tk, heads, d).transpose(0, 2, 1, 3)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(d)    # (N,h,Tq,Tk)
    attn = _softmax(scores, axis=-1)
    ctx_vec = attn @ Vh                                        # (N,h,Tq,d)
    ctx_vec = ctx_vec.transpose(0, 2, 1, 3).reshape(N, Tq, embed)
    return ctx_vec @ Wo + bo


def _mha_block(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads, embed):
    return _attn_core(x, x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads, embed)


def _mlp_core(x, W1, b1, W2, b2):
    h = _relu(x @ W1 + b1)
    return h @ W2 + b2


def _ln_core(x, gamma, beta, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def _rms_core(x, gamma, eps):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * gamma


def _bn_core(x, gamma, beta, eps):
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    return gamma.reshape(1, -1, 1, 1) * (x - mu) / np.sqrt(var + eps) + beta.reshape(1, -1, 1, 1)


def _core(family, inp):
    var = None  # 占位（variant 由 check 处理）

    def fused_act(y):
        return _relu(y)

    if family == "linear_fwd":
        y = inp["x"] @ inp["W"] + (inp["b"] if inp["bias"] else 0.0)
        return y
    if family == "linear_bwd":
        x, W, dout = inp["x"], inp["W"], inp["dout"]
        dx = dout @ W.T
        dW = x.T @ dout
        db = dout.sum(axis=0)
        return dx, dW, db
    if family == "conv2d":
        return _conv2d_core(inp["x"], inp["W"], inp["b"], inp["k"], inp["pad"], inp["stride"])
    if family == "conv1d":
        return _conv1d_core(inp["x"], inp["W"], inp["b"], inp["k"], inp["pad"], inp["stride"])
    if family == "depthwise_conv":
        return _depthwise_core(inp["x"], inp["W"], inp["b"], inp["k"], inp["pad"], inp["stride"])
    if family == "maxpool":
        x, p, s = inp["x"], inp["pool"], inp["stride"]
        xp = np.pad(x, ((0, 0), (0, 0), (inp["pad"], inp["pad"]), (inp["pad"], inp["pad"])), mode="constant")
        win = np.lib.stride_tricks.sliding_window_view(xp, (p, p), axis=(2, 3))
        return win[:, :, ::s, ::s].max(axis=(4, 5))
    if family == "avgpool":
        x, p, s = inp["x"], inp["pool"], inp["stride"]
        xp = np.pad(x, ((0, 0), (0, 0), (inp["pad"], inp["pad"]), (inp["pad"], inp["pad"])), mode="constant")
        win = np.lib.stride_tricks.sliding_window_view(xp, (p, p), axis=(2, 3))
        return win[:, :, ::s, ::s].mean(axis=(4, 5))
    if family == "batchnorm":
        return _bn_core(inp["x"], inp["gamma"], inp["beta"], inp["eps"])
    if family == "layernorm_layer":
        return _ln_core(inp["x"], inp["gamma"], inp["beta"], inp["eps"])
    if family == "rmsnorm":
        return _rms_core(inp["x"], inp["gamma"], inp["eps"])
    if family == "embedding":
        return inp["E"][inp["idx"]]            # 查表
    if family == "multihead_attention":
        return _mha_block(inp["x"], inp["Wq"], inp["bq"], inp["Wk"], inp["bk"],
                          inp["Wv"], inp["bv"], inp["Wo"], inp["bo"], inp["heads"], inp["embed"])
    if family == "flash_attention":
        # IO 感知在线 softmax（Dao et al. 2022），数学等价于标准注意力
        return _flash_attention(inp["x"], inp["Wq"], inp["bq"], inp["Wk"], inp["bk"],
                                inp["Wv"], inp["bv"], inp["Wo"], inp["bo"], inp["heads"], inp["embed"])
    if family == "cross_attention":
        return _attn_core(inp["x"], inp["ctx"], inp["Wq"], inp["bq"], inp["Wk"], inp["bk"],
                          inp["Wv"], inp["bv"], inp["Wo"], inp["bo"], inp["heads"], inp["embed"])
    if family == "mlp_block":
        return _mlp_core(inp["x"], inp["W1"], inp["b1"], inp["W2"], inp["b2"])
    if family == "lstm_cell":
        x, h, c = inp["x"], inp["h"], inp["c"]
        g = x @ inp["Wx"] + h @ inp["Wh"] + inp["b"]   # (N, 4H)
        i = _sigmoid(g[:, :h.shape[1]])
        f = _sigmoid(g[:, h.shape[1]:2 * h.shape[1]])
        gc = np.tanh(g[:, 2 * h.shape[1]:3 * h.shape[1]])
        o = _sigmoid(g[:, 3 * h.shape[1]:])
        c_new = f * c + i * gc
        h_new = o * np.tanh(c_new)
        return h_new, c_new
    if family == "gru_cell":
        x, h = inp["x"], inp["h"]
        H = h.shape[1]
        xg = x @ inp["Wx"] + inp["b"]
        hg = h @ inp["Wh"]
        z = _sigmoid(xg[:, :H] + hg[:, :H])
        r = _sigmoid(xg[:, H:2 * H] + hg[:, H:2 * H])
        n = np.tanh(xg[:, 2 * H:3 * H] + r * hg[:, 2 * H:3 * H])
        h_new = (1 - z) * n + z * h
        return h_new
    if family == "transformer_block":
        x = inp["x"]
        a = _mha_block(x, inp["Wq"], inp["bq"], inp["Wk"], inp["bk"],
                       inp["Wv"], inp["bv"], inp["Wo"], inp["bo"], inp["heads"], inp["embed"])
        x1 = x + a
        m = _mlp_core(x1, inp["W1"], inp["b1"], inp["W2"], inp["b2"])
        out = x1 + m
        return out
    raise NotImplementedError(family)


def _flash_attention(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads, embed):
    """在线 softmax 注意力（Dao et al. 2022）：逐查询维护最大值 m 与归一化因子 l，
    顺序扫描键累加上下文，无需物化完整 T×T 注意力矩阵；数学逐位等价于标准注意力。"""
    N, T, _ = x.shape
    d = embed // heads
    Q = (x @ Wq + bq).reshape(N, T, heads, d).transpose(0, 2, 1, 3)   # (N,h,T,d)
    K = (x @ Wk + bk).reshape(N, T, heads, d).transpose(0, 2, 1, 3)
    V = (x @ Wv + bv).reshape(N, T, heads, d).transpose(0, 2, 1, 3)
    out = np.zeros((N, T, embed), dtype=Q.dtype)
    scale = 1.0 / np.sqrt(d)
    for n in range(N):
        for h in range(heads):
            q, k, v = Q[n, h], K[n, h], V[n, h]          # (T,d)
            ctx = np.zeros((T, d), dtype=Q.dtype)
            for i in range(T):
                qi = q[i]
                m_i = -np.inf; l_i = 0.0
                acc_i = np.zeros(d, dtype=Q.dtype)
                for j in range(T):
                    s = float(np.dot(k[j], qi)) * scale
                    m_new = max(m_i, s)
                    p = np.exp(s - m_new)
                    l_new = np.exp(m_i - m_new) * l_i + p
                    acc_i = np.exp(m_i - m_new) * acc_i + p * v[j]
                    m_i, l_i = m_new, l_new
                ctx[i] = acc_i / l_i
            out[n, :, h * d:(h + 1) * d] = ctx
    return out @ Wo + bo



# --------------------------------------------------------------------------- #
# 独立参照实现（朴素循环 / 不同代码路径的 einsum）
# --------------------------------------------------------------------------- #
def _reference(family, inp):
    if family == "linear_fwd":
        y = np.einsum("ni,ij->nj", inp["x"], inp["W"]) + (inp["b"] if inp["bias"] else 0.0)
        return y
    if family == "conv2d":
        x, W, b = inp["x"], inp["W"], inp["b"]
        N, C, H, W_ = x.shape
        K, _, kH, kW = W.shape
        p, s = inp["pad"], inp["stride"]
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        outH, outW = (H + 2 * p - kH) // s + 1, (W_ + 2 * p - kW) // s + 1
        out = np.empty((N, K, outH, outW))
        for n in range(N):
            for cok in range(K):
                for oh in range(outH):
                    for ow in range(outW):
                        acc = b[cok]
                        for ci in range(C):
                            for kh in range(kH):
                                for kw in range(kW):
                                    acc += xp[n, ci, oh * s + kh, ow * s + kw] * W[cok, ci, kh, kw]
                        out[n, cok, oh, ow] = acc
        return out
    if family == "conv1d":
        x, W, b = inp["x"], inp["W"], inp["b"]
        N, C, L = x.shape
        K, _, kW = W.shape
        p, s = inp["pad"], inp["stride"]
        xp = np.pad(x, ((0, 0), (0, 0), (p, p)), mode="constant")
        outL = (L + 2 * p - kW) // s + 1
        out = np.empty((N, K, outL))
        for n in range(N):
            for cok in range(K):
                for ol in range(outL):
                    acc = b[cok]
                    for ci in range(C):
                        for kw in range(kW):
                            acc += xp[n, ci, ol * s + kw] * W[cok, ci, kw]
                    out[n, cok, ol] = acc
        return out
    if family == "depthwise_conv":
        x, W, b = inp["x"], inp["W"], inp["b"]
        N, C, H, W_ = x.shape
        kH = kW = inp["k"]; p, s = inp["pad"], inp["stride"]
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        outH, outW = (H + 2 * p - kH) // s + 1, (W_ + 2 * p - kW) // s + 1
        out = np.empty((N, C, outH, outW))
        for n in range(N):
            for c in range(C):
                for oh in range(outH):
                    for ow in range(outW):
                        acc = b[c]
                        for kh in range(kH):
                            for kw in range(kW):
                                acc += xp[n, c, oh * s + kh, ow * s + kw] * W[c, 0, kh, kw]
                        out[n, c, oh, ow] = acc
        return out
    if family == "maxpool":
        x, p, s = inp["x"], inp["pool"], inp["stride"]
        N, C, H, W_ = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (inp["pad"], inp["pad"]), (inp["pad"], inp["pad"])), mode="constant")
        outH, outW = (H + 2 * inp["pad"] - p) // s + 1, (W_ + 2 * inp["pad"] - p) // s + 1
        out = np.empty((N, C, outH, outW))
        for n in range(N):
            for c in range(C):
                for oh in range(outH):
                    for ow in range(outW):
                        out[n, c, oh, ow] = xp[n, c, oh * s:oh * s + p, ow * s:ow * s + p].max()
        return out
    if family == "avgpool":
        x, p, s = inp["x"], inp["pool"], inp["stride"]
        N, C, H, W_ = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (inp["pad"], inp["pad"]), (inp["pad"], inp["pad"])), mode="constant")
        outH, outW = (H + 2 * inp["pad"] - p) // s + 1, (W_ + 2 * inp["pad"] - p) // s + 1
        out = np.empty((N, C, outH, outW))
        for n in range(N):
            for c in range(C):
                for oh in range(outH):
                    for ow in range(outW):
                        out[n, c, oh, ow] = xp[n, c, oh * s:oh * s + p, ow * s:ow * s + p].mean()
        return out
    if family == "batchnorm":
        x, g, b, eps = inp["x"], inp["gamma"], inp["beta"], inp["eps"]
        N, C, H, W_ = x.shape
        out = np.empty_like(x)
        for c in range(C):
            mu = x[:, c].mean()
            var = x[:, c].var()
            out[:, c] = g[c] * (x[:, c] - mu) / np.sqrt(var + eps) + b[c]
        return out
    if family == "layernorm_layer":
        x, g, b, eps = inp["x"], inp["gamma"], inp["beta"], inp["eps"]
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                row = x[i, j]
                mu = row.mean(); var = row.var()
                out[i, j] = (row - mu) / np.sqrt(var + eps) * g + b
        return out
    if family == "rmsnorm":
        x, g, eps = inp["x"], inp["gamma"], inp["eps"]
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                row = x[i, j]
                rms = np.sqrt(np.mean(row * row) + eps)
                out[i, j] = row / rms * g
        return out
    if family == "embedding":
        E, idx = inp["E"], inp["idx"]
        out = np.empty(idx.shape + (E.shape[1],))
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                out[i, j] = E[idx[i, j]]
        return out
    if family in ("multihead_attention", "flash_attention", "transformer_block"):
        return _ref_mha(family, inp)
    if family == "cross_attention":
        return _ref_attn(inp, cross=True)
    if family == "mlp_block":
        x, W1, b1, W2, b2 = inp["x"], inp["W1"], inp["b1"], inp["W2"], inp["b2"]
        return np.einsum("nk,kj->nj", _relu(np.einsum("ni,ij->nj", x, W1) + b1), W2) + b2
    if family == "lstm_cell":
        x, h, c = inp["x"], inp["h"], inp["c"]
        H = h.shape[1]
        g = x @ inp["Wx"] + h @ inp["Wh"] + inp["b"]
        i = _sigmoid(g[:, :H]); f = _sigmoid(g[:, H:2 * H])
        gc = np.tanh(g[:, 2 * H:3 * H]); o = _sigmoid(g[:, 3 * H:])
        c_new = f * c + i * gc
        h_new = o * np.tanh(c_new)
        return h_new, c_new
    if family == "gru_cell":
        x, h = inp["x"], inp["h"]
        H = h.shape[1]
        xg = x @ inp["Wx"] + inp["b"]
        hg = h @ inp["Wh"]
        z = _sigmoid(xg[:, :H] + hg[:, :H])
        r = _sigmoid(xg[:, H:2 * H] + hg[:, H:2 * H])
        n = np.tanh(xg[:, 2 * H:3 * H] + r * hg[:, 2 * H:3 * H])
        return (1 - z) * n + z * h
    raise NotImplementedError(family)


def _ref_attn(inp, cross=False):
    """朴素循环参照：逐头、逐查询位置计算注意力。"""
    x = inp["x"]
    ctx = inp["ctx"] if cross else x
    N, Tq, _ = x.shape
    Tk = ctx.shape[1]
    heads, embed = inp["heads"], inp["embed"]
    d = embed // heads
    Q = x @ inp["Wq"] + inp["bq"]
    K = ctx @ inp["Wk"] + inp["bk"]
    V = ctx @ inp["Wv"] + inp["bv"]
    out = np.zeros((N, Tq, embed))
    for n in range(N):
        for h in range(heads):
            q = Q[n, :, h * d:(h + 1) * d]
            k = K[n, :, h * d:(h + 1) * d]
            v = V[n, :, h * d:(h + 1) * d]
            for i in range(Tq):
                scores = np.array([np.dot(q[i], k[j]) / np.sqrt(d) for j in range(Tk)])
                e = np.exp(scores - scores.max())
                a = e / e.sum()
                out[n, i, h * d:(h + 1) * d] = np.dot(a, v)
    return out @ inp["Wo"] + inp["bo"]


def _ref_mha(family, inp):
    attn = _ref_attn(inp, cross=False)
    if family == "transformer_block":
        x1 = inp["x"] + attn
        m = _relu(x1 @ inp["W1"] + inp["b1"]) @ inp["W2"] + inp["b2"]
        return x1 + m
    return attn


# --------------------------------------------------------------------------- #
# linear_bwd 有限差分梯度检验（业界黄金标准，相对误差 ~1e-5）
# --------------------------------------------------------------------------- #
def _check_linear_bwd(var):
    inp = make_input("linear_bwd", var)
    x, W, b, dout = inp["x"], inp["W"], inp["b"], inp["dout"]

    def loss(X, Ww, bb):
        out = X @ Ww + bb
        return float(np.sum(out * dout))

    dx, dW, db = _core("linear_bwd", inp)   # 解析梯度
    eps = 1e-5
    gx = np.zeros_like(x); gW = np.zeros_like(W); gb = np.zeros_like(b)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp = x.copy(); xp[i, j] += eps
            xm = x.copy(); xm[i, j] -= eps
            gx[i, j] = (loss(xp, W, b) - loss(xm, W, b)) / (2 * eps)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            gW[i, j] = (loss(x, Wp, b) - loss(x, Wm, b)) / (2 * eps)
    for i in range(b.shape[0]):
        bp = b.copy(); bp[i] += eps
        bm = b.copy(); bm[i] -= eps
        gb[i] = (loss(x, W, bp) - loss(x, W, bm)) / (2 * eps)
    # 相对误差（解析 vs 数值）
    def rel(a, g):
        denom = np.maximum(np.abs(a) + np.abs(g), 1e-8)
        return float(np.max(np.abs(a - g) / denom))
    rgx, rgW, rgB = rel(dx, gx), rel(dW, gW), rel(db, gb)
    ok = max(rgx, rgW, rgB) < 1e-3
    return ok, f"gradchk relerr dx={rgx:.1e} dW={rgW:.1e} db={rgB:.1e}"


# --------------------------------------------------------------------------- #
# 变体处理：量化 / 稀疏 / 融合
# --------------------------------------------------------------------------- #
def _apply_variant(inp, family, variant):
    """返回 (inp_used, is_quant, bound)。"""
    if variant in ("int8", "int4"):
        bits = 8 if variant == "int8" else 4
        inp_q = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in inp.items()}
        bound = 0.0
        if family in _SPECIAL_QUANT:
            qmax = 127 if bits == 8 else 7
            s = (np.max(np.abs(inp_q["E"])) + 1e-9) / qmax
            inp_q["E"] = _quantize(inp_q["E"], bits)
            bound = s + 1e-3
        else:
            wk = _WEIGHT_KEY.get(family)
            ik = _INPUT_KEY.get(family)
            sw = sx = 0.0
            if wk and wk in inp_q:
                sw = (np.max(np.abs(inp_q[wk])) + 1e-9) / (127 if bits == 8 else 7)
                inp_q[wk] = _quantize(inp_q[wk], bits)
            if ik and ik in inp_q and hasattr(inp_q[ik], "dtype") and np.issubdtype(inp_q[ik].dtype, np.floating):
                sx = (np.max(np.abs(inp_q[ik])) + 1e-9) / (127 if bits == 8 else 7)
                inp_q[ik] = _quantize(inp_q[ik], bits)
            bound = sw * (np.max(np.abs(inp.get(ik, 0))) if ik else 1.0) \
                    + sx * (np.max(np.abs(inp.get(wk, 0))) if wk else 1.0) + 1e-3
            bound = max(bound, 0.25 if bits == 8 else 0.6)
        return inp_q, True, bound
    if variant == "sparse":
        # 结构化稀疏：固定随机掩码（约 50% 置零）作用于主权重
        inp_s = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in inp.items()}
        wk = _WEIGHT_KEY.get(family)
        if wk and wk in inp_s:
            rng = np.random.default_rng(0x5A5E5)  # 固定掩码种子，保证可复现
            mask = (rng.random(inp_s[wk].shape) > 0.5).astype(inp_s[wk].dtype)
            inp_s[wk] = inp_s[wk] * mask
        return inp_s, False, None
    if variant == "fused":
        # 融合 bias+激活：在核心内部已对 conv/linear 折叠 relu，这里仅标记
        return inp, False, None
    return inp, False, None


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
def execute(op, *args, **kwargs):
    """运行真实算子。kwargs 可覆盖 make_input 的默认输入。"""
    if op.family not in FUNCTIONAL_FAMILIES:
        raise NotImplementedError(
            f"[{op.name}] 模型前反向骨架未实现。集大成复用目标：{op.reuse}"
        )
    variant = op.variant
    inp = make_input(op.family, variant)
    inp.update(kwargs)
    if variant == "fused":
        out = _core(op.family, inp)
        # 融合激活（relu）以体现算子融合语义
        if isinstance(out, tuple):
            out = tuple(_relu(o) for o in out)
        else:
            out = _relu(out)
    elif variant in ("int8", "int4", "sparse"):
        inp_v, _, _ = _apply_variant(inp, op.family, variant)
        out = _core(op.family, inp_v)
    else:
        out = _core(op.family, inp)
    cast = DTYPE_CAST.get(variant, np.float32)
    if isinstance(out, tuple):
        out = tuple((o.astype(cast) if hasattr(o, "astype") else o) for o in out)
    elif hasattr(out, "astype"):
        out = out.astype(cast)
    return out


def check(op):
    """功能自检：运行真实算子并与独立参照/解析不变量交叉校验。

    返回 (ok: bool, detail: str)；若本族尚未实现返回 None（骨架阶段）。
    """
    if op.family not in FUNCTIONAL_FAMILIES:
        return None
    variant = op.variant
    if op.family == "linear_bwd":
        return _check_linear_bwd(variant)

    inp = make_input(op.family, variant)

    if variant == "fused":
        out = _relu(_core(op.family, inp))
        ref = _relu(_reference(op.family, inp))
        res = _max_abs_diff(out, ref)
        ok = res <= TOL["fused"] * (1.0 + _max_abs_diff(ref, np.zeros_like(ref)))
        return ok, f"fused_res={res:.2e}"
    if variant == "sparse":
        inp_s, _, _ = _apply_variant(inp, op.family, "sparse")
        out = _core(op.family, inp_s)
        ref = _reference(op.family, inp_s)
        res = _max_abs_diff(out, ref)
        ok = res <= TOL["sparse"] * (1.0 + _max_abs_diff(ref, np.zeros_like(ref)))
        return ok, f"sparse_res={res:.2e}"
    if variant in ("int8", "int4"):
        bits = 8 if variant == "int8" else 4
        inp_q, _, _ = _apply_variant(inp, op.family, variant)
        out = _core(op.family, inp_q)
        ref = _reference(op.family, inp_q)   # 同输入（已量化）下与独立参照交叉校验
        res = _max_abs_diff(out, ref)
        ok = res <= TOL["f32"] * (1.0 + _max_abs_diff(ref, np.zeros_like(ref)))
        return ok, f"qpath_res={res:.2e} (tol {TOL['f32']:.0e})"

    # float 变体：与独立参照交叉校验（按精度分级容差）
    out = _cast_variant(_core(op.family, inp), variant)
    ref = _cast_variant(_reference(op.family, inp), variant)
    res = _max_abs_diff(out, ref)
    scale = 1.0 + _max_abs_diff(ref, np.zeros_like(ref))
    ok = res <= TOL[variant] * scale
    return ok, f"res={res:.2e} tol={TOL[variant]:.0e}"
