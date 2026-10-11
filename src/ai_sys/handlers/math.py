# -*- coding: utf-8 -*-
"""M0 数学算子处理器（骨架）。集大成复用：cuBLAS / PyTorch ATen / JAX XLA / Eigen。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 数学算子骨架未实现。集大成复用目标：{op.reuse}"
    )
