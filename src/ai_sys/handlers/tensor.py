# -*- coding: utf-8 -*-
"""M5 张量算子处理器（骨架）。集大成复用：PyTorch tensor / NumPy / XLA / Arrow。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 张量算子骨架未实现。集大成复用目标：{op.reuse}"
    )
