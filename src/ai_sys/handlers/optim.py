# -*- coding: utf-8 -*-
"""M1 优化与采样处理器（骨架）。集大成复用：PyTorch Optim / DeepSpeed / Lion / Adafactor。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 优化采样骨架未实现。集大成复用目标：{op.reuse}"
    )
