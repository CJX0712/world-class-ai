# -*- coding: utf-8 -*-
"""M3 训练循环处理器（骨架）。集大成复用：PyTorch DDP / DeepSpeed ZeRO / Megatron。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 训练循环骨架未实现。集大成复用目标：{op.reuse}"
    )
