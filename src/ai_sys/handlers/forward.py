# -*- coding: utf-8 -*-
"""M2 模型前反向处理器（骨架）。集大成复用：PyTorch nn / Megatron-LM / FlashAttention。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 模型前反向骨架未实现。集大成复用目标：{op.reuse}"
    )
