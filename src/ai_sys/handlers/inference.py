# -*- coding: utf-8 -*-
"""M4 推理服务处理器（骨架）。集大成复用：vLLM / TensorRT-LLM / llama.cpp / SGLang。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 推理服务骨架未实现。集大成复用目标：{op.reuse}"
    )
