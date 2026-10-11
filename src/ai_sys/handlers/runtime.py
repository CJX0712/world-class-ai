# -*- coding: utf-8 -*-
"""M7 运行时调度处理器（骨架）。集大成复用：Ray / asyncio / tcmalloc / mimalloc。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 运行时调度骨架未实现。集大成复用目标：{op.reuse}"
    )
