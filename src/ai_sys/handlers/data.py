# -*- coding: utf-8 -*-
"""M6 数据管道处理器（骨架）。集大成复用：Tokenizers(Rust) / FAISS / Pandas / pyarrow。"""


def execute(op, *args, **kwargs):
    raise NotImplementedError(
        f"[{op.name}] 数据管道骨架未实现。集大成复用目标：{op.reuse}"
    )
