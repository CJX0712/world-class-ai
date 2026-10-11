# -*- coding: utf-8 -*-
"""
handlers — 第十层各分类（模块）处理器的聚合与覆盖查询。

每个分类对应一个轻量 stub 模块（math/optim/forward/.../system），其 execute()
当前抛出 NotImplementedError，作为该操作族「集大成」实现的落脚点。覆盖信息从
catalog 派生，保证与单一事实源一致。
作者：晨星
"""
import os

from . import (math, optim, forward, training, inference,
               tensor, data, runtime, control, system)

_CAT_MODULES = {
    "math": math,
    "optim": optim,
    "forward": forward,
    "training": training,
    "inference": inference,
    "tensor": tensor,
    "data": data,
    "runtime": runtime,
    "control": control,
    "system": system,
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_CATALOG_PATH = os.path.join(_HERE, "..", "data", "micro_instructions.json")


def _load_catalog():
    import json
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


CATALOG = _load_catalog()

# category -> set(family) 覆盖表，源自单一事实源
COVERAGE = {}
for _e in CATALOG["micro_instructions"]:
    COVERAGE.setdefault(_e["category"], set()).add(_e["family"])


def get_handler(category):
    return _CAT_MODULES[category]


def execute(op, *args, **kwargs):
    """运行分发：委派给对应分类处理器模块的 execute（已实现族跑真实算子）。"""
    mod = _CAT_MODULES.get(op.category)
    if mod is None:
        raise NotImplementedError(
            f"[{op.name}] 无对应分类处理器。集大成复用目标：{op.reuse}"
        )
    return mod.execute(op, *args, **kwargs)


def covers(category, family):
    return family in COVERAGE.get(category, set())


def functional_check(op):
    """功能自检分发：若对应分类处理器已实现 check，则运行交叉校验。

    返回 (ok, detail) 或 None（该族仍为骨架阶段）。
    """
    mod = _CAT_MODULES.get(op.category)
    if mod is None:
        return None
    fn = getattr(mod, "check", None)
    if fn is None:
        return None
    return fn(op)
