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


def covers(category, family):
    return family in COVERAGE.get(category, set())
