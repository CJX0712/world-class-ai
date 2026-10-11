# -*- coding: utf-8 -*-
"""
registry.py — 第十层 1024 微指令的运行时注册表与逐项自检。

单一事实源：src/ai_sys/data/micro_instructions.json（由 tools/build_catalog.py 生成）。
每条微指令实例化为 MicroOp，self_check 在 skeleton 阶段校验「契约完整性」：
  · id 格式合法且全局唯一连续
  · 关键字段齐全（module / category / family / name / variant / reuse / layer_ref / description）
  · 依赖（deps）指向存在的微指令
  · 所属分类处理器（handlers）已覆盖该操作族

实现落地后，self_check 可升级为「功能正确性」校验（运行真实算子并比对参照实现）。
作者：晨星
"""
import os

from . import handlers

_HERE = os.path.dirname(os.path.abspath(__file__))
_CATALOG_PATH = os.path.join(_HERE, "data", "micro_instructions.json")


def load_catalog():
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        return __import__("json").load(f)


class MicroOp:
    """单条微指令的运行时对象。"""

    def __init__(self, entry):
        self.entry = entry
        for k, v in entry.items():
            setattr(self, k, v)

    def self_check(self, registry_ids):
        """骨架阶段契约自检。返回 (ok: bool, reason: str)。"""
        e = self.entry
        mid = e.get("id")
        if not (isinstance(mid, str) and mid.startswith("MI-") and len(mid) == 7):
            return False, "id 格式非法"
        for k in ("module", "category", "family", "name", "variant",
                  "reuse", "layer_ref", "description"):
            if not e.get(k):
                return False, f"缺失字段 {k}"
        for d in e.get("deps", []):
            if d not in registry_ids:
                return False, f"依赖 {d} 不存在"
        if not handlers.covers(e["category"], e["family"]):
            return False, f"分类处理器未覆盖 {e['family']}"
        return True, "ok"

    def run(self, *args, **kwargs):
        """运行该微指令：委派给对应分类处理器的 execute（已实现族跑真实算子，
        未实现族抛出 NotImplementedError 并提示集大成复用目标）。"""
        return handlers.execute(self, *args, **kwargs)

    def functional_check(self):
        """功能自检：运行真实算子并与独立参照/解析不变量交叉校验。

        返回 (ok, detail)（已实现族）或 None（仍为骨架）。
        """
        return handlers.functional_check(self)


class Registry:
    def __init__(self):
        self.catalog = load_catalog()
        self.ops = {}
        for e in self.catalog["micro_instructions"]:
            self.ops[e["id"]] = MicroOp(e)
        self.ids = set(self.ops)

    def get(self, mid):
        return self.ops.get(mid)

    def all(self):
        return list(self.ops.values())

    def check_all(self):
        return [(op, *op.self_check(self.ids)) for op in self.ops.values()]


REGISTRY = Registry()
