#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py — 第十层 1024 微指令 闭环验证器。

校验两项：
  1) 结构完整性：总数 == 1024、ID 唯一且连续 MI-0001..MI-1024。
  2) 逐项契约自检：每条 MicroOp 通过 registry.MicroOp.self_check。

退出码 0 = 全绿（skeleton 阶段），1 = 存在失败项。
零依赖（仅 Python 标准库），可离线、可复现运行。
作者：晨星
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ai_sys.registry import REGISTRY, load_catalog

LINE = "=" * 64


def main():
    cat = load_catalog()
    reg = REGISTRY
    ops = reg.ops

    errors = []
    if len(ops) != 1024:
        errors.append(f"微指令总数 {len(ops)} != 1024")
    expected = [f"MI-{i:04d}" for i in range(1, 1025)]
    if sorted(ops.keys()) != expected:
        errors.append("ID 非唯一或非连续 MI-0001..MI-1024")

    results = reg.check_all()
    fail = [(op.id, reason) for op, ok, reason in results if not ok]

    # 模块级汇总
    by_mod = defaultdict(lambda: [0, 0])
    for op, ok, _ in results:
        by_mod[op.module][0] += 1
        by_mod[op.module][1] += 1 if ok else 0

    print(LINE)
    print("世界顶级 AI 系统 · 第十层 1024 微指令 自检")
    print(LINE)
    print(f"总数: {len(ops)}   结构错误: {len(errors)}   逐项自检失败: {len(fail)}")
    print("-" * 64)
    print(f"{'模块':<5}{'名称':<10}{'条目':>6}{'通过':>6}  状态")
    for m in cat["meta"]["modules"]:
        k = m["key"]
        tot, ok_ = by_mod[k]
        state = "✅" if ok_ == tot else "⚠️"
        print(f"{k:<5}{m['name']:<10}{tot:>6}{ok_:>6}  {state}")
    print("-" * 64)

    if errors:
        for e in errors:
            print("结构错误:", e)
    if fail:
        for i, r in fail[:30]:
            print("失败:", i, r)

    ok_total = (not errors) and (not fail)
    print("结果:", "✅ 全绿（skeleton 阶段，逐项契约自检通过）" if ok_total else "⚠️ 存在失败项")
    print(LINE)
    return 0 if ok_total else 1


if __name__ == "__main__":
    sys.exit(main())
