#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py — 第十层 1024 微指令 闭环验证器。

校验三项：
  1) 结构完整性：总数 == 1024、ID 唯一且连续 MI-0001..MI-1024。
  2) 逐项契约自检：每条 MicroOp 通过 registry.MicroOp.self_check。
  3) 功能正确性：已实现族（status=implemented）运行真实算子，并与独立参照 /
     解析不变量交叉校验（residual → 机器精度级）。

退出码 0 = 全绿（契约 + 功能均通过），1 = 存在失败项。
零依赖（仅 Python 标准库 + 数值内核 NumPy），可离线、可复现运行。
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

    # 功能自检（已实现族）
    func_total = 0
    func_pass = 0
    func_fail = []
    func_by_mod = defaultdict(lambda: [0, 0])
    for op in reg.all():
        fc = op.functional_check()
        if fc is None:
            continue  # 骨架阶段，跳过
        func_total += 1
        ok, detail = fc
        func_by_mod[op.module][0] += 1
        if ok:
            func_pass += 1
            func_by_mod[op.module][1] += 1
        else:
            func_fail.append((op.id, op.name, detail))

    print(LINE)
    print("世界顶级 AI 系统 · 第十层 1024 微指令 自检")
    print(LINE)
    print(f"总数: {len(ops)}   结构错误: {len(errors)}   契约自检失败: {len(fail)}")
    print(f"功能自检: 已实现 {func_total} 条   通过 {func_pass}   失败 {len(func_fail)}")
    print("-" * 64)
    print(f"{'模块':<5}{'名称':<10}{'条目':>6}{'契约':>6}{'功能':>7}  状态")
    for m in cat["meta"]["modules"]:
        k = m["key"]
        tot, ok_ = by_mod[k]
        ft, fp = func_by_mod[k]
        state = "✅" if (ok_ == tot and (ft == 0 or fp == ft)) else "⚠️"
        fstr = f"{fp}/{ft}" if ft else "—"
        print(f"{k:<5}{m['name']:<10}{tot:>6}{ok_:>6}{fstr:>7}  {state}")
    print("-" * 64)

    if errors:
        for e in errors:
            print("结构错误:", e)
    if fail:
        for i, r in fail[:30]:
            print("契约失败:", i, r)
    if func_fail:
        for i, n, d in func_fail[:30]:
            print("功能失败:", i, n, d)

    ok_total = (not errors) and (not fail) and (len(func_fail) == 0)
    print("结果:", "✅ 全绿（契约 + 功能自检通过）" if ok_total else "⚠️ 存在失败项")
    if func_total:
        print(f"功能覆盖: M0 数学算子 {func_pass}/{func_total} 条真实算子交叉校验通过"
              if func_pass == func_total else
              f"功能覆盖: {func_pass}/{func_total} 通过（详见上方功能失败）")
    print(LINE)
    return 0 if ok_total else 1


if __name__ == "__main__":
    sys.exit(main())
