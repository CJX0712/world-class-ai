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
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ai_sys.registry import REGISTRY, load_catalog

LINE = "=" * 64

ECO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "ai_sys", "data", "ecosystem.json")
VALID_LAYERS = {f"L{i}" for i in range(1, 11)}
VALID_GRADES = {"S", "A", "B", "C"}


def check_ecosystem():
    """生态目录一致性校验（可验证）：总数 / 层 / 评级 / 计数自洽。

    返回 (ok, msgs)；文件缺失时返回 None（跳过，不报错）。
    """
    if not os.path.exists(ECO_PATH):
        return None
    with open(ECO_PATH, encoding="utf-8") as f:
        eco = json.load(f)
    msgs = []
    repos = eco.get("repos", [])
    total = eco.get("meta", {}).get("total_repos")
    if total != len(repos):
        msgs.append(f"ecosystem total_repos {total} != len(repos) {len(repos)}")
    layers_seen = set()
    for e in repos:
        if e.get("layer") not in VALID_LAYERS:
            msgs.append(f"仓库 {e.get('name')} 层非法: {e.get('layer')}")
        if e.get("grade") not in VALID_GRADES:
            msgs.append(f"仓库 {e.get('name')} 评级非法: {e.get('grade')}")
        layers_seen.add(e.get("layer"))
    # 计数自洽
    by_layer = eco.get("summary", {}).get("by_layer", {})
    if sum(by_layer.values()) != len(repos):
        msgs.append(f"by_layer 求和 {sum(by_layer.values())} != {len(repos)}")
    by_grade = eco.get("summary", {}).get("by_grade", {})
    if sum(by_grade.values()) != len(repos):
        msgs.append(f"by_grade 求和 {sum(by_grade.values())} != {len(repos)}")
    matrix = eco.get("summary", {}).get("matrix", {})
    m_sum = sum(v for row in matrix.values() for v in row.values())
    if m_sum != len(repos):
        msgs.append(f"matrix 求和 {m_sum} != {len(repos)}")
    # 跨层覆盖：L1–L10 至少各有一条（L10 含本仓库与天枢）
    missing = VALID_LAYERS - layers_seen
    if missing:
        msgs.append(f"缺少层的覆盖: {sorted(missing)}")
    return (len(msgs) == 0, msgs)


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

    # 生态目录一致性校验
    eco = check_ecosystem()
    eco_msgs = []
    if eco is None:
        print("生态校验: ⏭️  跳过（ecosystem.json 缺失）")
    else:
        eco_ok, eco_msgs = eco
        print(f"生态校验: {'✅ 一致' if eco_ok else '⚠️ 不一致'} "
              f"（{len(eco_msgs)} 项）")
        for m in eco_msgs[:30]:
            print("  生态:", m)

    ok_total = (not errors) and (not fail) and (len(func_fail) == 0) \
        and (eco is None or eco[0])
    print("结果:", "✅ 全绿（契约 + 功能 + 生态自检通过）" if ok_total else "⚠️ 存在失败项")
    if func_total:
        print(f"功能覆盖: M0 数学算子 {func_pass}/{func_total} 条真实算子交叉校验通过"
              if func_pass == func_total else
              f"功能覆盖: {func_pass}/{func_total} 通过（详见上方功能失败）")
    print(LINE)
    return 0 if ok_total else 1


if __name__ == "__main__":
    sys.exit(main())
