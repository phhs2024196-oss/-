#!/usr/bin/env python3
"""SYU-M24G (2024학년도 삼육대학교 논술가이드 모의문제 수학 1~4) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "SYU-M24G"
WIDTH = 2
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"
RESULTS = []
x = sp.symbols("x", real=True)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub="1"):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    for s in d["subparts"]:
        if str(s["id"]) == sub:
            return sp.sympify(str(s["answer"]["value"]))
    raise KeyError(sub)


def p1():
    f = x**2 - 5 * x - 1
    sols = [k for k in range(-100, 101) if sp.Integer(9) ** f.subs(x, k) - sp.Integer(3) ** (5 + f.subs(x, k)) < 0]
    report(1, len(sols) == load_answer(1), f"정수해 {sols} → {len(sols)}개(직접 대입 전수)")


def p2():
    tot = 0
    for n in range(1, 16):
        z = sp.Symbol("z")
        r = sp.roots(sp.Poly(z**2 - n * z + n + 3, z))
        rs = [k for k, mult in r.items() for _ in range(mult)]
        tot += sp.expand(rs[0] ** 2 + rs[1] ** 2)
    tot = sp.nsimplify(sp.simplify(tot))
    report(2, tot == load_answer(2), f"두 근(허근 포함)을 직접 구해 제곱합 누적 = {tot}")


def p3():
    cs = sp.symbols("c0:5")
    f = sum(cs[i] * x**i for i in range(5))
    eqs = [sp.Eq(f.subs(x, 3), 0)]                         # (1) 분모→0 이므로 분자→0
    # (2) 분자 f(x)-x+3 이 (x-2)^2 로 나뉘어야 함: 값·미분 0
    h = f - x + 3
    eqs += [sp.Eq(h.subs(x, 2), 0), sp.Eq(sp.diff(h, x).subs(x, 2), 0)]
    sol = sp.solve(eqs, [cs[0], cs[1], cs[2]], dict=True)[0]
    F = f.subs(sol)
    e1 = sp.Eq(sp.limit(x * F / (x - 3), x, 3), 6)
    e2 = sp.Eq(sp.limit((F - x + 3) / (x - 2) ** 2, x, 2), 2)
    s2 = sp.solve([e1, e2], [cs[3], cs[4]], dict=True)
    assert len(s2) == 1 and s2[0][cs[4]] != 0             # 사차함수
    F = sp.expand(F.subs(s2[0]))
    assert sp.limit(x * F / (x - 3), x, 3) == 6 and sp.limit((F - x + 3) / (x - 2) ** 2, x, 2) == 2
    val = F.subs(x, 1)
    report(3, val == load_answer(3), f"f={F}, f(1)={val}")


def p4():
    q = x**2 + 2 * x - 3                                   # (가) 의 절댓값 안
    line = x + 3
    total = 0
    for k in range(-1, 5):                                 # 조각 [-3+5k, 2+5k): f = |q(x-5k)| + 5k  ((나) 반복 적용)
        lo, hi = max(-3 + 5 * k, 0), min(2 + 5 * k, 20)
        if lo >= hi:
            continue
        inner = sp.expand(q.subs(x, x - 5 * k))
        cuts = {sp.Integer(lo), sp.Integer(hi)}
        for poly in (inner, sp.expand(inner - (line - 5 * k)), sp.expand(-inner - (line - 5 * k))):
            cuts |= {r for r in sp.solve(poly, x) if r.is_real and lo < r < hi}
        cuts = sorted(cuts)
        for u, v in zip(cuts, cuts[1:]):
            mid = (u + v) / 2
            fk = (inner if inner.subs(x, mid) >= 0 else -inner) + 5 * k
            diff = fk - line
            total += sp.integrate(diff if diff.subs(x, mid) >= 0 else -diff, (x, u, v))
    total = sp.nsimplify(total)
    import mpmath as mp
    def fnum(t):
        kk = int(mp.floor((t + 3) / 5))
        return abs(abs((t - 5 * kk) ** 2 + 2 * (t - 5 * kk) - 3) + 5 * kk - (t + 3))
    num = mp.quad(fnum, [0, 0.5, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20])
    report(4, total == load_answer(4) and abs(num - float(total)) < 1e-9, f"구간별 기호 적분 {total}, 수치 적분 {float(num):.10f}")


if __name__ == "__main__":
    fns = sorted((n for n in dir() if re.fullmatch(r"p\d+", n)), key=lambda s: int(s[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
