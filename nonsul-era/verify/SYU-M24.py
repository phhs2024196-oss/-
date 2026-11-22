#!/usr/bin/env python3
"""SYU-M24 (2024학년도 삼육대 모의논술 수학 5~8번) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "SYU-M24"
WIDTH = 2
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"
RESULTS = []
x, m = sp.symbols("x m", real=True)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub="1"):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    for s in d["subparts"]:
        if str(s["id"]) == sub:
            return sp.sympify(str(s["answer"]["value"]))
    raise KeyError(sub)


def p5():
    a0, a1 = sp.symbols("a0 a1")
    # 관계식을 a_{n+1}에 대한 이차방정식으로 풀기 → 근
    roots = sp.solve(a1**2 + 9 * a0**2 - 6 * a0 * a1, a1)
    assert roots == [3 * a0]
    r = sp.Rational(3)
    A1 = sp.solve(sp.Eq(sum(sp.Symbol("c") * r**k for k in range(5)), 121), sp.Symbol("c"))[0]
    ms = [k for k in range(1, 100) if A1 * r ** (k - 1) < 300]
    val = sum(ms)
    report(5, val == load_answer(5), f"a1={A1}, m={ms}, 합={val}")


def p6():
    a, b, c, d = sp.symbols("a b c d")
    # 다항함수 차수 탐색: (a)에서 deg 2f-x^3 = 2 → f 는 삼차. 일반 삼차로 두고 조건 대입
    f = a * x**3 + b * x**2 + c * x + d
    eqs = []
    expr = sp.expand(2 * f - x**3)
    eqs += [sp.Eq(expr.coeff(x, 3), 0), sp.Eq(expr.coeff(x, 2), 2)]
    eqs += [sp.Eq(f.subs(x, 0), 0)]
    sol = sp.solve(eqs, [a, b, d], dict=True)[0]
    g = f.subs(sol)
    csol = sp.solve(sp.Eq(sp.limit(g / x, x, 0), 1), c)[0]
    F = g.subs(c, csol)
    assert sp.limit((2 * F - x**3) / (2 * x**2), x, sp.oo) == 1 and sp.limit(F / x, x, 0) == 1
    val = F.subs(x, 1)
    report(6, sp.simplify(val - load_answer(6)) == 0, f"f={F}, f(1)={val}")


def p7():
    a, b = sp.symbols("a b", real=True)
    f = 2 * a - sp.sqrt(3) * sp.tan(x)
    # 감소 확인
    assert sp.simplify(sp.diff(f, x) + sp.sqrt(3) * sp.sec(x) ** 2) == 0
    asol = sp.solve(sp.Eq(f.subs(x, -sp.pi / 6), 11), a)[0]
    bsols = [s for s in sp.solve(sp.Eq(f.subs({a: asol, x: b}), 7), b) if -sp.pi / 6 < s < sp.pi / 2]
    assert len(bsols) == 1
    bb = bsols[0]
    # 수치로 구간 최대·최소 재확인
    import numpy as np
    fx = sp.lambdify(x, f.subs(a, asol), "numpy")
    xs = np.linspace(float(-sp.pi / 6), float(bb), 20001)
    ok_num = abs(fx(xs).max() - 11) < 1e-9 and abs(fx(xs).min() - 7) < 1e-9
    val = sp.simplify(asol * bb)
    report(7, ok_num and sp.simplify(val - load_answer(7)) == 0, f"a={asol}, b={bb}, ab={val}")


def p8():
    a = sp.symbols("a", real=True)
    curve = x**3 - 3 * x**2 - 3
    found = []
    for av in range(-50, 51):
        if curve.subs(x, av) == -3:
            continue  # 곡선 위의 점 제외
        eq = sp.expand(-3 - (sp.diff(curve, x).subs(x, m) * (av - m) + curve.subs(x, m)))
        real_roots = set(sp.Poly(eq, m).real_roots()) if sp.Poly(eq, m).degree() > 0 else set()
        if len(real_roots) == 1:
            found.append(av)
    val = sum(found)
    report(8, val == load_answer(8), f"정수 a={found}, 합={val}")


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
