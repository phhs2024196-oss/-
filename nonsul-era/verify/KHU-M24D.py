#!/usr/bin/env python3
"""KHU-M24D (경희대 2024학년도) — sympy 검산
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "KHU-M24D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    b = sp.symbols("b", positive=True)
    X = sp.symbols("X", positive=True)
    xq = sp.solve(sp.Eq(X + b, 1 / X), X)[0]
    B = sp.integrate(X + b, (X, -b, xq)) + sp.integrate(1 / X, (X, xq, 1 / b))
    t = (b + sp.sqrt(b**2 + 4)) / 2
    T, s = sp.symbols("T s", positive=True)
    f = T**2 / 2 + sp.log(T)
    ok1 = all(abs(sp.N((B + sp.log(b) - f.subs(T, t)).subs(b, bv), 30)) < 1e-20 for bv in [sp.Rational(1, 3), 1, 2, 7])
    L1 = sp.limit(f, T, 1, "+")
    g = sp.Rational(1, 2) + 1 / (2 * s) + sp.log(s)
    sv = t / b
    ok2 = all(abs(sp.N((B - b**2 / 2 - g.subs(s, sv)).subs(b, bv), 30)) < 1e-20 for bv in [sp.Rational(1, 3), 1, 2, 7])
    L2 = sp.limit(g, s, 1, "+")
    crit = sp.solve(sp.diff(g, s), s)
    cands = [g.subs(s, 2), g.subs(s, 3)] + [g.subs(s, cv) for cv in crit if 2 <= cv <= 3]
    gmax = max(cands, key=lambda e: float(e))
    ok = ok1 and ok2 and L1 == sp.Rational(1, 2) and L2 == 1 and sp.simplify(gmax - (sp.Rational(2, 3) + sp.log(3))) == 0
    report(1, ok, f"(1) f(t)=t^2/2+ln t (b 4값 대입 일치), lim={L1}  (2) g(s) 일치, lim={L2}, 최댓값={gmax}")


def p02():
    import math, random
    k = sp.symbols("k", real=True)
    # OA,OB,OC 단위벡터, (k/2+2)OA = OB+3OC 에서 OB·OC = ((k+4)^2/4 - 10)/6
    dot = ((k + 4)**2 / 4 - 1 - 9) / 6
    kmax = max([sp.Integer(-3), sp.Integer(3)] + sp.solve(sp.diff(dot, k), k), key=lambda v: float(dot.subs(k, v)) if -3 <= v <= 3 else -99)
    dmax = dot.subs(k, kmax)
    # 수치: B, C 를 원 위에서 돌려 |OB+3OC|/(k/2+2)=1 이 되는 k∈[-3,3] 중 OB·OC 최대
    best = -9
    for i in range(20000):
        th = 2 * math.pi * i / 20000
        n = math.sqrt((1 + 3 * math.cos(th))**2 + (3 * math.sin(th))**2)
        kk = 2 * n - 4
        if -3 <= kk <= 3:
            best = max(best, math.cos(th))
    ok1 = dmax == sp.Rational(3, 8) and abs(best - 3 / 8) < 1e-3
    # (2) 좌표로 넓이 계산
    th = sp.acos(dmax)
    Bv = sp.Matrix([1, 0]); Cv = sp.Matrix([sp.cos(th), sp.sin(th)])
    Av = (Bv + 3 * Cv) / (kmax / 2 + 2)
    ok_unit = sp.simplify(Av.dot(Av) - 1) == 0
    area = sp.Abs(sp.Matrix([[Bv[0] - Av[0], Cv[0] - Av[0]], [Bv[1] - Av[1], Cv[1] - Av[1]]]).det()) / 2
    area = sp.nsimplify(sp.simplify(area))
    ok2 = ok_unit and sp.simplify(area - sp.sqrt(55) / 112) == 0
    report(2, ok1 and ok2, f"(1) k={kmax}, 최댓값 OB·OC={dmax} (격자 {best:.6f})  (2) A 단위원 위={ok_unit}, △ABC={area}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
