#!/usr/bin/env python3
"""경희대 2026학년도 모의논술(의·약학계 수학) KHU-M26D-01~02 — sympy 검산. 정답 하드코딩 비교 금지, 문제 설정에서 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KHU-M26D"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I: 스위치 회로 — 64가지 상태 전수 열거, 그래프 연결로 판정."""
    p = sp.symbols("p", positive=True)
    # 노드: L, P(1-2 사이), X(2-3 사이), R, A(4-5 사이), B(5-6 사이)
    sw = {1: ("L", "P"), 2: ("P", "X"), 3: ("X", "R"), 4: ("L", "A"), 5: ("A", "B"), 6: ("B", "R")}

    def prob(extra):
        tot = 0
        for st in itertools.product([0, 1], repeat=6):
            edges = [sw[i + 1] for i in range(6) if st[i]] + extra
            seen = {"L"}; ch = True
            while ch:
                ch = False
                for u, v in edges:
                    if (u in seen) != (v in seen):
                        seen |= {u, v}; ch = True
            if "R" in seen:
                tot += sp.Mul(*[p if s else 1 - p for s in st])
        return sp.factor(sp.expand(tot))
    base = prob([])
    ok1 = sp.expand(base - (2*p**3 - p**6)) == 0 and sp.expand(base - (1 - (1 - p**3)**2)) == 0
    f = prob([("P", "A")]); g = prob([("P", "B")])
    diff = sp.factor(f - g)
    ok2 = sp.expand(f - p**3*(2 - p)*(2 - p**2)) == 0 and sp.expand(g - p**2*(1 + p - p**2)**2) == 0
    sol = sp.solve_univariate_inequality(f - g < 0, p, relational=False).intersect(sp.Interval.open(0, 1))
    ok3 = sol == sp.Interval.open(0, 1)
    report(1, ok1 and ok2 and ok3, f"(1) {base}; (2) f-g={diff}, 해={sol}")


def p02():
    """논제 II: 반원(반지름 1, A(-1,0), B(1,0)), C 는 B 에서의 수직선 위. 넓이를 적분으로 직접 계산."""
    th, X = sp.symbols("th X", real=True)
    import mpmath as mp
    mp.mp.dps = 30

    def f_num(t):
        t = mp.mpf(t)
        line = lambda x: mp.tan(t)*(x + 1)
        arc = lambda x: mp.sqrt(1 - x**2)
        xd = mp.cos(2*t)*(-1) + 0  # D: 직선과 원의 교점 x = (1-tan²)/(1+tan²) = cos2t
        xd = mp.cos(2*t)
        S1 = mp.quad(lambda x: arc(x) - line(x), [-1, xd])
        S2 = mp.quad(lambda x: line(x) - arc(x), [xd, 1])
        return S1 + S2
    closed = 2*sp.tan(th) - sp.sin(2*th) - 2*th + sp.pi/2
    ok = all(abs(f_num(v) - mp.mpf(sp.N(closed.subs(th, v), 30))) < mp.mpf("1e-20")
             for v in [mp.pi/6, mp.pi/5, mp.mpf("0.7"), mp.pi/4])
    # 최대·최소: 격자 + 임계점
    a, b_ = sp.pi/6, sp.pi/4
    grid = [a + (b_ - a)*sp.Rational(k, 400) for k in range(401)]
    vals = [float(closed.subs(th, g)) for g in grid]
    alpha = grid[vals.index(max(vals))]; beta_g = float(grid[vals.index(min(vals))])
    crit = sp.nsolve(sp.diff(closed, th), th, 0.65)
    ok &= alpha == sp.pi/4 and abs(float(crit) - beta_g) < 3e-3 and abs(float(sp.cos(crit)**4) - 0.5) < 1e-20
    val1 = sp.cos(alpha)**4 + sp.sin(crit)**2
    ok &= abs(float(val1 - (sp.Rational(5, 4) - sp.sqrt(2)/2))) < 1e-20
    # (2) 정적분
    beta = sp.acos(2**sp.Rational(-1, 4))
    I = sp.integrate(closed + 2*th - sp.pi/2, (th, alpha, beta))
    I = sp.nsimplify(sp.simplify(sp.expand_log(sp.simplify(I), force=True)))
    target = -sp.log(2)/2 + sp.sqrt(2)/2 - sp.Rational(1, 2)
    ok &= abs(float(I - target)) < 1e-25 or sp.simplify(I - target) == 0
    report(2, ok, f"(1) α=π/4, cos⁴β={sp.N(sp.cos(crit)**4, 15)}, 값={sp.N(val1, 15)}; (2) 적분={sp.N(I, 20)} (=-ln2/2+√2/2-1/2)")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
