#!/usr/bin/env python3
"""KW-G22B (광운대 2022학년도 논술고사 자연계열 2교시) — sympy 검산. 기대값은 V.ans() 로만 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: F401,F403
import math
import random
import fractions
import numpy as np

V = Verifier("KW-G22B", "items/KW/PAST")
X, Y, a, n_, t_ = sp.symbols("X Y a n_ t_")
x_, y_ = sp.symbols("x y")


def p01():
    # 1-1 영역
    pred = V.ans(1, "1-1").replace(",", " and ")
    random.seed(1)
    bad = []
    for _ in range(40000):
        b, c = random.uniform(-0.5, 2.5), random.uniform(-0.5, 2.5)
        truth = (0 <= b <= 2 and 0 <= c <= 2) and (b * b - 4 * c >= 0)
        if truth != bool(eval(pred)):
            bad.append((b, c))
    V.report(1, not bad, f"1-1 영역 불일치 {bad[:2]}")
    # 1-2 직선
    S_tot = sp.integrate(X**2 / 4, (X, 0, 2))
    aa = sp.Symbol("aa", positive=True)
    left_area = sp.integrate(X**2 / 4, (X, 0, aa)) + sp.integrate((aa**2 / 4) / (aa - 2) * (X - 2), (X, aa, 2))
    sols = [r for r in sp.solve(sp.Eq(left_area, sp.Rational(5, 16) * S_tot), aa) if r.is_real and 0 < r < 2 and r != 1] + [r for r in sp.solve(sp.Eq(left_area, sp.Rational(5, 16) * S_tot), aa) if r == 1]
    line = (aa**2 / 4) / (aa - 2) * (X - 2)
    lhs = sp.simplify(line.subs(aa, sols[0]))
    rhs = S(V.ans(1, "1-2").split("=")[1].replace("x", "X"))
    V.report(1, len(sols) == 1 and sp.simplify(lhs - rhs) == 0, f"1-2 a={sols} 직선 {lhs}")
    # 2-1 증명(수치)
    fs = [sp.sin(X), sp.exp(X), X**2 + X, sp.cos(2 * X) + X**3]
    V.proof(1, "2-1 (연속함수 표본)", [f for f in fs if abs(float(sp.Integral(f, (X, 0, sp.pi)).evalf(30)) - float(sp.Integral(f.subs(X, sp.pi - X), (X, 0, sp.pi)).evalf(30))) > 1e-12])
    I2 = sp.Integral(sp.sin(X) / (1 + sp.cos(X)**2), (X, 0, sp.pi)).evalf(30)
    V.check(1, "2-2", I2, V.ans(1, "2-2"), 1e-15)
    I3 = sp.Integral(X * sp.sin(X) / (1 + sp.cos(X)**2), (X, 0, sp.pi)).evalf(30)
    V.check(1, "2-3", I3, V.ans(1, "2-3"), 1e-15)
    # 3-1 좌표와 홀수
    F = fractions.Fraction
    ok = True
    for n in range(1, 15):
        bn = F(1, 2)**(n - 1)
        an = sum(F(1, 2)**(k - 1) for k in range(1, n + 1))
        ratio = an / bn
        ok &= an == F(str(sp.nsimplify(S(V.ans(1, "3-1", "a_n")).subs(sp.Symbol("n"), n))))
        ok &= bn == F(str(sp.nsimplify(S(V.ans(1, "3-1", "b_n")).subs(sp.Symbol("n"), n))))
        ok &= ratio.denominator == 1 and ratio.numerator % 2 == 1 and ratio == S(V.ans(1, "3-1", "ratio")).subs(sp.Symbol("n"), n)
    V.report(1, ok, "3-1 a_n,b_n,a_n/b_n 홀수 (n=1..14)")
    # 3-2
    ok = True
    for n in range(1, 8):
        k = sp.Symbol("k", integer=True, positive=True)
        r = sp.summation(sp.Rational(1, 4)**(k - 1), (k, n + 1, sp.oo)) / sp.Rational(1, 4)**(n - 1)
        ok &= sp.simplify(r - S(V.ans(1, "3-2"))) == 0
    V.report(1, ok, "3-2 비율")
    # 3-3
    ok = True
    d = sp.Symbol("d", positive=True)
    for n in range(2, 9):
        m = sp.Rational(1, 2**n - 1)
        cand = [r for r in sp.solve(sp.Eq((d - m)**2, (1 + d * m)**2), d) if r > 0]
        ok &= len(cand) == 1 and sp.simplify(cand[0] - S(V.ans(1, "3-3")).subs(sp.Symbol("n"), n)) == 0
    V.report(1, ok, "3-3 d_n (n=2..8, 각 45도)")
    # 4
    tot = sum(F(2 * n, 1 + n * n + n**4) for n in range(2, 11))
    V.check(1, "4", sp.Rational(tot.numerator, tot.denominator), V.ans(1, "4"))


def lattice_counts():
    """도로망: 5x5 격자, 가운데 구멍 x∈[2,4], y∈[1,4] (구멍 안 도로 없음, 둘레는 도로)."""
    def edge_ok(p, q):
        (x1, y1), (x2, y2) = p, q
        if y1 == y2:      # 가로 변 (x1,y1)-(x1+1,y1)
            return not (y1 in (2, 3) and x1 in (2, 3))
        return not (x1 == 3 and y1 in (1, 2, 3))     # 세로 변 (x1,y1)-(x1,y1+1)
    ways = {(0, 0): 1}
    for s in range(1, 11):
        for x in range(6):
            y = s - x
            if not 0 <= y <= 5:
                continue
            w = 0
            if x > 0 and edge_ok((x - 1, y), (x, y)):
                w += ways.get((x - 1, y), 0)
            if y > 0 and edge_ok((x, y - 1), (x, y)):
                w += ways.get((x, y - 1), 0)
            ways[(x, y)] = w
    return ways, edge_ok


def p02():
    # 1-1
    P = []
    ok = True
    for av in (sp.Rational(1, 2), 1, 2, 3, 7):
        x1, y1 = sp.symbols("x1 y1", positive=True)
        sol = sp.solve([x1**2 / av + y1**2 - 1, av / x1 - 1 / y1], [x1, y1], dict=True)
        exp = S(V.ans(2, "1-1", "P"))
        ok &= len(sol) == 1 and sp.simplify(sol[0][x1] - exp[0].subs(a, av)) == 0 and sp.simplify(sol[0][y1] - exp[1].subs(a, av)) == 0
    V.report(2, ok, "1-1 P 좌표")
    # 1-2
    av = sp.Symbol("av", positive=True)
    x1 = av / sp.sqrt(av + 1)
    y1 = 1 / sp.sqrt(av + 1)
    OA = av / x1
    OB = 1 / y1
    S1 = OA * y1 / 2
    S2 = OB * x1 / 2
    sol = sp.solve(sp.Eq(S1, 2 * S2), av)
    exp_l, exp_r = V.ans(2, "1-2").split("=")
    el = S(exp_l.replace("x", "X").replace("y", "Y"))
    # 타원 X^2/a + Y^2 = 1 에 a=1/2 → 2X^2+Y^2=1
    V.report(2, sol == [sp.Rational(1, 2)] and sp.simplify(el - (X**2 / sol[0] + Y**2)) == 0 and S(exp_r) == 1, f"1-2 a={sol}")
    # 1-3
    c2 = 1 - sp.Rational(1, 2)
    c = sp.sqrt(c2)
    F1, F2 = sp.Matrix([0, c]), sp.Matrix([0, -c])
    qs = sp.solve([2 * X**2 + Y**2 - 1, ((sp.Matrix([X, Y]) - F1).T * (sp.Matrix([X, Y]) - F2))[0]], [X, Y], dict=True)
    got = sorted([(sp.nsimplify(q[X]), sp.nsimplify(q[Y])) for q in qs], key=lambda p: float(p[0]))
    exp = sorted([tuple(S(t.strip()) for t in pr.split(",")) for pr in re.findall(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", V.ans(2, "1-3"))], key=lambda p: float(p[0]))
    V.report(2, len(got) == len(exp) and all(sp.simplify(u - v) == 0 for g, e in zip(got, exp) for u, v in zip(g, e)), f"1-3 Q {got}")
    # 2-1, 2-2
    ways, edge_ok = lattice_counts()
    back = {}
    # B 에서 거꾸로 센 경로 수 (그래프는 무방향)
    total = ways[(5, 5)]
    V.check(2, "2-1 A→B 최단경로 수", total, V.ans(2, "2-1"))
    # 각 지점을 지나는 경로 수 = ways(A→v) * ways(v→B), v→B 는 뒤집어 센다
    ways_b = {(5, 5): 1}
    for s in range(9, -1, -1):
        for x in range(6):
            y = s - x
            if not 0 <= y <= 5:
                continue
            w = 0
            if x < 5 and edge_ok((x, y), (x + 1, y)):
                w += ways_b.get((x + 1, y), 0)
            if y < 5 and edge_ok((x, y), (x, y + 1)):
                w += ways_b.get((x, y + 1), 0)
            ways_b[(x, y)] = w
    pas = lambda v: ways[v] * ways_b[v]
    # 두 사람은 시각 5 에 x+y=5 인 지점에서만 만난다. 연수의 경로는 B→A 이므로 같은 값
    Pn = fractions.Fraction(pas((1, 4)) * pas((1, 4)), total * total)
    Qn = fractions.Fraction(pas((2, 3)) * pas((2, 3)), total * total)
    V.report(2, sp.Rational(Pn.numerator, Pn.denominator) == S(V.ans(2, "2-2", "P")) and sp.Rational(Qn.numerator, Qn.denominator) == S(V.ans(2, "2-2", "Q")),
             f"2-2 P {Pn} Q {Qn} (경우수 {pas((1,4))**2}, {pas((2,3))**2}); x+y=5 지점 합 {sum(pas((x,5-x)) for x in range(6))}")
    # 3-1, 3-2
    A_ = [None, 2]
    for k in range(1, 12):
        A_.append(A_[k] + (k * k + 2 * k + 2) * math.factorial(k + 1))
    aa_, bb_ = sp.symbols("aa_ bb_")
    sol = sp.solve([(aa_ + bb_) * 1 - A_[1], (4 * aa_ + 2 * bb_) * 2 - A_[2]], [aa_, bb_])
    g = sol[aa_] * n_**2 + sol[bb_] * n_
    V.report(2, sp.simplify(g - S(V.ans(2, "3-1")).subs(sp.Symbol("n"), n_)) == 0, f"3-1 g(n)={g}")
    V.proof(2, "3-2 a_n=(n^2+n)n!", [k for k in range(1, 12) if A_[k] != (k * k + k) * math.factorial(k)])
    # 4-1: AC 를 변수로 놓고 ∠BAC+∠CAD=135° 를 만족하는 AC 를 전수 탐색
    s75 = math.sin(math.radians(75))
    roots = []
    cs = np.linspace(2 * s75 + 1e-9, 3.99, 400000)
    for branch in (1, -1):
        ang = lambda c: np.degrees(np.arccos(c / 4)) + (np.degrees(np.arcsin(2 * s75 / c)) if branch == 1 else 180 - np.degrees(np.arcsin(2 * s75 / c)))
        v = ang(cs) - 135
        idx = np.where(v[:-1] * v[1:] < 0)[0]
        roots += [round(float(cs[i]), 3) for i in idx]
    V.proof(2, "4-1 AC=2 유일(∠BAC+∠CAD=135° 탐색)", [r for r in roots if abs(r - 2) > 1e-2] + ([] if any(abs(r - 2) < 1e-2 for r in roots) else ["근 없음"]))
    # 4-2
    ad = math.sqrt(2**2 + 2**2 - 2 * 2 * 2 * math.cos(math.radians(30)))
    V.check(2, "4-2 AD", ad, V.ans(2, "4-2"), 1e-9)


import re  # noqa: E402

if __name__ == "__main__":
    V.run(globals())
