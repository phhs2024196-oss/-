#!/usr/bin/env python3
"""CUK-G25A (가톨릭대 2025 수시 논술 자연공학·간호 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G25A", "items/CUK/PAST")


def p01():
    a = lambda k: 3 + 7 * (k - 1)
    A = [a(k) for k in range(1, 101)]
    means = [sp.Rational(sum(A) - v, 99) for v in A]       # 하나씩 제외한 평균 전수
    V.check(1, "논제1 m", min(means), V.ans(1, "1", "m"))
    V.check(1, "논제1 M", max(means), V.ans(1, "1", "M"))
    tgt = 97 + sp.Rational(1, 9)
    sols = []
    for N in range(2, 400):
        B = [a(k) for k in range(1, N + 1)]
        for p in B:
            if sp.Rational(sum(B) - p, N - 1) == tgt:
                sols.append((N, p))
    V.check(1, "논제2 N", sols[0][0] if len(sols) == 1 else -1, V.ans(1, "2", "N"))
    V.check(1, "논제2 p", sols[0][1] if len(sols) == 1 else -1, V.ans(1, "2", "p"))


def p02():
    X, a, b, c, d, al, be, k_, cc, tt = sp.symbols("X a b c d al be k_ cc tt")
    # 1) g=ax^3+bx^2+cx+d. g-1 의 근 {α,1}(한쪽 중근), g+1 의 근 {-1,β}(한쪽 중근), g-1 와 g+1 의 차 = -2
    found = set()
    for m1, m2 in itertools.product([0, 1], repeat=2):
        r1 = (X - al) ** 2 * (X - 1) if m1 == 0 else (X - al) * (X - 1) ** 2
        r2 = (X + 1) ** 2 * (X - be) if m2 == 0 else (X + 1) * (X - be) ** 2
        eq = sp.Poly(sp.expand(a * r1 + 2 - a * r2), X).all_coeffs()
        for s in sp.solve(eq, [a, al, be], dict=True):
            if all(v.is_real for v in s.values()) and s[a] != 0 and s[al] < -1 and s[be] > 1:
                g = sp.expand(s[a] * r1.subs(s) + 1)
                found.add(sp.simplify(g))
    gy = sp.sympify(V.ans(2, "1").split("=", 1)[1], locals={"x": X})
    V.check(2, "논제1 g 유일", len(found) == 1 and sp.simplify(list(found)[0] - gy) == 0, True)
    g = list(found)[0]
    # 2) f=g+kx, h=f-x^2-cx-c 가 근 {t,2}(한쪽 중근), k>0, t<2
    cs = set()
    for m in (0, 1):
        r = (X - tt) ** 2 * (X - 2) if m == 0 else (X - tt) * (X - 2) ** 2
        h = g + k_ * X - X ** 2 - cc * X - cc
        lead = sp.Poly(h, X).LC()
        eq = sp.Poly(sp.expand(h - lead * r), X).all_coeffs()
        for s in sp.solve(eq, [k_, cc, tt], dict=True):
            if all(v.is_real for v in s.values()) and s[k_] > 0 and s[tt] < 2:
                cs.add(s[cc])
    V.check(2, "논제2 c 유일", cs == {sp.Integer(12)} and True, True)
    V.check(2, "논제2 c", list(cs)[0] if len(cs) == 1 else -1, V.ans(2, "2", "c"))


def p03():
    mp.mp.dps = 30
    F = lambda t: t * mp.e ** (-t)
    # a: 급수 sum r^n 이 r<=-1 이면 발산, -1<r<1 수렴: F(a)=-1, a<0
    a = mp.findroot(lambda t: F(t) + 1, -0.5)
    V.check(3, "a<0, F(a)=-1", bool(a < 0 and abs(F(a) + 1) < 1e-25), True)
    # 급수합 r/(1-r) 를 t>a 에서 수치 최대화 (격자 + 미분 영점)
    ts = [a + mp.mpf(i) / 2000 for i in range(1, 20001)]
    best = max(F(t) / (1 - F(t)) for t in ts)
    t1 = mp.findroot(lambda t: mp.diff(lambda u: F(u) / (1 - F(u)), t), 1)
    exact = F(t1) / (1 - F(t1))
    Mv = sp.N(sp.sympify(V.ans(3, "1", "M"), locals={"E": sp.E}), 25)
    V.check(3, "논제1 M", bool(abs(exact - Mv) < 1e-20 and best <= Mv + 1e-12), True)
    # 논제2 A: 격자
    ys = [2 * mp.pi * i / 20000 for i in range(20001)]
    inA = [y for y in ys if mp.cos(y) / mp.e ** mp.cos(y) <= mp.sin(y) / mp.e ** mp.sin(y) + mp.mpf(10) ** -25]
    lo, hi = min(inA), max(inA)
    ans = V.ans(3, "2", "A").strip("[]").split(",")
    e0, e1 = [sp.N(sp.sympify(s), 25) for s in ans]
    V.check(3, "논제2 A 끝점/연속", bool(abs(lo - e0) < 1e-3 and abs(hi - e1) < 1e-3 and len(inA) == int((hi - lo) / (2 * mp.pi / 20000)) + 1), True)
    q = lambda x: mp.e ** mp.cos(x) * mp.sin(x) - mp.e ** mp.sin(x) * mp.cos(x)
    S = mp.quad(lambda x: abs(q(x)), [0, mp.pi / 4, 5 * mp.pi / 4, 2 * mp.pi])
    Sv = sp.N(sp.sympify(V.ans(3, "2", "S")), 25)
    V.check(3, "논제2 S", bool(abs(S - Sv) < 1e-18), True)


if __name__ == "__main__":
    V.run(globals())
