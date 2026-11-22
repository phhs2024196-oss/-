#!/usr/bin/env python3
"""KAU-G25E-01~02 (한국항공대 2025 공학계열) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,  # noqa: E402
                                        parse_expr, standard_transformations)
from vlib import *  # noqa: E402,F403

V = Verifier("KAU-G25E", "items/KAU/PAST")
TR = standard_transformations + (convert_xor, implicit_multiplication_application)


def P(s):
    """답 문자열(e^x, ln, 암묵 곱)을 sympy 식으로."""
    return parse_expr(s, transformations=TR, local_dict={"e": sp.E, "ln": sp.log, "x": x, "k": k, "t": t, "n": n})


def parts(no, sub):
    """'(1)=…; (2)=…' → {'(1)': '…'}"""
    return dict(p.strip().split("=", 1) for p in V.ans(no, sub).split(";"))


def p01():
    a, k, al = sp.symbols("a k alpha", real=True)
    # 1-1 판별식 경계 → a_n 범위 → 조임정리
    an = sp.Symbol("a_n", real=True)
    up_e = sp.solve(sp.Eq((5 * (n + 2)) ** 2 - 4 * 2 * 3 * an, 0), an)[0]      # D1>=0  ⇔  a_n <= up_e
    lo_e = sp.solve(sp.Eq((5 * n) ** 2 - 4 * 2 * 3 * an, 0), an)[0]            # D2<0   ⇔  a_n >  lo_e
    V.check(1, "1-1 하한/n^2 의 극한", sp.limit(lo_e / n ** 2, n, sp.oo), V.ans(1, "1"))
    V.check(1, "1-1 상한/n^2 의 극한", sp.limit(up_e / n ** 2, n, sp.oo), V.ans(1, "1"))
    # 1-2
    A = parts(1, "2")
    f = (x - 5) * sp.exp(x)
    tan = sp.diff(f, x).subs(x, k) * (x - k) + f.subs(x, k)
    V.check(1, "1-2(1) 접선", tan, P(A["(1)"].split("=", 1)[1]))
    crit = sp.solve(sp.diff(f, x), x)
    V.check(1, "1-2(2) 극솟값(x=4)", f.subs(x, crit[0]), -sp.exp(4))
    sgn = [sp.sign(sp.diff(f, x).subs(x, c)) for c in (3, 5)]
    V.report(1, crit == [4] and sgn == [-1, 1] and "-e^4" in A["(2)"].replace(" ", ""), f"1-2(2) f'<0 후 >0 → 극소 {crit}, 극솟값 문자열 확인")
    eq = sp.expand((-(f.subs(x, k))) - sp.diff(f, x).subs(x, k) * (a - k))        # 접선이 (a,0) 통과
    q = sp.simplify(eq / sp.exp(k))
    D = sp.discriminant(sp.expand(q), k)
    sol = sp.solve_univariate_inequality(D > 0, a, relational=False)
    nums = sorted(int(v) for v in re.findall(r"-?\d+", A["(3)"]))
    V.report(1, sol == sp.Union(sp.Interval.open(-sp.oo, nums[0]), sp.Interval.open(nums[1], sp.oo)) and sp.expand(q - (k**2 - (a + 5) * k + 4 * a + 5)) == 0,
             f"1-2(3) 이차식 {sp.expand(q)}, D>0 해집합 {sol}")
    # 1-3 급수의 극한 = 정적분 (정확값 + 큰 n 수치합)
    g = lambda u: sp.log(3 * u)
    I = sp.integrate(g(x) / x, (x, 3, 5))
    V.check(1, "1-3 정적분", I, P(V.ans(1, "3")))
    N = 400000
    s = sum(2.0 / (3 * N + 2 * kk) * math.log(3 * (3 + 2 * kk / N)) for kk in range(1, N + 1))
    V.report(1, abs(s - float(I)) < 1e-4, f"1-3 n={N} 부분합 {s:.6f} ≈ {float(I):.6f}")
    # 1-4(1) 교점 개수
    B = parts(1, "4")
    h1 = sp.log(x) - 1 / (2 * x)
    mono = sp.simplify(sp.diff(h1, x) - (2 * x + 1) / (2 * x ** 2)) == 0          # h1' = (2x+1)/(2x^2) > 0 (x>0)
    grid = [0.5 + i * 0.001 for i in range(40001)]
    vals = [math.log(u) - 1 / (2 * u) for u in grid]
    cnt = sum(1 for v0, v1 in zip(vals, vals[1:]) if v0 * v1 < 0)
    V.check(1, "1-4(1) 교점 개수(단조증가 + 부호변화)", cnt if mono else -1, B["(1)"])
    # 1-4(2) h'(2): 연립방정식을 t마다 수치로 풀어 h(t)를 만들고 중심차분
    mp.mp.dps = 40

    def h(t):
        Pt = 4 * t ** 4 + 3 * t ** 2 + 1
        F = lambda kk, aa: (Pt * mp.log(kk - t) - 3 * mp.e ** (2 * kk - aa), Pt / (kk - t) - 6 * mp.e ** (2 * kk - aa))
        kk, aa = mp.findroot(F, (t + 1.8, 2 * t + 3))
        return aa
    eps = mp.mpf("1e-12")
    d = (h(mp.mpf(2) + eps) - h(mp.mpf(2) - eps)) / (2 * eps)
    V.report(1, abs(d - mp.mpf(sp.Rational(B["(2)"]).p) / sp.Rational(B["(2)"]).q) < 1e-9, f"1-4(2) h'(2) 수치미분 {mp.nstr(d, 15)} vs {B['(2)']} [h(t)는 접함 조건 연립식을 t마다 직접 풂]")


def p02():
    a_ = sp.Symbol("a", nonzero=True)
    b_, c_, d_, al, be, ga = sp.symbols("b c d alpha beta gamma")
    f = a_ * x ** 3 + b_ * x ** 2 + c_ * x + d_
    eqs = []
    for (lin, r1, r2, unk) in ((5 * x - 6, 2, 3, al), (6 * x - 8, 2, 4, be), (7 * x - 12, 3, 4, ga)):
        poly = sp.Poly(sp.expand(f - lin - a_ * (x - r1) * (x - r2) * (x - unk)), x)
        eqs += poly.all_coeffs()
    sol = sp.solve(eqs, [b_, c_, d_, al, be, ga], dict=True)[0]
    asum = sp.simplify(sol[al] + sol[be] + sol[ga] - 24)
    a_val = [r for r in sp.solve(asum, a_) if r != 0][0]
    f0 = sp.simplify(sol[d_].subs(a_, a_val))
    V.check(2, "2-1 f(0)", f0, V.ans(2, "1"))
    # 2-2 기하: BC=m 을 미지수로 두고 AP=3 조건에서 m, WQ 를 직접 수치로 구함
    mp.mp.dps = 40

    def config(m):
        Ax, Ay = mp.mpf(0), mp.mpf(0)
        B = mp.matrix([6, 0])
        cosA = (36 + 25 - m ** 2) / 60
        C = mp.matrix([5 * cosA, 5 * mp.sqrt(1 - cosA ** 2)])
        a_, b_, c_ = m, mp.mpf(5), mp.mpf(6)
        s = (a_ + b_ + c_) / 2
        I = (a_ * mp.matrix([0, 0]) + b_ * B + c_ * C) / (a_ + b_ + c_)
        area = mp.sqrt(s * (s - a_) * (s - b_) * (s - c_))
        r = area / s
        X = B + (s - b_) * (C - B) / a_                       # BX = s-b
        W = (s - a_) * C / b_                                  # AW = s-a
        # 직선 A→X 와 원의 두 번째 교점 Y
        dvec = X
        A0 = mp.matrix([0, 0])
        qa = dvec[0] ** 2 + dvec[1] ** 2
        qb = -2 * (dvec[0] * I[0] + dvec[1] * I[1])
        qc = I[0] ** 2 + I[1] ** 2 - r ** 2
        roots = [(-qb + sg * mp.sqrt(qb ** 2 - 4 * qa * qc)) / (2 * qa) for sg in (1, -1)]
        tY = min(roots, key=lambda t_: abs(t_ - 1) * -1 if False else t_)    # X 는 t=1 (먼 쪽), Y 는 가까운 쪽 t<1
        Y = tY * dvec
        nvec = Y - I                                           # 접선의 법선
        # 접선과 AB(y=0) 의 교점 P, AC 위 교점 Q
        Px = (nvec[0] * Y[0] + nvec[1] * Y[1]) / nvec[0]
        P_ = mp.matrix([Px, 0])
        u = C / b_
        t_q = (nvec[0] * Y[0] + nvec[1] * Y[1]) / (nvec[0] * u[0] + nvec[1] * u[1])
        Q = t_q * u
        return P_, Q, W, X, Y, I, r

    mm = mp.findroot(lambda m: config(m)[0][0] - 3, 3)
    P_, Q, W, X, Y, I, r = config(mm)
    WQ = mp.sqrt((W[0] - Q[0]) ** 2 + (W[1] - Q[1]) ** 2)
    ans = parts(2, "2")
    ok = abs(mm - 3) < 1e-20 and abs(WQ - sp.Rational(ans["(3)"]).p / mp.mpf(sp.Rational(ans["(3)"]).q)) < 1e-20
    # 각 확인: ∠PYX = ∠YXB
    B = mp.matrix([6, 0])

    def ang(U, Vv, Wv):
        a1, a2 = U - Vv, Wv - Vv
        return mp.acos((a1[0] * a2[0] + a1[1] * a2[1]) / (mp.sqrt(a1[0] ** 2 + a1[1] ** 2) * mp.sqrt(a2[0] ** 2 + a2[1] ** 2)))
    ang1, ang2 = ang(P_, Y, X), ang(Y, X, B)
    # ZP = 1: Z = AB 위 접점, AZ = s-a
    s_ = (mm + 11) / 2
    ZP = (s_ - mm) - 3
    V.report(2, ok and abs(ang1 - ang2) < 1e-20 and abs(ZP - 1) < 1e-20,
             f"2-2 AP=3 조건에서 BC={mp.nstr(mm, 12)}, WQ={mp.nstr(WQ, 12)}, ∠PYX-∠YXB={mp.nstr(ang1 - ang2, 3)}, ZP={mp.nstr(ZP, 12)}")


if __name__ == "__main__":
    V.run(globals())
