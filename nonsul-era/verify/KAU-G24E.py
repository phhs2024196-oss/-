"""KAU-G24E (한국항공대 2024 공학계열) 검산 — 설정에서 독립 재계산하고 YAML 정답(V.ans)과 비교."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
from fractions import Fraction
V = Verifier("KAU-G24E", "items/KAU/PAST")


def p01():
    # 1-1: f(20)=f'(20)=f(23)=f'(23)=0, f(21)=21 을 만족하는 사차함수의 계수를 선형연립으로 풀어 f(24)
    c = sp.symbols("c0:5")
    f = sum(c[i] * x**i for i in range(5))
    sol = sp.solve([f.subs(x, 20), sp.diff(f, x).subs(x, 20), f.subs(x, 23), sp.diff(f, x).subs(x, 23), f.subs(x, 21) - 21], c, dict=True)[0]
    V.check(1, "1-1 f(24)", f.subs(sol).subs(x, 24), V.ans(1, "1"))
    # 1-2: 방정식 4^x-2^(x+1)+1-1/k^2=0 의 근 합 -> 곱 Π(k^2-1)/k^2 (정확한 유리수), 2^N
    prod = Fraction(1)
    for kk in range(2, 2025):
        # 근: 2^x = 1 ± 1/k (직접 대입 확인)
        for sgn in (1, -1):
            y = 1 + sgn * Fraction(1, kk)
            assert y * y - 2 * y + 1 - Fraction(1, kk**2) == 0
        prod *= (1 + Fraction(1, kk)) * (1 - Fraction(1, kk))
    V.check(1, "1-2 2^N", sp.Rational(prod.numerator, prod.denominator), V.ans(1, "2"))
    # 1-3: 꼭짓점 조건을 격자 + 정확 끝점으로
    k_ = sp.symbols("k", positive=True)
    xk, yk = 1 / k_, k_ - 1 / k_ - 3
    ends = sorted(sp.solve(sp.Eq(yk, 0), k_) + sp.solve(sp.Eq(yk, 1), k_) + [1])
    lo_s, hi_s = V.ans(1, "3").replace(" ", "").split("<k<")
    lo, hi = S(lo_s), S(hi_s)
    ok = all(((0 < xk.subs(k_, t) < 1) and (0 < yk.subs(k_, t) < 1)) == (lo < t < hi) for t in
             [sp.Rational(i, 50) for i in range(1, 600)] if abs(float(t) - float(lo)) > 1e-9 and abs(float(t) - float(hi)) > 1e-9)
    V.report(1, ok and any(same(e, lo) for e in ends) and any(same(e, hi) for e in ends), f"1-3 범위 격자+끝점 ({lo_s} ~ {hi_s})")
    # 1-4: 판별식·근의 차 조건을 θ 격자로, 구간 끝점은 YAML 에서 읽는다
    segs = [seg.split("<=theta<=") for seg in V.ans(1, "4").split(";")]
    segs = [(S(a), S(b)) for a, b in segs]
    import math
    bad = []
    for i in range(0, 7200):
        th = 2 * math.pi * i / 7199
        s_ = math.sin(th)
        D4 = (s_ + 1) ** 2 - s_**2
        cond = D4 >= -1e-12 and (math.sqrt(max(8 * s_ + 4, 0)) <= 2 * math.sqrt(2) + 1e-12)
        inside = any(float(a) - 1e-9 <= th <= float(b) + 1e-9 for a, b in segs)
        if cond != inside and min(abs(th - float(e)) for a, b in segs for e in (a, b)) > 1e-3:
            bad.append(th)
    V.proof(1, "1-4 θ 구간 [수치확인 7200점]", bad)
    # 1-5: b=ar, c=ar^2, d=2c-b -> d/a = 2r^2 - r 의 최솟값 (a,b,c,d != 0 조건 포함 확인)
    r = sp.symbols("r", real=True)
    g = 2 * r**2 - r
    rc = sp.solve(sp.diff(g, r), r)[0]
    b_, c_ = rc, rc**2
    d_ = 2 * c_ - b_
    assert all(v != 0 for v in (rc, b_, c_, d_))
    V.check(1, "1-5 min d/a", g.subs(r, rc), V.ans(1, "5"))


def p02():
    t_, a_ = sp.symbols("t a", positive=True)
    # 2-1: 거리 적분
    xt, yt = t_**2 * sp.sin(t_), t_**2 * sp.cos(t_)
    sp_ = sp.simplify(sp.sqrt(sp.simplify(sp.diff(xt, t_)**2 + sp.diff(yt, t_)**2)))
    s_ = sp.integrate(t_ * sp.sqrt(4 + t_**2), (t_, 0, a_))
    assert sp.simplify(sp_ - t_ * sp.sqrt(4 + t_**2)) == 0
    sols = [v for v in sp.solve(sp.Eq(s_, sp.Rational(19, 3)), a_) if v.is_positive]
    V.check(2, "2-1 a", sols[0], V.ans(2, "1"))
    # 2-2: a>0 에 대해 S2-S1 (교점은 수치로)
    import mpmath as mp
    for av in (0.3, 1, 2, 5):
        x1 = mp.findroot(lambda z: mp.cos(z) - av * z * mp.sin(z), 0.5)
        S1 = mp.quad(lambda z: mp.cos(z) - av * z * mp.sin(z), [0, x1])
        S2 = mp.quad(lambda z: av * z * mp.sin(z) - mp.cos(z), [x1, mp.pi])
        exp = sp.N(S(V.ans(2, "2")).subs(sp.Symbol("a"), av), 20)
        V.report(2, abs((S2 - S1) - mp.mpf(str(exp))) < 1e-9, f"2-2 a={av} S2-S1={mp.nstr(S2 - S1, 10)} / YAML {exp}")
    # 2-3: tan(APB) 를 실제 좌표 벡터로 계산해 비교
    for av in (1.5, 2, 3, 4.5):
        x1 = 1 / (av - 1)
        m1, m2 = mp.e**x1, av * mp.e**(av * x1 - 1)
        P = (x1, mp.e**x1)
        A = (x1 - P[1] / m1, 0)
        B = (x1 - P[1] / m2, 0)
        PA = (A[0] - P[0], A[1] - P[1]); PB = (B[0] - P[0], B[1] - P[1])
        tan_ = abs(PA[0] * PB[1] - PA[1] * PB[0]) / (PA[0] * PB[0] + PA[1] * PB[1])
        exp = sp.N(S(V.ans(2, "3")).subs(sp.Symbol("a"), av), 20)
        V.report(2, abs(tan_ - mp.mpf(str(exp))) < 1e-9, f"2-3 a={av} tan={mp.nstr(tan_, 10)} / YAML {exp}")
    # 2-4: 평행 접선 조건, LM, 최댓값
    t, s = sp.symbols("t s", positive=True)
    f = x + 2 / x; g = sp.log(x)
    s_sol = [v for v in sp.solve(sp.Eq(sp.diff(f, x).subs(x, t), sp.diff(g, x).subs(x, s)), s)][0]
    V.check(2, "2-4(1) s", s_sol, sp.sympify(V.ans(2, "4", "s"), locals={"t": t}))
    L = (f - x * sp.diff(f, x)).subs(x, t)          # l 의 y절편
    M = (g - x * sp.diff(g, x)).subs(x, s_sol)      # m 의 y절편
    h = sp.simplify(L - M)
    crit = [c for c in sp.solve(sp.diff(h, t), t) if c.is_real and c > sp.sqrt(2)]
    assert len(crit) == 1 and sp.diff(h, t).subs(t, crit[0] - sp.Rational(1, 10)) > 0 > sp.diff(h, t).subs(t, crit[0] + sp.Rational(1, 10))
    V.check(2, "2-4(2) max LM", h.subs(t, crit[0]), V.ans(2, "4", "max"))


if __name__ == "__main__":
    V.run(globals())
