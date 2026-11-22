#!/usr/bin/env python3
"""CUK-G25B (가톨릭대 2025 수시 논술 의예·약학 1~4) — 문제 조건에서 독립 재계산."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
from fractions import Fraction as Fr
V = Verifier("CUK-G25B", "items/CUK/PAST")


def p01():
    # 상태 (a,b)=(흰,검은) 확률분포를 시행마다 전수 전개. 정지: a=3,b>0,b%3==0 or 3%b==0, 또는 k=10
    dist = {(0, 0): Fr(1)}
    PX = {}
    for k in range(1, 11):
        new = {}
        for (a, b), pr in dist.items():
            outs = [((a + 3, b), Fr(1, 6)), ((a, b + 2), Fr(1, 2))]
            if a + b == 0:
                outs.append(((a, b), Fr(1, 3)))
            else:
                outs.append(((a - 1, b), Fr(1, 3) * Fr(a, a + b)))
                outs.append(((a, b - 1), Fr(1, 3) * Fr(b, a + b)))
            for st, q in outs:
                new[st] = new.get(st, 0) + pr * q
        dist = {}
        for (a, b), pr in new.items():
            if (a == 3 and b > 0 and (b % 3 == 0 or 3 % b == 0)) or k == 10:
                PX[k] = PX.get(k, 0) + pr
            else:
                dist[(a, b)] = pr
    p = PX[4]
    q = PX[3] / sum(PX.get(i, 0) for i in range(1, 6))
    V.check(1, "p=P(X=4)", sp.Rational(p.numerator, p.denominator), V.ans(1, "1", "p"))
    V.check(1, "q=P(X=3|X<=5)", sp.Rational(q.numerator, q.denominator), V.ans(1, "1", "q"))


def p02():
    X, al = sp.symbols("X al", real=True)
    b2, b1, b0 = sp.symbols("b2 b1 b0", real=True)
    f = 3 * X ** 3 + b2 * X ** 2 + b1 * X + b0
    F = sp.Lambda(X, f)
    sols = set()
    for sgn in (4, -4):
        fa = al + sgn
        p = (al + fa) / 2
        eqs = [F(al) - fa, F(fa) - al, F(p) - p - 4,
               sp.integrate(f, (X, p, fa)) - 12]
        for s in sp.solve(eqs, [b2, b1, b0, al], dict=True):
            if all(v.is_real for v in s.values()):
                sols.add(sp.nsimplify(s[al]))
    exp = {sp.sympify(t) for t in V.ans(2, "1").split("=", 1)[1].split(",")}
    V.check(2, "alpha 집합", sorted(sols, key=float) == sorted(exp, key=float), True)


def p03():
    mp.mp.dps = 30
    # (1) 극한식 닫힌형 확인: 큰 n 에서 수치
    def f_n(x, a, n=60):
        x = mp.mpf(x)
        return 3 * a * mp.e ** (x + 1) * x ** (2 * n) / (3 * x ** (2 * n) + 1) + 2 * mp.fsum(x ** i for i in range(2 * n)) / (2 * x ** (2 * n) + 1)
    a0 = mp.mpf(1)
    ok = all(abs(f_n(x, a0) - (a0 * mp.e ** (x + 1) + 1 / (x - 1))) < 1e-8 for x in (1.5, 2, 3)) \
        and all(abs(f_n(x, a0) - 2 / (1 - mp.mpf(x))) < 1e-8 for x in (-0.5, 0, 0.5))
    V.report(3, ok, "극한 닫힌형 수치 확인")
    a, b, c = sp.symbols("a b c")
    # 연속·평균변화율 조건에서 a,b,c
    x = sp.symbols("x")
    fm1p = sp.Rational(2) / (1 - (-1))
    sol = sp.solve([sp.Eq(a + b, fm1p), sp.Eq(sp.Rational(3, 4) * a, fm1p), sp.Eq((c - sp.Rational(3, 4) * a) / 2, sp.Rational(1, 2))], [a, b, c], dict=True)[0]
    V.check(3, "a,b,c", (sol[a], sol[b], sol[c]), (sp.Rational(4, 3), -sp.Rational(1, 3), 2))
    A_, B_, C_ = 4, -1, 2
    e = mp.e
    f = lambda t: mp.mpf(4) / 3 * t * t - mp.mpf(1) / 3 if t < -1 else (mp.mpf(4) / 3 * e ** (t + 1) + 1 / (t - 1) if t > 1 else (2 if t == 1 else 2 / (1 - t)))
    fp = lambda r: mp.mpf(8) / 3 * r if r < -1 else (mp.mpf(4) / 3 * e ** (r + 1) - 1 / (r - 1) ** 2 if r > 1 else 2 / (1 - r) ** 2)   # 우미분계수
    def inA(p):
        s = (f(p) - 2) / (p - 1)
        lo, hi = (p, 1) if p < 1 else (1, p)
        N = 4000
        rs = [lo + (hi - lo) * i / N for i in range(1, N)]
        if p < -1:
            rs.append(mp.mpf(-1))              # r=-1 에서 우미분계수 1/2 (점프)
        pieces = [[fp(r) - s for r in rs if r < -1], [fp(r) - s for r in rs if -1 < r < 1], [fp(r) - s for r in rs if r > 1]]
        hit = any(v == 0 for pc in pieces for v in pc) or any(pc[i] * pc[i + 1] < 0 for pc in pieces for i in range(len(pc) - 1))
        if p < -1:
            hit = hit or (mp.mpf(1) / 2 - s == 0)     # r=-1 에서 우미분계수 1/2 (점프점)
        return not hit
    lowend = sp.sympify(V.ans(3, "1").split("=", 1)[1].split("<=")[0])
    lo_f = float(lowend)
    bad = []
    for p in [-6, -4, -3.5, -3.0, -2.9, -2.5, -2, -1.5, -1.2, -1.0, -0.5, 0, 0.5, 0.9, 1.1, 1.5, 1.9, 1.99, 2.05, 2.5, 3, 5]:
        want = (lo_f <= p < 1) or (1 < p <= 2)
        if abs(p - lo_f) < 0.02 or abs(p - 2) < 1e-9:
            continue
        got = inA(mp.mpf(p))
        if got != want:
            bad.append((p, got, want))
    V.proof(3, "A 판별 격자", bad)
    # 경계 -1-√19/2 에서 교점 방정식, p=2 포함(접점 t=2)
    s = sp.symbols("s")
    roots = sp.solve(sp.Eq(sp.Rational(4, 3) * s ** 2 - sp.Rational(1, 3), -sp.Rational(8, 3) * (s - 1) + 2), s)
    V.check(3, "경계 s<-1", [r for r in roots if r < -1][0], lowend)
    tt = sp.symbols("tt", real=True)
    tang = sp.solve(sp.Eq(2, sp.Rational(4, 3) * (2 - tt) * sp.E ** (tt + 1) + 2 / (tt - 1)), tt)
    V.report(3, any(abs(float(t0) - 2) < 1e-9 for t0 in tang if t0.is_real and t0 > 1) if tang else abs(mp.findroot(lambda t_: mp.mpf(4)/3*(2-t_)*e**(t_+1)+2/(t_-1)-2, 2) - 2) < 1e-20, "접점 t=2")


def p04():
    mp.mp.dps = 30
    # (다)로 a,b: b=3a^2/4 일 때 beta 의 sqrt-미분 = sqrt(6a)/2 = sqrt3 → a=2,b=3
    a_ = sp.symbols("a", positive=True); xx = sp.symbols("xx", positive=True)
    bb = sp.Rational(3, 4) * a_ ** 2
    beta = (-(xx - a_) + sp.sqrt(4 * bb - 3 * (xx - a_) ** 2)) / 2 + a_
    lim = sp.limit((beta - beta.subs(xx, 0)) / sp.sqrt(xx), xx, 0, "+")
    asol = sp.solve(sp.Eq(lim, sp.sqrt(3)), a_)
    V.check(4, "a=2", asol, [2])
    a, b = 2, 3
    beta_f = lambda t: (-(t - a) + mp.sqrt(4 * b - 3 * (t - a) ** 2)) / 2 + a
    alpha_f = lambda t: (-(t - a) - mp.sqrt(4 * b - 3 * (t - a) ** 2)) / 2 + a
    k, c = mp.mpf(7), mp.mpf(1)
    f = lambda t: k * ((t - a) ** 3 - b * (t - a)) + c
    fp = lambda t: k * (3 * (t - a) ** 2 - b)
    # 후보 가지 I, alpha, beta 가 f(g)=f(t) 를 만족, 정의역 [0,4]
    branch_ok = all(abs(f(br(mp.mpf(t) / 100)) - f(mp.mpf(t) / 100)) < 1e-15 for br in (alpha_f, beta_f, lambda s: s) for t in range(0, 401, 7))
    # 구간 [0,1],[1,3],[3,4] 각각 가지 선택 + 매듭 연속성 전수 → (다) 로 [0,1]은 beta
    br = {"I": lambda s: s, "al": alpha_f, "be": beta_f}
    node = {0: 0, 1: 1, 3: 3, 4: 4}
    cnt = []
    for ch in itertools.product(br, repeat=3):
        okc = all(abs(br[ch[i]](mp.mpf(node[t])) - br[ch[i + 1]](mp.mpf(node[t]))) < 1e-15 for i, t in ((0, 1), (1, 3)))
        # 시작 조건: g(x)-g(0) ~ sqrt3*sqrt(x)
        h = mp.mpf("1e-14")
        d0 = (br[ch[0]](h) - br[ch[0]](mp.mpf(0))) / mp.sqrt(h)
        if okc and abs(d0 - mp.sqrt(3)) < 1e-5:
            cnt.append(ch)
    # 3 번째 구간(3,4]: 연속 매듭은 2 개째까지만 검사했으므로 3 번째와 2 번째 연결도 확인
    cnt = [ch for ch in cnt if abs(br[ch[1]](mp.mpf(3)) - br[ch[2]](mp.mpf(3))) < 1e-15]
    V.check(4, "n", len(cnt), V.ans(4, "1", "n"))
    # S 계산: g1,g2 각각 k=5..10 에서 S=∫|f' g| (g>=0)
    def g1(x): return beta_f(x) if x <= 3 else x
    def g2(x): return beta_f(x)
    Ss = {}
    for nm, g in (("g1", g1), ("g2", g2)):
        for kk in (5, 10):
            k_ = mp.mpf(kk)
            fpp = lambda t, k_=k_: k_ * (3 * (t - a) ** 2 - b)
            Ss[(nm, kk)] = mp.quad(lambda x: abs(fpp(x) * g(x)), [1, 3, 4])
    M = max(Ss.values()); m = min(Ss.values())
    V.check(4, "M", sp.N(M, 20), sp.N(sp.sympify(V.ans(4, "1", "M")), 20), 1e-12)
    V.check(4, "m", sp.N(m, 20), sp.N(sp.sympify(V.ans(4, "1", "m")), 20), 1e-12)
    # S/k 비: g1 → 59/2? 59/4? (k 로 나눈 값) 확인
    V.report(4, abs(Ss[("g1", 10)] / 10 - mp.mpf(59) / 2) < 1e-15 and abs(Ss[("g2", 5)] / 5 - mp.mpf(91) / 4) < 1e-15 and branch_ok, "S/k=59/2, 91/4 및 가지 f(g)=f(t)")


if __name__ == "__main__":
    V.run(globals())
