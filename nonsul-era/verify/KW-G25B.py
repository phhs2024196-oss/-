"""광운대 2025학년도 자연 2교시(KW-G25B) 검산 — 기대값은 V.ans() 로만 읽는다."""
import sys, pathlib, itertools, math
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-G25B", "items/KW/PAST")
th = sp.symbols("th", real=True)


def p01():
    # [1] 일대일함수 전수
    tot = good = 0
    for f in itertools.permutations(range(3, 8), 3):
        tot += 1
        if f[0] < f[1] < f[2] and sum(f) % 2 == 1: good += 1
    V.check(1, "1 확률", sp.Rational(good, tot), V.ans(1, "1"))
    # [2](1) m(θ)=min_x f 를 완전제곱으로 (sympy 로 극소 x 를 구해 대입), 최댓값은 격자 + 정확식
    f = x**2 + 2*x*sp.sin(th) + 2*sp.cos(th)
    xs = sp.solve(sp.diff(f, x), x)[0]
    m = sp.simplify(f.subs(x, xs))
    best = max(float(m.subs(th, math.pi*i/20000)) for i in range(20001))
    V.check(1, "2-1 m(θ) 최댓값(격자)", best, sp.N(S(V.ans(1, "2-1"))), 1e-6)
    V.check(1, "2-1 정확", sp.simplify(m - ((sp.cos(th) + 1)**2 - 2)), 0)
    # [2](2) 실근을 갖는 θ 의 최소·최대(격자) → sin²α+sin²β
    ok = [math.pi*i/200000 for i in range(200001) if math.sin(math.pi*i/200000)**2 - 2*math.cos(math.pi*i/200000) >= 0]
    a_, b_ = min(ok), max(ok)
    val = math.sin(a_)**2 + math.sin(b_)**2
    V.check(1, "2-2", val, sp.N(S(V.ans(1, "2-2"))), 1e-4)
    # [3] 표준정규 표값으로: P(580<=X<=640)=P(-1<=Z<=2)=(0.6826+0.9544)/2 ; 이항 정확값과 비교(근사)
    table = (sp.Rational(6826, 10000) + sp.Rational(9544, 10000)) / 2
    V.check(1, "3 표값", table, V.ans(1, "3"))
    from math import comb
    exact = Fraction(sum(comb(450, k_) * 2**k_ for k_ in range(290, 321)), 3**450)   # X=2K, 580<=2K<=640
    V.report(1, abs(float(exact) - float(S(V.ans(1, "3")))) < 0.02, f"3 이항 정확값 {float(exact):.4f} ≈ 정규근사 {V.ans(1, '3')} (오차<0.02)")
    mean, var = 2*450*sp.Rational(2, 3), 4*450*sp.Rational(2, 3)*sp.Rational(1, 3)
    V.check(1, "3 E(X),V(X)", mean + var, 600 + 400)
    # [4] 교점의 x좌표와 극한
    nn = sp.symbols("nn", positive=True)
    an = [r for r in sp.solve(x**2 + (nn/x)**2 - 2*nn, x) if r.is_positive][0]
    bn = [r for r in sp.solve(x**2 + 2*nn*x - 2*nn, x) if sp.simplify(r.subs(nn, 1)) > 0][0]
    V.check(1, "4 a_n", an, sp.sqrt(nn)); V.check(1, "4 b_n", bn, sp.sqrt(nn**2 + 2*nn) - nn)
    V.check(1, "4 극한", sp.limit(bn/an**2, nn, sp.oo), V.ans(1, "4"))


def p02():
    # [1]
    f = 3*x**4 - 4*x**3 - 6*x**2 - 24*x
    g = -x**2 - 2*x + k
    crit = [r for r in sp.solve(sp.diff(f, x), x) if r.is_real]
    fmin = min(f.subs(x, r) for r in crit)
    gmax = max(g.subs(x, r) for r in sp.solve(sp.diff(g, x), x))
    kmax = sp.solve(sp.Eq(fmin, gmax), k)[0]
    V.check(2, "1 k 최댓값", kmax, V.ans(2, "1"))
    # [2](1) V(θ) 최대 θ
    th_ = sp.symbols("th_", positive=True)
    xr = (2*sp.pi - th_)/(2*sp.pi); hh = sp.sqrt(1 - xr**2)
    Vv = sp.pi*xr**2*hh/3
    sols = [r for r in sp.solve(sp.diff(Vv, th_), th_) if r.is_real and 0 < r < 2*sp.pi]
    V.check(2, "2-1 θ(미분 근)", sols[0], V.ans(2, "2-1"))
    # 최대임: 격자에서 V 최대 위치
    vals = [(float(Vv.subs(th_, 2*math.pi*i/4000)), 2*math.pi*i/4000) for i in range(1, 4000)]
    V.check(2, "2-1 격자 최대 위치", max(vals)[1], sp.N(S(V.ans(2, "2-1"))), 3e-3)
    # [2](2) 원뿔 옆면 최단거리: 밑면 둘레의 반(πx) = 전개도 중심각, 코사인법칙
    thv = S(V.ans(2, "2-1")); xx = (2*sp.pi - thv)/(2*sp.pi)
    l2 = 1 + 1 - 2*sp.cos(sp.pi*xx)
    V.check(2, "2-2 l^2", sp.N(l2), sp.N(S(V.ans(2, "2-2"))), 1e-10)
    # [3](1) 접선 x절편으로 점화식 직접 구성 후 반복 수렴
    a = sp.symbols("a", real=True)
    F = sp.tan(x - sp.pi/2)
    tang_x = sp.solve(sp.Eq(F.subs(x, a) + sp.diff(F, x).subs(x, a)*(x - a), 0), x)[0]
    seq = [sp.N(2*sp.pi/3, 60)]
    for _ in range(12): seq.append(sp.N(tang_x.subs(a, seq[-1]), 60))
    V.check(2, "3-1 수렴값", seq[-1], sp.N(S(V.ans(2, "3-1")), 60), 1e-15)
    V.check(2, "3-1 점화식", sp.simplify(tang_x - (a - sp.sin(2*(a - sp.pi/2))/2)), 0)
    # [3](2)
    al = sp.pi/2
    ratios = [(seq[i+1] - al)/(seq[i] - al) for i in range(5)]
    V.report(2, abs(ratios[-1]) < 1e-3, f"3-2 비 {[sp.N(r, 4) for r in ratios]} → 0")
    u = sp.symbols("u")
    V.check(2, "3-2 극한", 1 - sp.limit(sp.sin(2*u)/(2*u), u, 0), V.ans(2, "3-2"))


if __name__ == "__main__": V.run(globals())
