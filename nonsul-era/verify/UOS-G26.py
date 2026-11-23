import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
from fractions import Fraction
V = Verifier("UOS-G26", "items/UOS/PAST")
th, r, X = sp.symbols("theta r X", positive=True)

def p01():
    # 정적분을 θ(sinθ=a)의 함수로 직접 적분해 g(θ) 확인 후 최소화
    a = sp.sin(th)
    g = (8*th - sp.pi)*sp.sin(th) + 8*sp.cos(th) - 2
    import numpy as np
    def num_int(a_):
        xs = np.linspace(0, 3*np.pi, 600001); ys = np.abs(np.sin(xs) - a_)
        return float(np.trapz(ys, xs)) if hasattr(np, "trapz") else float(np.trapezoid(ys, xs))
    for tv in (0.3, 0.9, 1.3):  # 수치 적분과 g 비교
        V.check(1, f"g({tv}) 수치", num_int(np.sin(tv)), sp.N(g.subs(th, tv), 15), 1e-6)
    crit = sp.solveset(sp.diff(g, th), th, sp.Interval(0, sp.pi/2))
    V.check(1, "a", sp.sin(sp.pi/8), V.ans(1, "1-1", "a"))
    V.check(1, "min", g.subs(th, sp.pi/8), V.ans(1, "1-1", "min"))
    V.report(1, g.subs(th, 0) > g.subs(th, sp.pi/8) and g.subs(th, sp.pi/2) > g.subs(th, sp.pi/8), "경계값이 최솟값보다 큼")
    # 전역 수치 최소(a 격자)
    best = min(num_int(i/100) for i in range(101))
    V.report(1, best >= float(sp.N(g.subs(th, sp.pi/8))) - 1e-5 and best - float(sp.N(g.subs(th, sp.pi/8))) < 1e-2, f"a 격자 최소 {best:.5f} ≈ 해석 최솟값")

def p02():
    y1 = sp.symbols("y1", positive=True)
    x1 = sp.sqrt(64*(1 - y1**2))
    S = 32*(y1 + 1)**2/(x1*y1)
    # A, B 좌표로 넓이 직접 계산
    A = (64*(y1 + 1)/x1, -1); B = (0, 1/y1); C = (0, -1)
    area = sp.Abs(A[0])*(B[1] - C[1])/2
    V.check(2, "넓이식", sp.simplify((area - S).subs(y1, sp.Rational(1, 3))), 0)
    S2 = 16*(y1 + 1)**3/((1 - y1)*y1**2)
    V.check(2, "S^2", sp.simplify((S**2 - S2).subs(y1, sp.Rational(2, 5))), 0)
    crit = sp.solveset(sp.diff(S2, y1), y1, sp.Interval.open(0, 1))
    V.check(2, "임계점", list(crit)[0], sp.Rational(1, 2))
    V.check(2, "최솟값", sp.sqrt(S2.subs(y1, sp.Rational(1, 2))), V.ans(2, "2-1"))
    # 접선: x1 x + 64 y1 y = 64 가 타원에 접함(판별식 0)
    xx = sp.symbols("xx"); p = sp.Rational(3, 5); q = sp.sqrt(1 - p**2*0 )  # placeholder not used
    x0 = 8*sp.sqrt(1 - sp.Rational(1, 4)); y0 = sp.Rational(1, 2)
    sol = sp.solve([x0*xx + 64*y0*sp.Symbol("yy") - 64, xx**2/64 + sp.Symbol("yy")**2 - 1], [xx, sp.Symbol("yy")], dict=True)
    V.check(2, "접점 중근", len(sol), 1)

def p03():
    t_, rr = sp.symbols("t r")
    xx = sp.symbols("xx")
    # 임계조건 f'(x)=0, f=x^2-sqrt(2xt-t^2)
    fp = 2*xx - t_/sp.sqrt(2*xx*t_ - t_**2)
    tv = sp.Rational(1, 2)
    rel = sp.sympify(V.ans(3, "3-1", "rel"))
    rsol = sp.solve(rel.subs(t_, tv), rr)
    rsol = [s_ for s_ in rsol if s_.is_real and s_ > 0]
    V.check(3, "r(t=1/2)", rsol[0], V.ans(3, "3-1", "r"))
    V.check(3, "f'(r)=0 at t=1/2", fp.subs({xx: rsol[0], t_: tv}), 0)
    # 관계식이 f'=0 과 동치임을 임의의 (r)로 확인: t=8r^3/(4r^2+1) 일 때 f'(r)=0
    for rv in (sp.Rational(1, 3), sp.Rational(7, 5), 2):
        tt = sp.solve(rel.subs(rr, rv), t_)[0]
        V.check(3, f"관계식 r={rv}", sp.simplify(fp.subs({xx: rv, t_: tt})), 0)
    # 수치 최소: t=1/2 에서 y=r^2-sqrt(2rt-t^2) 의 최소 r
    ymin = min(((rv/1000.0)**2 - (max(2*(rv/1000.0)*0.5 - 0.25, 0))**0.5, rv/1000.0) for rv in range(250, 3000))
    V.check(3, "수치 최소 r", round(ymin[1], 2), 0.5)
    # 음함수 미분 dr/dt
    d = sp.symbols("d")
    F = rel.subs(rr, sp.Function("R")(t_))
    dr = sp.solve(sp.diff(F, t_), sp.Derivative(sp.Function("R")(t_), t_))[0]
    V.check(3, "dr/dt", dr.subs({sp.Function("R")(t_): rsol[0], t_: tv}), V.ans(3, "3-1", "drdt"))

def brute(n):
    tot = 0; c1 = 0; sx = 0
    for a, b, c in itertools.permutations(range(1, n + 1), 3):
        lo1, hi1 = sorted((a, b)); lo2, hi2 = sorted((a, c))
        L = max(0, min(hi1, hi2) - max(lo1, lo2))
        tot += 1; c1 += (L == 1); sx += L
    return Fraction(c1, tot), Fraction(sx, tot)

def p04():
    nn = sp.symbols("n", positive=True)
    pa = sp.sympify(V.ans(4, "4-1")); mean = sp.sympify(V.ans(4, "4-2"))
    for n_ in range(3, 9):
        pr, mu = brute(n_)
        V.check(4, f"n={n_} P(X=1)", sp.Rational(pr.numerator, pr.denominator), pa.subs(sp.Symbol("n"), n_))
        V.check(4, f"n={n_} E[X]", sp.Rational(mu.numerator, mu.denominator), mean.subs(sp.Symbol("n"), n_))
    # 합 공식
    k_, n_ = sp.symbols("k n")
    s = sp.summation(2*k_*(n_ - k_)*(n_ - k_ - 1)/(n_*(n_ - 1)*(n_ - 2)), (k_, 1, n_ - 2))
    V.check(4, "시그마 계산", sp.simplify(s - mean.subs(sp.Symbol("n"), n_)), 0)

if __name__ == "__main__": V.run(globals())
