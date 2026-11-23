import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G21A", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "k": k, "r": sp.Symbol("r", positive=True)})

def p01():
    a, b, c, tt = sp.symbols("a b c tt")
    A = sp.solve(sp.Eq(-(-1) + a, 2), a)[0]                     # f(-1)=2 (왼쪽 2x^2 의 극한값)
    B = sp.solve(sp.Eq(-sp.Rational(1, 3) + b, 0), b)[0]        # f(1)=0
    f1 = lambda z: 2*z**2 if z < -1 else (-z + A if z < 1 else -z/3 + B)
    t0 = [s for s in sp.solve(2*tt**2 - 8, tt) if s < -1][0]    # h(8)
    c0 = sp.solve(sp.Eq(-c/3 + B, t0), c)[0]                    # h(-2)=c, c>=1
    assert c0 >= 1 and f1(c0) == t0 and f1(t0) == 8
    V.check(1, "1-1 a+b+c", A + B + c0, E(V.ans(1, "1-1")))
    # 1-2: 리만합 극한 = (1/4)∫_{-2}^{2} f
    I = sp.integrate(2*x**2, (x, -2, -1)) + sp.integrate(-x + A, (x, -1, 1)) + sp.integrate(-x/3 + B, (x, 1, 2))
    V.check(1, "1-2", I/4, E(V.ans(1, "1-2")))
    n_ = 4000
    num = sum(float(f1(sp.Rational(-2) + sp.Rational(4*kk, n_))) for kk in range(1, n_ + 1))/n_
    V.report(1, abs(num - float(I/4)) < 2e-3, f"1-2 수치 합 {num:.5f}")
    # 1-3: g(alpha)=pi/3 -> alpha=tan(pi/3)
    al = sp.tan(sp.pi/3)
    val = sp.integrate(sp.sin(sp.atan(x))**3/(1 + x**2), (x, 0, al))
    V.check(1, "1-3", sp.nsimplify(sp.simplify(val)), E(V.ans(1, "1-3")))

def p02():
    r, X, y_ = sp.symbols("r X y_", positive=True)
    a = sp.Symbol("a")
    # 2-1: x=y^2/2 대입, 판별식 0
    q = sp.expand((X - a)**2 + 2*X - r**2)
    ar = sp.solve(sp.discriminant(q, X), a)[0]
    V.check(2, "2-1 a_r", ar, E(V.ans(2, "2-1")))
    # 2-2: (a_{2r}-1/2) e^{-r} + k = 0, r>1 서로 다른 두 실근
    a2 = ar.subs(r, 2*r)
    f = sp.simplify((a2 - sp.Rational(1, 2))*sp.exp(-r))
    crit = [s for s in sp.solve(sp.diff(f, r), r) if s > 1]
    fmax, f1 = sp.simplify(f.subs(r, crit[0])), sp.simplify(f.subs(r, 1))
    lo, hi = V.ans(2, "2-2").split("<k<")
    V.check(2, "2-2 하한 -f(2)", -fmax, E(lo)); V.check(2, "2-2 상한 -f(1)", -f1, E(hi))
    import numpy as np
    fl = sp.lambdify(r, f)
    xs = np.linspace(1 + 1e-9, 60, 600001)
    for kv, cnt in ((float(E(lo)) + 0.01, 2), (float(E(hi)) - 0.01, 2), (float(E(lo)) - 0.1, 0), (float(E(hi)) + 0.1, 1), (-0.01, 1)):
        g = fl(xs) + kv
        nz = int(np.sum(g[:-1]*g[1:] < 0))
        V.report(2, nz == cnt, f"2-2 k={kv:.3f} 실근 {nz}개 (기대 {cnt})")
    # 2-3
    rr, xx, yy = sp.symbols("rr xx yy", real=True)
    ar1 = ar.subs(r, rr + 1); arr = ar.subs(r, rr)
    xs_ = sp.solve(sp.expand((xx - ar1)**2 - (xx - arr)**2 - ((rr + 1)**2 - rr**2)), xx)[0]
    ys_ = sp.sqrt(sp.expand(rr**2 - (xs_ - arr)**2))
    tan = ys_/(ar1 - xs_)
    V.check(2, "2-3", sp.limit(tan, rr, sp.oo), E(V.ans(2, "2-3")))
    # 접점 확인: 원 C(r) 과 x=y^2/2 가 접함(수치)
    V.report(2, sp.simplify(sp.discriminant(sp.expand((X - ar)**2 + 2*X - r**2), X)) == 0, "2-1 접함 D=0")

def p03():
    import itertools
    from fractions import Fraction as Fr
    pm = {}
    for u, v in itertools.product(range(1, 7), repeat=2):
        if u != v: pm[max(u, v)] = pm.get(max(u, v), 0) + Fr(1, 30)
    EX = sum(kk*p for kk, p in pm.items())
    EY = 36*EX - 150
    Sv = {kk: Fr(1, 2)*(14 + 14 - 2*kk)*(49 - (kk - 7)**2) for kk in pm}
    pr = sum(p for kk, p in pm.items() if Sv[kk] >= 400)
    ans = V.ans(3, "3-1")
    V.check(3, "3-1 E(Y)", sp.Rational(EY.numerator, EY.denominator), E(V.ans(3, "3-1", "E")))
    V.check(3, "3-1 p+q", pr.numerator + pr.denominator, E(V.ans(3, "3-1", "sum")))
    T = sp.integrate(49 - (x - 7)**2, (x, 0, 14))
    kbest = max(Sv, key=lambda kk: Sv[kk])
    V.check(3, "3-2 k", kbest, E(V.ans(3, "3-2", "k")))
    V.check(3, "3-2 3T", 3*(T - sp.Rational(Sv[kbest].numerator, Sv[kbest].denominator)), E(V.ans(3, "3-2", "3T")))
    kk = 6; xa, xb = kk, 14 - kk
    h = sp.Piecewise((8*x, x <= 6), (48, x <= 8), (-8*x + 112, True))
    y1 = 8*x; assert y1.subs(x, 6) == (49 - 1)       # 포물선 위 D=(6,48)
    Vv = sp.integrate(8*x*sp.exp(-x), (x, 0, 6)) + sp.integrate(48*sp.exp(-x), (x, 6, 8)) + sp.integrate((-8*x + 112)*sp.exp(-x), (x, 8, 14))
    cf = sp.expand(Vv)
    for key, term in (("a", 1), ("b", sp.exp(-6)), ("c", sp.exp(-8)), ("d", sp.exp(-14))):
        V.check(3, f"3-3 {key}", cf.coeff(term) if term != 1 else cf.subs({sp.exp(-6): 0, sp.exp(-8): 0, sp.exp(-14): 0}), E(V.ans(3, "3-3", key)))

if __name__ == "__main__": V.run(globals())
