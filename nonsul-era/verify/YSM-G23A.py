import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G23A", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "n": n, "k": k})

def p01():
    nn = sp.Symbol("nn", positive=True)
    # 1-1: 합 닫힌 형태가 없으므로 상·하계(샌드위치) 극한 + 큰 n 에서 수치 합이 1/2 근방
    lo = sp.limit(sp.Rational(1, 2)*nn*(nn + 1)/(nn**2 + nn), nn, sp.oo)
    hi = sp.limit(sp.Rational(1, 2)*nn*(nn + 1)/(nn**2 + 1), nn, sp.oo)
    V.check(1, "1-1 하계", lo, E(V.ans(1, "1-1"))); V.check(1, "1-1 상계", hi, E(V.ans(1, "1-1")))
    for N in (10, 100, 1000):
        s = sum(sp.Rational(j, N**2 + j) for j in range(1, N + 1))
        lowb = sp.Rational(N*(N + 1), 2)/(N**2 + N); upb = sp.Rational(N*(N + 1), 2)/(N**2 + 1)
        V.report(1, lowb <= s <= upb, f"1-1 N={N} 하계≤합≤상계 (합={float(s):.5f})")
    # 1-2: 부분합 닫힌 식과 직접 합 비교, 극한
    Sn = lambda N: sum(sp.Rational(2, (2*j - 1)*(2*j + 3)) for j in range(1, N + 1))
    closed = lambda N: sp.Rational(1, 2)*(1 + sp.Rational(1, 3) - sp.Rational(1, 2*N + 1) - sp.Rational(1, 2*N + 3))
    V.report(1, all(Sn(N) == closed(N) for N in range(1, 40)), "1-2 S_n 닫힌 식 (n=1..39)")
    V.check(1, "1-2 급수의 합", sp.summation(sp.Rational(2, 1)/((2*k - 1)*(2*k + 3)), (k, 1, sp.oo)), E(V.ans(1, "1-2")))
    V.check(1, "1-2 lim S_n", sp.limit(closed(nn) if False else sp.Rational(1, 2)*(1 + sp.Rational(1, 3) - 1/(2*nn + 1) - 1/(2*nn + 3)), nn, sp.oo), E(V.ans(1, "1-2")))

def p02():
    a, b, c, d, kk = sp.symbols("a b c d kk", real=True)
    f1, f2, f3 = a*x + b, -x**2 + 9, c*x + d
    eqs = [sp.Eq(f1.subs(x, -3), f2.subs(x, -3)), sp.Eq(sp.diff(f1, x), sp.diff(f2, x).subs(x, -3)),
           sp.Eq(f2.subs(x, 0), f3.subs(x, 0)), sp.Eq(sp.diff(f3, x), sp.diff(f2, x).subs(x, 0))]
    sol = sp.solve(eqs, [a, b, c, d], dict=True)[0]
    for key, s in (("a", a), ("b", b), ("c", c), ("d", d)):
        V.check(2, f"2-1 {key}", sol[s], E(V.ans(2, "2-1", key)))
    # 접함: x^2+k x+3-k=0 중근, k>0
    kp = sp.Symbol("kp", positive=True)
    ks = sp.solve(sp.discriminant(sp.expand(f2 - (kp*(x - 1) + 12)), x), kp)
    V.check(2, "2-1 k", ks[0], E(V.ans(2, "2-1", "k")))
    # 2-2: 영역 넓이. 수치 격자 적분(위쪽 함수-아래쪽 함수, y>=0 부분)과 정확 적분 비교
    kv = ks[0]
    F = sp.Piecewise((sol[a]*x + sol[b], x < -3), (f2, x < 0), (sol[c]*x + sol[d], True))
    G = kv*(x - 1) + 12
    A = sp.integrate(G, (x, -5, -3)) + sp.integrate(G - f2, (x, -3, -1))
    V.check(2, "2-2 넓이", A, E(V.ans(2, "2-2")))
    import numpy as np
    xs = np.linspace(-6, 1, 700001); Fn = sp.lambdify(x, F, "numpy")(xs); Gn = float(kv)*(xs - 1) + 12
    # 영역: y>=0 이고 f<=y<=g 를 만족하는 점의 집합(f 는 x<-1 부근에서 g 아래)
    top = np.maximum(Gn, 0); lo = np.maximum(np.minimum(Fn, Gn), 0)
    area = np.trapezoid(np.where(Gn >= 0, np.maximum(Gn - np.maximum(Fn, 0), 0), 0)*(xs <= -1), xs)
    V.report(2, abs(area - float(A)) < 1e-3, f"2-2 수치 적분 {area:.5f} vs {float(A):.5f}")
    V.report(2, sp.simplify(sp.diff(f2, x).subs(x, -1) - kv) == 0, "2-2 접점 x=-1 에서 기울기 k")

def p03():
    f = lambda z: z**3 - 2*z**2 - 5
    t_ = sp.Symbol("t_", real=True)
    cps = [r for r in sp.solve(sp.diff(f(t_), t_), t_) if 1 <= r <= 5]
    vals = [f(1), f(5)] + [f(r) for r in cps]
    M, m = max(vals), min(vals)
    V.check(3, "3-1 M", M, E(V.ans(3, "3-1", "M"))); V.check(3, "3-1 m", m, E(V.ans(3, "3-1", "m")))
    V.check(3, "3-1 27m+M", 27*m + M, E(V.ans(3, "3-1", "ans")))
    import numpy as np
    xs = np.linspace(-200, 200, 4000001)
    yv = np.array(sp.lambdify(x, f(2*sp.sin(x/5) + 3), "numpy")(xs), dtype=float)
    V.report(3, abs(yv.max() - float(M)) < 1e-6 and abs(yv.min() - float(m)) < 1e-6, f"3-1 수치 최대 {yv.max():.6f} 최소 {yv.min():.6f}")
    # 3-2: g 의 극값을 수치로 판정(부호 변화) + 정확한 근
    g = sp.Rational(5, 2)*sp.Abs(f(x - 1) + 3)
    gn = sp.lambdify(x, g, "numpy")
    xs = np.linspace(-3, 6, 9000001); gv = gn(xs)
    dg = np.diff(gv); sg = np.sign(dg)
    idx = [i for i in range(1, len(sg)) if sg[i] != sg[i - 1] and sg[i] != 0 and sg[i - 1] != 0]
    # 부호 0 구간 처리를 피하려고 0 을 이웃 부호로 채움
    s2 = sg.copy()
    for i in range(1, len(s2)):
        if s2[i] == 0: s2[i] = s2[i - 1]
    idx = [i for i in range(1, len(s2)) if s2[i] != s2[i - 1]]
    pts = [(xs[i], gv[i], "min" if s2[i] > s2[i - 1] else "max") for i in idx]
    alpha = len(pts); beta = sum(v for _, v, t in pts if t == "min")
    V.check(3, "3-2 α", alpha, E(V.ans(3, "3-2", "alpha"))); V.check(3, "3-2 β", beta, E(V.ans(3, "3-2", "beta")), tol=1e-4)
    V.check(3, "3-2 αβ", alpha*beta, E(V.ans(3, "3-2", "ab")), tol=1e-5)
    # 정확 계산: C 는 x^3-2x^2-2 의 실근 +1
    C = [r for r in sp.Poly(sp.expand(f(x - 1) + 3), x).nroots() if abs(sp.im(r)) < 1e-12][0]
    V.report(3, abs(float(sp.re(C)) - [p[0] for p in pts if p[2] == 'min' and p[1] < 1e-3][0]) < 1e-3, f"3-2 C={float(sp.re(C)):.4f} 에서 극소(값 0)")
    V.check(3, "3-2 g(1)", g.subs(x, 1), 5)

if __name__ == "__main__": V.run(globals())
