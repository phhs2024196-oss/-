import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G26A", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "t": t, "n": n, "k": k})

def p01():
    a, b, h = sp.symbols("a b h", real=True)
    f2 = a*x*sp.exp(b*x)
    # 연속 + 좌우 미분계수 일치
    sol = sp.solve([sp.Eq(f2.subs(x, 2), 2*2 - 2), sp.Eq(sp.diff(f2, x).subs(x, 2), 2)], [a, b], dict=True)[0]
    V.check(1, "1-1 a", sol[a], E(V.ans(1, "1-1", "a")))
    V.check(1, "1-1 b", sol[b], E(V.ans(1, "1-1", "b")))
    # 우미분계수를 극한 정의로 직접
    av, bv = sol[a], sol[b]
    rd = sp.limit(((av*(2 + h)*sp.exp(bv*(2 + h)) - av*2*sp.exp(2*bv))/h), h, 0, "+")
    V.check(1, "1-1 우미분계수=좌미분계수 2", rd, 2)
    # 1-2: 넓이 (f,g 의 위아래를 구간별로 확인)
    f = lambda z: (2*z - 2) if z < 2 else av*z*sp.exp(bv*z)
    g = lambda z: sp.Abs((z - 1)*(z - 3)) - 5
    V.report(1, all(float(f(sp.Rational(i, 10)) - g(sp.Rational(i, 10))) >= -1e-12 for i in range(0, 41)), "1-2 [0,4]에서 f>=g")
    area = 0
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
        mid = sp.Rational(lo + hi, 2)
        fe = 2*x - 2 if hi <= 2 else av*x*sp.exp(bv*x)
        ge = (x - 1)*(x - 3) - 5 if mid < 1 or mid > 3 else -(x - 1)*(x - 3) - 5
        area += sp.integrate(fe - ge, (x, lo, hi))
    V.check(1, "1-2 넓이", sp.simplify(area), E(V.ans(1, "1-2")))
    import mpmath as mp
    fm = lambda z: (2*z - 2 if z < 2 else float(av)*z*mp.exp(float(bv)*z)) - (abs((z - 1)*(z - 3)) - 5)
    V.check(1, "1-2 수치적분", sp.Float(mp.quad(fm, [0, 1, 2, 3, 4]), 15), sp.N(E(V.ans(1, "1-2")), 15), 1e-8)

def p02():
    c, N = sp.symbols("c N", positive=True)
    nn = sp.symbols("nn", integer=True, positive=True)
    f = sp.exp(c*x)
    step = lambda a: a - f.subs(x, a)/sp.diff(f, x).subs(x, a)
    a_ = [N]
    for i in range(8): a_.append(sp.simplify(step(a_[-1])))
    tri = lambda i: sp.Rational(1, 2)*abs(a_[i] - a_[i + 1])*f.subs(x, a_[i])   # 0-index: S_{i+1}
    V.check(2, "2-1 S1", sp.simplify(tri(0)), E(V.ans(2, "2-1")).subs({"c": c, "N": N}) if False else sp.sympify(V.ans(2, "2-1"), locals={"c": c, "N": N}))
    S2 = sp.sympify(V.ans(2, "2-2"), locals={"c": c, "N": N, "n": nn})
    V.report(2, all(sp.simplify(tri(i) - S2.subs(nn, i + 1)) == 0 for i in range(8)), "2-2 S_n (n=1..8)")
    V.report(2, all(sp.simplify(a_[i] - (N - sp.Rational(i, 1)/c)) == 0 for i in range(9)), "2-2 a_n=N-(n-1)/c")
    S3 = sp.sympify(V.ans(2, "2-3"), locals={"c": c, "N": N})
    tot = sp.summation(S2, (nn, 1, sp.oo))
    V.check(2, "2-3 급수", sp.simplify(tot - S3), 0)
    # 2-4: c=1, e=2.7 와 실제 e 모두에서 최소 n = N+4
    for Nv in (2, 3, 7):
        for ev in (2.7, float(sp.E)):
            m = next(i for i in range(1, 100) if ev**(Nv - i + 1) < 0.1)
            V.check(2, f"2-4 N={Nv} e={ev:.3f}", m, sp.sympify(V.ans(2, "2-4"), locals={"N": N}).subs(N, Nv))

def p03():
    k_ = sp.symbols("k_", real=True)
    f = x**3 - 12*x**2 + 45*x - k_
    crit = sorted(sp.solve(sp.diff(f, x), x))
    V.check(3, "3-1 극점", crit, [3, 5])
    lo = sp.solve(f.subs(x, 5), k_)[0]; hi = sp.solve(f.subs(x, 3), k_)[0]
    V.check(3, "3-1 하한", lo, E(V.ans(3, "3-1", "k_lo"))); V.check(3, "3-1 상한", hi, E(V.ans(3, "3-1", "k_hi")))
    ok = True
    for kv in [sp.Rational(i, 4) for i in range(160, 240)]:
        import numpy as np
        rr = np.roots([1, -12, 45, -float(kv)])
        allreal = all(abs(r.imag) < 1e-6 for r in rr)
        ok &= (allreal == (lo <= kv <= hi))
    V.report(3, ok, "3-1 k∈[40,60) 격자에서 '세 근 모두 실수 ⇔ 50<=k<=54'")
    # 3-2
    trip = sorted({tuple(sorted((a, b, 12 - a - b))) for a in range(1, 12) for b in range(1, 12) if 12 - a - b > 0
                   if a*b + b*(12 - a - b) + a*(12 - a - b) == 45})
    V.report(3, trip == [(2, 5, 5), (3, 3, 6)], "3-2 순서쌍 (2,5,5),(3,3,6)")
    ks = sorted({a*b*c for a, b, c in trip})
    V.check(3, "3-2 k", ks, [E(V.ans(3, "3-2", "k1")), E(V.ans(3, "3-2", "k2"))])
    V.report(3, all(sp.expand((x - a)*(x - b)*(x - c) - (x**3 - 12*x**2 + 45*x - a*b*c)) == 0 for a, b, c in trip), "3-2 해당 k 에서 근이 (a,b,c)")
    # 3-3: 삼각형이 되는 조합으로 실제 각을 계산
    tri = [(a, b, c) for a, b, c in trip if a + b > c and b + c > a and c + a > b]
    V.report(3, tri == [(2, 5, 5)], "3-3 삼각형은 (2,5,5)뿐")
    a_, b_, c_ = [sp.Integer(v) for v in tri[0]]
    A = sp.acos((b_**2 + c_**2 - a_**2)/(2*b_*c_)); B = sp.acos((c_**2 + a_**2 - b_**2)/(2*c_*a_)); C = sp.acos((a_**2 + b_**2 - c_**2)/(2*a_*b_))
    V.check(3, "3-3 내각합 π", sp.N(A + B + C, 20), sp.N(sp.pi, 20), 1e-12)
    V.check(3, "3-3 값", sp.N(a_*b_*sp.cos(C) + b_*c_*sp.cos(A) + c_*a_*sp.cos(B), 20), sp.N(E(V.ans(3, "3-3")), 20), 1e-12)

if __name__ == "__main__": V.run(globals())
