import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DSW-G25", "items/DSW/PAST")
kk, tt, dd, xs = sp.symbols("k t d xs", real=True)
pos = sp.symbols("kp", positive=True)
E = lambda v: sp.sympify(v, locals={"x": x, "n": n, "k": pos})

def p01():
    k_ = pos
    f = lambda z: z**3 + 3*k_*z**2
    # 1-1: 접점 t, 접선이 (1,d)를 지나는 방정식의 서로 다른 실근 3개 <=> 두 극값 곱 < 0
    tan = sp.diff(f(x), x).subs(x, tt)*(x - tt) + f(tt)
    l = sp.expand(sp.expand(tan.subs(x, 1)) - dd)  # = -(l(t)) 꼴
    l = sp.expand(-l)
    crit = sp.solve(sp.diff(l, tt), tt)
    vals = sorted([sp.simplify(l.subs(tt, c)) for c in crit], key=lambda e: sp.N(e.subs({dd: 0, k_: 2})))
    lo, hi = sp.solve(vals[0], dd)[0], sp.solve(vals[1], dd)[0]
    lo, hi = sorted([lo, hi], key=lambda e: sp.N(e.subs(k_, 2)))
    a, b = V.ans(1, "1-1").split("<d<")
    V.check(1, "1-1 하한", lo, E(a)); V.check(1, "1-1 상한", hi, E(b))
    # 수치 확인: k=2 에서 d 가 범위 안/밖일 때 실근 수
    for kv in (1, 2, 5):
        lo_v, hi_v = float(lo.subs(k_, kv)), float(hi.subs(k_, kv))
        for dv, cnt in ((lo_v + 0.01, 3), (hi_v - 0.01, 3), (lo_v - 1, 1), (hi_v + 1, 1)):
            r = sp.Poly(l.subs({k_: kv, dd: dv}), tt).nroots()
            real = len([z for z in r if abs(sp.im(z)) < 1e-9])
            V.report(1, real == cnt, f"1-1 k={kv} d={dv:.2f} 접선 {real}개 (기대 {cnt})")
    # 1-2: L(k) = (세 접선 불가 m<0 의 최댓값) = lo
    L = lo
    V.check(1, "1-2 L", L, E(V.ans(1, "1-2", "L")))
    S = sp.integrate(f(x) - L*x, (x, 0, 1))
    V.check(1, "1-2 S", sp.expand(S), E(V.ans(1, "1-2", "S")))
    V.check(1, "1-2 lim", sp.limit(S/L, k_, sp.oo), E(V.ans(1, "1-2", "lim")))
    # 1-3: 평균값 정리 c=1 -> k, 극값 없음 -> p 상한, 적분 계수 -> q
    kc = sp.symbols("kc")
    kval = sp.solve(sp.Eq((kc*0 + (27 + 27*kc) - (-27 + 27*kc))/6, 3 + 6*kc), kc)[0]
    fx = x**3 + 3*kval*x**2
    p, q = sp.symbols("p q", real=True)
    h = fx - (x**2 + p*x + q)
    disc = sp.discriminant(sp.diff(h, x), x)
    pmax = sp.solve(sp.Eq(disc, 0), p)[0]
    V.check(1, "1-3 p_max", pmax, E(V.ans(1, "1-3", "p_max")))
    I = sp.integrate((x - t)*q*t**2, (t, -1, x)) + sp.integrate((t - x)*q*t**2, (t, x, 1))
    qv = sp.solve(sp.Eq(sp.Poly(sp.expand(I), x).coeff_monomial(x**4), 1), q)[0]
    V.check(1, "1-3 q", qv, E(V.ans(1, "1-3", "q")))

def p02():
    O = sp.Matrix([0, 0]); P = sp.Matrix([2, 1]); Q = sp.Matrix([sp.Rational(24, 11), 0])
    def foot_on_OP(Qm):  # (Qm,0)에서 OP로 내린 수선의 발
        u = P/sp.sqrt(5)
        return (sp.Matrix([Qm, 0]).dot(u))*u
    Q1x = P[0]
    Ps = {}; Qx = {1: Q1x}
    for i in range(1, 15):
        Ps[i] = foot_on_OP(Qx[i])
        Qx[i + 1] = Ps[i][0]
    # 2-1: 직선 P1Q1 과 A2
    slope = (Ps[1][1] - 0)/(Ps[1][0] - Qx[1])
    line = sp.simplify(slope*(x - Qx[1]))
    V.check(2, "2-1 직선", line, E(V.ans(2, "2-1", "line")))
    tri = lambda a, b, c: abs((b - a).cross(c - a)[0] if False else ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])))/2
    P1, Q1, Q2 = Ps[1], sp.Matrix([Qx[1], 0]), sp.Matrix([Qx[2], 0])
    V.check(2, "2-1 A2", tri(P1, Q2, Q1), E(V.ans(2, "2-1", "A2")))
    # 2-2: x_n = P_n 의 x좌표
    for i in range(1, 15):
        V.report(2, sp.simplify(Ps[i][0] - E(V.ans(2, "2-2")).subs(n, i)) == 0, f"2-2 x_{i}")
    # 2-3
    A = {1: tri(P, sp.Matrix([Qx[1], 0]), Q)}
    for kv in range(2, 15):
        A[kv] = tri(Ps[kv - 1], sp.Matrix([Qx[kv], 0]), sp.Matrix([Qx[kv - 1], 0]))
    V.check(2, "2-3 A1", A[1], E(V.ans(2, "2-3", "A1")))
    kf = sp.sympify(V.ans(2, "2-3", "A_k"), locals={"k": k})
    V.report(2, all(sp.simplify(A[kv] - kf.subs(k, kv)) == 0 for kv in range(2, 15)), "2-3 A_k (k=2..14)")
    nmin = next(m for m in range(2, 15) if A[1] + sum(A[i] for i in range(2, m + 1)) > sp.Rational(5, 11))
    V.check(2, "2-3 n", nmin, E(V.ans(2, "2-3", "n")))

if __name__ == "__main__": V.run(globals())
