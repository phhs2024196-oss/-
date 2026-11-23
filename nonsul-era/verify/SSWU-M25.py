import sys, pathlib, math; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SSWU-M25", "items/SSWU/PAST")
N_, T_ = sp.Symbol("n"), sp.Symbol("t")

def kv(s): return {p.split("=", 1)[0]: S(p.split("=", 1)[1]) for p in s.replace(" ", "").split(";")}

def p01():
    a1 = kv(V.ans(1, "1"))
    f = lambda nn: sum(sp.sin(k*x + sp.pi/4)*sp.sin((k+1)*x + sp.pi/4) for k in range(1, nn+1))
    V.check(1, "1-1 f(0)", f(1).subs(x, 0), a1["f0"]); V.check(1, "1-1 f'(0)", sp.diff(f(1), x).subs(x, 0), a1["fp0"])
    a2 = kv(V.ans(1, "2")); ok = True
    for nn in range(1, 13):
        ok &= same(f(nn).subs(x, 0), a2["f0"].subs(N_, nn)) and same(sp.diff(f(nn), x).subs(x, 0), a2["fp0"].subs(N_, nn))
    V.report(1, ok, "1-2 f(0), f'(0) 식 (n=1..12 정확값)")
    k_ = sp.Symbol("k", integer=True, positive=True); nn = sp.Symbol("nn", integer=True, positive=True)
    fp = sp.summation((k_ + (k_+1))*sp.Rational(1, 2), (k_, 1, nn)); f0 = sp.summation(sp.Rational(1, 2), (k_, 1, nn))
    A = sp.Rational(1, 2)*abs(-f0/fp)*f0   # 접선 y=fp x+f0: x절편 -f0/fp, y절편 f0
    V.check(1, "1-3 lim A_n", sp.limit(sp.simplify(A), nn, sp.oo), V.ans(1, "3"))

def p02():
    a, b, c = sp.symbols("a b c", real=True); ex = x**2 + a*x + b
    sol = sp.solve([ex.subs(x, 3), sp.diff(ex, x).subs(x, 3) - 2], [a, b])   # 극한 존재: f(3)=0, f'(3)=2
    ans = {p.split("=")[0]: S(p.split("=")[1]) for p in V.ans(2, "1").replace(" ", "").split(";")}
    V.report(2, sol[a] == ans["a"] and sol[b] == ans["b"] and same(sp.limit((x**2 + sol[a]*x + sol[b])/(x - 3), x, 3), 2), f"2-1 a,b = {sol}")
    f = x**2 + sol[a]*x + sol[b]; g = x**3 + x**2 - 7*x + c
    cs = set()
    for xv in sp.solve(sp.diff(g, x) - sp.diff(f, x), x):
        cs |= set(sp.solve((g - f).subs(x, xv), c))
    exp = {S(s) for s in V.ans(2, "2").replace(" ", "").split(";")}
    V.report(2, cs == exp, f"2-2 c {cs} / YAML {exp}")
    h = sp.expand((g - f).subs(c, min(cs)))
    rts = sorted(sp.solve(h, x)); area_ = integral(abs(sp.Piecewise((h, True))) if False else -h, x, rts[0], rts[-1])
    V.check(2, "2-3 넓이(h≤0 구간 확인)", area_, V.ans(2, "3"))
    V.report(2, all(h.subs(x, v) <= 0 for v in [-0.9, 0, 1, 1.9]), "2-3 구간 부호")

def p03():
    xt = 2*sp.log(T_**2 + 1); yt = T_**2 + 1/(T_**2 + 1)
    dydx = sp.diff(yt, T_)/sp.diff(xt, T_)
    V.check(3, "3-1 dy/dx(t=1)", sp.simplify(dydx).subs(T_, 1), V.ans(3, "1"))
    f2 = sp.simplify(sp.diff(xt, T_)**2 + sp.diff(yt, T_)**2)
    fe = S(V.ans(3, "2").replace("t", "T_")).subs(sp.Symbol("T_"), T_) if False else sp.sympify(V.ans(3, "2"), locals={"t": T_})
    V.report(3, all(abs(float((f2 - fe**2).subs(T_, v))) < 1e-9 for v in [0, 0.3, 1, 2.5, 7]) and sp.simplify(f2 - fe**2) == 0, "3-2 f(t)^2 = x'^2+y'^2 (기호 일치)")
    L = sp.integrate(sp.simplify(fe), (T_, 0, 1))
    V.check(3, "3-3 이동거리", L, V.ans(3, "3"))
    Lnum = sp.Integral(sp.sqrt(f2), (T_, 0, 1)).evalf(20)
    V.check(3, "3-3 수치적분", Lnum, V.ans(3, "3"), 1e-12)

def p04():
    def step(poly):
        out = []; m = len(poly); tot = 0.0; cnt = 0
        for i in range(m):
            P, Q = poly[i], poly[(i+1) % m]
            d = (Q[0]-P[0], Q[1]-P[1]); l = math.hypot(*d); u = (d[0]/l, d[1]/l); nrm = (-u[1], u[0])  # 반시계: 왼쪽=안쪽
            P1 = (P[0]+0.4*d[0], P[1]+0.4*d[1]); P2 = (P[0]+0.6*d[0], P[1]+0.6*d[1])
            h = math.sqrt((0.4*l)**2 - (0.1*l)**2)
            M = ((P1[0]+P2[0])/2 + h*nrm[0], (P1[1]+P2[1])/2 + h*nrm[1])
            out += [P, P1, M, P2]
        return out
    ar = lambda P: abs(sum(P[i][0]*P[(i+1)%len(P)][1] - P[(i+1)%len(P)][0]*P[i][1] for i in range(len(P))))/2
    poly = [(math.cos(math.pi/3*k), math.sin(math.pi/3*k)) for k in range(6)]
    Ns, Ss = [], [ar(poly)]
    for _ in range(6):
        Ns.append(len(poly)); poly = step(poly); Ss.append(ar(poly))
    xs = [Ss[i] - Ss[i+1] for i in range(6)]
    Nn = S(V.ans(4, "1"))
    V.report(4, all(Ns[i] == Nn.subs(N_, i+1) for i in range(6)), f"4-1 N_n=변의 수 n=1..6 {Ns}")
    kv2 = kv(V.ans(4, "2"))
    rok = all(abs(xs[i+1]/xs[i] - float(kv2["r"])) < 1e-9 for i in range(5)) and abs(xs[0] - float(kv2["x1"])) < 1e-9
    V.report(4, rok, f"4-2 x1={xs[0]:.10f}, 공비 {xs[1]/xs[0]:.10f} (기하 구성, n=1..6)")
    lim = Ss[0] - xs[0]/(1 - xs[1]/xs[0])
    V.check(4, "4-3 lim S_n", sp.Float(lim, 20), V.ans(4, "3"), 1e-9)

if __name__ == "__main__": V.run(globals())
