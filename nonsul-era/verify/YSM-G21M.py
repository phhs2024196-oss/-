import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G21M", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "k": k, "t": t})

def p01():
    a, b, N = sp.Rational(1, 2), sp.Rational(1, 20), 100
    f, g = sp.symbols("f g", positive=True)
    # 1-1: g' = (a/N f g - b g) < 0  <=>  f < bN/a
    gp = a/N*g*f - b*g
    sol = sp.solve(sp.Eq(gp, 0), f)[0]
    V.check(1, "1-1 경계 bN/a", sol, E(V.ans(1, "1-1").split("<")[1]))
    V.report(1, gp.subs({f: sol - 1, g: 3}) < 0 and gp.subs({f: sol + 1, g: 3}) > 0, "1-1 부등호 방향(수치)")
    # 1-2: 연립 ODE 수치적분(RK4)으로 h=20 일 때의 f
    a2, b2 = 0.5, 0.1
    def rhs(s):
        F, G, H = s
        return [-a2/N*G*F, a2/N*G*F - b2*G, b2*G]
    s = [99.0, 1.0, 0.0]; dt = 0.001
    import math
    for _ in range(400000):
        k1 = rhs(s); k2 = rhs([s[i] + dt/2*k1[i] for i in range(3)])
        k3 = rhs([s[i] + dt/2*k2[i] for i in range(3)]); k4 = rhs([s[i] + dt*k3[i] for i in range(3)])
        new = [s[i] + dt/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(3)]
        if new[2] >= 20:
            frac = (20 - s[2])/(new[2] - s[2]); s = [s[i] + frac*(new[i] - s[i]) for i in range(3)]; break
        s = new
    exp = float(E(V.ans(1, "1-2")))
    V.report(1, abs(s[0] - exp) < 1e-3, f"1-2 ODE 수치해 f(t*)={s[0]:.5f} / YAML {exp:.5f}")
    # 기호 풀이: ln f = -(a/(bN)) h + C
    h = sp.Symbol("h")
    fh = 99*sp.exp(-sp.Rational(1, 2)/(sp.Rational(1, 10)*100)*h)
    V.check(1, "1-2 f(h=20)", fh.subs(h, 20), E(V.ans(1, "1-2")))
    # 1-3: P_{k+1} = k/(k+2) P_k, P_1 = 1, 합 = 1
    P = {1: sp.Integer(1)}
    for kk in range(1, 100): P[kk + 1] = sp.Rational(kk, kk + 2)*P[kk]
    px = {kk: P[kk] - (P[kk + 1] if kk < 100 else 0) for kk in range(1, 101)}
    assert sum(px.values()) == 1 and P[100] == px[100]
    Pr = lambda lo, hi: sum(px[i] for i in range(lo, hi + 1))
    V.check(1, "1-3 비", Pr(1, 3)/Pr(6, 19), E(V.ans(1, "1-3", "ratio")))
    kmin = next(kk for kk in range(1, 101) if Pr(1, kk) > sp.Rational(49, 50))
    V.check(1, "1-3 k", kmin, E(V.ans(1, "1-3", "k")))

def p02():
    al, be = sp.symbols("al be", real=True)
    ell = 6
    # 위치: A(-3,0), B(3,0), tan al = cy/(3+cx), tan be = cy/(3-cx)
    cx_, cy_ = sp.symbols("cx_ cy_")
    s = sp.solve([sp.Eq(sp.tan(al)*(3 + cx_), cy_), sp.Eq(sp.tan(be)*(3 - cx_), cy_)], [cx_, cy_], dict=True)[0]
    CX, CY = s[cx_], s[cy_]
    # 2-1: al+be = 5pi/12 에서 al 이 pi/12 → pi/4 로 움직일 때 C 자취 길이
    be1 = 5*sp.pi/12 - al
    px, py = CX.subs(be, be1), CY.subs(be, be1)
    sp_ = sp.sqrt(sp.diff(px, al)**2 + sp.diff(py, al)**2)
    L = sp.Integral(sp_, (al, sp.pi/12, sp.pi/4)).evalf(30)
    V.check(2, "2-1 자취 길이(수치적분)", L, E(V.ans(2, "2-1")), tol=1e-9)
    # 2-2: t 매개변수
    tt = sp.Symbol("tt")
    a_t, b_t = sp.pi/12 + 3*tt, sp.pi/3 - 3*tt
    cxt, cyt = CX.subs({al: a_t, be: b_t}), CY.subs({al: a_t, be: b_t})
    t0 = sp.pi/36
    for key, expr in (("cx", cxt), ("cy", cyt), ("vx", sp.diff(cxt, tt)), ("vy", sp.diff(cyt, tt))):
        V.check(2, f"2-2 {key}", sp.simplify(expr.subs(tt, t0)), E(V.ans(2, "2-2", key)))
    # 2-3: 기댓값
    pr = {sp.pi/6: sp.Rational(1, 6), sp.pi/4: sp.Rational(1, 2), sp.pi/3: sp.Rational(1, 3)}
    Ex = sum(p*sp.simplify(CX.subs({al: a_, be: 5*sp.pi/12 - a_})) for a_, p in pr.items())
    Ey = sum(p*sp.simplify(CY.subs({al: a_, be: 5*sp.pi/12 - a_})) for a_, p in pr.items())
    V.check(2, "2-3 E(cx)", sp.nsimplify(sp.simplify(Ex)), E(V.ans(2, "2-3", "Ecx")))
    V.check(2, "2-3 E(cy)", sp.nsimplify(sp.simplify(Ey)), E(V.ans(2, "2-3", "Ecy")))

if __name__ == "__main__": V.run(globals())
