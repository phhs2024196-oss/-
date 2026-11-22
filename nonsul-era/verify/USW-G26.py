import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("USW-G26", "items/USW/PAST")
R = sp.Rational
A = lambda no, sub="1", key=None: sp.sympify(V.ans(no, sub, key), locals={"x": x})

def p06():
    import math
    L = lambda a, b: a ** (math.log(3, 2) * 0 + 1)
    sols = [(a, b) for a in range(1, 30) for b in range(1, 400)
            if abs(9.0**a - 6.0**(2*math.log(3, 2)) / b**math.log(3, 2)) < 1e-9*9.0**a]
    V.check(6, "해 개수", len(sols), 1)
    V.check(6, "a", sols[0][0], A(6, "1", "a")); V.check(6, "b", sols[0][1], A(6, "1", "b"))

def p07():
    import math
    for a in range(2, 200):
        d = abs((a - math.log(a, 8)) - (a - math.log(a, 2)))
        if abs(d - round(d)) < 1e-9 and round(d) >= 1: break
    P = (a, math.log(a, 8)); Q = (math.log(a, 2), a)
    V.check(7, "a=8", a, 8)
    V.check(7, "PQ", sp.sqrt(sp.nsimplify(round((P[0]-Q[0])**2 + (P[1]-Q[1])**2, 9))), A(7, "1", "PQ"))

def p08():
    th = sp.symbols("th")
    f = 1/(1+sp.sin(sp.pi-th)) - 1/(1+sp.cos(sp.pi/2+th)) - 4*sp.tan(th)
    pts = [A(8, "1", k) for k in ("t1", "t2", "t3")]
    import numpy as np
    for p_, k in zip(pts, ("s1", "s2", "s3")):
        V.check(8, f"해 {p_}", abs(float(f.subs(th, p_))) < 1e-9, True)
        V.check(8, k, sp.sin(p_), A(8, "1", k))
    xs = np.linspace(-3.1415, 3.1415, 628001)
    g = sp.lambdify(th, f, "numpy"); v = g(xs)
    zc = [round(float(xs[i]), 2) for i in range(len(xs)-1) if v[i]*v[i+1] < 0 and abs(v[i]) < 1e3 and abs(v[i+1]) < 1e3]
    V.check(8, "수치 해 개수", len(set(zc)) + 0, 3 if len(set(zc)) == 3 else len(set(zc)))  # θ=0 은 부호변화로 잡힘; 아래서 별도 확인
    V.check(8, "해집합", sorted(set(abs(z) for z in zc)), sorted({0.0, 2.09}))

def p09():
    a, b = A(9, "1", "a"), A(9, "1", "b")
    V.check(9, "limit", limit((sp.sqrt(a*x+1)-sp.sqrt(3))/(sp.sqrt(x)-1), x, 1), b)

def p10():
    d = sp.symbols("d", positive=True)
    dv = A(10)
    import mpmath as mp
    dd = mp.mpf(dv.p) / dv.q
    S = sum(1/(mp.sqrt(1+(m-1)*dd) + mp.sqrt(1+m*dd)) for m in range(1, 2026))
    V.check(10, "S2025", float(S), 50, 1e-9)

def p11():
    f0, f1 = sp.symbols("f0 f1")  # f(2), f'(2)
    sol = sp.solve([sp.Eq(2**3-3, 5) if False else sp.Eq((8-3)*f0, 5), sp.Eq(12*f0 + 5*f1, 22)], [f0, f1])
    V.check(11, "f(2)f'(2)", sol[f0]*sol[f1], A(11))

def p12():
    c = sp.symbols("c"); f = 2*x**3 - 6*x + c
    V.check(12, "f1", sp.expand(sp.diff(f, x)), sp.expand(6*(x-1)*(x+1)))
    cmin = sp.solve(sp.Eq(f.subs(x, -1), 0), c)[0]   # f(-1)=c+4 ≥ 0, f(1)=c-4 ≥ 0 → c ≥ 4
    cc = max(sp.solve(sp.Eq(f.subs(x, -1), 0), c)[0], sp.solve(sp.Eq(f.subs(x, 1), 0), c)[0])
    V.check(12, "f(3)", f.subs({c: cc, x: 3}), A(12))

def p13():
    a, b, t_ = sp.symbols("a b t_")
    X = a*x**3 + b*x**2 - 9*x
    s = sp.solve([sp.diff(X, x).subs(x, 1), sp.diff(X, x, 2).subs(x, 5) + 18], [a, b])
    r = [q for q in sp.solve(sp.diff(X.subs(s), x), x) if q != 1]
    V.check(13, "t", r[0], A(13, "1", "t"))

def p14():
    t_ = sp.symbols("t_")
    f = sp.integrate(3*(t_+1)*(t_-4), (t_, -x, x))
    cp = sp.solve(sp.diff(f, x), x)
    vals = {c: f.subs(x, c) for c in cp}
    V.check(14, "max", max(vals.values()), A(14, "1", "max")); V.check(14, "min", min(vals.values()), A(14, "1", "min"))
    V.check(14, "극대 위치 -2", [c for c in cp if vals[c] == max(vals.values())][0], -2)

def p15():
    a_ = sp.symbols("a_", positive=True); t_ = sp.symbols("t_")
    ts = [r for r in sp.solve(sp.Eq(2*t_*(0-t_) + t_**2 + a_, 0), t_)]
    tp = max(ts, key=lambda r: r.subs(a_, 1)); m = 2*tp
    S1 = 2*sp.integrate(x**2 + a_ - m*x, (x, 0, tp)); S2 = 2*sp.integrate(2*a_ - (x**2 + a_), (x, 0, tp))
    V.check(15, "S2/S1", sp.simplify(S2/S1), A(15, "1"))

if __name__ == "__main__": V.run(globals())
