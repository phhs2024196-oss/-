import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-G26C", "items/EUL/PAST")

def _xs(v):
    return sp.sympify(v).subs(sp.Symbol("x"), x)

def p08():
    a = sp.symbols("a")
    q = 2*x**2 - 10*x + a
    xm = sp.solve(sp.diff(q, x), x)[0]  # 밑 1/3 <1 : f 최대 <=> 진수 최소
    V.check(8, "1 x", xm, V.ans(8, "1"))
    av = sp.solve(sp.Eq(sp.log(q.subs(x, xm), sp.Rational(1, 3)), 1), a)[0]
    V.check(8, "2 a", av, V.ans(8, "2"))
    V.report(8, av > sp.Rational(25, 2), "a>25/2")

def p09():
    f = sp.sin(x)**2 + sp.cos(x) - 1
    c = sp.symbols("c")
    g = f.subs(sp.sin(x)**2, 1 - c**2).subs(sp.cos(x), c)
    M = extrema(sp.expand(g), c, -1, 1)
    V.check(9, "1 M", M[1], V.ans(9, "1"))
    V.check(9, "2 M-m", M[1] - M[0], V.ans(9, "2"))
    vals = [f.subs(x, sp.Rational(i, 200)*sp.pi) for i in range(0, 401)]
    V.check(9, "2 grid", float(max(vals) - min(vals)), float(sp.Rational(V.ans(9, "2"))), 1e-3)

def p10():
    a1, r = sp.symbols("a1 r", positive=True)
    sol = sp.solve([a1*r*a1*r**3 - 100, a1*r**5 - 80], [a1, r], dict=True)[0]
    V.check(10, "1 r", sol[r], V.ans(10, "1"))
    V.check(10, "2 a10", (a1*r**9).subs(sol), V.ans(10, "2"))

def p11():
    a, b, c = sp.symbols("a b c")
    f = 2*x**3 + a*x**2 + b*x + c
    cc = sp.solve(f.subs(x, 0), c)[0]
    f0 = f.subs(c, cc)
    bv = sp.solve(sp.Eq(sp.limit(f0/x, x, 0), -8), b)[0]
    V.check(11, "1 b", bv, V.ans(11, "1"))
    av = sp.solve(sp.diff(f0, x).subs(x, 1), a)[0]
    fx = f0.subs({b: bv, a: av})
    V.check(11, "2 f", sp.expand(fx - _xs(V.ans(11, "2"))), 0)
    V.report(11, sp.diff(fx, x, 2).subs(x, 1) > 0, "x=1 극솟값")

def p12():
    k = sp.symbols("k")
    f = (x**2 - 9)/(x - 3)
    kv = sp.solve(sp.Eq(sp.limit(f, x, 3), k), k)[0]
    V.check(12, "1 k", kv, V.ans(12, "1"))
    V.check(12, "2 f(k)", f.subs(x, kv), V.ans(12, "2"))

def p13():
    a, b = sp.symbols("a b")
    f = sp.symbols("f0:4")
    # f 는 다항함수: (x-1)f(x)=4a x^3+4bx 이고 f(x) 는 이차식
    F = sp.Poly(sp.symbols("u0:3")[0] + sp.symbols("u0:3")[1]*x + sp.symbols("u0:3")[2]*x**2, x).as_expr()
    u = sp.symbols("u0:3")
    cond = sp.integrate(sp.expand((x - 1)*F).subs(x, sp.Symbol("t")).subs(sp.Symbol("t"), sp.Symbol("t")), x)  # 사용 안 함
    t_ = sp.Symbol("t")
    Fi = sp.integrate((t_ - 1)*F.subs(x, t_), (t_, 1, x))
    eqs = sp.Poly(sp.expand(Fi - (a*x**4 + 2*b*x**2 + 1)), x).coeffs()
    sol = sp.solve(eqs, [*u, a, b], dict=True)
    sol = [s for s in sol if s[a] != 0][0]
    V.check(13, "1 a", sol[a], V.ans(13, "1", "a"))
    V.check(13, "1 b", sol[b], V.ans(13, "1", "b"))
    fx = F.subs(sol)
    V.check(13, "2 min", extrema(fx, x, -5, 5)[0], V.ans(13, "2"))

def p14():
    tt = sp.symbols("tt", real=True)
    # 접점 (tt, tt^2), 접선이 (0,-2) 를 지남
    ts = sp.solve(sp.Eq(-2, tt**2 + 2*tt*(0 - tt)), tt)
    pts = {(s, s**2) for s in ts}
    V.check(14, "1 pts", len(pts), 2)
    exp = {(sp.sympify(s), sp.sympify(c)) for s, c in [(-sp.sqrt(2), 2), (sp.sqrt(2), 2)]}
    V.report(14, pts == exp and "sqrt(2)" in V.ans(14, "1"), f"1 접점 {sorted(map(str, pts))} / YAML {V.ans(14, '1')}")
    s2 = sp.sqrt(2)
    upper = x**2
    lo = sp.Piecewise((-2*s2*x - 2, x < 0), (2*s2*x - 2, True))
    A = integral(upper - (-2*s2*x - 2), x, -s2, 0) + integral(upper - (2*s2*x - 2), x, 0, s2)
    V.check(14, "2 area", A, V.ans(14, "2"))

if __name__ == "__main__": V.run(globals())
