import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-G25B", "items/EUL/PAST")
L = {"x": x, "t": t, "n": n}
def E(s): return sp.sympify(s, locals=L)
def rhs(s): return E(s.split("=", 1)[1])
def pt(s): return tuple(E(s))

def p08():
    a = sp.symbols("a")
    av = sp.solve(3**2 - a, a)[0]
    b = limit((x**2 - av)/(x - 3), x, 3)
    V.check(8, "1 a", av, V.ans(8, "1")); V.check(8, "2 b", b, V.ans(8, "2"))
    V.check(8, "3 f(b)", (b**2 - av)/(b - 3), V.ans(8, "3"))

def p09():
    d = sp.symbols("d")
    ds = [r for r in sp.solve((27+6*d)**2 - (27+12*d)**2, d) if r != 0]
    an = 27 + (n-1)*ds[0]
    V.check(9, "1 d", ds[0], V.ans(9, "1")); V.check(9, "2 an", an, E(V.ans(9, "2")))
    V.check(9, "3 a20", an.subs(n, 20), V.ans(9, "3"))

def p10():
    f = x**4 - 2*x**3 + x + 1; m = sp.diff(f, x).subs(x, 1); line = f.subs(x, 1) + m*(x-1)
    V.check(10, "1 l", line, rhs(V.ans(10, "1")))
    g = x**2 - 7*x + 11
    tt = sp.solve(sp.diff(g, x) - m, x)[0]
    B = (tt, g.subs(x, tt))
    V.report(10, sp.simplify(B[1] - line.subs(x, tt)) == 0, "B 가 접선 위의 점")
    V.check(10, "2 B", list(B), list(pt(V.ans(10, "2"))))
    V.check(10, "3 k", sp.sqrt((B[0]-1)**2 + (B[1]-1)**2), V.ans(10, "3"))

def p11():
    c = sp.symbols("c0:4"); u = sp.symbols("u")
    f = lambda v: sum(c[i]*v**i for i in range(4))
    eq = 2*f(x) - x*f(x) - (x**3 - sp.Rational(15, 2)*x**2 + 18*x - sp.integrate(f(u), (u, 1, x)))
    s = sp.solve(sp.Poly(sp.expand(eq), x).coeffs(), c, dict=True)[0]
    fx = f(x).subs(s)
    V.check(11, "1 f'", sp.diff(fx, x), E(V.ans(11, "1")))
    V.check(11, "2 f", fx, E(V.ans(11, "2")))
    V.check(11, "3 f(2)", fx.subs(x, 2), V.ans(11, "3"))

def p12():
    a1 = sp.symbols("a1")
    b = lambda k: (a1+5)*2**(k-1)
    S = lambda k: b(k) - 5
    a = lambda k: S(1) if k == 1 else S(k) - S(k-1)
    a1v = sp.solve(a(3) - 20, a1)[0]
    V.check(12, "1 a1", a1v, V.ans(12, "1"))
    an = sp.simplify(sp.powsimp(a(n).subs(a1, a1v))) if False else 5*2**(n-1)
    V.report(12, all(a(k).subs(a1, a1v) == an.subs(n, k) for k in range(1, 12)), "a_n 전항 일치(n=1 포함)")
    V.check(12, "2 an", an, E(V.ans(12, "2")))
    V.check(12, "3 S5", S(5).subs(a1, a1v), V.ans(12, "3"))

def p13():
    f = -(1-t**2) - 4*t + 1
    V.check(13, "1 f(t)", sp.expand(f), E(V.ans(13, "1")))
    mn, mx = extrema(sp.expand(f), t, -1, 1)
    V.check(13, "2 a+b", mn+mx, V.ans(13, "2"))

def p14():
    a = sp.symbols("a", positive=True)
    C = ((3*0 + 8)/sp.Integer(4), (3*a + 0)/sp.Integer(4))
    V.check(14, "1 k", C[1]/a, V.ans(14, "1"))
    av = sp.solve(sp.Eq(C[1], sp.log(C[0]+1, 3)), a)[0]
    V.check(14, "2 C", [C[0], C[1].subs(a, av)], list(pt(V.ans(14, "2"))))

if __name__ == "__main__": V.run(globals())
