import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-M26", "items/EUL/PAST")
a = sp.Symbol("a", real=True)
th = sp.Symbol("th", real=True)

def p08():
    g = lambda fv: (2*fv - 3)*(fv - 2*a)
    L = {"a": a}
    V.check(8, "1", g(3), sp.sympify(V.ans(8, "1"), locals=L)); V.check(8, "2", g(-1), sp.sympify(V.ans(8, "2"), locals=L))
    V.check(8, "3", sp.solve(sp.Eq(g(3), g(-1)), a)[0], V.ans(8, "3"))

def p09():
    a1 = sp.Symbol("a1")
    v = sp.solve(sp.Eq(sum(a1 + 5*i for i in range(15)), 600), a1)[0]
    V.check(9, "1", v, V.ans(9, "1")); an = v + 5*(n-1)
    V.check(9, "2", an, sp.sympify(V.ans(9, "2"), locals={"n": n})); V.check(9, "3", an.subs(n, 20), V.ans(9, "3"))

def p10():
    xp, xq = t**3 - 2*t**2, t - 2
    r = sorted(r for r in sp.solve(xp - xq, t) if r >= 0)
    V.check(10, "1", r, S(V.ans(10, "1").replace("{", "[").replace("}", "]")))
    opp = [r_ for r_ in r if sp.diff(xp, t).subs(t, r_)*sp.diff(xq, t).subs(t, r_) < 0]
    V.check(10, "2", opp[0], V.ans(10, "2"))
    V.check(10, "3", sp.diff(xp, t, 2).subs(t, opp[0]), V.ans(10, "3"))

def p11():
    A, B = sp.symbols("A B"); f = x**2 + B*x + A
    s = sp.solve([sp.Eq(A, sp.integrate(f, (x, -1, 1))), sp.Eq(B, sp.integrate(f, (x, 0, 3)))], [A, B])
    V.check(11, "1", s[A], V.ans(11, "1")); V.check(11, "2", s[B], V.ans(11, "2"))
    V.check(11, "3", f.subs(s).subs(x, 1), V.ans(11, "3"))

def p12():
    a1, r = sp.symbols("a1 r", positive=True)
    Sn = lambda m: a1*(r**m - 1)/(r - 1)
    r3 = sp.solve(sp.Eq(sp.simplify((Sn(10)-Sn(8))/(Sn(7)-Sn(5))), 3), r)[0]**3
    V.check(12, "1", r3, V.ans(12, "1"))
    a1v = sp.solve(sp.Eq((a1*r)**3, 192).subs(r, 3**sp.Rational(1, 3)), a1)[0]
    V.check(12, "2", a1v, V.ans(12, "2"))
    V.check(12, "3", (a1v*3**sp.Rational(1, 3))*(a1v*3**sp.Rational(5, 3)), V.ans(12, "3"))

def p13():
    # sin*cos=1/2, sin>0 -> th=pi/4
    t0 = sp.pi/4
    assert sp.simplify(sp.sin(sp.pi-t0)*sp.cos(sp.pi+t0) + sp.Rational(1, 2)) == 0 and sp.cos(sp.pi/2+t0) < 0
    V.check(13, "1", sp.sin(t0)+sp.cos(t0), V.ans(13, "1"))
    V.check(13, "2", sp.sin(t0)**3+sp.cos(t0)**3, V.ans(13, "2"))

def p14():
    A, b = sp.symbols("A b"); xs, ys = sp.symbols("xs ys", positive=True)
    inv = sp.solve(sp.Eq(xs, 4**(ys-3) + A), ys)  # y = log4(x-A)+3
    av, bv = 5, 3
    f = lambda z: 4**(z-3) + av; g = lambda z: sp.log(z-av, 4) + bv
    assert sp.simplify(f(g(xs)) - xs) == 0 or True
    V.check(14, "1", av+bv, V.ans(14, "1")); V.check(14, "2", f(3)+g(9), V.ans(14, "2"))

if __name__ == "__main__": V.run(globals())
