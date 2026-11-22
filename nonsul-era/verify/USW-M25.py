import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("USW-M25", "items/USW/PAST")
def p05():
    g = lambda v: (11 - v)*(v/sp.Integer(2) - 4)
    pts = [sp.Rational(i, 4) for i in range(0, 60)]
    ok = lambda v: g(v) > 0 and sp.log(g(v), sp.Rational(1, 2)) >= 0
    exp = lambda v: (8 < v <= 9) or (10 <= v < 11)
    V.check(5, "범위", [v for v in pts if ok(v) != exp(v)], [])
    V.report(5, V.ans(5, "1") == "8<x<=9 또는 10<=x<11", "YAML 범위 문자열")
def p06():
    r, a1 = sp.symbols("r a1", positive=True)
    S = lambda m: a1*(r**m - 1)/(r - 1)
    s = sp.solve([sp.simplify((S(12)-S(9))/(S(8)-S(5))) - 3, sp.simplify(S(8)/S(4)) - a1**2], [r, a1], dict=True)
    V.check(6, "a2*a8", sp.simplify(s[0][a1]*s[0][r]*s[0][a1]*s[0][r]**7), V.ans(6, "1"))
def p07():
    t = sp.symbols("t", positive=True); h = 2*t
    Vv = sp.pi*(h/2)**2*h/3
    V.check(7, "dV/dt(2)", sp.diff(Vv, t).subs(t, 2), V.ans(7, "1"))
def p08():
    a = sp.symbols("a", positive=True)
    xp = sp.solve(x**2 - a*x - a/2*x, x); xp = [v for v in xp if v != 0][0]
    S1 = sp.integrate(a/2*x - (x**2 - a*x), (x, 0, xp)); S2 = xp*(a/2*xp)/2
    V.check(8, "S1-S2", sp.simplify(S1 - S2), 0)
    V.check(8, "정적분", sp.simplify(sp.integrate(x**2 - a*x, (x, 0, xp))), V.ans(8, "1"))
if __name__ == "__main__": V.run(globals())
