import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy.parsing.sympy_parser import parse_expr
V = Verifier("SWU-M26E", "items/SWU/PAST")
a, b = sp.symbols("a b", real=True)
P = lambda s: parse_expr(s, local_dict={"a": a, "b": b, "x": x})

def p01():
    f = lambda u: u**2
    # (1) 현 AB
    l = sp.expand((f(a) - f(b)) / (a - b) * (x - a) + f(a))
    V.check(1, "1 l", l, P(V.ans(1, "1", "l")))
    S1 = sp.simplify(sp.integrate(l - x**2, (x, b, a)))
    V.check(1, "1 S1", S1, P(V.ans(1, "1", "S1")))
    # (2) C, D, 삼각형 ABD
    Cy = (f(a) + f(b)) / 2; Dy = f((a + b) / 2)
    S2 = sp.simplify(sp.Rational(1, 2) * (a - b) * (Cy - Dy))
    V.check(1, "2 S2", S2, P(V.ans(1, "2", "S2")))
    V.check(1, "2 비", sp.simplify(S1 / S2), V.ans(1, "2", "ratio"))
    # (3) 접선 교점 E
    m = sp.diff(x**2, x).subs(x, a) * (x - a) + a**2
    n = sp.diff(x**2, x).subs(x, b) * (x - b) + b**2
    Ex = sp.solve(sp.Eq(m, n), x)[0]; Ey = sp.simplify(m.subs(x, Ex))
    ans = V.ans(1, "3", "E")
    V.check(1, "3 E x", Ex, P(ans[1:-1].split(",")[0]))
    V.check(1, "3 E y", Ey, P(ans[1:-1].split(",")[1]))
    S3 = sp.simplify(sp.Rational(1, 2) * (a - b) * (Cy - Ey))
    V.check(1, "3 S3", S3, P(V.ans(1, "3", "S3")))
    V.check(1, "3 비", sp.simplify(S1 / S3), V.ans(1, "3", "ratio"))

if __name__ == "__main__": V.run(globals())
