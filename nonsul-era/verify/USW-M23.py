import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("USW-M23", "items/USW/PAST")
A = lambda no, sub="1", key=None: sp.sympify(V.ans(no, sub, key), locals={"x": x})

def p05():
    sols = [s for s in sp.solve(sp.Eq(sp.log(3*x+2, 2), 1 + 2*sp.log(x-1, 2)), x) if s > 1] or \
           [s for s in sp.solve(sp.Eq(3*x+2, 2*(x-1)**2), x) if s > 1]
    V.check(5, "x", sols[0], A(5))

def p06():
    a, b = sp.symbols("a b")
    c2 = sp.solve(sp.Eq((2*a+b)**2, 4*a*(a+2*b)), b)  # b=0, b=4a
    b4 = [s for s in c2 if s != 0][0]
    V.check(6, "c1", 2*a+b, sp.sympify(V.ans(6, "1", "c1"), locals={"a": a, "b": b}))
    V.check(6, "c2", b4, sp.sympify(V.ans(6, "1", "c2"), locals={"a": a, "b": b}))
    V.check(6, "c3", sp.simplify(((2*a+b)/(4*a)).subs(b, b4)), V.ans(6, "1", "c3"))

def p07():
    f0, f1 = sp.symbols("f0 f1")
    g1 = (4*2-3)*f0 + (2*4-6)*f1
    V.check(7, "f'(2)", sp.solve(sp.Eq(g1.subs(f0, 3), 4), f1)[0], A(7))

def p08():
    a = sp.symbols("a", positive=True)
    area = sp.integrate(-(x**2 - a*x), (x, 0, a))
    V.check(8, "a", sp.solve(sp.Eq(area, sp.Rational(4, 3)), a)[0], A(8))

if __name__ == "__main__": V.run(globals())
