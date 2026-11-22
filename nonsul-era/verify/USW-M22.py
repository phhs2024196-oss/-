import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("USW-M22", "items/USW/PAST")
A = lambda no, sub="1", key=None: sp.sympify(V.ans(no, sub, key), locals={"x": x})

def p05():
    r = sp.symbols("r", positive=True)
    rv = sp.solve(sp.Eq(8*r**2, sp.Rational(1, 2)*8*r + 12), r)[0]   # a3+a4+a5=8 이므로 합은 8r^k
    V.check(5, "a6+a7+a8", 8*rv**3, A(5))

def p06():
    f = x**2 - 6*x + 11
    cs = [1, 3, 7]; vals = [sp.log(f.subs(x, c), 3) for c in cs]   # g 증가 → f 의 극값/끝점
    V.check(6, "차", sp.simplify(sp.expand_log(max(vals) - min(vals), force=True) - 0), A(6) * 1 if False else sp.simplify(max(vals) - min(vals)))
    V.check(6, "차 값", float(max(vals) - min(vals)), float(A(6)))

def p07():
    f0, f1 = sp.symbols("f0 f1")
    s = sp.solve([sp.Eq(8*f0, 16), sp.Eq(12*f0 + 8*f1, 0)], [f0, f1])
    V.check(7, "f'(2)", s[f1], A(7))

def p08():
    f1 = 2*x**2 - 4*x + 3; f2 = -x**2 + 8*x - 6
    r = sorted(sp.solve(sp.Eq(f1, f2), x))
    V.check(8, "넓이", sp.integrate(sp.Abs(1) * (f2 - f1), (x, r[0], r[1])), A(8))

if __name__ == "__main__": V.run(globals())
