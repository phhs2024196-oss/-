import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("USW-M24", "items/USW/PAST")
R = sp.Rational
def p05():
    n = [m for m in range(-10, 11) if m + 2 > 0 and 2*m + 5 > 0 and sp.log(m + 2, 2) + sp.log(2*m + 5, sp.Rational(1, 4)) < sp.Rational(1, 2)]
    V.check(5, "정수 개수", len(n), V.ans(5, "1"))
def p06():
    a = {1: sp.sqrt(2) - 1}
    for m in range(1, 16): a[m+1] = sp.expand(sp.sqrt(2)*a[m] - (1 - sp.sqrt(2))*m - 1)
    V.check(6, "a16", sp.simplify(a[16]), V.ans(6, "1"))
    V.check(6, "④ a16+16", sp.simplify(a[16] + 16), 256)
def p07():
    a = sp.symbols("a")
    assert sp.solve(sp.sqrt(a + 4) - 3, a) == [5]
    f = (sp.sqrt(5 - 2*x) - 3) / (x**2 - 3*x - 10)
    V.check(7, "극한", sp.limit(f, x, -2), V.ans(7, "1"))
def p08():
    f = sp.integrate(3*x**2 - 12*x, x)
    V.check(8, "넓이", -sp.integrate(f, (x, 0, 6)), V.ans(8, "1"))
if __name__ == "__main__": V.run(globals())
