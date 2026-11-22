import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("USW-M26", "items/USW/PAST")
def p05():
    CA = sp.root(2, 4); CB = sp.root(18, 4)
    AB = sp.sqrt(CA**2 + CB**2)
    V.check(5, "AH", sp.simplify(CA**2/AB), V.ans(5, "1"))
def p06():
    V.check(6, "합", sum(sp.Rational(1, sum(range(1, m+1))) for m in range(1, 11)), V.ans(6, "1"))
def p07():
    a, b = sp.symbols("a b", positive=True)
    f = x**3 + a*x**2 - a**2*x + b
    s = sp.solve([sp.diff(f, x).subs(x, 1), f.subs(x, 1) - 3], [a, b], dict=True)[0]
    g = f.subs(s)
    crit = sp.solve(sp.diff(g, x), x)
    assert sp.diff(g, x, 2).subs(x, 1) > 0
    V.check(7, "극댓값", max(g.subs(x, c) for c in crit), V.ans(7, "1"))
def p08():
    k = sp.symbols("k", positive=True)
    xa = k + 1
    area = sp.integrate((k+1)*x - (x**3 - k*x**2), (x, 0, xa)); tri = xa*(k+1)*xa/2
    sol = sp.solve(sp.simplify(area - tri), k)
    V.check(8, "k", sol, [V.ans(8, "1")])
if __name__ == "__main__": V.run(globals())
