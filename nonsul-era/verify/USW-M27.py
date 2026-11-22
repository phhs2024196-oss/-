import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("USW-M27", "items/USW/PAST")
def p05():
    t = sp.log(sp.sqrt(80), 2)
    V.check(5, "f(sqrt80)", sp.simplify(sp.expand_log(3*t + sp.log(sp.sqrt(10), 2) - sp.log(10, sp.sqrt(2)), force=True)), V.ans(5, "1"))
def p06():
    r, h = sp.symbols("r h", positive=True)
    tri = r*r*sp.sin(sp.pi/4)/2
    V.check(6, "부피", sp.simplify(8*tri*h/3 - sp.sympify(V.ans(6, "1"), locals={"r": r, "h": h})), 0)
def p07():
    t = sp.symbols("t", positive=True)
    S = t*(2*t - t**2)/2
    c = sp.solve(sp.diff(S, t), t)
    V.check(7, "최댓값", max(S.subs(t, v) for v in c if 0 < v <= 2), V.ans(7, "1"))
def p08():
    nn = sp.symbols("nn", positive=True)
    Sn = 2*sp.integrate(x - x**nn, (x, 0, 1))
    pr = 1
    for m in (2, 4, 6, 8, 10): pr *= sp.simplify(Sn.subs(nn, m))
    V.check(8, "곱", pr, V.ans(8, "1"))
if __name__ == "__main__": V.run(globals())
