import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KNG-M27", "items/KNG/PAST")

def p01():
    X, Y = sp.symbols("X Y", positive=True)
    # 4^{log2 x}=x^2, log_{sqrt2}(1/4)^x=-4x, 2^{log2 y}=y 를 수치로 확인
    for xv in (sp.Rational(1, 3), 2, 5):
        lhs = 4 ** sp.log(xv, 2) + xv * sp.log(sp.Rational(1, 4)) / sp.log(sp.sqrt(2))
        V.check(1, f"항 정리 x={xv}", sp.simplify(sp.N(lhs, 30)), sp.N(xv**2 - 4 * xv, 30))
    yx = -x**2 + 4 * x
    al = sp.solve(sp.diff(yx, x), x)[0]
    be = yx.subs(x, al)
    V.check(1, "alpha", al, V.ans(1, "1", "alpha"))
    V.check(1, "beta", be, V.ans(1, "1", "beta"))
    V.check(1, "log", sp.log(be, al).simplify(), V.ans(1, "1", "ans"))

def p02():
    a, r4 = sp.symbols("a r4", positive=True)
    sol = sp.solve([2 * (a + a * r4) - 8, a * r4 + a * r4**2 - 8], [a, r4], dict=True)
    s = sol[0]
    V.check(2, "r4", s[r4], V.ans(2, "1", "r4"))
    V.check(2, "a", s[a], V.ans(2, "1", "a"))
    V.check(2, "a9", s[a] * s[r4] ** 2, V.ans(2, "1", "a9"))

def p03():
    f = x**3 + x
    g = x**3 - x
    c = [r for r in sp.solve(sp.diff(g, x), x) if r < 0][0]
    V.check(3, "max", g.subs(x, c), V.ans(3, "1", "max"))
    V.check(3, "min", f.subs(x, 0), V.ans(3, "1", "min"))
    # x>=0 에서 f'>0, x=0 좌우 부호 음→양, c 좌우 양→음
    V.report(3, bool(sp.diff(g, x).subs(x, c - 1) > 0), "f' 좌")
    V.report(3, bool(sp.diff(g, x).subs(x, c / 2) < 0), "f' 우(c,0)")
    V.report(3, bool(sp.diff(f, x).subs(x, sp.Rational(1, 7)) > 0), "f' x>0")

def p04():
    a, b, c = sp.symbols("a b c")
    F = x**3 + a * x**2 + b * x + c
    sol = sp.solve([F.subs(x, 0), sp.integrate(F, (x, 0, 2)) - 16 - sp.integrate(F, (x, 0, -2)), F.subs(x, -1) - 6], [a, b, c], dict=True)[0]
    V.check(4, "a", sol[a], V.ans(4, "1", "a"))
    V.check(4, "b", sol[b], V.ans(4, "1", "b"))
    V.check(4, "f'(1)", sp.diff(F.subs(sol), x).subs(x, 1), V.ans(4, "1", "ans"))

if __name__ == "__main__": V.run(globals())
