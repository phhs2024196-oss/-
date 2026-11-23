import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-M27", "items/EUL/PAST")
a = sp.symbols("a", real=True)

def p08():
    b = [v for v in sp.solve(sp.Abs(t - 1) - 2, t) if v > -1]  # f(x)>-1
    av = sp.solve(2**(x + 1) - 1 - b[0], x)[0]
    V.check(8, "1", b[0], V.ans(8, "1"))
    V.check(8, "2", av + b[0], V.ans(8, "2"))

def p09():
    # b=c, a:b=4:3, 외접원 지름 9
    cb = sp.Rational(2, 3)
    V.check(9, "1", cb, V.ans(9, "1"))
    sB = sp.sqrt(1 - cb**2); bb = 9 * sB; aa = bb * sp.Rational(4, 3)
    V.check(9, "1-ck", (aa / 2) / bb, V.ans(9, "1"))
    V.check(9, "2", sp.Rational(1, 2) * aa * sp.sqrt(bb**2 - (aa / 2)**2), V.ans(9, "2"))

def p10():
    r = sp.symbols("r", positive=True); A = sp.symbols("A", positive=True)
    S10 = sum(A * r**i for i in range(10)); T5 = sum(A * r**(2 * i + 2) for i in range(5))
    V.check(10, "1", T5, sp.sympify(V.ans(10, "1").replace("a", "A"), locals={"A": A, "r": r}))
    rv = [s for s in sp.solve(sp.simplify((S10 - 6 * T5) / A), r) if s > 0 and s != 1]
    V.check(10, "2", rv[0]**4, V.ans(10, "2"))

def p11():
    p, q = sp.symbols("p q")
    s = sp.solve([1 + p + q, p + 2 - 4], [p, q])
    V.check(11, "1", s[p], V.ans(11, "1"))
    V.check(11, "2", s[p] - 3 * s[q], V.ans(11, "2"))
    V.check(11, "ck", sp.limit((x**2 + s[p] * x + s[q]) / (x - 1), x, 1), 4)

def p12():
    f = sp.sqrt(a**2 + (2 * a + 1)**2); g = sp.sqrt(2)
    V.check(12, "1", f * g, sp.sympify(V.ans(12, "1"), locals={"a": a}))
    V.check(12, "2", sp.limit((f - g) / (a + 1), a, -1), V.ans(12, "2"))

def p13():
    p, q = sp.symbols("p q"); F = x**2 + p * x + q
    s = sp.solve([(F.subs(x, 3) - F.subs(x, 0)) / 3 - 5, sp.diff(F, x).subs(x, 1) / 2 - F.subs(x, 1)], [p, q])
    f = F.subs(s)
    V.check(13, "1", f, sp.sympify(V.ans(13, "1"), locals={"x": x}))
    V.check(13, "2", f.subs(x, 3) - sp.diff(f, x).subs(x, 3), V.ans(13, "2"))

def p14():
    f = x**3 - 2 * x**2
    tt = [r for r in sp.solve(f.subs(x, t) + sp.diff(f, x).subs(x, t) * (0 - t) + 8, t) if r.is_real][0]
    g = f.subs(x, tt) + sp.diff(f, x).subs(x, tt) * (x - tt)
    xs = sorted(sp.solve(f - g, x))
    V.check(14, "1x", xs, [-2, 2])
    V.check(14, "1y", [f.subs(x, r) for r in xs], [-16, 0])
    V.check(14, "2", integral(f - g, x, xs[0], xs[-1]), V.ans(14, "2"))

if __name__ == "__main__": V.run(globals())
