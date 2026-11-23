import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy.parsing.sympy_parser import parse_expr
V = Verifier("SHU-G25H", "items/SHU/PAST")
P = lambda s: parse_expr(s.replace("^", "**"), local_dict={"x": x, "pi": sp.pi})
def kv(s):
    return dict(p.split("=", 1) for p in s.replace(" ", "").split(";") if "=" in p)

def p10():
    a = sp.symbols("a", real=True)
    f = lambda t: 2**t + a
    av = sp.solve(sp.Eq(f(3), 7), a)
    V.check(10, "a", av[0], P(V.ans(10, "1", "a")))
    V.check(10, "m", f(-2).subs(a, av[0]), P(V.ans(10, "1", "m")))
    mn, mx = extrema(2**x + av[0], x, -2, 3)
    V.check(10, "구간 최댓값 7", mx, 7)
    V.check(10, "구간 최솟값 m", mn, P(V.ans(10, "1", "m")))

def p11():
    a1, d = sp.symbols("a1 d", real=True)
    an = lambda n: a1 + (n - 1) * d
    s = sp.solve([sp.Eq(an(7), 22), sp.Eq(an(2) + an(5), 23)], [a1, d], dict=True)
    V.check(11, "해 개수", len(s), 1)
    V.check(11, "d", s[0][d], P(V.ans(11, "1", "d")))
    V.check(11, "a1", s[0][a1], 4)
    V.check(11, "a10", an(10).subs(s[0]), P(V.ans(11, "1", "a10")))

def p12():
    f2 = sp.symbols("f2")
    q = sp.cancel((x - 2) / (x**2 + 2 * x - 8))
    V.check(12, "(가) 1/(가)=(x-2)/(x^2+2x-8)", 1 / q, P(V.ans(12, "1")))
    V.check(12, "(나) 극한 1/(x+4) at 2", sp.limit(q, x, 2), P(V.ans(12, "2")))
    V.check(12, "(다) f(2)", sp.solve(sp.Eq(sp.limit(q, x, 2) * f2, 5), f2)[0], P(V.ans(12, "3")))
    V.check(12, "수치 극한", sp.limit((x - 2) * 30 / (x**2 + 2 * x - 8), x, 2), 5)

def p13():
    a, t = sp.symbols("a t", real=True)
    A = sp.integrate(x**2 + a, (x, -2, 1))
    sol = sp.solve(sp.Eq(A, 9), a)
    V.check(13, "a", sol[0], P(V.ans(13, "1", "a")))
    V.report(13, all(float(x**2 + sol[0]) > 0 for x in (-2, 0, 1)) , "곡선이 x축 위(넓이=적분)")
    V.check(13, "S", sp.integrate(x**2 + 2 * sol[0], (x, -2, 1)), P(V.ans(13, "1", "S")))

def p14():
    a, b, c = sp.symbols("a b c", real=True)
    f = 2 * x**3 + a * x**2 + b * x + c
    s = sp.solve([sp.diff(f, x).subs(x, 1), sp.diff(f, x).subs(x, 2)], [a, b], dict=True)[0]
    cc = sp.solve(sp.Eq(f.subs(s).subs(x, 1), 4), c)[0]
    V.check(14, "a", s[a], P(V.ans(14, "1", "a")))
    V.check(14, "b", s[b], P(V.ans(14, "1", "b")))
    V.check(14, "c", cc, P(V.ans(14, "1", "c")))
    F = f.subs(s).subs(c, cc)
    V.check(14, "극솟값 f(2)", F.subs(x, 2), P(V.ans(14, "1", "min")))
    d = sp.diff(F, x)
    V.report(14, d.subs(x, 0) > 0 and d.subs(x, sp.Rational(3, 2)) < 0 and d.subs(x, 3) > 0, "x=1 극대, x=2 극소(증감표)")

def p15():
    f = sp.sin(x)**2 - 2 * sp.cos(x - sp.pi) + 1
    lo, hi = -sp.pi / 2, sp.pi / 3
    mn, mx = extrema(f, x, lo, hi)
    V.check(15, "m", mn, P(V.ans(15, "1", "m")))
    V.check(15, "M", mx, P(V.ans(15, "1", "M")))
    pts = [lo, hi] + list(sp.solveset(sp.diff(f, x), x, sp.Interval(lo, hi)))
    amin = [p for p in pts if sp.simplify(f.subs(x, p) - mn) == 0]
    amax = [p for p in pts if sp.simplify(f.subs(x, p) - mx) == 0]
    V.check(15, "a", amin, [P(V.ans(15, "1", "a"))])
    V.check(15, "b", amax, [P(V.ans(15, "1", "b"))])
    t = sp.symbols("t")
    V.check(15, "치환식", sp.expand(f.subs(x, sp.acos(t)) if False else (1 - t**2) + 2 * t + 1), sp.expand(-(t - 1)**2 + 3))

if __name__ == "__main__": V.run(globals())
