import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("USW-G22", "items/USW/PAST")
a, b, C = sp.symbols("a b C", real=True)

def p07():
    sol = sp.solve(sp.Eq(4**(x+2), sp.Rational(1, 4)**(-x) + 30), x)
    V.check(7, "x", sol[0], V.ans(7, "1", "x"))

def p08():
    sc = sp.sqrt(sp.Rational(2, 3))  # 3 sin^2 C = 2
    V.check(8, "sinC", sc, V.ans(8, "1", "c1"))
    V.check(8, "2R", 2*sp.sqrt(3), V.ans(8, "1", "c2"))
    V.check(8, "AB", 2*sp.sqrt(3)*sc, V.ans(8, "1", "c3"))

def p09():
    an = lambda m: 3 + 4*(m-1)
    V.check(9, "sum", sum(sp.Rational(1, (an(m)-1)*(an(m+1)-1)) for m in range(1, 33)), V.ans(9, "1"))

def p10():
    sol = sp.solve(sp.Eq(8/(1+a), 2), a)
    V.check(10, "a", sol[0], V.ans(10, "1", "a"))
    V.check(10, "left", limit((x**3-x)/(x-1), x, 1, "-"), 2)

def p11():
    f = x**3 - 2*x**2 - 3*x + 1
    m, y0 = sp.diff(f, x).subs(x, 1), f.subs(x, 1)
    V.check(11, "tangent", m*(x-1)+y0, S(V.ans(11, "1").split("=",1)[1], locals={'x': x}))

def p12():
    f1 = sp.diff(x**3/3 + a*x**2 + (2-a)*x + 2*a, x)
    disc = sp.discriminant(f1, x)
    sol = sp.solve_univariate_inequality(disc <= 0, a, relational=False)
    V.check(12, "lo", sol.start, V.ans(12, "1", "lo")); V.check(12, "hi", sol.end, V.ans(12, "1", "hi"))

def p13():
    A = sp.symbols("A")
    f = A*x**3/3 - A*x**2 + b
    s = sp.solve([f.subs(x, 0) - 2, f.subs(x, 2) + 2], [A, b], dict=True)[0]
    V.check(13, "f", f.subs(s), S(V.ans(13, "1"), locals={'x': x}))
    V.report(13, bool(s[A] > 0), 'a>0 (그래프가 아래로 볼록)')

def p14():
    f = lambda u: 4*u**3 + 2*u
    inv = sp.nsolve(f(t) - 3, t, 1)  # sample of inverse
    V.check(14, "f(1)=6", f(1), 6)
    # int_0^6 g = 6 - int_0^1 f
    V.check(14, "int g", 1*6 - integral(f(x), x, 0, 1), V.ans(14, "1"))
    import mpmath as mp
    g = lambda y: mp.findroot(lambda u: 4*u**3 + 2*u - y, 1)
    V.check(14, "numeric", mp.quad(g, [0, 6]), V.ans(14, "1"), tol=1e-6)

def p15():
    v = -3*t**2 + 6*t
    turn = sp.solve(v, t)
    pos = integral(v, t, 0, n)
    zero = [r for r in sp.solve(pos, n) if r > 0][0]
    V.check(15, "time", zero - max(turn), V.ans(15, "1"))

if __name__ == "__main__": V.run(globals())
