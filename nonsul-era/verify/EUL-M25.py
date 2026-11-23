import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-M25", "items/EUL/PAST")
a = sp.Symbol("a", real=True)
def rhs(s): return sp.sympify(s.split("=")[1], locals={"n": n, "a": a, "x": x})

def p05():
    f = 2*x**3 + (4*a+7)*x**2 + (4*a+5)*x + a + 2
    c0, c1 = sp.Poly(f, a).all_coeffs()
    xp = sp.solve(c0, x)[0]; assert c1.subs(x, xp) == 1
    yp = sp.simplify(f.subs(x, xp).subs(a, 0))
    ex = S(V.ans(5, "1").strip("()").split(",")[0]); ey = S(V.ans(5, "1").strip("()").split(",")[1])
    V.check(5, "1 x", xp, ex); V.check(5, "1 y", yp, ey)
    m = sp.simplify(sp.diff(f, x).subs(x, xp))
    line = m*(x - xp) + yp
    V.check(5, "2", line, rhs(V.ans(5, "2")))
    V.check(5, "3", sp.Rational(1, 2)*(-line.subs(x, 0)/m)*line.subs(x, 0) if False else sp.Rational(1,2)*sp.solve(line, x)[0]*line.subs(x, 0), V.ans(5, "3"))

def p06():
    A, b = sp.symbols("A b", positive=True)
    av = -(-1 - 3)  # 최솟값 -A+3=-1
    A_ = sp.solve(sp.Eq(-A + 3, -1), A)[0]
    b_ = sp.solve(sp.Eq(2*sp.pi/b, 10*sp.pi), b)[0]
    V.check(6, "1", A_, V.ans(6, "1")); V.check(6, "2", b_, V.ans(6, "2"))
    f = A_*sp.sin(b_*(x - sp.pi/6)) + 3
    V.check(6, "3", f.subs(x, 11*sp.pi/6), V.ans(6, "3"))

def p07():
    a1 = sp.Symbol("a1"); an = lambda m: a1 + 3*(m-1)
    s = sp.solve(sp.Eq(an(7)**2, an(3)*an(10)), a1)
    V.check(7, "1", s[0], V.ans(7, "1"))
    g = an(n).subs(a1, s[0])
    V.check(7, "2", g, sp.sympify(V.ans(7, "2"), locals={"n": n}))
    V.check(7, "3", g.subs(n, 20), V.ans(7, "3"))

def p08():
    c = sp.symbols("c0:3"); d = sp.symbols("d0:3")
    # f = 3 + (x-2)(2(x+2)) 형태의 한 예: f(2)=3, f'(2)=8 ; g=(x-2)/6
    f = 3 + 2*(x**2 - 4)
    g = (x - 2)/6
    assert limit((f-3)/(x**2-4), x, 2) == 2 and limit((x-2)/g, x, 2) == 6
    V.check(8, "1", sp.diff(f, x).subs(x, 2), V.ans(8, "1"))
    V.check(8, "2", sp.diff(g, x).subs(x, 2), V.ans(8, "2"))
    V.check(8, "3", sp.diff(f*g, x).subs(x, 2), V.ans(8, "3"))

if __name__ == "__main__": V.run(globals())
