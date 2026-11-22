import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("USW-G23", "items/USW/PAST")
a, b, c, d, a1 = sp.symbols("a b c d a1", positive=True)
X = lambda s: S(s, locals={"x": x, "t": t})

def p07():
    s = sp.symbols("s", positive=True)  # a=s^3, b=s^2, c=s^4 from c=b^2, b^3=a^2
    A, B, Cc = s**3, s**2, s**4
    V.check(7, "log_a bc", sp.log(B*Cc)/sp.log(A), 2)
    V.check(7, "2log_a b - log_a c", (2*sp.log(B) - sp.log(Cc))/sp.log(A), 0)
    V.check(7, "ans", sp.log(A)/sp.log(B) + sp.log(A)/sp.log(Cc), V.ans(7, "1"))

def p08():
    BC = sp.sqrt(25 + 64 - 2*5*8*sp.Rational(1, 2))
    V.check(8, "BC", BC, 7)
    V.check(8, "R", BC/(2*sp.sin(sp.pi/3)), X(V.ans(8, "1")))

def p09():
    A_, D_ = sp.symbols("A_ D_")
    an = lambda m: A_ + (m-1)*D_
    s = sp.solve([an(9)+an(19)+an(29)-12, an(29)+an(39)+an(49)-21], [A_, D_])
    V.check(9, "a29", an(29).subs(s), V.ans(9, "1"))

def p10():
    sol = {}
    b_ = sp.symbols("b_")
    lim1 = limit((-x+b_+x)*(-x+b_-x**2+2*x)/(x**2+2), x, sp.oo)
    V.check(10, "b", sp.solve(sp.Eq(lim1, 3), b_)[0], V.ans(10, "1", "c3"))
    c_ = sp.symbols("c_")
    lim2 = limit((x**2-2*x+c_+x)*c_/(x**2+2), x, sp.oo)
    V.check(10, "c", sp.solve(sp.Eq(lim2, 3), c_)[0], V.ans(10, "1", "c4"))
    deg = sp.degree(sp.expand((c_+x)*(c_-x**2+2*x)), x)
    V.report(10, deg == 3 and V.ans(10, "1", "c1") == "삼차식", f"c=상수일 때 분자 차수 {deg} (①)")
    V.report(10, V.ans(10, "1", "c2") == "이차식", "② 2차식(분모와 같은 차수)")

def p11():
    f = x**3 - 2*x**2 + 3*x - 4
    h = sp.symbols("h")
    V.check(11, "lim", limit((f.subs(x, 1+h) - f.subs(x, 1-h))/h, h, 0), V.ans(11, "1"))
    V.check(11, "2f'(1)", 2*sp.diff(f, x).subs(x, 1), V.ans(11, "1"))

def p12():
    a_ = sp.symbols("a_", real=True)
    f = x**2 - 2*x + 1
    tan = f.subs(x, a_) + sp.diff(f, x).subs(x, a_)*(x - a_)
    P = sp.solve(tan, x)[0]; Q = tan.subs(x, 0)
    area = sp.Rational(1, 2)*P*Q
    V.check(11+1, "area", area, sp.Rational(1, 4)*(-a_**3 - a_**2 + a_ + 1))
    r = [r for r in sp.solve(sp.diff(area, a_), a_) if 0 < r < 1]
    V.check(12, "a", r[0], V.ans(12, "1"))
    V.report(12, bool(sp.diff(area, a_, 2).subs(a_, r[0]) < 0), "극대(최대)")

def p13():
    X_ = t**3 - 3*t**2 - 9*t
    v = sp.diff(X_, t)
    turn = [r for r in sp.solve(v, t) if r > 0][0]
    V.check(13, "accel", sp.diff(X_, t, 2).subs(t, turn), V.ans(13, "1"))

def p14():
    f = sp.Function("f")
    A, B, C0 = sp.symbols("A B C0")
    F = A*x**2 + B*x + C0
    eq = sp.expand(x*F - (sp.integrate(F.subs(x, t), (t, 1, x)) + 2*x**3 - 3*x**2))
    s = sp.solve(sp.Poly(eq, x).all_coeffs(), [A, B, C0], dict=True)[0]
    V.check(14, "f", F.subs(s), X(V.ans(14, "1")))

def p15():
    k_ = sp.symbols("k_", positive=True)
    g = x**2*(x-6)*(x-k_)
    V.check(15, "factor", x**4 - (6+k_)*x**3 + 6*k_*x**2, g)
    r = [r for r in sp.solve(sp.integrate(g, (x, 0, k_)), k_) if r > 6]
    V.check(15, "k", r[0], V.ans(15, "1"))
    kk = r[0]
    V.check(15, "areas equal", sp.integrate(g.subs(k_, kk), (x, 0, 6)), -sp.integrate(g.subs(k_, kk), (x, 6, kk)))

if __name__ == "__main__": V.run(globals())
