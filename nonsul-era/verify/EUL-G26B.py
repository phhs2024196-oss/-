import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-G26B", "items/EUL/PAST")

X = sp.symbols("X", positive=True)
h = sp.symbols("h")

def p08():
    sol = [s for s in sp.solve(sp.Eq(X - sp.Rational(1, 3), 1/X), X) if s > 0][0]  # X=3^x, 3^x-1/3=(1/3)^x
    yB = sol - sp.Rational(1, 3)
    V.check(8, "1 yB", yB, V.ans(8, "1"))
    xA = sp.solve(sp.Eq(3**x - sp.Rational(1, 3), 0), x)[0]
    V.check(8, "2 area", sp.Rational(1, 2) * abs(xA) * yB, V.ans(8, "2"))

def p09():
    a, b = sp.symbols("a b", positive=True)
    f = sp.sin(a*(x - sp.pi/6)) + b
    av = sp.solve(sp.Eq(2*sp.pi/a, 3*sp.pi), a)[0]
    bv = sp.solve(sp.Eq(f.subs(a, av).subs(x, 2*sp.pi/3), sp.sqrt(3)), b)[0]
    V.check(9, "1 ab", av*bv, V.ans(9, "1"))
    V.check(9, "2 f", sp.simplify(f.subs({a: av, b: bv}).subs(x, -sp.pi/12)), V.ans(9, "2"))

def p10():
    a1, d = sp.symbols("a1 d")
    sol = sp.solve([a1 + 2*d - 11, (a1 + 4*d) + 8*(2*a1 + 7*d)/2 - 155], [a1, d])
    V.check(10, "1 a1", sol[a1], V.ans(10, "1"))
    S = lambda m: m*(2*sol[a1] + (m-1)*sol[d])/2
    V.check(10, "2 n", min(m for m in range(1, 100) if S(m) > 100), V.ans(10, "2"))

def p11():
    p, q, kk, l = sp.symbols("p q k l")
    f = p*x**3 + q*x**2 + kk*x + l
    # (가): f(x)+f(-x)=0 for all x
    c = sp.Poly(sp.expand(f + f.subs(x, -x)), x).coeffs()
    s1 = sp.solve(c, [q, l], dict=True)[0]
    V.check(11, "1 q/p", (q/p).subs(s1), V.ans(11, "1"))
    f1 = f.subs(s1)
    sol = sp.solve([f1.subs(x, 1) + 2, sp.diff(f1, x).subs(x, 1) - 4], [p, kk], dict=True)[0]
    V.check(11, "2 f(3)", f1.subs(sol).subs(x, 3), V.ans(11, "2"))

def p12():
    a = sp.symbols("a")
    av = sp.solve(sp.Eq(4 + 2*a, 1 - a), a)[0]
    V.check(12, "1 a", av, V.ans(12, "1"))
    g1 = x**2 + av*x; g2 = x/2 - av
    vals = [g1.subs(x, 0), g1.subs(x, sp.Rational(1, 2)), g2.subs(x, 2), g2.subs(x, 3)]
    V.check(12, "2 M+m", max(vals) + min(vals), V.ans(12, "2"))
    grid = [sp.Rational(i, 100) for i in range(0, 301)]
    fv = [(g1 if t < 2 else g2).subs(x, t) for t in grid]
    V.check(12, "2 grid", max(fv) + min(fv), V.ans(12, "2"))

def p13():
    f = sp.Function("f")
    fp = sp.symbols("fp")  # f'(x) 를 기호로: lim 은 2f'(x), 3f'(x)
    d1 = (-4*x**3 + 2*x**2 + 2) / 2
    V.check(13, "1 f'", d1, sp.sympify(V.ans(13, "1")).subs(sp.Symbol("x"), x))
    # 다항함수로 직접 확인: F=∫d1, 극한을 sympy 로
    F = sp.integrate(d1, x)
    lim1 = sp.limit((F.subs(x, x + h) - F.subs(x, x - h)) / h, h, 0)
    V.check(13, "1 lim", lim1, -4*x**3 + 2*x**2 + 2)
    g = sp.limit((F.subs(x, x + 4*h) - F.subs(x, x - 2*h)) / (2*h), h, 0)
    V.check(13, "2 g0+g1", g.subs(x, 0) + g.subs(x, 1), V.ans(13, "2"))

def p14():
    a = sp.symbols("a")
    t = sp.symbols("t")
    # f 는 다항함수: ∫_{x-1}^{x+1} f = a x^2-4x+4 를 미분하면 f(x+1)-f(x-1)=2a x-4 (이차 이하 f 로 확인)
    c0, c1, c2 = sp.symbols("c0 c1 c2")
    f = c0 + c1*x + c2*x**2
    I = sp.integrate(f, (x, x - 1, x + 1)) + 4*x
    eq = sp.Poly(sp.expand(I - (a*x**2 + 4)), x).coeffs()
    sol = sp.solve(eq, [c0, c1, c2], dict=True)[0]
    fs = f.subs(sol)
    cond = sp.integrate(fs, (x, 0, 3)) - sp.integrate(fs, (x, 0, 1)) - 20
    V.check(14, "2 a", sp.solve(cond, a)[0], V.ans(14, "2"))
    expr1 = sp.sympify(V.ans(14, "1")).subs(sp.Symbol("x"), x)
    V.check(14, "1 vs f", sp.expand(sp.integrate(fs, (x, x - 1, x + 1)) - expr1), 0)

if __name__ == "__main__": V.run(globals())
