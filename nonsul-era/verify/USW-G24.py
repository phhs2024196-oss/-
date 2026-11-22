import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("USW-G24", "items/USW/PAST")
R = sp.Rational

def p06():
    a = 1 / sp.log(8, 3); b = 1 / sp.log(R(1, 5), 3)   # a log_3 8 = 1, b log_3 (1/5) = 1
    V.check(6, "값", sp.simplify(3**(1/a - 3/b) + sp.root(3**10, 3*a)), V.ans(6, "1"))

def p07():
    ya = 2; xa = 2**ya; b = sp.symbols("b", positive=True)
    xb = sp.nsimplify(sp.nsolve(sp.log(b, 2) + 2*b, b, 0.5))  # OB 기울기 -2: log2(b)/b=-2
    A = sp.Matrix([xa, ya]); B = sp.Matrix([xb, sp.log(xb, 2)])
    assert abs(A.dot(B)) < 1e-12      # 직각
    V.check(7, "넓이", sp.Rational(1, 2) * A.norm() * B.norm(), V.ans(7, "1"), 1e-9)

def p08():
    sol = [s for s in sp.solveset(sp.sqrt(2)/2*sp.cos(sp.pi/2 + x) + sp.cos(x)**2 - 2*sp.cos(sp.pi/2 - x) - 1 - sp.sqrt(2), x, sp.Interval(0, 2*sp.pi))]
    V.check(8, "해 1", sol, [V.ans(8, "1", "x1"), V.ans(8, "1", "x2")])

def p09():
    a1, d = sp.symbols("a1 d")
    s = sp.solve([a1 + 2*d - 14, 5*a1 + 10*d - (7*a1 + 21*d)], [a1, d])
    S = lambda m: m*(2*s[a1] + (m - 1)*s[d])/2
    V.check(9, "최댓값", max(S(m) for m in range(1, 60)), V.ans(9, "1"))

def p10():
    b, a = sp.symbols("b a"); c = sp.symbols("c")
    f = x**2 + 2*x + b
    assert limit((f - x**2)/(x - 1), x, sp.oo) == 2
    bb = sp.solve(f.subs(x, 1) - 2, b)[0]
    V.check(10, "a", limit((f.subs(b, bb) - 2)/(x - 1), x, 1), V.ans(10, "1"))

def p11():
    f = 3*x - 1; g = f**2 - 2*f   # f 를 접선으로 두어도 f(1), f'(1) 만 쓰이므로 동일
    m = sp.diff(g, x).subs(x, 1); c = g.subs(x, 1) - m
    V.check(11, "기울기", m, V.ans(11, "1", "m")); V.check(11, "절편", c, V.ans(11, "1", "c"))

def p12():
    h = sp.symbols("h", positive=True)
    r2 = 9 - (h - 3)**2
    Vh = sp.pi*r2*h/3
    V.check(12, "높이", sp.solve(sp.diff(Vh, h), h)[0] if len(sp.solve(sp.diff(Vh, h), h)) == 1 else max(sp.solve(sp.diff(Vh, h), h)), V.ans(12, "1"))

def p13():
    k = sp.symbols("k", positive=True)
    f = x**3/3 - k*x**2/2
    kk = sp.solve(sp.Eq(f.subs(x, 0) - f.subs(x, k), R(4, 3)), k)[0]
    V.check(13, "적분", integral((x**2 - k*x).subs(k, kk), x, -kk, kk), V.ans(13, "1"))

def p14():
    F = sp.Function("F"); t_ = sp.symbols("t_")
    f = 2*x**2 + 1
    lhs = (x**2 + 1)*f; rhs = (x**2 + 1)**2 + 2*sp.integrate(t*f.subs(x, t), (t, 0, x))
    V.check(14, "항등", sp.expand(lhs - rhs), 0)
    # 유일성: 차수 4 이하 다항함수 계수 탐색
    cs = sp.symbols("c0:5"); g = sum(c*x**i for i, c in enumerate(cs))
    eq = sp.expand((x**2 + 1)*g - (x**2 + 1)**2 - 2*sp.integrate(t*g.subs(x, t), (t, 0, x)))
    s = sp.solve(sp.Poly(eq, x).all_coeffs(), cs, dict=True)[0]
    V.check(14, "f(x)", sp.expand(g.subs(s)), sp.sympify(V.ans(14, "1"), locals={"x": x}))

def p15():
    ns = sp.symbols("n", positive=True)
    res = []
    for nn in range(2, 60):
        f1 = x - x**2; f2 = x/nn - (x/nn)**2
        xs = sorted(sp.solve(f1 - f2, x))
        res.append((nn, sp.integrate(sp.Abs(f1 - f2), (x, xs[0], xs[1])) if False else abs(integral(f1 - f2, x, xs[0], xs[1]))))
    V.check(15, "n", [n for n, s in res if s == R(5, 54)], [V.ans(15, "1")])

if __name__ == "__main__": V.run(globals())
