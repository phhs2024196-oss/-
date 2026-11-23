import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-G25A", "items/EUL/PAST")
L = {"x": x, "t": t, "n": n}
def E(s): return sp.sympify(s, locals=L)
def rhs(s): return E(s.split("=", 1)[1])
def pt(s): return tuple(E(s))

def p08():
    f = (sp.sqrt(x**2 - 3*x + 16) - 4) / (x - 3)
    a = limit(f, x, sp.oo); b = limit(f, x, 3)
    V.check(8, "1 a", a, V.ans(8, "1")); V.check(8, "2 b", b, V.ans(8, "2"))
    V.check(8, "3 f(a+8b)", sp.simplify(f.subs(x, a + 8*b)), V.ans(8, "3"))

def p09():
    a1, d = sp.symbols("a1 d")
    s = sp.solve([(a1+2*d)+(a1+7*d)-63, (a1+2*d)-(a1+7*d)-15], [a1, d])
    an = s[a1] + (n-1)*s[d]
    V.check(9, "1 a1", s[a1], V.ans(9, "1")); V.check(9, "2 an", an, E(V.ans(9, "2")))
    V.check(9, "3 n", sp.solve(an, n)[0], V.ans(9, "3"))

def p10():
    f = x**3 - 6*x + 5; m = sp.diff(f, x).subs(x, 1); line = m*(x-1)
    V.check(10, "1 tangent", line, rhs(V.ans(10, "1")))
    g = -x**2 - 5*x + 2
    ts = sp.solve(sp.Eq(g, line), x)  # double root
    tt = sp.solve(sp.diff(g, x) - m, x)[0]
    ok = sp.simplify(g.subs(x, tt) - line.subs(x, tt)) == 0 and sp.factor(g - line).has(sp.Pow)
    B = (tt, g.subs(x, tt))
    V.check(10, "2 B", list(B), list(pt(V.ans(10, "2"))))
    V.check(10, "3 AB", sp.sqrt((B[0]-1)**2 + B[1]**2), V.ans(10, "3"))
    V.report(10, ok, "B 가 접점(이중근)")

def p11():
    c = sp.symbols("c0:4"); a = sp.symbols("a")
    f = lambda v: sum(c[i]*v**i for i in range(4))
    u = sp.symbols("u")
    eq = sp.integrate(x*f(u), (u, 0, x)) - (x**3/3 - 8*x - sp.Rational(32, 3) + 2*sp.integrate(f(u), (u, a, x)) + sp.integrate(u*f(u), (u, a, x)))
    eqs = sp.Poly(sp.expand(eq), x).coeffs()
    sols = [s for s in sp.solve(eqs, list(c) + [a], dict=True) if s.get(a, 1) != 0]
    s = sols[0]; fx = f(x).subs(s)
    V.check(11, "1 f(0)", fx.subs(x, 0), V.ans(11, "1"))
    V.check(11, "2 a", s[a], V.ans(11, "2"))
    V.check(11, "3 f(a)", fx.subs(x, s[a]), V.ans(11, "3"))

def p12():
    a = {2: -1, 1: -1}  # a1=a2, a2=S2-S1 의 n=1
    for k in range(1, 40):
        a[2*k] = -4*k + 3; a[2*k+1] = 5*k - 1
    S_ = {0: 0}
    for k in range(1, 80): S_[k] = S_[k-1] + a[k]
    ok = all(S_[2*k+1]-S_[2*k] == 5*k-1 and S_[2*k]-S_[2*k-1] == -4*k+3 for k in range(1, 39))
    V.report(12, ok and a[1] == a[2], "조건 재현")
    V.check(12, "1 a1+a3", a[1]+a[3], V.ans(12, "1")); V.check(12, "2 S5", S_[5], V.ans(12, "2"))
    V.check(12, "3 max k", max(k for k in range(1, 78) if S_[k] < 0), V.ans(12, "3"))

def p13():
    t_ = t
    f = lambda s: s**2 - (1-s**2) - s
    ft = sp.expand(f(t_))
    V.check(13, "1 f(t)", ft, E(V.ans(13, "1")))
    import math
    vals = [math.sin(2*math.pi*i/20000)**2 - math.cos(2*math.pi*i/20000)**2 - math.sin(2*math.pi*i/20000) for i in range(20000)]
    mn, mx = extrema(ft, t, -1, 1)
    V.check(13, "2 M-m", mx-mn, V.ans(13, "2"))
    V.report(13, abs(max(vals)-min(vals) - float(mx-mn)) < 1e-6, "수치 격자 확인")

def p14():
    p, q = sp.symbols("p q")
    g = lambda v: sp.Rational(1, 8)*2**(3*v) - 2
    s = sp.solve([8**(0-p)+q - g(0), 8**(1-p)+q - g(1)], [p, q], dict=True)[0]
    V.check(14, "1 p", s[p], V.ans(14, "1", "p")); V.check(14, "1 q", s[q], V.ans(14, "1", "q"))
    xa = sp.solve(sp.Eq(2**(3*x), 16), x)[0]
    A = (xa, 0); B = (0, g(0))
    V.check(14, "2 area", abs(A[0]*B[1])/2, V.ans(14, "2"))

if __name__ == "__main__": V.run(globals())
