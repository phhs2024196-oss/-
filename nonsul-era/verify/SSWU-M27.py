import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import math
V = Verifier("SSWU-M27", "items/SSWU/PAST")
X = sp.Symbol('x'); N = sp.Symbol('n'); P_ = sp.Symbol('p')
A_, B_ = sp.symbols('a b')

def p01():
    f = X**2 + 4*X + 8; g = X**2 - 4*X + 16
    sol = sp.solve([sp.discriminant(f - A_*X - B_, X), sp.discriminant(g - A_*X - B_, X)], [A_, B_], dict=True)
    a, b = sol[0][A_], sol[0][B_]
    V.check(1, "1-(1) a", a, V.ans(1, "1", "a")); V.check(1, "1-(1) b", b, V.ans(1, "1", "b"))
    line = a*X + b
    c = sp.solve(sp.diff(f, X) - a, X)[0]; d = sp.solve(sp.diff(g, X) - a, X)[0]
    assert same(f.subs(X, c), line.subs(X, c)) and same(g.subs(X, d), line.subs(X, d))
    V.check(1, "1-(2) c", c, V.ans(1, "2", "c")); V.check(1, "1-(2) d", d, V.ans(1, "2", "d"))
    xi = sp.solve(f - g, X)[0]
    area = sp.integrate(f - line, (X, c, xi)) + sp.integrate(g - line, (X, xi, d))
    V.check(1, "1-(3) 넓이", area, V.ans(1, "3"))

def p02():
    # 좌표: A(-4,0), B(4,0), O(0,3) (OM=3), P 는 O 반대편 작은 호
    sa = sp.Rational(4, 5)
    V.check(2, "2-(1) sin(alpha)=AM/OA", sp.Rational(4, 5), V.ans(2, "1", "sin_alpha"))
    al = math.asin(0.8)
    def P(th):  # ∠PAB=θ 인 원 위의 점(아래쪽)
        # A 에서 방향 (cosθ,-sinθ), 원 (x)^2+(y-3)^2=25 와의 두 번째 교점
        dx, dy = math.cos(th), -math.sin(th)
        ax, ay = -4, 0
        # |A + s d - O|^2 = 25 → s = -2 d·(A-O)
        s = -2*(dx*(ax-0) + dy*(ay-3))
        return (ax + s*dx, ay + s*dy), s
    ok = all(abs(P(th)[1] - 10*math.sin(al-th)) < 1e-9 for th in [0.05*i for i in range(1, 15)])
    V.report(2, ok, "2-(1) AP=10 sin(α-θ) (좌표 수치)")
    expr = S(V.ans(2, "1", "AP"))
    al_s, th_s = sp.symbols('alpha theta')
    assert sp.simplify(expr.subs({al_s: sp.asin(sa)}) .subs(th_s, 0.3) - 10*sp.sin(sp.asin(sa)-0.3)) == 0
    th = sp.Symbol('theta', positive=True)
    Sx = 40*sp.sin(sp.asin(sa)-th)*sp.sin(th)
    # S 를 좌표 넓이로 확인
    for t0 in [0.2, 0.5]:
        (px, py), _ = P(t0)
        area = abs((4-(-4))*(py-0) - (px-(-4))*0)/2
        assert abs(area - float(Sx.subs(th, t0))) < 1e-9
    V.check(2, "2-(2) 극한", sp.limit(Sx/th, th, 0, '+'), V.ans(2, "2"))
    # (3) T(θ)=삼각형 PAO 넓이 최대
    def T(t0):
        (px, py), _ = P(t0); ox, oy = 0, 3
        return abs((-4-ox)*(py-oy) - (px-ox)*(0-oy))/2
    best = max((T(0.0005*i), 0.0005*i) for i in range(1, 1500))
    V.check(2, "2-(3) 최대 θ에서 sinθ(수치)", math.sin(best[1]), float(S(V.ans(2, "3"))), 1e-3)
    thx = sp.asin(sa) - sp.pi/4
    V.check(2, "2-(3) sin(α-π/4)", sp.sin(thx), V.ans(2, "3"))

def p03():
    f = A_*X**2 + B_; c = sp.Symbol('c')
    sol = sp.solve([f.subs(X, 1), (f.subs(X, 2)-f.subs(X, 1)), ], [A_, B_], dict=True)  # placeholder
    a = sp.solve(sp.Eq((f.subs(X, 2) - f.subs(X, 1)).subs(B_, -A_), sp.Rational(3, 2)), A_)[0]
    b = -a
    cc = sp.solve(sp.Eq((c*sp.log(sp.E) - c*sp.log(1))/(sp.E-1), 1/(sp.E-1)), c)[0]
    V.check(3, "3-(1) a", a, V.ans(3, "1", "a")); V.check(3, "3-(1) b", b, V.ans(3, "1", "b")); V.check(3, "3-(1) c", cc, V.ans(3, "1", "c"))
    h = (a*X**2 + b) - cc*sp.log(X)
    crit = [r for r in sp.solve(sp.diff(h, X), X) if r > 0]
    pp = crit[0]; q = h.subs(X, pp)
    assert sp.diff(h, X, 2).subs(X, pp) > 0 and all(float(h.subs(X, v)) >= float(q) - 1e-12 for v in [0.01*i for i in range(1, 600)])
    V.check(3, "3-(2) p", pp, V.ans(3, "2", "p")); V.check(3, "3-(2) q", q, V.ans(3, "2", "q"))
    F = a*X**2 + b; G = cc*sp.log(X)
    assert all(F.subs(X, v) > G.subs(X, v) for v in [0.5, 0.9, 1.1, 2, 5])
    lim = sp.limit((sp.exp(F) - sp.exp(G))/(F - G), X, pp)
    V.check(3, "3-(3) 극한", lim, V.ans(3, "3"))

def p04():
    pp = sp.Symbol('p'); nn = sp.Symbol('n', positive=True); x = sp.Symbol('x')
    an = sp.integrate(1/(x + pp), (x, nn, nn + 1))
    an_f = lambda k, pv: sp.log((k+1+pv)/(k+pv))
    sols = [s_ for s_ in sp.solve(sp.Eq(sp.exp(an_f(1, pp) + an_f(6, pp)), sp.Rational(3, 2)), pp) if s_ > -1]
    V.check(4, "4-(1) p", sols[0], V.ans(4, "1", "p"))
    pv = sols[0]
    an2 = sp.log((nn+1+pv)/(nn+pv))
    N_ = sp.Symbol('n')
    V.check(4, "4-(1) a_n(수치 n=1..5 적분 대조)", [sp.N(sp.integrate(1/(x+pv), (x, k, k+1)), 20) for k in range(1, 6)], [sp.N(S(V.ans(4, "1", "an")).subs(N_, k), 20) for k in range(1, 6)], 1e-12)
    bn = sp.simplify(sp.exp(an2) - 1)
    k = sp.Symbol('k', integer=True, positive=True)
    Sn = sp.summation(bn.subs(nn, k)*bn.subs(nn, k+1), (k, 1, nn))
    q = sp.limit(Sn, nn, sp.oo)
    V.check(4, "4-(2) q", q, V.ans(4, "2"))
    b = bn; l_x = sp.Symbol('t')
    # 접선 y=3b^2(x-b)+b^3, 넓이
    tang = 3*b**2*(x - b) + b**3
    An = sp.integrate(x**3 - tang, (x, 0, b))
    lim = sp.limit(sp.simplify(An/(q - Sn)**4), nn, sp.oo)
    V.check(4, "4-(3) 극한", lim, V.ans(4, "3"))

if __name__ == "__main__": V.run(globals())
