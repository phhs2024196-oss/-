#!/usr/bin/env python3
"""KUT-G26-01~10 (한국기술교육대 2026학년도 실전) — 문제 설정에서 독립 재계산, 기대값은 YAML(V.ans)."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-G26", "items/KUT/PAST")
R = sp.Rational


def p01():
    f = (R(1, 2)) ** x + 1
    lo, hi = extrema(f, x, 0, 2)  # m, M
    a, b = sp.symbols("a b", real=True)
    g = lambda c: -a**c + b
    sols = [q for q in sp.solve([sp.Eq(g(0), -lo), sp.Eq(g(1), -hi)], [a, b], dict=True) if q[a] > 1]
    assert len(sols) == 1
    sol = sols[0]
    V.check(1, "a+b", sol[a] + sol[b], V.ans(1, "1"))


def p02():
    f = lambda u: sp.log(10 * u**2 + 35, 5)
    inv = lambda v: sp.sqrt((5**v - 35) / 10)  # x>0 에서 f 의 역함수
    assert abs(sp.N(f(inv(3)) - 3)) < 1e-12
    V.check(2, "f^-1(f^-1(3))", sp.simplify(inv(inv(3))), V.ans(2, "1"))


def p03():
    lo = (5*x + 1)**2 / (5 * (7*x**2 + 4)); hi = (5*x + 3)**2 / (5 * (7*x**2 + 4))
    L, H = sp.limit(lo, x, sp.oo), sp.limit(hi, x, sp.oo)
    V.check(3, "조임 하계=상계", L, H)
    V.check(3, "극한값", L, V.ans(3, "1"))
    # 치환 확인: f(t)<=(5t+3)^2/5, (5s+1)^2/5<=f(s) 는 x=5t+3, 5s+1 대입
    t_, s_ = sp.symbols("t_ s_")
    assert sp.simplify(((5*t_ + 3) - 3) / 5 - t_) == 0 and sp.simplify(((5*s_ + 1) - 1) / 5 - s_) == 0


def p04():
    c = sp.symbols("c")
    f2, df2 = 3, -3  # 접선 y=-3x+9 의 x=2 에서
    assert -3 * 2 + 9 == f2
    fx, dfx = sp.Function("f"), None
    X = sp.symbols("X")
    F, dF = sp.symbols("F dF")
    g2 = R(1, 8) * 8 * F
    dg2 = R(3, 8) * 4 * F + R(1, 8) * 8 * dF
    g2v, dg2v = g2.subs(F, f2), dg2.subs({F: f2, dF: df2})
    slope_m = -1 / dg2v
    m = slope_m * (X - 2) + g2v
    l = -3 * X + 9
    xi_l, xi_m = sp.solve(l, X)[0], sp.solve(m, X)[0]
    xc = sp.solve(sp.Eq(l, m), X)[0]
    yc = l.subs(X, xc)
    area = sp.Abs(xi_m - xi_l) * sp.Abs(yc) / 2
    V.check(4, "넓이", area, V.ans(4, "1"))
    # 구체 다항함수로 확인: f(x)=3-3(x-2)+k(x-2)^2 (k 임의)
    for kk in (0, 1, -2):
        ff = 3 - 3*(x - 2) + kk*(x - 2)**2
        gg = R(1, 8) * x**3 * ff
        assert sp.simplify(gg.subs(x, 2) - g2v) == 0 and sp.simplify(sp.diff(gg, x).subs(x, 2) - dg2v) == 0


def p05():
    import numpy as np
    def fmin(th):
        s = math.sin(th)
        xv = max(s, 0.0)
        return xv*xv - 2*s*xv + math.cos(th)**2 - s + 1
    th = np.linspace(0, 2*math.pi, 2_000_001, endpoint=False)
    s = np.sin(th); xv = np.maximum(s, 0)
    val = xv**2 - 2*s*xv + np.cos(th)**2 - s + 1 - 1
    roots = []
    for i in range(len(th) - 1):
        if val[i] == 0 or val[i]*val[i+1] < 0:
            roots.append(th[i])
    # 연속인 이웃 묶음
    cl = []
    for r in roots:
        if not cl or r - cl[-1] > 1e-3: cl.append(r)
    tot = sum(cl)
    V.check(5, "해의 개수", len(cl), 2)
    V.check(5, "합(수치)", sp.Float(tot, 12), sp.N(sp.pi, 12), tol=1e-4)
    V.check(5, "θ=π/6,5π/6 에서 최솟값 1", sum(fmin(q) for q in (math.pi/6, 5*math.pi/6)), 2, tol=1e-12)
    V.check(5, "합(YAML)", sp.pi, sp.sympify(V.ans(5, "1")))


def p06():
    a, p, q, r = sp.symbols("a p q r")
    f = lambda u: u**3 + p*u**2 + q*u + r
    # g 의 좌·우 값(불연속 후보 x=±a)
    gl = lambda u: 2*f(u)           # |x|<=a 쪽 값 (u=±a 에서 f(|x|)=f(a))
    g_in, g_out = 2*f(a), -f(a)     # x=a, -a 에서 g 의 안/밖 극한
    eqs = [f(0)*(g_in - g_out), f(-2*a)*(g_in - g_out), f(2*a)*(g_in - g_out)]  # f(x-a)g(x)@x=a,-a ; f(x)g(x-a)@x=2a ; (x=0 은 f(0))
    # x=a: f(0)*g(a-)=f(0)*g(a+) ; x=-a: f(-2a)*g ; x=2a: f(2a)*g ; x=0: f(0)*g(-a): g(-a) 안/밖 같은 값이므로 f(0)=0
    sols = sp.solve(eqs + [f(a) + R(81, 8)], [p, q, r, a], dict=True)
    sols = [s_ for s_ in sols if s_[a].is_real and s_[a] > 0 and sp.simplify(f(a).subs(s_) + R(81, 8)) == 0 and sp.simplify(f(a).subs(s_)) != 0]
    V.check(6, "해 개수", len(sols), 1)
    V.check(6, "a", sols[0][a], V.ans(6, "1"))
    fx = sp.expand(f(x).subs(sols[0]))
    V.check(6, "f=x(x-2a)(x+2a)", fx, sp.expand(x*(x - 2*sols[0][a])*(x + 2*sols[0][a])))


def p07():
    a, b = sp.symbols("a b")
    # (나): (x-3)(x-1)(x^2+ax+b)>=0 항상 -> x=3,1 에서 부호 변화: 이차식이 (x-1)(x-3)
    f = sp.expand((x - 1)**2 * (x - 3))
    assert sp.expand((x - 3)*f) == sp.expand((x - 3)**2 * (x - 1)**2) and f.coeff(x, 3) == 1 and f.subs(x, 1) == 0
    # 후보 확인: 이차식 x^2+ax+b 의 다른 선택은 불가(부호 변화 조건)
    import itertools
    for aa, bb in itertools.product(range(-6, 7), repeat=2):
        h = sp.expand((x - 3)*(x - 1)*(x**2 + aa*x + bb))
        pol = sp.Poly(h, x)
        ok = all(h.subs(x, sp.Rational(v, 4)) >= 0 for v in range(-40, 60))
        if ok:
            assert (aa, bb) == (-4, 3), (aa, bb)
    k = sp.symbols("k", real=True)
    fp = sp.diff(f, x)
    def cnt(kv):
        eq = sp.expand(f - (x - 1)*fp.subs(x, kv))
        return len(set(sp.real_roots(sp.Poly(eq, x))))
    ks = set(sp.solve(sp.Eq(fp.subs(x, k), -1), k)) | set(sp.solve(sp.Eq(fp.subs(x, k), 0), k))
    assert all(cnt(kv) == 2 for kv in ks)
    # 격자 점검: 개수 2 인 k 는 위 4개뿐
    for i in range(-60, 100):
        kv = sp.Rational(i, 12)
        assert (cnt(kv) == 2) == (kv in ks), kv
    V.check(7, "k 합", sum(ks), V.ans(7, "1"))


def p08():
    t_, s_ = sp.symbols("t_ s_", real=True)
    v1 = 3*t**2 + 4*t - 20; v2 = 2*t + 9
    s1 = 3 + sp.integrate(v1.subs(t, s_), (s_, 0, t)); s2 = 20 + sp.integrate(v2.subs(t, s_), (s_, 0, t))
    d = sp.expand(s1 - s2)
    roots = sorted(r for r in sp.solve(sp.Eq(sp.Abs(d), 12), t) if r.is_real and r >= 0)
    # 절댓값 방정식: d=12 또는 d=-12 의 비음 실근
    cand = sorted(r for r in sp.real_roots(sp.Poly(d - 12, t)) + sp.real_roots(sp.Poly(d + 12, t)) if r >= 0)
    t1 = cand[0]
    V.check(8, "처음 시각", t1, 5)
    dist = sp.integrate(sp.Abs(v1), (t, 0, t1)) if False else sp.integrate(-v1, (t, 0, 2)) + sp.integrate(v1, (t, 2, t1))
    V.check(8, "P 움직인 거리", dist, V.ans(8, "1"))
    V.check(8, "수치 적분", sp.N(sp.Integral(sp.Abs(v1), (t, 0, 5))), V.ans(8, "1"), tol=1e-6)


def p09():
    import numpy as np
    xs = np.linspace(0, math.pi/3, 400001)
    def mm(a, b):
        fv = np.cos(xs - math.pi/6)**2 + a*np.cos(xs + math.pi/3) + b
        return fv.max(), fv.min()
    M, m = mm(0.5, 1)
    V.check(9, "(1) M", sp.Float(M, 10), sp.N(sp.sympify(V.ans(9, "9-1", "M")), 10), tol=1e-7)
    V.check(9, "(1) m", sp.Float(m, 10), sp.N(sp.sympify(V.ans(9, "9-1", "m")), 10), tol=1e-7)
    # (2) 기호 풀이: 네 경우
    a, b = sp.symbols("a b", real=True)
    g = lambda u: -(u + a/2)**2 + a**2/4 + b + 1
    sols = set()
    for M_pt, m_pt, cond in [(-R(1, 2), R(1, 2), lambda s_: s_[a] >= 1), (-a/2, R(1, 2), lambda s_: 0 <= s_[a] < 1),
                             (-a/2, -R(1, 2), lambda s_: -1 < s_[a] < 0), (R(1, 2), -R(1, 2), lambda s_: s_[a] <= -1)]:
        for s_ in sp.solve([sp.Eq(g(M_pt), 5), sp.Eq(g(m_pt), -3)], [a, b], dict=True):
            if all(v.is_real for v in s_.values()) and cond(s_):
                sols.add((sp.nsimplify(s_[a]), sp.nsimplify(s_[b])))
    exp = {tuple(sp.sympify(v) for v in pr) for pr in
           [tuple(w.strip() for w in chunk.split(",")) for chunk in V.ans(9, "9-2").replace(" ", "").strip("()").split("),(")]}
    V.report(9, sols == exp, f"(2) 순서쌍 {sorted(sols, key=str)} / YAML {sorted(exp, key=str)}")
    # 수치 점검: 격자 탐색으로 (M,m)=(5,-3) 을 만드는 (a,b) 가 위 두 개뿐인지 확인
    for (av, bv) in [(8, 0.25), (-8, 0.25)]:
        M2, m2 = mm(av, bv)
        V.report(9, abs(M2 - 5) < 1e-6 and abs(m2 + 3) < 1e-6, f"(a,b)=({av},{bv}) -> M={M2:.6f}, m={m2:.6f}")


def p10():
    n_, tt, X = sp.symbols("n_ tt X", positive=True)
    def parts(nv):
        f = X**2 - 2*nv*X + nv**2
        l = sp.diff(f, X).subs(X, tt)*(X - tt) + f.subs(X, tt)
        an = sp.integrate(f - l, (X, 0, tt))
        xi = sp.solve(l, X)[0]; yi = l.subs(X, 0)
        bn = sp.Rational(1, 2)*xi*yi
        return sp.expand(an + bn), sp.simplify(an), sp.simplify(bn)
    def cn(nv):
        g, _, _ = parts(nv)
        assert sp.simplify(sp.diff(g, tt) - (tt - nv)**2/4) == 0  # 0<t<n 에서 증가
        lo, hi = g.subs(tt, 0), g.subs(tt, nv)
        return len([j for j in range(int(sp.floor(lo)) - 1, int(sp.ceiling(hi)) + 2) if lo < j < hi])
    V.check(10, "(1) t 개수(n=3)", cn(3), V.ans(10, "10-1"))
    V.check(10, "(2) c5+c6+c7", cn(5) + cn(6) + cn(7), V.ans(10, "10-2"))
    # 수치 점검: 격자에서 a_n+b_n 이 정수 근방 통과하는 횟수
    g3, a3, _ = parts(3)
    V.check(10, "a_3=t^3/3", a3, tt**3/3)


if __name__ == "__main__":
    V.run(globals())
