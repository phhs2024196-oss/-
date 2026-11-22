import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
E = lambda v: sp.sympify(v, locals={"x": x, "n": n})  # 식 답은 vlib 의 실수 기호로 읽기

V = Verifier("SMU-G25N", "items/SMU/PAST")
d, a, b = sp.symbols("d a b", real=True)

def p01():
    res = []
    for dd in sp.solve((d - 3)**2 - (-3 + 2*d), d):
        A, B = -3 + dd, -3 + 2*dd
        nums = [-3, A, B, A*B]
        if len(set(nums)) == 4 and A != 0 and sp.simplify(B/A - A*B/B) == 0:
            res.append((dd, B/A, A*B))
    V.report(1, len(res) == 1, f"서로 다른 네 수 조건 만족 해 {res}")
    V.check(1, "1 d", res[0][0], V.ans(1, "1")); V.check(1, "2 r", res[0][1], V.ans(1, "2")); V.check(1, "3 ab", res[0][2], V.ans(1, "3"))

def p02():
    L = sp.symbols("L", real=True)
    iv = sp.solve_univariate_inequality((L + 1)**2 - (L + 7) < 0, L, relational=False)
    V.check(2, "1 m", iv.left, V.ans(2, "1")); V.check(2, "2 n", iv.right, V.ans(2, "2"))
    m_, n_ = iv.left, iv.right
    ok = [X for X in range(-100, 101) if (X**2 + (m_ - n_)*X + m_*n_) > 0 and sp.log(X**2 + (m_ - n_)*X + m_*n_, 4) <= sp.log(sp.sqrt(m_*n_*(m_ - n_)), 2) + sp.Rational(1, 10**12)]
    V.check(2, "3 합", sum(ok), V.ans(2, "3"))

def p03():
    a1, dd = sp.symbols("a1 dd")
    so = sp.solve([a1 + 3*dd - 7, a1 + 10*dd - 21], [a1, dd])
    an = lambda kk: so[a1] + (kk - 1)*so[dd]
    V.check(3, "1 an", an(n), E(V.ans(3, "1")))
    V.check(3, "2 합", sum(1/(an(kk)*an(kk + 2)) for kk in range(1, 11)), V.ans(3, "2"))
    ms = [m for m in range(1, 200) if sum(sp.Rational(4)/(an(kk)*an(kk + 2)) for kk in range(1, m + 1)) < sp.Rational(13, 12)]
    V.check(3, "3 m", ms, E(V.ans(3, "3")))

def p04():
    seq = []; S = 0
    for i in range(1, 11):
        sols = [z for z in sp.solve(6*(S + a) - (a**2 + 3*a - 10), a) if z > 0]
        seq.append(sols[0]); S += sols[0]
    V.check(4, "1 a10", seq[9], V.ans(4, "1")); V.check(4, "2 a1", seq[0], V.ans(4, "2")); V.check(4, "3 합", sum(seq), V.ans(4, "3"))

def p05():
    lo, hi = -2*x**2 + 2*x, x**2 + 2*x
    L = sp.limit(lo/x, x, 0); H = sp.limit(hi/x, x, 0)
    V.report(5, L == H == 2, "조임: f(x)/x → 2")
    u = sp.symbols("u")
    V.check(5, "1", sp.limit(u**2/(3 + u), u, 2), V.ans(5, "1"))   # (f/x)^2/(3+f/x)
    # f(x+2) 를 경계 두 함수로 바꿔도 극한이 같아야 함
    A_, B_ = sp.symbols("A_ B_")
    res = []
    for fx in (lo, hi):
        e = (fx.subs(x, x + 2) - A_*x + B_)/(x**2 - 4)
        so = sp.solve([sp.numer(sp.together(e)).subs(x, -2), sp.limit(e.subs(B_, sp.solve(sp.numer(sp.together(e)).subs(x, -2), B_)[0]), x, -2) - 4], [A_, B_], dict=True)
        res.append(so[0])
    V.report(5, res[0] == res[1], f"경계 함수 두 개로 같은 a,b {res}")
    V.check(5, "2 a", res[0][A_], V.ans(5, "2", "a")); V.check(5, "2 b", res[0][B_], V.ans(5, "2", "b"))

def p06():
    tt = sp.symbols("tt")
    res = []
    for aa in sp.solve(sp.Symbol("z"), sp.Symbol("z")) or []: pass
    av = sp.symbols("av", real=True)
    s1 = (av + 1)/2
    for A in sp.solve(s1*(s1 + 1) - (3 - av), av):
        if A < 0:
            rts = sorted(sp.solve(tt**2 - (A + 2)*tt - A + 3, tt))
            res.append((A, 10**rts[0], 10**rts[1]))
    V.report(6, len(res) == 1 and res[0][2] == 10*res[0][1], f"음수 a 유일, q=10p {res}")
    V.check(6, "1 a", res[0][0], V.ans(6, "1")); V.check(6, "2 p", res[0][1], V.ans(6, "2")); V.check(6, "3 q", res[0][2], V.ans(6, "3"))

def p07():
    A = sp.solve(-3*2 - 3*(2 - a), a)[0]
    V.check(7, "1 a", A, V.ans(7, "1"))
    f = sp.Piecewise((-3*t, t < 2), (3*(t - A), True))
    g1 = sp.integrate((t - 2)*(-3*t), (t, 0, x))
    g2 = g1.subs(x, 2) + sp.integrate((t - 2)*3*(t - A), (t, 2, x))
    G = E(V.ans(7, "2"))
    V.report(7, all(sp.simplify(G.subs(x, X) - (g1 if X < 2 else g2).subs(x, X)) == 0 for X in [-3, -1, 0, 1, sp.Rational(3, 2), 2, 3, 5, 7]), "g(x) 구간별 일치")
    def cnt(c):
        r1 = [z for z in sp.real_roots(sp.Poly(sp.expand(g1 - c), x)) if z < 2]
        r2 = [z for z in sp.real_roots(sp.Poly(sp.expand(g2 - c), x)) if z >= 2]
        return len(set(r1 + r2))
    V.check(7, "3 b+c", cnt(0) + cnt(3), V.ans(7, "3"))

def p08():
    C = sp.symbols("C")
    xf = sp.integrate(4*x**3 - 3*x**2 + 3, x) + C
    F = sp.cancel(xf.subs(C, 0)/x)
    V.report(8, sp.simplify(sp.diff(sp.integrate(F - 3, x) + sp.integrate(x*sp.diff(F, x), x) - (x**4 - x**3), x)) == 0, "원식(미분 형태) 만족")
    crit = sp.solve(sp.diff(F, x), x)
    mx = [c for c in crit if sp.diff(F, x, 2).subs(x, c) < 0]
    V.check(8, "1 f(a)", F.subs(x, mx[0]), V.ans(8, "1"))
    l = sp.diff(F, x).subs(x, 1)*(x - 1) + F.subs(x, 1)
    V.check(8, "2 l", l, E(V.ans(8, "2")))
    xs = sorted(set(sp.solve(F - l, x)))
    V.check(8, "3 넓이", sp.integrate(abs(F - l), (x, xs[0], xs[-1])), V.ans(8, "3"))

if __name__ == "__main__": V.run(globals())
