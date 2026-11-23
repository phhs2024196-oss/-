#!/usr/bin/env python3
"""KUT-G24B-01~10 (한국기술교육대 2024학년도 수리논술 오후) — sympy/전수 검산. 기대값은 YAML(V.ans)에서 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import sympy as sp

V = Verifier("KUT-G24B", "items/KUT/PAST")


def p01():
    xv = sp.log(6, 3)                                   # 3^x = 6
    yv = sp.log(8, 3)
    V.check(1, "6x-2y", sp.N(6*xv - 2*yv, 30), V.ans(1, "1"))


def p02():
    a, b = sp.symbols("a b", positive=True)
    f = lambda xv: a**(2*xv + 1) + b
    # 0<a<1 → 감소함수: 최댓값 f(-2), 최솟값 f(1). 수치 전수로 단조성 확인
    for av in [sp.Rational(1, 10), sp.Rational(1, 2), sp.Rational(9, 10)]:
        vals = [av**(2*sp.Rational(i, 10) + 1) for i in range(-20, 11)]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    sol = [s for s in sp.solve([f(-2) - 5, f(1) - sp.Rational(7, 2)], [a, b], dict=True) if 0 < s[a] < 1]
    assert len(sol) == 1
    V.check(2, "a^3 b", sol[0][a]**3*sol[0][b], V.ans(2, "1"))


def p03():
    p, q, r, s = sp.symbols("p q r s")
    f = p*x**3 + q*x**2 + r*x + s
    s0 = sp.solve(f.subs(x, 0), s)[0]                    # lim f/x 존재 → f(0)=0
    f = f.subs(s, s0)
    r0 = sp.solve(sp.limit(sp.cancel(f/x), x, 0) + 6, r)[0]
    f = f.subs(r, r0)
    # f(3)=0 이어야 두 번째 극한 존재, 극한값 24
    f3 = sp.solve(f.subs(x, 3), q)[0]
    f = f.subs(q, f3)
    p0 = sp.solve(sp.limit(sp.cancel(f/(x - 3)), x, 3) - 24, p)[0]
    F = f.subs(p, p0)
    V.check(3, "lim f/(2x^2+3x+1)", sp.limit(F/(2*x**2 + 3*x + 1), x, 1), V.ans(3, "1"))


def p04():
    sols = {}
    for n in range(1, 5):                                # 최고차항 계수 1 인 n 차 다항식 전수
        cs = sp.symbols(f"c0:{n}")
        f = x**n + sum(c*x**i for i, c in enumerate(cs))
        eq = sp.expand(f*sp.diff(f, x) - f + x*sp.diff(f, x) - (2*x**3 + 13*x**2 + 22*x + 9))
        sl = sp.solve(sp.Poly(eq, x).all_coeffs(), cs, dict=True)
        if sl:
            sols[n] = f.subs(sl[0])
    assert list(sols) == [2]
    V.check(4, "f(2)", sols[2].subs(x, 2), V.ans(4, "1"))


def p05():
    X = sp.symbols("X", positive=True)
    # 반지름 X, 9-X, X, 9-X, X ; 중심각 = 정육각형 내각 2π/3
    ang = 2*sp.pi/3
    total = ang*(X + (9 - X) + X + (9 - X) + X)
    xs = [v for v in sp.solve(sp.Eq(total, sp.Rational(41, 3)*sp.pi), X) if 0 < v < 9]
    assert len(xs) == 1
    V.check(5, "CRS 넓이", sp.Rational(1, 2)*xs[0]**2*ang, V.ans(5, "1"))


def p06():
    a = sp.symbols("a")
    p, q = sp.symbols("p q")
    f = (x + 1)*(x**2 + p*x + q)                         # 최고차 계수 1, f(-1)=0
    # f(x)g(x) = f(x)/(x-2) (x≠2): 극한 존재 → f(2)=0
    qs = sp.solve(f.subs(x, 2), q)[0]
    f = f.subs(q, qs)                                    # f = (x+1)(x-2)(x+a') 꼴
    lim = sp.limit(sp.cancel(f/(x - 2)), x, 2)           # = f(2) g(2) = 0*5 = 0
    ps = sp.solve(lim - 0, p)[0]
    F = f.subs(p, ps)
    V.check(6, "f(3)", F.subs(x, 3), V.ans(6, "1"))


def p07():
    a = sp.symbols("a")
    asol = sp.solve(sp.Rational(1, 2)*(-2)**3 - a*(-2) + 1 - 48, a)[0]   # f(-2)=48
    f = lambda t: sp.Rational(1, 2)*t**3 - asol*t + 1
    good = []
    for n in range(1, 20):
        pts = [-2 + sp.Rational(i, 8) for i in range(1, 8*(n + 2))]
        pts = [t for t in pts if t < n]
        dec = all(f(pts[i]) > f(pts[i + 1]) for i in range(len(pts) - 1))
        grid = [-2 + sp.Rational(i, 8) for i in range(0, 8*(n + 2) + 1) if -2 + sp.Rational(i, 8) <= n]
        mx = max(f(t) for t in grid)
        if dec and mx == 48:
            good.append(n)
    V.check(7, "sum n", sum(good), V.ans(7, "1"))


def p08():
    a, b = sp.symbols("a b")
    f = x**3 - 4*x**2 + a*x
    g = x**3 + b*x**2 + 2*x
    eq1 = sp.Poly(sp.expand(f + f.subs(x, -x) + g + g.subs(x, -x) - 2*x**2), x).all_coeffs()
    bs = sp.solve(eq1, b)[b] if isinstance(sp.solve(eq1, b), dict) else sp.solve(eq1, b)[0]
    g = g.subs(b, bs)
    I = sp.integrate(sp.diff(f, x) + x*sp.diff(g, x), (x, -1, 1))
    as_ = sp.solve(I - sp.Rational(38, 3), a)[0]
    f = f.subs(a, as_)
    V.check(8, "int(xf+g)", sp.integrate(x*f + g, (x, -1, 1)), V.ans(8, "1"))


def p09():
    def nxt(v):
        return v + 7 if v % 2 else v//2 + 5
    # (1) a4=20 에서 앞으로 전개
    seq = {4: 20}
    for n in range(4, 23):
        seq[n + 1] = nxt(seq[n])
    V.check(9, "9-1 a23", seq[23], V.ans(9, "9-1"))
    # (2) a1..a3 전수(자연수 a1 상한 1000)
    sums = []
    for a1 in range(1, 1000):
        a2 = nxt(a1); a3 = nxt(a2)
        if nxt(a3) == 20:
            sums.append(a1 + a2 + a3)
    M, m = max(sums), min(sums)
    V.check(9, "9-2 M", M, V.ans(9, "9-2", "M"))
    V.check(9, "9-2 m", m, V.ans(9, "9-2", "m"))
    V.check(9, "9-2 M+m", M + m, V.ans(9, "9-2", "M+m"))


def p10():
    A, B, C = sp.symbols("A B C")
    f = x**3 + A*x**2 + B*x + C
    fp = sp.diff(f, x)
    num = sp.expand(fp.subs(x, x + 1) - fp.subs(x, x - 3))
    As = sp.solve(num.subs(x, 3), A)[0]                  # x→3- 에서 극한 존재 (분자→0)
    f1 = f.subs(A, As); fp1 = sp.diff(f1, x)
    gl = sp.cancel((fp1.subs(x, x + 1) - fp1.subs(x, x - 3))/(x - 3))   # x<3 의 g
    gr = x*f1 + 6                                         # x≥3 의 g
    conds = [sp.Eq(gl.subs(x, 3), gr.subs(x, 3)),         # 연속
             sp.Eq(sp.diff(gl, x).subs(x, 3), sp.diff(gr, x).subs(x, 3))]   # 미분가능
    sol = sp.solve(conds, [B, C], dict=True)[0]
    F = f1.subs(sol)
    V.check(10, "10-1 f(1)", F.subs(x, 1), V.ans(10, "10-1"))
    h = F + 2*x
    cr = sorted(sp.solve(sp.diff(h, x), x))               # 작은 쪽 = 극대, 큰 쪽 = 극소 (최고차 양수)
    al, be = h.subs(x, cr[0]), h.subs(x, cr[1])
    V.check(10, "alpha", al, V.ans(10, "10-2", "alpha"))
    V.check(10, "beta", be, V.ans(10, "10-2", "beta"))
    V.check(10, "alpha+beta", al + be, V.ans(10, "10-2", "alpha+beta"))


if __name__ == "__main__":
    V.run(globals())
