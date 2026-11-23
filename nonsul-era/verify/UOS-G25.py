import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, math
from fractions import Fraction
V = Verifier("UOS-G25", "items/UOS/PAST")

def seq(N):
    out = []; v = 13
    while len(out) < N:
        if math.gcd(v, 12) == 1: out.append(v)
        v += 1
    return out

def p01():
    a = seq(200)  # a[k-1]
    k_min = next(i + 1 for i, v in enumerate(a) if v >= 100)
    V.check(1, "(a) k 최솟값", k_min, V.ans(1, "1-1"))
    f = sp.sympify(V.ans(1, "1-2"))
    n_ = sp.Symbol("n")
    for nv in range(1, 9):
        s = sum(a[k - 1]*a[k] - a[k + 1] for k in range(1, 2*nv + 1))
        V.check(1, f"(b) n={nv}", s, f.subs(n_, nv))
    # 일반항 a_{2m-1}=6m+7, a_{2m}=6m+11
    V.report(1, all(a[2*m - 2] == 6*m + 7 and a[2*m - 1] == 6*m + 11 for m in range(1, 40)), "일반항 a_{2m-1}, a_{2m}")

def p02():
    a_, b_, c_ = sp.symbols("a b c")
    f = sp.sympify(V.ans(2, "2-1"))
    def proj(av, bv, cv):
        av, bv, cv = [sp.nsimplify(v) for v in (av, bv, cv)]
        u = 1 + (bv**2 - cv**2)/4          # A 의 x 좌표(AE=b, BF=c 조건)
        h = sp.sqrt(bv**2 - 1 - u**2)
        E = sp.Matrix([0, 0, 0]); H = sp.Matrix([0, av + 2, 0]); F = sp.Matrix([av + 2, 0, 0])
        A = sp.Matrix([u, 1, h]); D = sp.Matrix([u, 1 + av, h]); B = sp.Matrix([u + av, 1, h])
        assert sp.simplify((A - E).norm() - bv) == 0 and sp.simplify((B - F).norm() - cv) == 0 and sp.simplify((D - H).norm() - bv) == 0
        n = (F - B).cross(H - E); n = n/n.norm()          # 평면 BFGC 의 법선
        def pr(P): return P - n*(n.dot(P))
        pts = [pr(P) for P in (A, E, H, D)]               # 사각형 AEHD 의 정사영(순서 A,E,H,D)
        ar = sum(((pts[i].cross(pts[(i + 1) % 4])) for i in range(4)), sp.zeros(3, 1))
        return sp.N(ar.norm()/2, 15), sp.N(f.subs({a_: av, b_: bv, c_: cv}), 15)
    for tr in [(1, 2, 2), (2, sp.Rational(5, 2), sp.Rational(11, 5)), (3, 3, 2), (sp.Rational(1, 2), 2, 3)]:
        got, exp = proj(*tr)
        V.check(2, f"정사영 넓이 a,b,c={tr}", got, exp, 1e-8)

def p03():
    def deg_cells(nv):
        import itertools
        cells = [(i, j) for i in range(nv) for j in range(nv)]
        nb = {c: [(c[0] + di, c[1] + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= c[0] + di < nv and 0 <= c[1] + dj < nv] for c in cells}
        return cells, nb
    fa = sp.sympify(V.ans(3, "3-1")); fb = sp.sympify(V.ans(3, "3-2")); n_ = sp.Symbol("n")
    for nv in (3, 4, 5, 6):
        cells, nb = deg_cells(nv)
        tot_a = 0; win_a = 0
        for s1 in cells:
            for s2 in cells:
                if s2 == s1: continue
                tot_a += 1; win_a += (s2 in nb[s1])
        V.check(3, f"(a) n={nv}", sp.Rational(win_a, tot_a), fa.subs(n_, nv))
        tot = 0; win = 0
        for s1 in cells:
            for s2 in cells:
                if s2 == s1 or s2 in nb[s1]: continue   # 시립이가 첫 번에 승리하지 않는 경우
                for s3 in cells:
                    if s3 in (s1, s2): continue
                    tot += 1; win += (s3 in nb[s2])
        # 전체 표본공간: 서울이 s1, 시립이 s2, 서울이 s3 의 모든 순서열 n^2(n^2-1)(n^2-2)
        V.check(3, f"(b) n={nv}", sp.Rational(win, nv**2*(nv**2 - 1)*(nv**2 - 2)), fb.subs(n_, nv))
    V.check(3, "(b) 분자 합 공식", sp.simplify(8*(n_**2 - 3) + 12*(n_ - 2)**2*(n_ + 2) + 4*(n_ - 2)**2*(n_**2 - 5) - 4*(n_**4 - n_**3 - 5*n_**2 + 8*n_ - 2)), 0)

def p04():
    import numpy as np
    def quad(fn, lo, hi, points=None, limit=0):
        xs = np.linspace(lo, hi, 400001); ys = fn(xs)
        return float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(ys, xs)), 0
    lim = sp.sympify(V.ans(4, "4-1"))
    def T(nv):
        tot = 0.0
        for k in range(1, 3*nv + 1):
            pts = [2*nv*np.pi*j for j in range(1, 3) if 2*nv*np.pi*j < k*np.pi]
            val, _ = quad(lambda x: np.abs(np.sin(x/(2*nv))), 0, k*np.pi, points=pts or None, limit=200)
            tot += val/nv**2
        return tot
    # k 번째 적분값 공식 확인
    for nv, k in [(3, 2), (3, 5), (4, 8), (4, 11)]:
        val, _ = quad(lambda x: np.abs(np.sin(x/(2*nv))), 0, k*np.pi, points=[2*nv*np.pi] if k > 2*nv else None)
        form = 2*nv*(1 - np.cos(k*np.pi/(2*nv))) if k <= 2*nv else 2*nv*(3 + np.cos(k*np.pi/(2*nv)))
        V.check(4, f"적분값 n={nv},k={k}", val, form, 1e-7)
    # T_n 의 닫힌 형태(10 + (1/n)Σ…)와 수치 합 비교, 극한 수렴
    for nv in (5, 20):
        closed = 10 + sum(-4*np.cos(k*np.pi/(2*nv)) + 2*np.sin(k*np.pi/(2*nv)) for k in range(1, nv + 1))/nv
        V.check(4, f"T_n 정리식 n={nv}", T(nv), closed, 1e-6)
    V.check(4, "T_n(400) → 극한", T(400), sp.N(lim), 1e-2)
    V.check(4, "정적분", sp.integrate(-4*sp.cos(sp.pi*x/2) + 2*sp.sin(sp.pi*x/2), (x, 0, 1)) + 10, lim)

if __name__ == "__main__": V.run(globals())
