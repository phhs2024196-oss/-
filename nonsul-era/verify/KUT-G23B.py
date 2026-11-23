#!/usr/bin/env python3
"""KUT-G23B-01~10 (2023학년도 한국기술교육대 실전 오후) — 문제 설정으로부터 독립 재계산 후 YAML 답과 비교."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: F401,F403
V = Verifier("KUT-G23B", "items/KUT/PAST")
R = sp.Rational


def p01():
    ns = []
    for m in range(1, 200):
        v = sp.N((sp.sqrt(2*sp.root(8, 4)))**m, 50)
        if abs(v - sp.floor(v + R(1, 2))) < 1e-30 and 100 <= round(v) <= 999:
            ns.append(m)
    V.report(1, len(ns) == 1, f"조건 만족 n={ns}")
    V.check(1, "n", ns[0], V.ans(1, "1"))


def p02():
    tot, good = 0, []
    for a in range(-200, 201):
        base = abs(a - 2)
        if base == 0 or base == 1:
            continue
        # x^2+2ax+5a+36 의 최솟값(x=-a)이 양수
        if (x**2 + 2*a*x + 5*a + 36).subs(x, -a) > 0:
            good.append(a)
    V.check(2, f"a={good} 의 합", sum(good), V.ans(2, "1"))


def p03():
    # f(3)=1, f'(3)=1 (f=x-2), g(3)=5, g'(3)=5 (g=5x-10) : f(3)g(3)=5, (fg)'(3)=10
    f, g = x - 2, 5*x - 10
    V.check(3, "예시 함수가 조건 만족", sp.limit((f*g - 5)/(x - 3), x, 3), 10)
    val = sp.diff(f, x).subs(x, 3)/f.subs(x, 3) + sp.diff(g, x).subs(x, 3)/g.subs(x, 3)
    # 일반: f3,g3,f1,g1 로 놓고 f3*g3=5, f1*g3+f3*g1=10 이면 값은 항상 10/5
    f3, g3, f1, g1 = sp.symbols("f3 g3 f1 g1")
    gen = sp.simplify((f1/f3 + g1/g3).subs(g1, (10 - f1*g3)/f3).subs(f3, 5/g3))
    V.check(3, "예시값", val, V.ans(3, "1"))
    V.check(3, "일반값", gen, V.ans(3, "1"))


def p04():
    X = sp.Symbol("X", positive=True)
    ang = sp.pi*2/3
    B = sp.Matrix([2, 0]); C = X*sp.Matrix([sp.cos(ang), sp.sin(ang)])
    D = (X*B + 2*C)/(2 + X)          # 각의 이등분선 정리: BD:DC = AB:AC
    ad2 = sp.simplify(D.dot(D))
    sol = [s for s in sp.solve(sp.Eq(ad2, X**4), X) if s.is_real and s > 0]
    V.check(4, f"AC 후보 {sol}", sol[0], V.ans(4, "1"))


def p05():
    c = sp.symbols("c0:5")
    f = sum(c[i]*x**i for i in range(5))
    t = sp.Symbol("t")
    rhs = 4*x**3 + 3*x**2 + sp.integrate(f.subs(x, t), (t, -2, x))
    s = sp.solve(sp.Poly(sp.expand(x*f - rhs), x).all_coeffs(), c, dict=True)[0]
    V.check(5, "f(2)", f.subs(s).subs(x, 2), V.ans(5, "1"))


def p06():
    tt = sp.Symbol("tt")
    cnt = lambda tv: len(sp.solveset(x**2 + 2*x + tv, x, sp.S.Reals))
    eps = R(1, 1000)
    L, M, Rr = cnt(1 - eps), cnt(1), cnt(1 + eps)
    a = sp.Symbol("a")
    sol = sp.solve(sp.Eq((1 - eps + a)*L, (1 + eps + a)*Rr), a)  # 근사: 좌우 값 일치 조건
    exact = sp.solve(sp.Eq((1 + a)*L, (1 + a)*Rr), a)
    V.report(6, (L, M, Rr) == (2, 1, 0), f"f(1-)={L}, f(1)={M}, f(1+)={Rr}")
    V.check(6, f"연속 조건의 해 {exact}", exact[0], V.ans(6, "1"))


def p07():
    a = sp.Symbol("a")
    f = (x - 2)*(x - 5)*(x + a)
    L = sp.limit(f/((x - 5)*sp.diff(f, x)**2), x, 5)
    sol = [s for s in sp.solve(sp.Eq(L, R(1, 9)), a) if s != -5]
    V.check(7, f"a={sol} 일 때 f(3)", f.subs(a, sol[0]).subs(x, 3), V.ans(7, "1"))


def p08():
    t = sp.Symbol("t")
    f = sp.integrate(3*(t + x)*(t + x**2), (t, 0, 2))
    mn, mx = extrema(f, x, 0, 2)
    V.report(8, all(sp.diff(f, x).subs(x, v) > 0 for v in [R(i, 10) for i in range(-30, 31)]), "f' > 0 (격자)")
    V.check(8, "M+m", mn + mx, V.ans(8, "1"))


def p09():
    k = sp.Symbol("k", positive=True)
    xb = sp.Symbol("xb", positive=True)
    # AB = sqrt(2)*xb = sqrt(2) -> xb = 1; B on y=log2 x + 6
    kk = sp.solve(sp.Eq(k - 1, sp.log(1, 2) + 6), k)[0]
    V.check(9, "1-1 k", kk, V.ans(9, "1-1"))
    xc = sp.Symbol("xc", positive=True)
    # BC^2 = AB*CD (등비중항), 모든 거리는 sqrt2*(x좌표 차)
    sol = [s for s in sp.solve(sp.Eq((xc - 1)**2, 1*(kk - xc)), xc) if 1 < s < kk]
    xC = sol[0]
    a = sp.Symbol("a")
    aa = sp.solve(sp.Eq(kk - xC, a*sp.log(xC, 9)), a)[0]
    V.check(9, f"1-2 a (x_C={xC})", sp.simplify(aa), V.ans(9, "1-2"))


def p10():
    kk = sp.Symbol("k")
    u = sp.Symbol("u")
    f = x*(x - kk)*(x - 4)
    fp = sp.diff(f, x)
    ts = sp.solve(sp.Eq(f.subs(x, u), u*fp.subs(x, u)), u)   # 접점 (u=0 은 원점)
    slopes = sorted({sp.simplify(fp.subs(x, s)) for s in ts}, key=lambda e: float(e.subs(kk, 2)))
    slopes = [s for s in slopes]
    l1, l2 = slopes[-1], slopes[0]
    V.check(10, "1-1 l1", l1, V.ans(10, "1-1", "l1"))
    V.check(10, "1-1 l2", l2, V.ans(10, "1-1", "l2"))
    g = sp.simplify(l1*l2)
    V.check(10, "1-2 Σg(k)", sum(g.subs(kk, i) for i in (1, 2, 3)), V.ans(10, "1-2"))


if __name__ == "__main__":
    V.run(globals())
