#!/usr/bin/env python3
"""SSWU-G20 (성신여대 2020학년도 논술고사 자연계열 문제 1~4) 검산. 기대값은 V.ans() 로 YAML 에서 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
import math

V = Verifier("SSWU-G20", "items/SSWU/PAST")
X = sp.Symbol("x")
A_ = sp.Symbol("a")


def ev(s, **kw):
    e = sp.sympify(s)
    return e.subs({sp.Symbol(k): v for k, v in kw.items()})


def pair(s):
    return [sp.sympify(t) for t in s.strip("() ").split(",")]


def p01():
    f = (X + 3) / (4 * X + 8)
    g = -(X - 3) / (X - 4)
    # (1) 점근선: 극한으로
    fx, fy = -2, sp.limit(f, X, sp.oo)
    gx, gy = 4, sp.limit(g, X, sp.oo)
    assert sp.limit(f, X, -2, "+") == sp.oo and sp.limit(g, X, 4, "+") == -sp.oo
    V.check(1, "1-1 넓이", abs(gx - fx) * abs(gy - fy), V.ans(1, "1"))
    # (2) A=(s,f(s)), C = P 에 대한 대칭점이 f 위, s<-2, B=(x_C, y_A)
    s = sp.Symbol("s")
    Px, Py = fx, fy
    # C 는 P 에 대한 A 의 대칭점(대각선 교점이 P). f 는 P 에 대칭이라 C 는 자동으로 f 위. 정사각형: 가로=세로
    eq = (2 * Px - 2 * s) - (2 * Py - 2 * f.subs(X, s))
    sols = [r for r in sp.solve(sp.numer(sp.together(eq)), s) if r < -2]
    assert len(sols) == 1
    sA = sols[0]
    B = (2 * Px - sA, f.subs(X, sA))
    ex = pair(V.ans(1, "2"))
    V.check(1, "1-2 B.x", B[0], ex[0]); V.check(1, "1-2 B.y", B[1], ex[1])
    # 정사각형 확인: 변 길이 같음, 네 변 축 평행
    C = (2 * Px - sA, 2 * Py - f.subs(X, sA))
    V.report(1, abs((C[0] - sA) - (C[1] - f.subs(X, sA))) == 0 and f.subs(X, C[0]) == C[1], "1-2 정사각형·C 가 f 위")
    # (3) h=-f, k=g+b: 점근선 x=-2,y=-1/4 / x=4,y=-1+b, 가로=세로
    b = sp.Symbol("b")
    w = abs(4 - (-2)); hy = sp.limit(-f, X, sp.oo); ky = sp.limit(g, X, sp.oo) + b
    bs = set(sp.solve(sp.Eq((ky - hy) ** 2, w ** 2), b))
    V.check(1, "1-3 b", bs, [sp.sympify(t) for t in V.ans(1, "3").split(";")])


def p02():
    a = A_
    # 좌표: O=원점, A=(1,0), P=(cos a, sin a), Q=(cos 2a, -sin 2a)
    def pts(av):
        P = mp.matrix([mp.cos(av), mp.sin(av)]); Q = mp.matrix([mp.cos(2 * av), -mp.sin(2 * av)])
        return P, Q
    def seg_area(av):  # 활꼴: 부채꼴(각 3a) - 삼각형 OPQ, 삼각형 OPQ 는 외적으로
        P, Q = pts(av)
        tri = abs(P[0] * Q[1] - P[1] * Q[0]) / 2
        return (3 * av) / 2 - tri
    def pHq(av):
        P, Q = pts(av)
        u = Q / mp.norm(Q)           # OQ 방향 (|OQ|=1)
        H = (P.T * u)[0] * u
        d1, d2 = P - H, Q - H
        return abs(d1[0] * d2[1] - d1[1] * d2[0]) / 2
    fe, ge = V.ans(2, "1", "f"), V.ans(2, "1", "g")
    ok = True
    for av in [0.05, 0.2, 0.3, 0.45, 0.5]:
        ok &= abs(seg_area(av) - mp.mpf(str(ev(fe, a=av).evalf(30)))) < 1e-12
        ok &= abs(pHq(av) - mp.mpf(str(ev(ge, a=av).evalf(30)))) < 1e-12
    V.report(2, ok, "2-1 f,g 를 좌표 기하와 비교")
    fa, ga = sp.sympify(fe), sp.sympify(ge)
    V.check(2, "2-2 limit f/a", sp.limit(fa / a, a, 0, "+"), V.ans(2, "2"))
    V.check(2, "2-3 limit g/a^3", sp.limit(ga / a ** 3, a, 0, "+"), V.ans(2, "3"))


def p03():
    f = X ** 3 / 3 - X + 10
    A = (2, f.subs(X, 2)); m = sp.diff(f, X).subs(X, 2)
    line = m * (X - 2) + A[1]
    V.check(3, "3-1 접선", sp.expand(line), sp.expand(sp.sympify(V.ans(3, "1"))))
    rts = sp.solve(sp.expand(f - line), X)
    xB = [r for r in rts if r != 2]
    V.check(3, "3-2 xB", xB[0], V.ans(3, "2"))
    V.report(3, sp.factor(f - line).as_poly(X).degree() == 3 and sp.roots(sp.expand(f - line), X) == {2: 2, -4: 1}, "3-2 x=2 중근")
    V.check(3, "3-3 넓이", sp.integrate(sp.Abs(1) * (f - line), (X, xB[0], 2)), V.ans(3, "3"))
    # 구간에서 곡선이 접선 위
    V.report(3, all((f - line).subs(X, sp.Rational(i, 10)) >= 0 for i in range(-40, 21)), "3-3 [-4,2] 에서 f>=line")


def lattice(n):
    """행 y=0..n, 행 y 의 꼭짓점 (y,i) i=0..n-y, 위치 x=i+y/2. 이웃 반환."""
    V_ = [(y, i) for y in range(n + 1) for i in range(n - y + 1)]
    nb = {v: [] for v in V_}
    for (y, i) in V_:
        for (y2, i2) in [(y, i - 1), (y, i + 1), (y + 1, i), (y + 1, i - 1), (y - 1, i), (y - 1, i + 1)]:
            if (y2, i2) in nb:
                nb[(y, i)].append((y2, i2))
    return V_, nb


def p04():
    k_, n_ = sp.symbols("k n_")
    res_ok = True
    for n in range(1, 8):
        Vs, nb = lattice(n)
        B = (n, 0)
        # 최단 경로 수: 거리 BFS + 개수
        dist = {B: 0}; cnt = {B: 1}; order = [B]
        for v in order:
            for w in nb[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1; cnt[w] = 0; order.append(w)
                if dist[w] == dist[v] + 1:
                    cnt[w] += cnt[v]
        f_vals = [cnt[(0, kk)] for kk in range(n + 1)]
        if n == 1:
            V.report(4, f_vals == [int(t) for t in V.ans(4, "1", "R1").split(",")], f"4-1 R1 {f_vals}")
        if n == 2:
            V.report(4, f_vals == [int(t) for t in V.ans(4, "1", "R2").split(",")], f"4-1 R2 {f_vals}")
        fk = sp.sympify(V.ans(4, "1", "f"))
        res_ok &= f_vals == [int(fk.subs({sp.Symbol("n"): n, sp.Symbol("k"): kk})) for kk in range(n + 1)] if False else True
        ff = sp.sympify(V.ans(4, "1", "f"), locals={"binomial": sp.binomial})
        V.report(4, f_vals == [int(ff.subs({sp.Symbol("n"): n, sp.Symbol("k"): kk})) for kk in range(n + 1)], f"4-1 n={n} f(k) 전수")
        gn = sp.sympify(V.ans(4, "2", "g")).subs(sp.Symbol("n"), n)
        V.report(4, sum(f_vals) == gn, f"4-2 n={n} g(n)={gn}")
    # 4-2 급수
    nn = sp.Symbol("nn", integer=True, positive=True)
    gsym = sp.sympify(V.ans(4, "2", "g")).subs(sp.Symbol("n"), nn)
    V.check(4, "4-2 sum", sp.summation(1 / gsym, (nn, 1, sp.oo)), V.ans(4, "2", "sum"))
    # 4-3 최장 경로: DFS (n<=5), B=(n,0) -> A=(0,n)
    sys.setrecursionlimit(10000)
    e3 = sp.sympify(V.ans(4, "3"))
    for n in range(1, 6):
        Vs, nb = lattice(n)
        best = 0
        start, goal = (n, 0), (0, n)
        def dfs(v, seen, L):
            nonlocal best
            if v == goal:
                best = max(best, L)
                return
            for w in nb[v]:
                if w not in seen:
                    seen.add(w); dfs(w, seen, L + 1); seen.discard(w)
        dfs(start, {start}, 0)
        V.report(4, best == e3.subs(sp.Symbol("n"), n) == (len(Vs) - 1), f"4-3 n={n} 최장={best}, 식={e3.subs(sp.Symbol('n'), n)}, 꼭짓점-1={len(Vs)-1}")


if __name__ == "__main__":
    V.run(globals())
