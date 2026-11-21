#!/usr/bin/env python3
"""경희대 2023학년도 논술(자연계, 11.19 오후) KHU-G23A-01~03 — sympy 검산. 정답 하드코딩 비교 금지, 문제 설정에서 독립 재계산."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "KHU-G23A"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I: 타원 x^2/a^2+y^2/b^2=1, A(k,0)에서 그은 접선의 접점(y>0)."""
    a, b, k, x1, y1 = sp.symbols("a b k x1 y1", positive=True)
    # (1) 접선 x1 x/a^2 + y1 y/b^2 = 1 이 (k,0)을 지나고 P가 타원 위
    sols = sp.solve([x1*k/a**2 - 1, x1**2/a**2 + y1**2/b**2 - 1], [x1, y1], dict=True)
    P = [(s[x1], s[y1]) for s in sols]
    md = (a**2/k, b*sp.sqrt(1 - a**2/k**2))
    ok = any(sp.simplify(px - md[0]) == 0 and sp.simplify(py - md[1]) == 0 for px, py in P)
    # 수치: 접선 조건 재확인(판별식) a=5,b=4,k=13
    vals = {a: 5, b: 4, k: 13}
    px, py = [sp.nsimplify(v.subs(vals)) for v in md]
    X, Y = sp.symbols("X Y", real=True)
    line_y = sp.solve(px*X/25 + py*Y/16 - 1, Y)[0]
    disc = sp.discriminant(sp.expand((X**2/25 + line_y**2/16 - 1)*400), X)
    ok &= sp.simplify(disc) == 0 and line_y.subs(X, 13) == 0
    # (2) 법선 l' 과 x축 교점 Q, 초점 (±3,0)
    slope_t = sp.diff(line_y, X)
    normal = py + (-1/slope_t)*(X - px)
    qx = sp.solve(normal, X)[0]
    c = sp.sqrt(25 - 16)
    F1, F2 = (-c, 0), (c, 0)
    dist = lambda p, q: sp.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
    Pp, Q = (px, py), (qx, 0)
    val = sp.nsimplify(sp.simplify(dist(Pp, F1)/dist(Q, F1) + dist(Pp, F2)/dist(Q, F2)))
    ok &= val == sp.Rational(10, 3) and qx == sp.Rational(9, 13)
    report(1, ok, f"(1) P=(a^2/k, b√(1-a^2/k^2)) 연립해 일치, 판별식 0; (2) Q=({qx},0), 합={val}")


def p02():
    """논제 II: 두 정사각형 겹침 넓이 f(t), ∫|f-g|, S(a) 최소."""
    t, a, s = sp.symbols("t a s", real=True)
    yc = -t**2 + 3*t

    def overlap(tv):
        y = -tv**2 + 3*tv
        w = max(0, min(3, tv+1) - max(1, tv-1))
        h = max(0, min(3, y+1) - max(1, y-1))
        return w*h
    pieces = [(-t**3+3*t**2, 0, 1), (t**3-3*t**2+4*t, 1, 2), (t**3-7*t**2+12*t, 2, 3)]
    ok = True
    for expr, lo, hi in pieces:   # 다항식이 기하학적 겹침 넓이와 일치(격자 300점)
        for i in range(1, 300):
            tv = lo + (hi-lo)*i/300
            ok &= abs(float(expr.subs(t, tv)) - overlap(tv)) < 1e-9
    g1, g2 = 2*t, 12 - 4*t
    I = sum(sp.integrate(sp.Abs(e - (g1 if hi <= 2 else g2)).rewrite(sp.Piecewise), (t, lo, hi))
            for e, lo, hi in pieces)
    I = sp.nsimplify(sp.N(I, 40))
    # 독립 수치 적분 (기하 넓이 그대로)
    N = 60000
    num = sum(abs(overlap((i+0.5)*3/N) - (2*(i+0.5)*3/N if (i+0.5)*3/N < 2 else 12-4*(i+0.5)*3/N)) for i in range(N))*3/N
    ok &= I == sp.Rational(7, 12) and abs(num - 7/12) < 1e-6
    # (2) S(a) = ∫_1^2 |f(t)-f(a)| dt, f 는 [1,2]에서 증가
    f2 = pieces[1][0]
    S = sp.integrate(f2.subs(t, a) - f2, (t, 1, a)) + sp.integrate(f2 - f2.subs(t, a), (t, a, 2))
    crit = [r for r in sp.solve(sp.diff(S, a), a) if r.is_real and 1 < r < 2]
    grid_min = min((float(S.subs(a, 1 + i/1000)), 1 + i/1000) for i in range(1, 1000))
    ok &= crit == [sp.Rational(3, 2)] and abs(grid_min[1] - 1.5) < 2e-3
    ok &= all(float(sp.diff(f2, t).subs(t, 1 + i/100)) > 0 for i in range(101))
    report(2, ok, f"(1) f(t) 3구간 = 기하 겹침, ∫|f-g|={I} (수치 {num:.6f}); (2) S'(a)=0 → a={crit}, 격자최소 {grid_min[1]}")


def p03():
    """논제 III: 불꽃 분기 — (1) P(B|A), A: 22<=X<=26; (2) f(t) 미분불가능점."""
    from fractions import Fraction as Fr
    PA = PAB = Fr(0)
    for first in (4, 6):
        for sec in itertools.product((4, 6), repeat=first):
            pr = Fr(1, 2)**(1 + first)
            X = sum(sec)
            if 22 <= X <= 26:
                PA += pr
                if first == 6:
                    PAB += pr
    cond = PAB/PA
    ok = cond == Fr(7, 27)

    def pts(tv):
        P = []
        for a0 in (120, 60, 180, 0, 240, -60):
            bx, by = 10*math.cos(math.radians(a0)), 10*math.sin(math.radians(a0))
            for o in (30, -30, 90, -90, 150, -150):
                b = math.radians(a0 + o)
                P.append((bx + 2*tv*math.cos(b), by + 2*tv*math.sin(b)))
        return P

    def f(tv):
        P = pts(tv)
        return min(math.dist(p, q) for p, q in itertools.combinations(P, 2))
    # 수치로 한쪽 미분계수 변화 지점 탐색
    dt = 1e-3
    grid = [i*dt for i in range(1, 2500)]
    vals = [f(tv) for tv in grid]
    sl = [(vals[i+1]-vals[i])/dt for i in range(len(vals)-1)]
    kinks = [grid[i+1] for i in range(len(sl)-1) if abs(sl[i+1]-sl[i]) > 0.1]
    tt = sp.symbols("tt", positive=True)
    t0 = sp.solve(sp.Eq(2*tt, 10 - 2*sp.sqrt(3)*tt), tt)[0]
    t0f = float(t0)
    ok &= len(kinks) > 0 and all(abs(k - t0f) < 2e-3 for k in kinks)
    ok &= sp.simplify(t0 - 5*(sp.sqrt(3)-1)/2) == 0
    dl = (f(t0f) - f(t0f - 1e-7))/1e-7
    dr = (f(t0f + 1e-7) - f(t0f))/1e-7
    ok &= abs(dl - 2) < 1e-4 and abs(dr + 2*math.sqrt(3)) < 1e-4
    report(3, ok, f"(1) 2^7 경우 전수: P(B|A)={cond}; (2) 36점 최소거리 수치 꺾임 t≈{kinks[0] if kinks else None}, t0={t0} (좌미분 {dl:.4f}, 우미분 {dr:.4f})")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
