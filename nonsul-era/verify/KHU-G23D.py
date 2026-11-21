#!/usr/bin/env python3
"""경희대 2023학년도 논술(의·약학계 수학, 11.19 오후) KHU-G23D-01~02 — sympy 검산. 정답 하드코딩 비교 금지, 문제 설정에서 독립 재계산."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "KHU-G23D"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I-1: 두 정사각형 겹침 넓이 f(t), ∫|f-g|, S(a) 최소."""
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
    f3 = pieces[2][0]   # [2,3]에서 감소
    S = sp.integrate(f3 - f3.subs(t, a), (t, 2, a)) + sp.integrate(f3.subs(t, a) - f3, (t, a, 3))
    crit = [r for r in sp.solve(sp.diff(S, a), a) if r.is_real and 2 < r < 3]
    grid_min = min((float(S.subs(a, 2 + i/1000)), 2 + i/1000) for i in range(1, 1000))
    ok &= crit == [sp.Rational(5, 2)] and abs(grid_min[1] - 2.5) < 2e-3
    ok &= all(float(sp.diff(f3, t).subs(t, 2 + i/100)) < 0 for i in range(101))
    report(1, ok, f"(1) f(t) 3구간 = 기하 겹침, ∫|f-g|={I} (수치 {num:.6f}); (2) S'(a)=0 → a={crit}, 격자최소 {grid_min[1]}")


def p02():
    """논제 I-2: 불꽃 분기 — (1) P(B|A), A: 20<=X<=28; (2) 미분불가능점과 f(t0)."""
    from fractions import Fraction as Fr
    PA = PAB = Fr(0)
    for first in (4, 6):
        for sec in itertools.product((4, 6), repeat=first):
            pr = Fr(1, 2)**(1 + first)
            X = sum(sec)
            if 20 <= X <= 28:
                PA += pr
                if first == 6:
                    PAB += pr
    cond = PAB/PA
    ok = cond == Fr(1, 3)

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
    ft0 = f(t0f)
    ok &= abs(ft0 - float(5*(sp.sqrt(3)-1))) < 1e-9
    report(2, ok, f"f(t0)={ft0:.10f}; (1) 2^7 경우 전수: P(B|A)={cond}; (2) 36점 최소거리 수치 꺾임 t≈{kinks[0] if kinks else None}, t0={t0} (좌미분 {dl:.4f}, 우미분 {dr:.4f})")


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
