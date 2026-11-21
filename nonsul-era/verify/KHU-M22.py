#!/usr/bin/env python3
"""KHU-M22 (경희대 2022학년도 모의논술 자연계 논제 1~4) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "KHU-M22"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def exp_cubic_check():
    """g(x)=f(e^x), f=4x^3+ax^2+bx+c. (ㄱ)(ㄴ)(ㄷ) 아래 h 의 최솟값 k(a)의 최대 존재 여부."""
    a, b, c = sp.symbols("a b c", real=True)
    g = 4 * sp.exp(3 * x) + a * sp.exp(2 * x) + b * sp.exp(x) + c
    gp = sp.diff(g, x)
    cond = sp.expand(gp - sp.exp(4 * x) * gp.subs(x, -x))
    bsol = sp.solve([sp.simplify(cond.subs(x, 1)), sp.simplify(cond.subs(x, 2))], [b], dict=True)
    bval = bsol[0][b]
    ok = bval == 12 and sp.simplify(cond.subs(b, 12)) == 0
    cval = sp.limit(g.subs(b, bval), x, -sp.oo) - c  # g→c
    ok &= cval == 0                                     # lim g = c ⇒ c=-9a
    # (ㄷ): 6A^2+aA+6 의 부호가 바뀌는 양근 존재 ⇔ a<-12 (수치 확인)
    def has_extremum(av):
        q = sp.Poly(6 * t**2 + av * t + 6, t)
        rs = [r for r in q.nroots() if abs(sp.im(r)) < 1e-12 and sp.re(r) > 0]
        return len(rs) == 2 and abs(rs[0] - rs[1]) > 1e-9
    ok &= all(has_extremum(av) == (av < -12) for av in [-30, -13, -12.0001, -12, -11.9, 0, 5, 13, 30])
    # h 의 최솟값: 수치 최소화와 k(a)=3/4 a^3 - 9a 비교
    def hmin(av):
        h = lambda xv: 4 * math.exp(3 * xv) + av * math.exp(2 * xv) - 2 * av**2 * math.exp(xv) - 9 * av
        lo, hi = -20.0, 10.0
        for _ in range(200):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if h(m1) < h(m2):
                hi = m2
            else:
                lo = m1
        return h((lo + hi) / 2)
    av_s = sp.symbols("a", real=True)
    k = sp.Rational(3, 4) * av_s**3 - 9 * av_s
    ok &= all(abs(hmin(av) - float(k.subs(av_s, av))) < 1e-6 * max(1, abs(hmin(av))) for av in [-13, -15, -20, -40])
    # k 는 a<-12 에서 증가(k'>0) ⇒ 열린 구간이라 최댓값 없음
    kp = sp.diff(k, av_s)
    ok &= sp.solve_univariate_inequality(kp > 0, av_s, relational=False).contains(-12) and \
        all(kp.subs(av_s, v) > 0 for v in [-12.001, -50, -1000])
    return ok, "b=12, c=-9a, 극값⇔a<-12, min h=3a^3/4-9a 증가·끝점 제외 ⇒ 최대가 되는 a 없음"


def p01():
    # (1) 100x^2+240xy, x^2+y^2=1 → 삼각치환 수치 최대 + 이차형식 고윳값
    vals = [100 * math.cos(u)**2 + 240 * math.cos(u) * math.sin(u) for u in [i * 2 * math.pi / 200000 for i in range(200000)]]
    lam = max(sp.Matrix([[100, 120], [120, 0]]).eigenvals())
    ok1 = lam == 180 and abs(max(vals) - 180) < 1e-6
    ok2, msg = exp_cubic_check()
    report(1, ok1 and ok2, f"(1) 최댓값={lam}; (2) {msg}")


def count_int_roots(As, Bs, Cs):
    cnt = 0
    for a, b, c in itertools.product(As, Bs, Cs):
        D = a * a - 4 * b * c
        if D > 0:
            s = math.isqrt(D)
            if s * s == D and (a + s) % 2 == 0:
                cnt += 1
    return cnt


def p02():
    n1 = count_int_roots(range(0, 6), range(1, 6), [1, -1])
    n2 = count_int_roots(range(2, 17), [2, 3, 5, 7, 11, 13, 17, 19], [1, -1])
    prob = sp.Rational(n2, 15 * 8 * 2)
    report(2, n1 == 9 and prob == sp.Rational(1, 20), f"(1) {n1}가지; (2) {n2}/240={prob}")


def tri_area(P):
    return abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))) / 2


def clip_above(poly, A, B):
    """다각형 poly 중 직선 AB 의 왼쪽(위쪽) 부분 (Sutherland–Hodgman)."""
    side = lambda P: (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])
    out = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        sp_, sq = side(P), side(Q)
        if sp_ >= 0:
            out.append(P)
        if sp_ * sq < 0:
            tt = sp_ / (sp_ - sq)
            out.append((P[0] + tt * (Q[0] - P[0]), P[1] + tt * (Q[1] - P[1])))
    return out if len(out) >= 3 else []


def p03():
    a, n = sp.symbols("a n", positive=True)
    ok = True
    for av, nv in [(1.0, 2), (2.0, 3), (1.5, 5), (1.0, 9)]:
        w = av / nv
        h = math.sqrt(av**2 - (w / 2)**2)
        tris = [[(i * w, 0.0), ((i + 0.5) * w, h), ((i + 1) * w, 0.0)] for i in range(nv)]
        S = sum(tri_area(T) for T in tris)
        Bn = ((nv - 0.5) * w, h)
        Tn = sum(tri_area(clip_above(T, (0.0, 0.0), Bn)) if clip_above(T, (0.0, 0.0), Bn) else 0 for T in tris)
        Sf = av**2 * math.sqrt(4 * nv**2 - 1) / (4 * nv)
        Tf = (2 * nv - 1) * math.sqrt(4 * nv**2 - 1) / (24 * nv**2) * av**2
        ok &= abs(S - Sf) < 1e-12 and abs(Tn - Tf) < 1e-12
    Ssym = a**2 * sp.sqrt(4 * n**2 - 1) / (4 * n)
    Tsym = (2 * n - 1) * sp.sqrt(4 * n**2 - 1) / (24 * n**2) * a**2
    l1, l2 = sp.limit(Ssym, n, sp.oo), sp.limit(Tsym, n, sp.oo)
    ok &= l1 == a**2 / 2 and l2 == a**2 / 6
    report(3, ok, f"S_n·T_n 식을 n=2,3,5,9 다각형 절단 넓이로 확인, 극한 {l1}, {l2}")


def p04():
    r = 1.0
    best_area, areas, best_per = None, [], 0
    for i in range(1, 20000):
        c = -r + 2 * r * i / 20000
        # A=(c-a, a), E=(c+b, b) 가 원 위: (c-a)^2+a^2=r^2 → a, (c+b)^2+b^2=r^2 → b (양근)
        a = (c + math.sqrt(2 * r * r - c * c)) / 2
        b = (-c + math.sqrt(2 * r * r - c * c)) / 2
        if a <= 0 or b <= 0:
            continue
        areas.append(a * a / 2 + b * b / 2)
        best_per = max(best_per, (2 + math.sqrt(2)) * (a + b))
    ok = max(areas) - min(areas) < 1e-12 and abs(max(areas) - r * r / 2) < 1e-12
    ok &= abs(best_per - (2 + 2 * math.sqrt(2)) * r) < 1e-6
    rs = sp.symbols("r", positive=True)
    bb = sp.symbols("b", positive=True)
    f = (2 + sp.sqrt(2)) * (bb + sp.sqrt(rs**2 - bb**2))
    crit = sp.solve(sp.diff(f, bb), bb)
    ok &= sp.simplify(f.subs(bb, crit[0]) - (2 + 2 * sp.sqrt(2)) * rs) == 0
    report(4, ok, f"(1) 넓이 합 상수 {max(areas):.6f}=r^2/2; (2) 최대 둘레 {best_per:.6f}r=(2+2√2)r")


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
