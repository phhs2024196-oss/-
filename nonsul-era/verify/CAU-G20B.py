#!/usr/bin/env python3
"""CAU-G20B (2020학년도 중앙대 수시 논술 자연계열Ⅱ 수학 문제 1~3) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
PREFIX = "CAU-G20B"
import re
import sys
import itertools
from fractions import Fraction as Fr

import sympy as sp

WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")



def p01():
    """주머니 규칙을 그대로 따라 4명의 배정을 정확한 분수로 전수 → P(A,B,C 모두 배정) (답 1/4)."""
    order = "ABC"

    def step(bag, assigned, k):
        """bag: {'A':개수,..,'':빈 공}, assigned: 배정된 과제 집합. k명 남음. 반환: 성공 확률."""
        if k == 0:
            return Fr(int(set(assigned) == set(order)))
        tot = sum(bag.values())
        res = Fr(0)
        for ball, cnt in bag.items():
            if cnt == 0:
                continue
            pr = Fr(cnt, tot)
            if ball:  # 과제명이 쓰인 공 → 그 과제, 빈 공 하나에 그 과제명 기록
                nb = dict(bag); nb[ball] = nb.get(ball, 0) + 1; nb[""] -= 1
                res += pr * step(nb, assigned | {ball}, k - 1)
            else:
                un = [c for c in order if c not in assigned]
                if un:  # 우선순위 높은 미배정 과제, 뽑은 공에 기록
                    c = un[0]
                    nb = dict(bag); nb[c] = nb.get(c, 0) + 1; nb[""] -= 1
                    res += pr * step(nb, assigned | {c}, k - 1)
                else:  # 모두 배정됨 → 임의 배정(이후 사건에 영향 없음)
                    for c in order:
                        res += pr * Fr(1, 3) * step(dict(bag), assigned | {c}, k - 1)
        return res

    P = step({"A": 2, "": 4}, frozenset(), 4)
    report(1, P == Fr(1, 4), f"P(세 과제 모두 배정)={P}")


def p02():
    """(2-1) 중근 g(t) 수치 추적 + 음함수 미분, ∫_0^1 t g'' dt 수치적분 (답 1/6) / (2-2) k=1..8 연립으로 p(x) 결정 (답 2x^2-2πx)."""
    import mpmath as mp
    X, T = sp.symbols("X T", real=True)
    F = 4 * X**3 - 6 * (T + 1) * X**2 + 7 * T**2 + 1
    def mid_root(tv):
        rs = sorted(complex(r).real for r in sp.Poly(F.subs(T, tv), X).nroots())
        return rs[1], rs
    # g' , g'' 음함수 미분
    gp = -sp.diff(F, T) / sp.diff(F, X)
    gpp = sp.diff(gp, T) + sp.diff(gp, X) * gp
    gpp_f = sp.lambdify((X, T), gpp, "mpmath")
    integrand = lambda tv: tv * gpp_f(mid_root(float(tv))[0], tv)
    val = mp.quad(integrand, [0, 0.5, 1])
    ok_real = all(len([r for r in sp.Poly(F.subs(T, sp.Rational(k, 20)), X).nroots() if abs(sp.im(r)) < 1e-12]) == 3 for k in range(-2, 23))
    g0 = mid_root(0)[0]; g1 = mid_root(1)[0]
    ok1 = ok_real and abs(val - sp.Rational(1, 6)) < 1e-8 and abs(g0 - 0.5) < 1e-12 and abs(g1 - 1) < 1e-12
    # (2-2)
    a, b, c, d = sp.symbols("a b c d", real=True)
    k = sp.symbols("k", integer=True, positive=True)
    p = a * x**3 + b * x**2 + c * x + d
    eqs = [sp.integrate((kk**2 * p + 4) * sp.sin(kk * x), (x, 0, sp.pi)) for kk in range(1, 9)]
    sol = sp.solve(eqs, [a, b, c, d], dict=True)
    ok2 = len(sol) == 1
    pp = sp.expand(p.subs(sol[0]))
    ok2 = ok2 and sp.simplify(pp - (2 * x**2 - 2 * sp.pi * x)) == 0
    ok2 = ok2 and all(sp.simplify(sp.integrate((kk**2 * pp + 4) * sp.sin(kk * x), (x, 0, sp.pi))) == 0 for kk in range(1, 21))
    report(2, ok1 and ok2, f"(2-1) g(0)={g0:.6f}, g(1)={g1:.6f}, ∫t g''={float(val):.10f}  (2-2) p(x)={pp}")


def p03():
    """(3-1) 등차수열 연립 → a_254 (답 2019) / (3-2) 초점 P,Q 타원 위 ∠PRQ=π/3 인 최대 x 점을 수치로 찾아 dy/dx (답 -16√3/73)."""
    c, d, e, f = sp.symbols("c d e f", real=True)
    an = lambda m: c * m + d
    bn = lambda m: e * m + f
    m = sp.symbols("m", integer=True)
    eqs = sp.Poly(sp.expand(an(2 * m) - bn(m) - 3), m).coeffs()
    eqs += [bn(1) - 756, sum(an(j**2) for j in range(1, 12)) - sum((bn(j) - an(j)) ** 2 for j in range(1, 12))]
    sols = [s for s in sp.solve(eqs, [c, d, e, f], dict=True) if s[c] > 0]
    a254 = an(254).subs(sols[0])
    ok1 = len(sols) == 1 and a254 == 2019
    # (3-2)
    import mpmath as mp
    mp.mp.dps = 30
    def R(tv):
        bmin = mp.sqrt(100 - tv**2)
        def ang(th):
            px, py = tv + 10 * mp.cos(th), bmin * mp.sin(th)
            v1 = (-px, -py); v2 = (2 * tv - px, -py)
            return mp.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (mp.hypot(*v1) * mp.hypot(*v2))) - mp.pi / 3
        roots = []
        N = 400
        for i in range(1, N):
            t0, t1 = mp.pi * i / N, mp.pi * (i + 1) / N
            if ang(t0) * ang(t1) < 0:
                roots.append(mp.findroot(ang, (t0, t1), solver="bisect"))
        pts = [(tv + 10 * mp.cos(th), bmin * mp.sin(th)) for th in roots]
        pts = [q for q in pts if q[0] > 0 and q[1] > 0]
        return max(pts)
    t0 = 2 * mp.sqrt(7); h = mp.mpf("1e-8")
    (x1, y1), (x2, y2) = R(t0 - h), R(t0 + h)
    slope = (y2 - y1) / (x2 - x1)
    target = -16 * mp.sqrt(3) / 73
    Rx, Ry = R(t0)
    # 공식 좌표식과 대조
    fx = t0 + 20 / mp.sqrt(3) * mp.sqrt(t0**2 - 25) / t0; fy = (100 / t0 - t0) / mp.sqrt(3)
    ok2 = abs(slope - target) < 1e-6 and abs(Rx - fx) < 1e-15 and abs(Ry - fy) < 1e-15
    report(3, ok1 and ok2, f"(3-1) c,d,e,f={sols[0]}, a_254={a254}  (3-2) R(2√7)=({float(Rx):.6f},{float(Ry):.6f}), dy/dx≈{float(slope):.10f}, -16√3/73={float(target):.10f}")

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

