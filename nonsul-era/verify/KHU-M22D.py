#!/usr/bin/env python3
"""KHU-M22D (경희대 2022학년도 모의논술 의학계 수학 논제 I-1~I-3) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import importlib.util
import math
import pathlib
import re
import sys

import sympy as sp

PREFIX = "KHU-M22D"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 자연계 논제 1-(2)와 같은 문항: 같은 독립 재계산 함수를 불러 쓴다
    spec = importlib.util.spec_from_file_location("khu_m22", pathlib.Path(__file__).with_name("KHU-M22.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok, msg = mod.exp_cubic_check()
    report(1, ok, msg)


def max_circle(theta):
    """AB⊥BC, AD∥BC, AD:BC=1:2, 둘레 1, ∠C=θ. AB·BC 에 접하고 내부에 있는 원의 최대 반지름."""
    x = 1 / (1 + 3 * math.cos(theta) + math.sin(theta))
    AB, BC = x * math.sin(theta), 2 * x * math.cos(theta)
    C, D = (BC, 0.0), (BC / 2, AB)
    # 중심 (r,r): r ≤ AB/2, 직선 CD 까지 거리 ≥ r (같은 쪽)
    nx, ny = D[1] - C[1], -(D[0] - C[0])
    nn = math.hypot(nx, ny)
    dist = lambda r: abs(nx * (r - C[0]) + ny * (r - C[1])) / nn
    lo, hi = 0.0, AB / 2
    if dist(hi) >= hi:
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if dist(mid) >= mid:
            lo = mid
        else:
            hi = mid
    return lo


def p02():
    N = 200000
    best, bt = 0, 0
    for i in range(1, N):
        th = (math.pi / 2) * i / N
        r = max_circle(th)
        if r > best:
            best, bt = r, th
    ok = abs(best - 1 / 9) < 1e-8 and abs(math.cos(bt) - 0.6) < 1e-4
    # 기호: cosθ=3/5 에서 r1=r2
    t = sp.symbols("t", positive=True)
    r1 = sp.sin(t) / (2 * (3 * sp.cos(t) + sp.sin(t) + 1))
    r2 = (sp.cos(t) + sp.sin(t) - 1) / (3 * sp.cos(t) + sp.sin(t) + 1)
    al = sp.acos(sp.Rational(3, 5))
    ok &= sp.simplify((r1 - r2).subs(t, al)) == 0 and sp.nsimplify(r1.subs(t, al)) == sp.Rational(1, 9)
    # 해설 B 의 도함수 식(예시답안의 계수 2 오기 수정본) 확인
    ok &= sp.simplify(sp.diff(r1, t) - (3 + sp.cos(t)) / (2 * (3 * sp.cos(t) + sp.sin(t) + 1)**2)) == 0
    ok &= sp.simplify(sp.diff(r2, t) - 2 * (sp.cos(t) - 2 * sp.sin(t) + 1) / (3 * sp.cos(t) + sp.sin(t) + 1)**2) == 0
    report(2, ok, f"격자 최대 r={best:.10f}, cosθ={math.cos(bt):.5f}; cosθ=3/5 에서 r1=r2=1/9")


def p03():
    p = sp.Rational(2, 7)
    q = 1 - p
    ok = True
    for m in [1, 2, 5, 100, 2021]:
        lhs = sum(sp.binomial(k, m) * p**m * q**(k - m) * q**(-k) for k in range(m, 2023)) if m <= 5 else None
        rhs = p**m * q**(-m) * sp.binomial(2023, m + 1)
        if lhs is not None:
            ok &= sp.simplify(lhs - rhs) == 0
        else:   # 하키스틱 항등식 직접 확인
            ok &= sum(sp.binomial(k, m) for k in range(m, 2023)) == sp.binomial(2023, m + 1)
    a_sum = 1 + (-1) + 2023   # a1=m, a2=-m (p,q 지수), a3=2023 — m 에 무관한 합
    # (2) A64≥B64 ⇔ X≥16 확인 (u^16 d^48 = r^64)
    u, d = 3.0, 0.5
    rr = (u * d**3) ** 0.25
    ok &= all((u**X * d**(64 - X) >= rr**64 * (1 - 1e-12)) == (X >= 16) for X in range(65))
    pp = sp.symbols("p", positive=True)
    sols = [s for s in sp.solve(sp.Eq((16 - 64 * pp)**2, 64 * pp * (1 - pp)), pp) if 16 - 64 * s > 0 and sp.Rational(1, 10) < s < sp.Rational(9, 10)]
    ok &= sols == [sp.Rational(1, 5)]
    report(3, ok, f"(1) Σ=p^m q^-m ·2023C(m+1) 확인(m=1,2,5 직접합) ⇒ a1+a2+a3=2023; (2) X≥16, z=1 ⇒ p={sols}, q={[1 - s for s in sols]}")


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
