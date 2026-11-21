#!/usr/bin/env python3
"""KKU-G22A (건국대 2022학년도 수시 논술 자연계 A 수학 문제 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KKU-G22A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
LOC = {"e": sp.E, "E": sp.E}


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")




def p01():
    """1-1: O1=(0,0), O2=(3r,0), 공통 법선 n=(cosφ,sinφ). 두 접선 사이 거리 1 → (O2-O1)·n=2r+1.
    A1=O1+rn, A2=O2-rn 로 비를 만들어 r→∞ 극한. 1-2: 계수 61개를 직접 계산해 정렬."""
    r = sp.symbols("r", positive=True)
    c = (2 * r + 1) / (3 * r)
    s = sp.sqrt(1 - c**2)
    A1 = sp.Matrix([r * c, r * s]); A2 = sp.Matrix([3 * r - r * c, -r * s])
    # 접선 l1: A1 을 지나고 n 에 수직, l2: A2 를 지나고 n 에 수직 → 두 직선 거리 = |(A2-A1)·n|
    dist = sp.simplify((A2 - A1).dot(sp.Matrix([c, s])))
    ratio = sp.sqrt(sp.expand((A2 - A1).dot(A2 - A1))) / (3 * r)
    L = sp.limit(ratio, r, sp.oo)
    ok1 = sp.simplify(dist - 1) == 0 and sp.simplify(L - sp.sympify(md_answer(1, "1"))) == 0
    a = [sp.binomial(60, k) * 5**(60 - k) * 2**k for k in range(61)]
    order = sorted(range(61), key=lambda k: -a[k])
    ok2 = tuple(order[:2]) == tuple(sp.sympify(md_answer(1, "2"))) and a[order[0]] > a[order[1]] > a[order[2]]
    report(1, ok1 and ok2, f"거리={dist} lim={L}; (p,q)={tuple(order[:2])}, a16/a18={sp.N(a[16]/a[18], 8)}")


def p02():
    """2-1: θ(t)=∠AOB 를 벡터 내적 acos 로 정의해 t=2 에서 수치미분(30자리).
    2-2: C=0, D=d, E=e 를 격자로 두고 F 를 ∠CPD=π/3 이 되도록 이분법으로 찾아 CD+EF 최대를 탐색."""
    import mpmath as mp
    mp.mp.dps = 30

    def theta(tv):
        xs = sorted(sp.Poly(x**2 - tv * x - tv, x).nroots(n=40))
        A = (mp.mpf(str(xs[0])), mp.mpf(str(xs[0]))**2); B = (mp.mpf(str(xs[1])), mp.mpf(str(xs[1]))**2)
        return mp.acos((A[0]*B[0] + A[1]*B[1]) / (mp.sqrt(A[0]**2 + A[1]**2) * mp.sqrt(B[0]**2 + B[1]**2)))

    def theta_f(tv):
        tv = mp.mpf(tv); d = mp.sqrt(tv**2 + 4*tv)
        x1, x2 = (tv - d) / 2, (tv + d) / 2
        A = (x1, x1**2); B = (x2, x2**2)
        return mp.acos((A[0]*B[0] + A[1]*B[1]) / (mp.sqrt(A[0]**2 + A[1]**2) * mp.sqrt(B[0]**2 + B[1]**2)))

    dth = mp.diff(theta_f, 2)
    ans1 = sp.N(sp.sympify(md_answer(2, "1")), 30)
    ok1 = abs(dth - mp.mpf(str(ans1))) < mp.mpf("1e-20") and abs(theta(2) - theta_f(2)) < 1e-15

    R = 2.0
    import math

    def pt(a):
        return (R * math.cos(a), R * math.sin(a))

    def inter(P1, P2, P3, P4):
        d1 = (P2[0]-P1[0], P2[1]-P1[1]); d2 = (P4[0]-P3[0], P4[1]-P3[1])
        den = d1[0]*d2[1] - d1[1]*d2[0]
        u = ((P3[0]-P1[0])*d2[1] - (P3[1]-P1[1])*d2[0]) / den
        return (P1[0] + u*d1[0], P1[1] + u*d1[1])

    def ang(Pc, X, Y):
        v = (X[0]-Pc[0], X[1]-Pc[1]); w = (Y[0]-Pc[0], Y[1]-Pc[1])
        return math.acos(max(-1, min(1, (v[0]*w[0]+v[1]*w[1]) / (math.hypot(*v)*math.hypot(*w)))))

    best = 0
    N = 120
    for i in range(1, N):
        dd = 2*math.pi*i/N
        for j in range(i+1, N):
            ee = 2*math.pi*j/N
            C, D, E = pt(0), pt(dd), pt(ee)
            def g(ff):
                F = pt(ff); P = inter(C, E, D, F)
                return ang(P, C, D) - math.pi/3
            lo, hi = ee + 1e-9, 2*math.pi - 1e-9
            if g(lo) * g(hi) > 0:
                continue
            for _ in range(60):
                mid = (lo+hi)/2
                if g(lo)*g(mid) <= 0: hi = mid
                else: lo = mid
            F = pt(lo)
            val = math.dist(C, D) + math.dist(E, F)
            best = max(best, val)
    ok2 = abs(best - float(sp.sympify(md_answer(2, "2")))) < 1e-3 and best <= float(sp.sympify(md_answer(2, "2"))) + 1e-9
    report(2, ok1 and ok2, f"θ'(2)={mp.nstr(dth, 15)}; 격자 탐색 max(CD+EF)={best:.6f}")


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
