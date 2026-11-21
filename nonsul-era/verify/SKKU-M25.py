#!/usr/bin/env python3
"""SKKU-M25 (성균관대 2025학년도 모의논술 수리 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "SKKU-M25"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def good(S):
    """<제시문 3> 조건: 두 원소 합이 모두 다르고 정렬하면 등차수열."""
    sums = sorted(a + b for a, b in itertools.combinations(S, 2))
    if len(set(sums)) != len(sums):
        return None
    d = sums[1] - sums[0]
    return sums if all(sums[i + 1] - sums[i] == d for i in range(len(sums) - 1)) else None


def p01():
    # (i) n=3, 합 600: 원소 상한 = 합/2
    c3 = 0
    for S in itertools.combinations(range(1, 200), 3):
        sm = good(S)
        if sm and sum(sm) == 600:
            c3 += 1
    ok1 = c3 == int(md_answer(1, "1"))
    # (ii) n=4, 합 2025 = 3*(원소 합) → 원소 합 675; a1<a2<a3 이 결정되면 a4 결정
    c4 = 0
    for a1 in range(1, 675):
        for a2 in range(a1 + 1, 675):
            if a1 + 2 * a2 + 3 > 675:
                break
            for a3 in range(a2 + 1, 675):
                a4 = 675 - a1 - a2 - a3
                if a4 <= a3:
                    break
                if good((a1, a2, a3, a4)):
                    c4 += 1
    ok2 = c4 == int(md_answer(1, "2"))
    # (iii) n=5,6 에서 원소 <= 40 전수 탐색으로 존재하지 않음 확인(증명 대신 수치 근거)
    ex5 = any(good(S) for S in itertools.combinations(range(1, 41), 5))
    ex6 = any(good(S) for S in itertools.combinations(range(1, 25), 6))
    ok3 = (not ex5) and (not ex6) and md_answer(1, "3") == "none"
    report(1, ok1 and ok2 and ok3, f"n=3:{c3} n=4:{c4} n=5존재:{ex5} n=6존재:{ex6}")


def p02():
    k = sp.symbols("k", real=True)
    par = (1 - k**2) / 2 * x**2
    line = k * x + 1
    eq = sp.expand(2 * (par - line))           # (1-k^2)x^2-2kx-2
    # (i) 중점 x좌표 = 2 (직선이 y축에 평행)
    xs_sum = -sp.Poly(eq, x).all_coeffs()[1] / sp.Poly(eq, x).all_coeffs()[0]
    disc = sp.discriminant(eq, x)
    cand = sp.solve(sp.Eq(xs_sum / 2, 2), k)
    sols = [c for c in cand if disc.subs(k, c) > 0 and c not in (1, -1)]
    md1 = sp.sympify(md_answer(2, "1"))
    ok1 = set(sp.nsimplify(s) for s in sols) == set(md1)
    # (ii) 0<x1<x2 인 k 범위를 격자로 찾고, c 의 치역에서 빠진 구간 길이 계산
    import numpy as np
    cs = []
    for kv in np.linspace(-3, 3, 600001):
        if abs(abs(kv) - 1) < 1e-9:
            continue
        r = np.roots([1 - kv**2, -2 * kv, -2])
        if np.iscomplexobj(r) and np.max(abs(r.imag)) > 1e-12:
            continue
        r = np.sort(r.real)
        if r[0] > 0 and r[1] - r[0] > 1e-9:
            mx, my = r.mean(), kv * r.mean() + 1
            if abs(mx - 2) > 1e-12:
                cs.append(my * (0 - 2) / (mx - 2))          # (2,0)에서 (mx,my) 방향으로 x=0
    cs = np.array(cs)
    hole_lo, hole_hi = cs[cs < 0].max(), cs[cs > 0].min()
    ok2 = abs((hole_hi - hole_lo) - float(sp.sympify(md_answer(2, "2")))) < 1e-3
    # (iii) 중근이고 x1<0
    ks = [s for s in sp.solve(disc, k) if (xs_sum / 2).subs(k, s) < 0]
    kk = ks[0]
    P = sp.Rational(1, 2) * xs_sum.subs(k, kk)
    pa = par.subs(k, kk); P1 = (sp.nsimplify(P), pa.subs(x, P))
    perp = -1 / kk * (x - P1[0]) + P1[1]
    Tx = [r for r in sp.solve(sp.Eq(perp, pa), x) if sp.simplify(r - P1[0]) != 0][0]
    T = (Tx, pa.subs(x, Tx))
    tang = sp.diff(pa, x).subs(x, Tx) * (x - Tx) + T[1]
    Bx = sp.solve(sp.Eq(tang, line.subs(k, kk)), x)[0]
    B = (Bx, line.subs({k: kk, x: Bx}))
    tri = sp.Abs((T[0] - P1[0]) * (B[1] - P1[1]) - (B[0] - P1[0]) * (T[1] - P1[1])) / 2
    lower = sp.integrate(pa - perp, (x, P1[0], Tx))           # 현 P1T 와 포물선 사이
    upper = sp.simplify(tri - lower)
    a_, b_ = md_answer(2, "3").split(":")
    ok3 = sp.simplify(lower / upper - sp.Rational(a_) / sp.Rational(b_)) == 0
    report(2, ok1 and ok2 and ok3, f"k={sols} hole=({hole_lo:.6f},{hole_hi:.6f}) T={T} B={B} tri={sp.simplify(tri)} lower={sp.simplify(lower)}")


def count_inter(a, seg):
    """원 C_a 와 선분 seg=(P,Q) 교점 개수(정확 계산)."""
    t = sp.symbols("t", real=True)
    (px, py), (qx, qy) = seg
    X = px + (qx - px) * t; Y = py + (qy - py) * t
    eq = sp.expand((X - a)**2 + (Y - a**2 / sp.sqrt(3))**2 - sp.Rational(4, 3))
    rts = sp.Poly(eq, t).nroots(n=50, maxsteps=200)
    rr = set()
    for r in rts:                      # 중근(접할 때) 수치오차 ~1e-25 를 허용
        if abs(sp.im(r)) < 1e-12 and -1e-12 <= sp.re(r) <= 1 + 1e-12:
            rr.add(round(float(sp.re(r)), 9))
    return len(rr)


def p03():
    A = (0, 0); Bp = (2, 0); C = (1, sp.sqrt(3))
    m = [r for r in sp.Poly(sp.sympify(md_answer(3, "2"), locals={"x": x}), x).nroots(n=60) if r.is_real and r > 1][0]
    # (ii) f1(a)=2 인 a>1 중 최솟값을 수치로 찾고 md 다항식의 근과 비교
    import mpmath as mp
    lo, hi = mp.mpf(1) + mp.mpf("1e-6"), mp.sqrt(2) - mp.mpf("1e-6")
    mp.mp.dps = 50
    f1 = lambda a: count_inter(sp.Float(str(a), 50), (A, Bp))
    assert f1(lo) == 1 and f1(hi) == 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if f1(mid) == 2:
            hi = mid
        else:
            lo = mid
    ok2 = abs(hi - m) < 1e-9 and sp.Poly(sp.sympify(md_answer(3, "2"), locals={"x": x}), x).LC() == 1
    # (i),(iii),(iv) 격자 a ∈ [1,3] 에서 집합 비교
    grid = [sp.Rational(100 + i, 100) for i in range(0, 201)] + [sp.sqrt(2), sp.Integer(2), sp.Float(m, 60)]
    def ins(expr_set, a):
        return bool(expr_set.contains(a))
    s1 = sp.sympify(md_answer(3, "1"))
    s3 = sp.sympify(md_answer(3, "3"))
    s4 = sp.sympify(md_answer(3, "4"), locals={"m": sp.Float(m, 60)})
    bad = []
    for a in grid:
        af = sp.N(a, 60)
        n1 = count_inter(af, (A, Bp)); n2 = count_inter(af, (A, C)); n3 = count_inter(af, (Bp, C))
        if (n1 >= 1) != ins(s1, a):
            bad.append(("f1", a))
        if (n2 >= 1) != ins(s3, a):
            bad.append(("f2", a))
        if (n3 >= 1) != ins(s4, a):
            bad.append(("f3", a))
    report(3, ok2 and not bad, f"m={sp.N(m, 15)} bisect={mp.nstr(hi, 15)} 불일치={bad[:5]}")


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
