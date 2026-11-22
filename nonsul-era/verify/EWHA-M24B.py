#!/usr/bin/env python3
"""EWHA-M24B (이화여대 2024학년도 모의논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import random
import re
import sys

import sympy as sp

PREFIX = "EWHA-M24B"
SET = "B"
WIDTH = 2
RESULTS = []
x, y = sp.symbols("x y", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def tangent_chord(t):
    """y=x^2+1/4 위 (t, t^2+1/4)의 접선과 y=x^2 의 교점 x좌표, 접선식"""
    g = x**2 + sp.Rational(1, 4)
    L = sp.diff(g, x).subs(x, t) * (x - t) + g.subs(x, t)
    r = sp.solve(sp.Eq(L, x**2), x)
    return sorted(r, key=lambda v: float(v.subs(t, 1)) if hasattr(v, "subs") else float(v)), L


def p01():
    t = sp.symbols("t", real=True)
    # (1) P(0,1/4)
    (u1, u2), L0 = tangent_chord(sp.Integer(0))
    area1 = sp.integrate(L0 - x**2, (x, u1, u2))
    ok1 = sp.simplify(area1 - sp.sympify(md_answer(1, "1"))) == 0
    # (2) 증명: 기호적으로 (a2-a1)^2 = 1  [증명-수치확인 겸 기호 확인]
    (r1, r2), L = tangent_chord(t)
    ok2 = sp.simplify(sp.expand((r2 - r1)**2) - 1) == 0
    ok2 = ok2 and all(abs(float((r2 - r1).subs(t, v))) - 1 < 1e-12 for v in [random.uniform(-50, 50) for _ in range(20)])
    # (3) QR=7 → 기울기
    qr2 = (r2 - r1)**2 + (r2**2 - r1**2)**2
    ts = sp.solve(sp.Eq(qr2, 49), t)
    slopes = {sp.nsimplify(sp.simplify(2 * tv)) for tv in ts}
    md3 = {sp.simplify(v) for v in sp.sympify(md_answer(1, "3"))}
    ok3 = {sp.simplify(s) for s in slopes} == md3
    # (4)
    k = sp.symbols("k", positive=True, integer=True)
    (s1, s2), Lk = tangent_chord(k)
    Ak = sp.simplify(sp.integrate(Lk - x**2, (x, s1, s2)))
    if SET == "A":
        S = sp.summation(Ak**2 / 2**k, (k, 1, sp.oo))
    else:
        mk = sp.diff(x**2 + sp.Rational(1, 4), x).subs(x, k)
        S = sp.summation(1 / (mk**2 - 24 * Ak), (k, 2, sp.oo))
    ok4 = sp.simplify(S - sp.sympify(md_answer(1, "4"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4,
           f"(1)={area1} (2)(a2-a1)^2=1 [증명-수치확인] (3)slopes={slopes} (4)A_k={Ak} sum={sp.simplify(S)}")


def p02():
    m, n = sp.symbols("m n")
    a0 = sp.sqrt(3) + sp.sqrt(5)
    val = sp.expand(a0**4 + m * a0**2 + n)
    rat = val.subs(sp.sqrt(15), 0)
    irr = sp.expand((val - rat) / sp.sqrt(15))
    sol = sp.solve([rat, irr], [m, n], dict=True)[0]
    ok1 = sp.sympify(md_answer(2, "1")) == sp.Tuple(sol[m], sol[n])
    f = sp.Poly(x**4 + sol[m] * x**2 + sol[n], x)
    rr = sp.real_roots(f)
    nreal = len(set(rr))
    ok2 = nreal == int(md_answer(2, "2"))
    # (3) [증명-수치확인] 유리근 정리 후보 ±1,±2,±4 모두 근 아님 + 실근 전부 무리수
    cands = [sp.Rational(s * p, q) for s in (1, -1) for p in (1, 2, 4) for q in (1,)]
    ok3 = all(f.eval(c) != 0 for c in cands) and all(sp.degree(sp.minimal_polynomial(r, x), x) > 1 for r in rr)
    # 공식 해설의 짝홀 논증: p^4 = 4q^2(4p^2-q^2) 을 만족하는 서로소 (p,q) 없음(1..200 전수)
    import math
    ok3 = ok3 and not any(p**4 == 4 * q * q * (4 * p * p - q * q)
                          for p, q in itertools.product(range(1, 201), repeat=2) if math.gcd(p, q) == 1)
    # (4) √3+√5 는 f 의 근이고 실수이며 무리수
    ok4 = sp.simplify(f.as_expr().subs(x, a0)) == 0 and a0.is_real and sp.degree(sp.minimal_polynomial(a0, x), x) > 1
    report(2, ok1 and ok2 and ok3 and ok4,
           f"(m,n)=({sol[m]},{sol[n]}) 실근 {nreal}개 (3)(4)[증명-수치확인] 유리근 없음·√3+√5 근")


def circles(axis, allow_internal):
    """중심이 axis 축 위, 단위원과 한 점에서 만나고(외접/내접) y=2x 에 접하는 원 (cx, cy, r) 목록"""
    c = sp.symbols("c", real=True)
    out = []
    cen = (0, c) if axis == "y" else (c, 0)
    dist = sp.Abs(cen[1] - 2 * cen[0]) / sp.sqrt(5)
    d0 = sp.Abs(c)
    cases = [d0 - 1]                      # 외접(중심 밖): 중심거리 = r + 1
    if allow_internal:
        cases += [1 - d0, d0 + 1]         # 내접: 작은 원이 안(r=1-|c|), 큰 원이 단위원을 품음(r=|c|+1)
    for rexpr in cases:
        for cv in sp.solve(sp.Eq(dist, rexpr), c):
            rv = sp.simplify(rexpr.subs(c, cv))
            if rv > 0 and cv != 0:
                cc = tuple(sp.simplify(sp.sympify(v).subs(c, cv)) for v in cen)
                out.append((cc[0], cc[1], rv))
    return out


def parse_circles(s):
    res = set()
    for eq in s.split(";"):
        lhs, rhs = eq.split("=")
        e = sp.expand(sp.sympify(lhs, locals={"x": x, "y": y}) - sp.sympify(rhs))
        cx = -sp.Poly(e, x, y).coeff_monomial(x) / 2
        cy = -sp.Poly(e, x, y).coeff_monomial(y) / 2
        r2 = cx**2 + cy**2 - sp.Poly(e, x, y).coeff_monomial(1)
        res.add((sp.nsimplify(sp.simplify(cx)), sp.nsimplify(sp.simplify(cy)), sp.nsimplify(sp.sqrt(sp.simplify(r2)))))
    return res


def p03():
    internal = SET == "B"
    cy = circles("y", internal)
    got = {(sp.nsimplify(a), sp.nsimplify(b), sp.nsimplify(r)) for a, b, r in cy}
    ok1 = {tuple(sp.simplify(v) for v in t) for t in got} == {tuple(sp.simplify(v) for v in t) for t in parse_circles(md_answer(3, "1"))}
    # (2) [증명-수치확인] 각 원의 중심과 y=-2x 거리 = 반지름
    ok2 = all(sp.simplify(sp.Abs(b + 2 * a) / sp.sqrt(5) - r) == 0 for a, b, r in cy) and len(cy) > 0
    cx = circles("x", internal)
    allc = cy + cx
    if SET == "A":
        pts = [(float(a), float(b)) for a, b, r in allc]
        import math
        pts.sort(key=lambda p: math.atan2(p[1], p[0]))
        area_num = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                           for i in range(len(pts)))) / 2
        dy = max(b for a, b, r in cy) - min(b for a, b, r in cy)
        dx = max(a for a, b, r in cx) - min(a for a, b, r in cx)
        area = sp.simplify(dx * dy / 2)
        ans = sp.sympify(md_answer(3, "3"))
        ok3 = sp.simplify(area - ans) == 0 and abs(area_num - float(ans)) < 1e-9 and len(allc) == 4
        msg3 = f"마름모 넓이={area}"
    else:
        rs = [r for a, b, r in allc]
        val = sp.simplify(max(rs, key=float) + min(rs, key=float))
        ok3 = sp.simplify(val - sp.sympify(md_answer(3, "3"))) == 0
        msg3 = f"반지름 {sorted(rs, key=float)} 최대+최소={val}"
    report(3, ok1 and ok2 and ok3, f"(1)원 {len(cy)}개 (2)y=-2x 접함[증명-수치확인] (3){msg3}")


if __name__ == "__main__":
    random.seed(0)
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
