#!/usr/bin/env python3
"""HYU-M25 (한양대 2025학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "HYU-M25"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) 구분 없는 주사위 3개 a>=b>=c 전수 조사
    cnt = 0
    for a, b, c in itertools.product(range(1, 7), repeat=3):
        if a >= b >= c:
            v = Fraction(a, 3) * (a - c) ** 3 + Fraction(b, 2) * (a - c) ** 2 + (a - c)
            cnt += v.denominator == 1 and v > 0
    ok1 = cnt == int(sp.sympify(md_answer(1, "1")))
    # (2) 원 (x-p)^2+(y-1)^2=1, 직선 y=-x/3+2 → 삼각형 PQR 넓이 최대
    p, X = sp.symbols("p X", real=True)
    pts = sp.solve((X - p) ** 2 + (-X / 3 + 2 - 1) ** 2 - 1, X)
    x1, x2 = pts
    l = sp.sqrt((x1 - x2) ** 2 * (1 + sp.Rational(1, 9)))
    d = sp.Abs(p + 3 * 1 - 6) / sp.sqrt(10)
    S2 = sp.simplify((l * d / 2) ** 2)
    crit = [r for r in sp.solve(sp.diff(S2, p), p) if r.is_real and 0 < r < 3]
    best = max(crit, key=lambda r: S2.subs(p, r))
    Smax = sp.sqrt(sp.simplify(S2.subs(p, best)))
    md = sp.sympify(md_answer(1, "2"))
    # 경계(0, 3) 쪽 값보다 큰지도 확인
    edge = max(S2.subs(p, sp.Rational(1, 10**6)), S2.subs(p, 3 - sp.Rational(1, 10**6)))
    ok2 = sp.simplify(best - md[0]) == 0 and sp.simplify(Smax - md[1]) == 0 and edge < S2.subs(p, best)
    # (3) 한 변 1인 정육각형 20개 삼각형 넓이(A1) → 한 변 n^(1/4) 이면 A1*sqrt(n)
    V = [(sp.cos(sp.pi * k / 3), sp.sin(sp.pi * k / 3)) for k in range(6)]
    areas = []
    for i, j, k in itertools.combinations(range(6), 3):
        (ax, ay), (bx, by), (cx, cy) = V[i], V[j], V[k]
        areas.append(sp.nsimplify(sp.Abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2))
    tot = 0
    for A1 in set(areas):
        m = areas.count(A1)
        tot += m * sum(1 for nn in range(1, 2026) if (A1 * sp.sqrt(nn)).is_integer)
    ok3 = tot == int(sp.sympify(md_answer(1, "3")))
    report(1, ok1 and ok2 and ok3, f"(1) {cnt}  (2) p={best}, S={Smax}  (3) sum={tot}")


def p02():
    # (1) E=(0,0,0),F=(1,0,0),G=(1,1,0),H=(0,1,0),A=(0,0,1),D=(0,1,1),C=(1,1,1)
    t = sp.symbols("t", real=True)
    P = sp.Matrix([0, 1 - t, 1]); Q = sp.Matrix([t, 0, 0]); R = sp.Matrix([1, 1, t])
    nrm = (Q - P).cross(R - P)
    cos2 = sp.simplify(nrm[2] ** 2 / nrm.dot(nrm))
    ok1 = sp.simplify(cos2 - sp.Rational(1, 3)) == 0          # [증명-수치확인] cos θ = 1/√3 상수
    # (2) 2×7 격자 3색 적정 색칠: 열 단위 전이(DP)
    cols = [(i, j) for i in range(3) for j in range(3) if i != j]
    dp = {c: 1 for c in cols}
    for _ in range(6):
        dp = {c: sum(v for e, v in dp.items() if e[0] != c[0] and e[1] != c[1]) for c in cols}
    n2 = sum(dp.values())
    ok2 = n2 == int(sp.sympify(md_answer(2, "2")))
    # (3) k 격자에서 g(t)의 불연속점 수(= f 의 서로 다른 극값의 개수) == 1 인지 vs md 집합
    x = sp.symbols("x", real=True)
    md_set = sp.sympify(md_answer(2, "3"))

    def n_disc(k):
        f = x ** 4 / 4 - x ** 3 / 3 + k * x
        fp = sp.Poly(x ** 3 - x ** 2 + k, x)
        vals = set()
        for r in sp.real_roots(fp):
            r = sp.nsimplify(r) if r.is_rational else r
            e = sp.Rational(1, 10**8)
            s1 = sp.sign(sp.N((x ** 3 - x ** 2 + k).subs(x, r - e), 30))
            s2 = sp.sign(sp.N((x ** 3 - x ** 2 + k).subs(x, r + e), 30))
            if s1 != s2:
                vals.add(sp.N(f.subs(x, r), 25))
        return len({round(float(v), 12) for v in vals})
    ks = [sp.Rational(i, 270) for i in range(-60, 121)] + [sp.Rational(4, 27), 0]
    ok3 = all((n_disc(k) == 1) == bool(md_set.contains(k)) for k in ks)
    report(2, ok1 and ok2 and ok3, f"(1) cos^2θ={cos2} [증명-수치확인]  (2) {n2}  (3) {len(ks)}개 k 판정 일치")


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
