#!/usr/bin/env python3
"""SJU-M27F (세종대 2027학년도 모의논술 자유전공학부-수학문항) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "SJU-M27F"
WIDTH = 2
RESULTS = []
n = sp.symbols("n", positive=True, integer=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def vertices(N):
    """n=N 일 때 모든 정사각형의 꼭짓점 (A=(0,0), B=(N,0))."""
    pts = set()
    for row in range(N):                       # row 0 = 첫 줄(N개)
        cnt = N - row
        left = sp.Rational(row, 2)
        for j in range(cnt):
            for dx, dy in itertools.product((0, 1), (0, 1)):
                pts.add((left + j + dx, row + dy))
    return [(float(a), float(b)) for a, b in pts]


def min_enclosing_radius(P):
    """2점·3점으로 정해지는 원 중 모두를 품는 최소 원(전수)."""
    best = math.inf
    def ok(cx, cy, r):
        return all((x - cx)**2 + (y - cy)**2 <= r * r + 1e-9 for x, y in P)
    for (x1, y1), (x2, y2) in itertools.combinations(P, 2):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = math.hypot(x1 - cx, y1 - cy)
        if r < best and ok(cx, cy, r):
            best = r
    for (x1, y1), (x2, y2), (x3, y3) in itertools.combinations(P, 3):
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-12:
            continue
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
        r = math.hypot(x1 - ux, y1 - uy)
        if r < best and ok(ux, uy, r):
            best = r
    return best


def p02():
    loc = {"n": n}
    # (2-1) 좌표로 sin(CAB)
    C = (sp.Rational(1, 2) * (n - 1), n)
    sin_a = C[1] / sp.sqrt(C[0]**2 + C[1]**2)
    md1 = sp.sympify(md_answer(2, "1"), locals=loc)
    ok1 = all(sp.simplify((sin_a - md1).subs(n, k)) == 0 for k in range(1, 30))
    # (2-2) 꼭짓점 전체의 최소 포함원(전수) vs md r(n)
    md2 = sp.sympify(md_answer(2, "2"), locals=loc)
    rs = {N: min_enclosing_radius(vertices(N)) for N in range(1, 6)}
    ok2 = all(abs(rs[N] - float(md2.subs(n, N))) < 1e-9 for N in rs)
    # (2-3) f(n)=64 n^2 r^2 (위에서 수치로 확인된 r 공식의 근원: 외접원 반지름을 직접 기호 계산)
    yc = sp.symbols("yc")
    ycv = sp.solve(sp.Eq((n / 2)**2 + yc**2, (n / 2 - C[0])**2 + (C[1] - yc)**2), yc)[0]
    r2 = sp.simplify((n / 2)**2 + ycv**2)                     # 중심 (n/2, yc)
    fpoly = sp.expand(64 * n**2 * r2)
    from fractions import Fraction
    coeffs = sp.Poly(fpoly, n).all_coeffs()
    fval = lambda k: sum(int(c) * k**i for i, c in enumerate(reversed(coeffs)))
    goods = []
    for d in range(1, 200):
        ks = range(1, 3 * d + 20)
        vals = [fval(k) - Fraction(206 + 50 * d, 3) * k**3 + Fraction(3 * d, 2) * k**2 for k in ks]
        mn = min(vals)
        if 2 * d - 6 in [k for k, v in zip(ks, vals) if v == mn]:
            goods.append(d)
    ok3 = goods == [int(md_answer(2, "3"))]
    report(2, ok1 and ok2 and ok3, f"r(1..5)={[round(rs[N], 6) for N in rs]} f(n)={fpoly} d={goods}")


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
