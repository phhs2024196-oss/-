#!/usr/bin/env python3
"""HYU-M26 (한양대 2026학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "HYU-M26"
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
    x = sp.symbols("x", positive=True)
    # (1) r = 넓이/반둘레, R = 빗변/2
    c = sp.sqrt(x**2 + 1 / x**2)
    r = (sp.Rational(1, 2)) / ((x + 1 / x + c) / 2)
    fg = sp.pi * r**2 * sp.pi * (c / 2) ** 2
    L1 = sp.limit(fg, x, sp.oo)
    ok1 = sp.simplify(L1 - sp.sympify(md_answer(1, "1"))) == 0
    # (2) 접선 조건(판별식)으로 절편을 직접 구하고 S^2 의 최댓값을 (a,b) 표본에서 비교
    u, k, X = sp.symbols("u k X", positive=True)
    a, b = sp.symbols("a b", positive=True)
    md2 = sp.sympify(md_answer(1, "2"), locals={"a": a, "b": b})

    def tangent_c(slope, A, B):
        cc = sp.symbols("cc", positive=True)
        disc = sp.discriminant(sp.expand((X**2 / A**2 + (slope * X + cc) ** 2 / B**2 - 1) * A**2 * B**2), X)
        return sp.solve(disc, cc)[0]
    ok2 = True
    for A, B in [(3, 1), (2, 5), (sp.Rational(3, 2), 1), (1, 1)]:
        c1 = tangent_c(u, A, B)                 # 기울기 u=√m
        c3 = tangent_c(-1 / u, A, B)
        w1 = 2 * c1 / sp.sqrt(1 + u**2)          # 평행선 사이 거리
        w3 = 2 * c3 / sp.sqrt(1 + 1 / u**2)
        S2 = sp.simplify((w1 * w3) ** 2)
        crit = [s for s in sp.solve(sp.diff(S2, u), u) if s.is_real and 0 < s <= sp.sqrt(2026)]
        cands = [S2.subs(u, s) for s in crit] + [sp.limit(S2, u, 0, "+"), S2.subs(u, sp.sqrt(2026))]
        mx = max(cands, key=lambda v: float(v))
        ok2 = ok2 and sp.simplify(mx - md2.subs({a: A, b: B})) == 0
    # (3) |PA+PB| = 4|PG| 자취(원) 위 점 P 에 대해 삼각형 APC 넓이 최대
    px, py, th = sp.symbols("px py th", real=True)
    Ap, Bp, Cp = sp.Matrix([6, 2]), sp.Matrix([-2, -4]), sp.Matrix([5, -1])
    G = (Ap + Bp + Cp) / 3
    P = sp.Matrix([px, py])
    eq = sp.expand(((Ap - P) + (Bp - P)).dot((Ap - P) + (Bp - P)) - 16 * (G - P).dot(G - P))
    cx = sp.solve(sp.diff(eq, px), px)[0]; cy = sp.solve(sp.diff(eq, py), py)[0]
    rad = sp.sqrt(-eq.subs({px: cx, py: cy}) / eq.coeff(px, 2))
    Pt = sp.Matrix([cx + rad * sp.cos(th), cy + rad * sp.sin(th)])
    area = ((Pt - Ap)[0] * (Cp - Ap)[1] - (Pt - Ap)[1] * (Cp - Ap)[0]) / 2
    # 넓이 = |상수 + α cos th + β sin th| → 최대 = |상수| + sqrt(α²+β²)
    const = sp.simplify((area.subs(th, 0) + area.subs(th, sp.pi)) / 2)
    al = sp.simplify((area.subs(th, 0) - area.subs(th, sp.pi)) / 2)
    be = sp.simplify((area.subs(th, sp.pi / 2) - area.subs(th, 3 * sp.pi / 2)) / 2)
    mx3 = sp.Abs(const) + sp.sqrt(al**2 + be**2)
    import math
    num = max(abs(float(area.subs(th, 2 * math.pi * i / 20000))) for i in range(20000))
    ok3 = sp.simplify(mx3 - sp.sympify(md_answer(1, "3"))) == 0 and abs(num - float(mx3)) < 1e-6
    report(1, ok1 and ok2 and ok3, f"(1) {L1}  (2) 표본 4쌍 최댓값 일치  (3) 중심=({cx},{cy}) r={rad} max={sp.nsimplify(mx3)}")


def p02():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.symbols("a", positive=True)
    md1 = sp.sympify(md_answer(2, "1"), locals={"a": a})
    ok1 = True
    for av in [sp.Rational(1, 3), sp.Rational(1, 2), 1, 2, sp.Rational(7, 2)]:
        L = sp.limit(sp.log(av ** (2 * n) + av ** (5 * n)) / n, n, sp.oo)
        ok1 = ok1 and sp.simplify(L - md1.subs(a, av)) == 0
    # (2) 전수: k>=3 자연수 중 n^k <= k^n (k 는 3n 까지 조사)
    seq = []
    for nn in range(3, 151):
        seq.append(sum(1 for kk in range(3, 3 * nn + 1) if nn**kk <= kk**nn))
    lin = all(seq[i] == (i + 3) - 2 for i in range(len(seq)))      # a_n 이 n-2 와 같은지 전수 확인
    L2 = sp.limit((n - 2) / n, n, sp.oo) if lin else None
    ok2 = lin and sp.simplify(L2 - sp.sympify(md_answer(2, "2"))) == 0
    # (3) p=1 고정, b^2=a^2-1. 교점·접선 교점 A·각 60° 를 수치로 풀어 a/p 비교
    import mpmath as mp
    mp.mp.dps = 40

    def angle_minus(A):
        B2 = A**2 - 1
        x1 = mp.findroot(lambda xx: xx**2 / A**2 + 4 * xx / B2 - 1, 0.5)
        y1 = mp.sqrt(4 * x1)
        xa = -x1                                  # 두 접선 y1 y = 2(x+x1), -y1 y = 2(x+x1) 의 교점
        v1 = (x1 - xa, y1); v2 = (x1 - xa, -y1)
        c = (v1[0] * v2[0] + v1[1] * v2[1]) / (mp.sqrt(v1[0]**2 + v1[1]**2) * mp.sqrt(v2[0]**2 + v2[1]**2))
        return c - mp.mpf(1) / 2
    root = mp.findroot(angle_minus, 4.5)
    ok3 = abs(root - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "3")), 40)))) < 1e-20
    report(2, ok1 and ok2 and ok3, f"(1) 5개 a 일치  (2) a_n=n-2 (n<=150 전수), 극한 {L2}  (3) a/p={mp.nstr(root, 15)}")


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
