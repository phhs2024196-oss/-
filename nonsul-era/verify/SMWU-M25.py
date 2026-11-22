#!/usr/bin/env python3
"""SMWU-M25 (숙명여대 2025학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SMWU-M25"
WIDTH = 2
RESULTS = []
x, t, n, k, a, c, m = sp.symbols("x t n k a c m", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {kk.strip(): sp.sympify(v) for kk, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _count_ffx_quadratic(av):
    f = lambda z: av * (z + 2) * (z - 3)
    poly = sp.Poly(sp.expand(f(f(x))), x)
    return len(set(sp.real_roots(poly)))


def q_composite_quadratic():
    """f(x)=a(x+2)(x-3), f(f(x))=0 의 서로 다른 실근 3개 → a. 근 개수가 바뀌는 a(판별식 0)를 후보로 실근 개수 직접 셈."""
    ap = sp.symbols("ap", positive=True)
    cands = set()
    for r in (3, -2):                      # f 의 근
        disc = sp.discriminant(sp.expand(ap * (x + 2) * (x - 3) - r), x)
        cands |= {s for s in sp.solve(disc, ap) if s.is_positive}
    tests = sorted(cands | {sp.Rational(1, 10), sp.Rational(1, 3), 1, 2})
    good = [av for av in tests if _count_ffx_quadratic(av) == 3]
    return good


def q_composite_sine_count(av):
    """f(x)=a sin x (0<=x<=pi), 카드 해석대로 f(f(x))=0 <=> a sin x = j*pi (j>=0) 의 [0,pi] 서로 다른 해 개수."""
    import math
    sols = set()
    j = 0
    while j * math.pi <= av + 1e-12:
        r = j * math.pi / av
        if r <= 1 + 1e-12:
            r = min(r, 1.0)
            s1 = math.asin(r)
            sols.add(round(s1, 9)); sols.add(round(math.pi - s1, 9))
        j += 1
    return len(sols)


def q_common_tangent_line(f1, f2):
    """두 이차함수에 동시에 접하는 직선 y=m x+n 들(판별식=0 연립)."""
    mm, nn = sp.symbols("mm nn")
    e1 = sp.discriminant(sp.expand(f1 - (mm * x + nn)), x)
    e2 = sp.discriminant(sp.expand(f2 - (mm * x + nn)), x)
    return sp.solve([e1, e2], [mm, nn], dict=True), mm, nn


def q_perp_tangent_a():
    """y=-x^2-2, y=x^2-2ax+a^2+a 의 공통접선 두 개가 수직 → a."""
    av = sp.symbols("av", real=True)
    mm, nn = sp.symbols("mm nn")
    e1 = sp.discriminant(sp.expand(-x**2 - 2 - (mm * x + nn)), x)
    e2 = sp.discriminant(sp.expand(x**2 - 2 * av * x + av**2 + av - (mm * x + nn)), x)
    nsol = sp.solve(e1, nn)[0]
    qm = sp.Poly(sp.expand(e2.subs(nn, nsol)), mm)            # m 에 대한 이차식
    A2, B2, C2 = qm.all_coeffs()
    sols = sp.solve(sp.Eq(C2 / A2, -1), av)                    # 두 기울기 곱 = -1
    ok_two = all(sp.simplify(B2**2 - 4 * A2 * C2).subs(av, s) > 0 for s in sols)
    return sols, ok_two


def q_king_c_values():
    """f(x)+f(c-x)=c, ∫_0^c f = 2 → c. 조건을 만족하는 함수 두 개로 ∫_0^c f 를 직접 계산."""
    cs = sp.symbols("cs", real=True)
    fams = [cs / 2 + (x - cs / 2)**3, cs / 2 + sp.sin(x - cs / 2)]
    vals = []
    for fx in fams:
        assert sp.simplify(fx + fx.subs(x, cs - x) - cs) == 0
        I = sp.simplify(sp.integrate(fx, (x, 0, cs)))
        vals.append(set(sp.solve(sp.Eq(I, 2), cs)))
    return vals


def p01():
    good = q_composite_quadratic()
    ok1 = len(good) == 1 and sp.simplify(good[0] - sp.sympify(md_answer(1, "1"))) == 0
    I = sp.sympify(md_answer(1, "2"), locals={"pi": sp.pi})
    lo, hi = float(I.start), float(I.end)
    import numpy as np
    grid = np.linspace(0.05, 4 * np.pi, 4000)
    inside = [g for g in grid if q_composite_sine_count(g) == 4]
    ok2 = (I.left_open and I.right_open and abs(min(inside) - lo) < 5e-3 and abs(max(inside) - hi) < 5e-3
           and q_composite_sine_count(lo) != 4 and q_composite_sine_count(hi) != 4
           and all(lo < g < hi for g in inside))
    report(1, ok1 and ok2, f"a={good} count4 in [{min(inside):.4f},{max(inside):.4f}]")


def p02():
    sols, mm, nn = q_common_tangent_line(x**2, x**2 - x)
    lines = {sp.simplify(s[mm] * x + s[nn]) for s in sols}
    ymd = sp.sympify(md_answer(2, "1").split("=", 1)[1], locals={"x": x})
    ok1 = len(lines) == 1 and sp.simplify(lines.pop() - ymd) == 0
    avs, two = q_perp_tangent_a()
    ok2 = two and len(avs) == 1 and sp.simplify(avs[0] - sp.sympify(md_answer(2, "2"))) == 0
    report(2, ok1 and ok2, f"line={ymd} a={avs}")


def p03():
    f = sp.Rational(1, 2) + (x - sp.Rational(1, 2))**5 * sp.cos(2 * sp.pi * x)
    I = sp.integrate(f, (x, 0, 1))
    ok1 = sp.simplify(I - sp.sympify(md_answer(3, "1"))) == 0
    vals = q_king_c_values()
    md = set(sp.sympify(md_answer(3, "2")))
    ok2 = all(v == md for v in vals)
    report(3, ok1 and ok2, f"I={sp.simplify(I)} c={vals[0]}")


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
