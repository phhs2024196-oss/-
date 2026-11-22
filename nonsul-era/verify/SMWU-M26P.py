#!/usr/bin/env python3
"""SMWU-M26P (숙명여대 2026학년도 모의논술 자연계열(약학부)) — sympy 검산. 문항 1은 일반(SMWU-M26-01)과 동일하므로 verify/SMWU-M26.py::p01 에서 검산."""
import re
import sys

import sympy as sp

PREFIX = "SMWU-M26P"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
n = sp.symbols("n", positive=True, integer=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    """'b_n = ...; c_n = ...' 형식을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    y = sp.symbols("y", real=True)
    loc = {"n": n, "factorial": sp.factorial}
    P1 = parts(md_answer(2, "1"))
    bn = sp.sympify(P1["b_n"], locals=loc)
    cn = sp.sympify(P1["c_n"], locals=loc)
    dn = sp.sympify(parts(md_answer(2, "2"))["d_n"], locals=loc)
    ok = True
    for k in range(1, 9):
        an = sp.integrate(x**k * (1 - x)**k, (x, 0, 1))
        rhs = sp.integrate(x**(k + 1) * (1 - x)**(k - 1), (x, 0, 1))
        pn = sp.integrate((1 - y**2)**k, (y, 0, 1))
        ok &= sp.simplify(an / rhs - bn.subs(n, k)) == 0          # b_n
        ok &= sp.simplify(an / sp.factorial(k) - cn.subs(n, k)) == 0  # c_n
        ok &= sp.simplify(an / pn - dn.subs(n, k)) == 0            # d_n
    report(2, ok, "b_n, c_n, d_n 이 n=1..8 에서 직접 적분과 일치")


def p03():
    y = sp.symbols("y", real=True)
    # (3-1) YAML 의 두 포물선 식이 곡선 위에 있고 조건 범위가 곡선 전체를 덮는지 확인
    segs = md_answer(3, "1").split(";")
    exprs = [sp.sympify(seg.split("=", 1)[1].split("(")[0], locals={"x": x}) for seg in segs]
    up, lo = exprs
    ok1 = sp.simplify(3 * (up - 2 * x) + 2 * x**2 - 8) == 0 and sp.simplify(-3 * (lo - 2 * x) + 2 * x**2 - 8) == 0
    ok1 &= set(sp.solve(sp.Eq(up, 2 * x), x)) == {-2, 2} and set(sp.solve(sp.Eq(lo, 2 * x), x)) == {-2, 2}
    # 곡선 위 점 샘플: 각 x 에 대해 곡선의 y 값 = {2x ± (8-2x^2)/3}
    import numpy as np
    for xv in np.linspace(-2, 2, 81):
        r = (8 - 2 * xv**2) / 3
        ok1 &= abs(float(up.subs(x, xv)) - (2 * xv + r)) < 1e-12 and abs(float(lo.subs(x, xv)) - (2 * xv - r)) < 1e-12
    # (3-2) 영역 {3|y-2x|+2x²<=8, y>=0} 의 넓이: x 별 세로 길이를 직접 적분(대칭성 미사용)
    r = (8 - 2 * x**2) / 3
    top, bot = 2 * x + r, sp.Max(0, 2 * x - r)
    # 길이 = top - bot (top>=0 일 때). top>=0 ⇔ x∈[-1,2], 2x-r>=0 ⇔ x∈[1,2]
    xr = sp.solve(sp.Eq(top, 0), x); xb = sp.solve(sp.Eq(2 * x - r, 0), x)
    x0 = max(v for v in xr if v <= 0); x1 = max(xb)
    area = sp.integrate(top, (x, x0, x1)) + sp.integrate(top - (2 * x - r), (x, x1, 2))
    ok2 = sp.simplify(area - sp.sympify(md_answer(3, "2"))) == 0
    report(3, ok1 and ok2, f"[그래프-수치확인] area={area} (x0={x0}, x1={x1})")


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
