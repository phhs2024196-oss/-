#!/usr/bin/env python3
"""SGU-M25A (서강대 2025학년도 모의논술 1차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M25A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(str(s["answer"]["value"]) for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    import numpy as np
    X, T, A = sp.symbols("X T A", real=True)
    # (1-1) 2x+sin2x: f'≥0, f'' 부호, 변곡점 (π/2, π)
    f = 2 * X + sp.sin(2 * X)
    f1, f2 = sp.diff(f, X), sp.diff(f, X, 2)
    xs = np.linspace(1e-6, np.pi - 1e-6, 2001)
    f1l, f2l = sp.lambdify(X, f1), sp.lambdify(X, f2)
    ok1 = all(f1l(v) >= -1e-12 for v in xs) and all(f2l(v) < 0 for v in xs if v < np.pi / 2 - 1e-3) \
        and all(f2l(v) > 0 for v in xs if v > np.pi / 2 + 1e-3) and f.subs(X, sp.pi / 2) == sp.pi \
        and f.subs(X, sp.pi) == 2 * sp.pi and f1.subs(X, sp.pi / 2) == 0
    # (1-2) 접선 → B, C 직접 계산
    tl = sp.tan(T) + sp.sec(T)**2 * (X - T)
    bx = sp.solve(sp.Eq(tl, 0), X)[0]
    cy = tl.subs(X, sp.pi / 2)
    S = sp.Rational(1, 2) * (sp.pi / 2 - bx) * cy
    Sl = sp.lambdify(T, S)
    ts = np.linspace(0, np.pi / 2 - 1e-4, 4001)
    vals = [Sl(v) for v in ts]
    incr = all(b2 > b1 for b1, b2 in zip(vals, vals[1:]))
    parts = dict(s.split("=", 1) for s in md_answer(1, "2").replace(" ", "").split(";"))
    ok2 = incr and sp.simplify(S.subs(T, 0) - sp.sympify(parts["min"])) == 0 and parts["max"] == "none" and all(v < 2 for v in vals) and vals[-1] > 2 - 1e-3   # 상한 2 에 수렴, 도달 안 함
    # (1-3) S(t)=(a-t)tan t 의 최대가 유일, g(a)=1/2 인 a (수치 탐색)
    from scipy.optimize import minimize_scalar, brentq
    def g(av):
        r = minimize_scalar(lambda tt: -(av - tt) * np.tan(tt), bounds=(1e-9, av - 1e-9), method="bounded", options={"xatol": 1e-13})
        return -r.fun, r.x
    def ncrit(av):
        grid = np.linspace(1e-6, av - 1e-6, 20001)
        d = (1 / np.cos(grid)**2) * (av - grid - np.sin(grid) * np.cos(grid))
        return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
    uniq = all(ncrit(av) == 1 for av in np.linspace(0.05, 1.55, 31))
    a_star = brentq(lambda av: g(av)[0] - 0.5, 0.5, 1.55, xtol=1e-14)
    ok3 = uniq and abs(a_star - float(sp.sympify(md_answer(1, "3")))) < 1e-8
    # (1-4) h/g, a→0+ (t0 수치, 작은 a 에서 수렴)
    ratios = []
    for av in (1e-2, 1e-3, 1e-4):
        gv, t0 = g(av)
        hv = float(sp.integrate(sp.tan(X), (X, 0, t0)))
        ratios.append(hv / gv)
    ok4 = abs(ratios[-1] - float(sp.sympify(md_answer(1, "4")))) < 1e-4
    report(1, ok1 and ok2 and ok3 and ok4, f"S(0)={S.subs(T, 0)} incr={incr} a*={a_star:.12f} ratios={[round(r, 6) for r in ratios]}")


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
