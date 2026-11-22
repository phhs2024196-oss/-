#!/usr/bin/env python3
"""HYU-G21A (한양대 2021학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G21A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
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
    T = sp.symbols("T", positive=True)
    y, X = sp.symbols("y X", positive=True)
    # (1) A: y축에 수직 단면 정사각형(한 변 ln y), B: x축에 수직 단면 정삼각형(한 변 T-e^x)
    V = sp.integrate(sp.log(y)**2, (y, 1, T))
    W = sp.sqrt(3) / 4 * sp.integrate((T - sp.exp(X))**2, (X, 0, sp.log(T)))
    expr = sp.log(T) / T * W / V
    L1 = sp.limit(sp.simplify(expr), T, sp.oo)
    ok1 = sp.simplify(L1 - sp.sympify(md_answer(1, "1"))) == 0
    # (2) 곡선 길이와 거리
    yy = (X + 1)**sp.Rational(3, 2)
    l = sp.integrate(sp.sqrt(1 + sp.diff(yy, X)**2), (X, -1, T))
    d = sp.sqrt(T**2 + (T + 1)**3)
    L2 = sp.limit(l / d, T, sp.oo)
    ok2 = sp.simplify(L2 - sp.sympify(md_answer(1, "2"))) == 0
    # (3) 자연수 n 전수(1..200)로 a_n/b_n 최소
    vals = {k: sp.Rational((k * k - 12 * k + 37)**2, (2 * k + 1)**2) for k in range(1, 201)}
    nmin = min(vals, key=vals.get)
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "3").replace(" ", "").split(";"))
    ok3 = int(parts["n"]) == nmin and sp.sympify(parts["ratio"]) == vals[nmin]
    report(1, ok1 and ok2 and ok3, f"lim1={L1} lim2={L2} n={nmin} ratio={vals[nmin]}")


def p02():
    a, b, c, X = sp.symbols("a b c X", positive=True)
    import random
    random.seed(1)
    # (1) f'(x) 를 직접 미분해 (ㄱ)/x^2 * {…} 와 일치하는지
    S = a * b * X + 1 / a + 1 / b + 1 / X
    f = a + b + X + 1 / (a * b * X) + 1 / S
    gk = sp.sympify(md_answer(2, "1"), locals={"a": a, "b": b, "x": X})
    ok1 = sp.simplify(sp.diff(f, X) - gk / X**2 * (1 / S**2 - 1 / (a * b))) == 0
    # f'<=0 표본 확인
    fp = sp.lambdify((a, b, X), sp.diff(f, X))
    ok1 = ok1 and all(fp(random.uniform(.01, 1), random.uniform(.01, 1), random.uniform(.01, 1)) <= 1e-12
                      for _ in range(2000))

    # (2)(3) 원래 부등식의 좌변 최소를 무작위 표본 + a=b=c=1 값으로 확인
    def lhs(A, B, av, bv, cv):
        dv = 1 / (av * bv * cv)
        return A * (av + bv + cv + dv) + B / (av * bv * cv + av * bv * dv + av * cv * dv + bv * cv * dv)

    def check(A, B, md):
        target = float(sp.sympify(md))
        at1 = lhs(A, B, 1, 1, 1)
        mn = min(lhs(A, B, random.uniform(.05, 1), random.uniform(.05, 1), random.uniform(.05, 1))
                 for _ in range(20000))
        return abs(at1 - target) < 1e-12 and mn >= target - 1e-12, at1

    ok2, m = check(1, 1, md_answer(2, "2"))
    ok3, k = check(2, 17, md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"(ㄱ) ok={ok1} M={m} K={k}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
