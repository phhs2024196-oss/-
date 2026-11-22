#!/usr/bin/env python3
"""SGU-M22B (서강대 2022학년도 모의논술 2차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M22B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    import random
    import mpmath as mp
    random.seed(1)
    # (1-1) [증명-수치확인] -1<x<0 격자
    ok1 = all(mp.sqrt(1 + xv) < 1 + xv / 2 for xv in [-1 + i / 1000 for i in range(1, 1000)])
    # (1-2) [증명-수치확인] 양수 함수 몇 개에서 c 존재(부호 변화) 확인
    X = sp.symbols("X", real=True)
    ok2 = True
    for fx, (a, b) in [(sp.exp(X), (0, 1)), (X**2 + 1, (-2, 3)), (sp.sin(X) + 2, (0, 5)), (sp.sqrt(X), (0, 4))]:
        g = (X - a) * (X - b) * fx
        gp = sp.lambdify(X, sp.diff(g, X))
        xs = [a + (b - a) * i / 2000 for i in range(1, 2000)]
        ok2 &= any(gp(u) * gp(v) <= 0 for u, v in zip(xs, xs[1:]))
    # (1-3) 무작위 다항식 g 로 f 를 만들고 ∫f=-3 이 되게 스케일 → 목표 적분 계산
    vals = []
    for _ in range(4):
        g = sum(random.randint(-3, 3) * X**i for i in range(4)) + 1
        f = g * (sp.sin(X)**2 + 2 * sp.sin(X))
        I = sp.integrate(sp.expand(f), (X, 0, 2 * sp.pi))
        if I == 0:
            continue
        fs = f * (-3) / I
        J = sp.integrate(sp.expand(X * (2 * sp.pi - X) * sp.diff(fs, X, 2)), (X, 0, 2 * sp.pi))
        vals.append(sp.nsimplify(sp.simplify(J)))
    ok3 = len(vals) >= 3 and all(sp.simplify(v - sp.sympify(md_answer(1, "3"))) == 0 for v in vals)
    # (1-4) 여러 a_n→0 에서 큰 n 수치 극한
    target = float(sp.sympify(md_answer(1, "4")))
    def S(nv, an):
        return mp.fsum(mp.sqrt(mp.mpf(k) / nv + an) for k in range(1, nv + 1)) / nv
    ok4 = all(abs(S(nv, an(nv)) - target) < 3e-3 for nv in (200000,) for an in (lambda m: 1 / m, lambda m: 1 / mp.sqrt(m), lambda m: 1 / mp.log(m) ** 4))
    ok4 = ok4 and sp.integrate(sp.sqrt(X), (X, 0, 1)) == sp.sympify(md_answer(1, "4"))
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] (1-2) 4개 함수 c 존재, (1-3) 무작위 g 값={vals}, (1-4)→{target:.6f}")


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
