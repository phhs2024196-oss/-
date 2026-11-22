#!/usr/bin/env python3
"""SGU-M24B (서강대 2024학년도 모의논술 2차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M24B"
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
    import itertools
    from fractions import Fraction as Fr
    a, b, X = sp.symbols("a b X", positive=True)
    # (1-1) f'(3)=0, f(3)=1 연립
    f = a * sp.log(X - 1, 3) + b * sp.log(5 - X, 5)
    sol = sp.solve([sp.diff(f, X).subs(X, 3), f.subs(X, 3) - 1], [a, b], dict=True)[0]
    parts = dict(s.split("=", 1) for s in md_answer(1, "1").replace(" ", "").split(";"))
    ok1 = all(abs(float(sol[sym] - sp.sympify(parts[k]))) < 1e-12 for sym, k in ((a, "a"), (b, "b")))
    # 최댓값 확인(격자)
    fl = sp.lambdify(X, f.subs(sol))
    ok1 = ok1 and all(fl(3) >= fl(1 + 4 * i / 1000) - 1e-12 for i in range(1, 1000)) and abs(fl(3) - 1) < 1e-12
    # (1-2) 넓이 = ∫_{atan(1/A)}^{π/2} (A sin x - cos x) dx, 자연수 A
    area = lambda A: sp.integrate(A * sp.sin(X) - sp.cos(X), (X, sp.atan(sp.Rational(1, A)), sp.pi / 2))
    As = [A for A in range(1, 50) if 0 <= float(area(A)) <= 10]
    ok2 = As == list(sp.sympify(md_answer(1, "2")))
    # (1-3) 접점 t 의 방정식 판별 — 정수 a 를 -20..20 에서 실근 개수 직접 조사
    T = sp.symbols("T", real=True)
    y = (10 * T - 1) * sp.exp(T)
    ints = []
    for av in range(-20, 21):
        eq = sp.simplify((sp.diff(y, T) * (av - T) + y) * sp.exp(-T))
        roots = sp.Poly(sp.expand(eq), T).real_roots()
        if len(set(roots)) < 2:
            ints.append(av)
    ok3 = ints == list(sp.sympify(md_answer(1, "3")))
    # (1-4) 6번의 던지기(검정·흰 번갈아) 전수, 어느 시점이든 위치 일치 여부
    good = 0
    for seq in itertools.product((0, 1), repeat=6):
        B, W = [0, 0], [3, 3]
        met = False
        for i, c in enumerate(seq):
            if i % 2 == 0:
                if c == 1:
                    B[1] = min(3, B[1] + 1)
                else:
                    B[0] += 1
            else:
                if c == 1:
                    W[1] = max(0, W[1] - 1)
                else:
                    W[0] -= 1
            met = met or B == W
        good += met
    pr = Fr(good, 64)
    ok4 = pr == Fr(md_answer(1, "4"))
    report(1, ok1 and ok2 and ok3 and ok4, f"a={float(sol[a]):.6f} b={float(sol[b]):.6f} A={As[0]}..{As[-1]} ints={ints} P={pr}")


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
