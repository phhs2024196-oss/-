#!/usr/bin/env python3
"""KKU-G25F (건국대 2025학년도 KU논술우수자 KU자유전공학부 [문제 2] 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KKU-G25F"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    # (2-1) A(a, a^3+a+3), a>=0; B = 직선 OA 와 y=4x-2 의 교점(O 반대쪽). 비 OA/OB 의 최솟값
    a = sp.symbols("a", nonnegative=True)
    X = sp.symbols("X", real=True)
    ya = a**3 + a + 3
    xb = sp.solve(sp.Eq(ya * X, a * (4 * X - 2)), X)[0]        # 직선 OA: a*y = ya*x (a=0 포함)
    yb = 4 * xb - 2
    ratio = sp.sqrt(a**2 + ya**2) / sp.sqrt(xb**2 + yb**2)
    import mpmath as mp
    rf = sp.lambdify(a, ratio, "mpmath")
    grid = [i / 1000 for i in range(0, 5001)]
    a0 = min(grid, key=lambda v: rf(v))
    amin = mp.findroot(lambda v: mp.diff(rf, v), a0)
    rmin = rf(amin)
    opp = all(float(xb.subs(a, v)) <= 0 and float(yb.subs(a, v)) < 0 for v in (0, 0.5, 1, 3))  # B 가 O 반대쪽
    ok1 = opp and abs(rmin - float(sp.sympify(md_answer(2, "1")))) < 1e-9 and rf(0) > rmin
    # (2-2) 1..9 에서 4개, 합이 4의 배수 — 전수 열거
    cnt = sum(1 for c in itertools.combinations(range(1, 10), 4) if sum(c) % 4 == 0)
    by_k = {k: sum(1 for c in itertools.combinations(range(1, 10), 4) if sum(c) == k) for k in (12, 16, 20, 24, 28)}
    ok2 = cnt == int(md_answer(2, "2"))
    # (2-3)(1) t=1/2: A(0,0), B(4,0), 중심 E(t,1), 반지름 1. A·B 에서 그은 AB 가 아닌 접선의 교점 C
    t = sp.Rational(1, 2)
    m, n_ = sp.symbols("m n", real=True)
    ms = [s for s in sp.solve(sp.Eq((m * t - 1)**2, m**2 + 1), m) if s != 0]
    ns = [s for s in sp.solve(sp.Eq((n_ * (t - 4) - 1)**2, n_**2 + 1), n_) if s != 0]
    xc = sp.solve(sp.Eq(ms[0] * X, ns[0] * (X - 4)), X)[0]
    yc = ms[0] * xc
    area = sp.Rational(1, 2) * 4 * yc
    # 독립 확인: 접선 길이 t, 4-t, x 와 헤론 공식 = r*s  →  x = 4/(t(4-t)-1)
    xx = sp.symbols("xx", positive=True)
    tt = sp.symbols("tt", positive=True)
    xsol = sp.solve(sp.Eq((4 + xx) * tt * (4 - tt) * xx, (4 + xx)**2), xx)
    area2 = (4 + xsol[0]).subs(tt, t)
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "3").replace(" ", "").split(";"))
    ok3a = sp.simplify(area - sp.sympify(parts["area"])) == 0 and sp.simplify(area2 - area) == 0
    # (2-3)(2) 존재 범위: x>0 인 해가 있을 조건을 0<t<4 에서 풀고, 수치 기하로도 확인
    cond = sp.solve_univariate_inequality(tt * (4 - tt) - 1 > 0, tt, relational=False)
    lo, hi = sp.sympify(parts["range"])
    ok3b = cond == sp.Interval.open(lo, hi)

    def exists(tv):   # 두 접선(AB 제외)이 AB 위쪽에서 만나는가 — 각 A=2atan(1/t), B=2atan(1/(4-t))
        import math
        A_, B_ = 2 * math.atan(1 / tv), 2 * math.atan(1 / (4 - tv))
        return A_ + B_ < math.pi - 1e-12
    grid_t = [i / 1000 for i in range(1, 4000)]
    ok3c = all(exists(v) == (float(lo) < v < float(hi)) for v in grid_t
               if abs(v - float(lo)) > 2e-3 and abs(v - float(hi)) > 2e-3)
    report(2, ok1 and ok2 and ok3a and ok3b and ok3c,
           f"min OA/OB={float(rmin):.12f} at a={float(amin):.6f}; count={cnt} {by_k}; "
           f"C=({xc},{yc}) area={area}; range={cond}")


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
