#!/usr/bin/env python3
"""SGU-M26A (서강대 2026학년도 모의논술 1차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M26A"
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
    from scipy.optimize import minimize_scalar
    ph = sp.symbols("phi", real=True)
    r = sp.sqrt(6) / 2
    # Q 의 x좌표: |PQ|=√6/2, P=(cos φ, sin φ). q>0 인 해(두 해 중 양수인 것; 큰 근)
    q = sp.cos(ph) + sp.sqrt(r**2 - sp.sin(ph)**2)
    ql = sp.lambdify(ph, q)
    # 다른 근 cos φ - sqrt(...) 는 φ∈(0,π) 에서 항상 ≤0 인지 확인(유일성)
    q2 = sp.lambdify(ph, sp.cos(ph) - sp.sqrt(r**2 - sp.sin(ph)**2))
    grid = np.linspace(1e-4, np.pi - 1e-4, 20001)
    uniq = all(q2(v) <= 1e-12 for v in grid) and all(ql(v) > 0 for v in grid)
    # (1-1) 삼각형 PHQ 넓이 최대
    A1 = lambda v: 0.5 * np.sin(v) * abs(ql(v) - np.cos(v))
    m1 = max(A1(v) for v in grid)
    res = minimize_scalar(lambda v: -A1(v), bounds=(grid[np.argmax([A1(v) for v in grid])] - 0.01, grid[np.argmax([A1(v) for v in grid])] + 0.01), method="bounded", options={"xatol": 1e-13})
    ok1 = uniq and abs(-res.fun - float(sp.sympify(md_answer(1, "1")))) < 1e-10
    # (1-2) f(θ)=½·OQ·sinθ, 기호 적분
    fth = sp.Rational(1, 2) * q * sp.sin(ph)
    I = sp.integrate(fth * sp.cos(ph), (ph, sp.pi / 4, sp.pi / 3))
    ok2 = abs(float(I) - float(sp.sympify(md_answer(1, "2")))) < 1e-12
    # (1-3) f 의 최댓값 위치
    fl = sp.lambdify(ph, fth)
    v0 = grid[np.argmax([fl(v) for v in grid])]
    res3 = minimize_scalar(lambda v: -fl(v), bounds=(v0 - 0.01, v0 + 0.01), method="bounded", options={"xatol": 1e-14})
    ok3 = abs(np.cos(res3.x) - float(sp.sympify(md_answer(1, "3")))) < 1e-7
    # (1-4) Q=(t,0) 일 때 P=(c(t), sqrt(1-c^2)), c=(t^2-1/2)/(2t)
    tt = sp.symbols("tt", positive=True)
    c = (tt**2 + 1 - r**2) / (2 * tt)
    Px, Py = c, sp.sqrt(1 - c**2)
    a_sol = [s for s in sp.solve(sp.Eq(c, sp.Rational(1, 2)), tt) if (r - 1) < s < (r + 1)]
    av = a_sol[0]
    speed = sp.sqrt(sp.diff(Px, tt)**2 + sp.diff(Py, tt)**2).subs(tt, av)
    ok4 = len(a_sol) == 1 and abs(float(speed) - float(sp.sympify(md_answer(1, "4")))) < 1e-12 and \
        abs(float(Py.subs(tt, av)) - float(sp.sqrt(3) / 2)) < 1e-12
    report(1, ok1 and ok2 and ok3 and ok4, f"maxPHQ={-res.fun:.12f} I={float(I):.12f} cos(alpha)={np.cos(res3.x):.9f} a={sp.nsimplify(av)} speed={float(speed):.12f}")


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
