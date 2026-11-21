#!/usr/bin/env python3
"""DGU-M26 (2026학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-M26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """장축 2·단축 1 타원, P 와 장축 대칭점 Q, 타원 위 R → 삼각형 PQR 넓이 최댓값."""
    import numpy as np
    # 타원 x^2/1 + y^2/(1/4) = 1 의 매개 (cos u, sin(u)/2)
    us = np.linspace(0, 2 * np.pi, 1441)
    best = 0.0
    for u in us:
        P = np.array([np.cos(u), np.sin(u) / 2])
        Q = np.array([P[0], -P[1]])
        R = np.stack([np.cos(us), np.sin(us) / 2], axis=1)
        area = 0.5 * np.abs((Q[0] - P[0]) * (R[:, 1] - P[1]) - (Q[1] - P[1]) * (R[:, 0] - P[0]))
        best = max(best, area.max())
    # 기호: P=(α,β), β>0, α≥0 일 때 R=(-1,0), S(β)=β(1+√(1-4β²))
    bb = sp.symbols("beta", positive=True)
    S = bb * (1 + sp.sqrt(1 - 4 * bb**2))
    crit = [c for c in sp.solve(sp.diff(S, bb), bb) if 0 < c < sp.Rational(1, 2)]
    cand = [sp.simplify(S.subs(bb, c)) for c in crit] + [S.subs(bb, sp.Rational(1, 2))]
    Smax = max(cand, key=lambda v: float(v))
    ok = sp.simplify(Smax - 3 * sp.sqrt(3) / 8) == 0 and abs(best - float(Smax)) < 1e-4
    report(1, ok, f"β*={crit}, 최대넓이={Smax} (격자 {best:.6f})")


def p02():
    """표적 사격 3발, 총점 ≥29 확률 — 점수 분포 전수(3발 독립)."""
    import itertools
    side = 550
    rad = [25 * k for k in range(1, 11)]
    hit = 1 - sp.Rational(19, 100)
    tot = sp.Integer(side) ** 2
    prob = {}
    # 반지름 25k 이내(경계 포함)면 원 (11-k) 개 이상 포함 → 점수 = 포함하는 원 개수
    prev = 0
    for k, r in enumerate(rad, start=1):
        ring = sp.pi * (r**2 - prev**2)
        prob[11 - k] = hit * ring / tot
        prev = r
    assert rad[-1] <= side / 2  # 가장 큰 원이 표적지 안
    prob[0] = 1 - sum(prob.values())
    P = sum(prob[a] * prob[b] * prob[c] for a, b, c in itertools.product(prob, repeat=3) if a + b + c >= 29)
    P = sp.simplify(P)
    target = 10 * sp.pi**3 * sp.Rational(9, 220) ** 6
    ok = sp.simplify(P - target) == 0
    report(2, ok, f"P(총점≥29)={sp.factor(P)} ≈ {float(P):.6e}")


def p03():
    """P1(1+cos t, sin t), P2(2cosθ, 2sinθ), 속력비 1:4 → θ=2t, (1) cos t0·최대거리 (2) 위치."""
    th = sp.Function("theta")(t)
    v1 = sp.sqrt(sp.simplify(sp.diff(1 + sp.cos(t), t) ** 2 + sp.diff(sp.sin(t), t) ** 2))
    v2 = sp.sqrt(sp.simplify(sp.diff(2 * sp.cos(th), t) ** 2 + sp.diff(2 * sp.sin(th), t) ** 2))
    w = sp.symbols("w", positive=True)  # 반시계 → θ'=w>0, θ(0)=0
    wsol = sp.solve(sp.Eq(v2.subs(sp.Derivative(th, t), w) / v1, 4), w)
    assert wsol == [2]
    T = 2 * t
    d2 = sp.simplify((1 + sp.cos(t) - 2 * sp.cos(T)) ** 2 + (sp.sin(t) - 2 * sp.sin(T)) ** 2)
    crit = sp.solve(sp.diff(d2, t), t)
    import numpy as np
    grid = np.linspace(0, 2 * np.pi, 400001)[:-1]
    dg = sp.lambdify(t, d2, "numpy")(grid)
    dmax2 = max(sp.nsimplify(sp.simplify(d2.subs(t, c))) for c in crit)
    tmax = [c for c in crit if sp.simplify(d2.subs(t, c) - dmax2) == 0]
    tmax = sorted({float(sp.N(c)) % (2 * np.pi) for c in tmax})
    cos0 = {sp.nsimplify(round(np.cos(v), 12)) for v in tmax}
    dmax = sp.sqrt(dmax2)
    pos = []
    for v in tmax:
        tv = [c for c in crit if abs(float(sp.N(c)) % (2 * np.pi) - v) < 1e-9][0]
        P1 = (sp.nsimplify(sp.simplify(1 + sp.cos(tv))), sp.radsimp(sp.simplify(sp.sin(tv))))
        P2 = (sp.simplify(2 * sp.cos(2 * tv)), sp.radsimp(sp.simplify(2 * sp.sin(2 * tv))))
        pos.append((P1, P2))
    s63 = 3 * sp.sqrt(7)
    want = {((sp.Rational(7, 8), s63 / 8), (sp.Rational(-31, 16), -s63 / 16)),
            ((sp.Rational(7, 8), -s63 / 8), (sp.Rational(-31, 16), s63 / 16))}
    got = {tuple(tuple(sp.nsimplify(sp.simplify(c)) for c in pt) for pt in pr) for pr in pos}
    ok = (cos0 == {sp.Rational(-1, 8)} and sp.simplify(dmax - 9 * sp.sqrt(2) / 4) == 0
          and abs(dg.max() - float(dmax2)) < 1e-8 and len(tmax) == 2 and got == want)
    report(3, ok, f"θ'={wsol[0]} (1) cos t0={cos0}, d_max={sp.nsimplify(dmax)} (2) {got}")


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
