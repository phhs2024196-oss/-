#!/usr/bin/env python3
"""HYU-G21B (한양대 2021학년도 논술 자연계열 오후1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G21B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(str(s["answer"]["value"]) for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import itertools
    R = sp.Rational
    pA = {2: R(1, 6), 1: R(1, 3), 0: R(1, 2)}
    pB = {2: R(1, 4), 1: R(1, 12), 0: R(2, 3)}
    # (1) Y~B(50,p) 정규근사 후 제시문 <나> 표로 계산(연속성 보정 없음 — 공식 해설 방식)
    p = 1 - pB[0]
    m, sd = 50 * p, sp.sqrt(50 * p * (1 - p))
    z1, z2 = (10 - m) / sd, (20 - m) / sd
    table = {R(1, 2): 0.1915, 1: 0.3413, R(3, 2): 0.4332, 2: 0.4772, R(5, 2): 0.4938}
    prob1 = table[abs(z1)] + table[abs(z2)]                      # z1<0<z2
    ok1 = z1 < 0 < z2 and abs(prob1 - float(md_answer(1, "1"))) < 1e-12
    # (2) 5회 전수
    prob2 = sum(sp.prod([pA[k] for k in seq]) for seq in itertools.product((0, 1, 2), repeat=5) if sum(seq) >= 7)
    ok2 = prob2 == sp.sympify(md_answer(1, "2"))
    # (3) 각 2회 전수
    prob3 = sum(pA[a1] * pA[a2] * pB[b1] * pB[b2]
                for a1, a2, b1, b2 in itertools.product((0, 1, 2), repeat=4) if b1 + b2 > a1 + a2)
    ok3 = prob3 == sp.sympify(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"z=({z1},{z2}) P1={prob1:.4f} P2={prob2} P3={prob3}")


def p02():
    import numpy as np
    from scipy import integrate
    # 좌표: A(0,0), D(1,0), B(0,-1), C(1,-1). Q=(1,-tan t) (CD 위), AP 는 AD 와 t+π/4
    def pts(tt):
        Q = np.array([1.0, -np.tan(tt)])
        ang = tt + np.pi / 4
        P = np.array([np.cos(ang) / np.sin(ang), -1.0])
        d = Q - P
        H = P + d * (-(P @ d)) / (d @ d)                   # A=원점에서 PQ 에 내린 수선의 발
        return P, Q, H
    # (1) H 자취의 길이: H(t) 를 기호식으로 만들어 |H'(t)| 를 수치 적분
    T = sp.symbols("T", real=True)
    Qs = sp.Matrix([1, -sp.tan(T)])
    Ps = sp.Matrix([sp.cot(T + sp.pi / 4), -1])
    ds = Qs - Ps
    Hs = Ps + ds * (-(Ps.dot(ds))) / ds.dot(ds)
    sp_f = sp.lambdify(T, sp.sqrt(sp.diff(Hs[0], T)**2 + sp.diff(Hs[1], T)**2), "numpy")
    L1, _ = integrate.quad(sp_f, 0, np.pi / 4, epsabs=1e-13, epsrel=1e-13)
    ok1 = abs(L1 - float(sp.sympify(md_answer(2, "1")))) < 1e-9
    # (2) 넓이 f(t) 적분
    def area(tt):
        P, Q, _ = pts(tt)
        return abs(P[0] * Q[1] - P[1] * Q[0]) / 2
    I2, _ = integrate.quad(area, 0, np.pi / 4)
    ok2 = abs(I2 - float(sp.sympify(md_answer(2, "2")))) < 1e-9
    # (3) T=(xT,0), S=T+a(cos s,-sin s) (x=1), R=T+a(cos(s+π/3),-sin(s+π/3)) (x=0) — 정사각형 안인 s 범위 탐색
    def tri(s):
        a = 1 / (np.cos(s) - np.cos(s + np.pi / 3))
        xT = -a * np.cos(s + np.pi / 3)
        yS, yR = -a * np.sin(s), -a * np.sin(s + np.pi / 3)
        okp = -1e-12 <= xT <= 1 + 1e-12 and -1 - 1e-12 <= yS <= 1e-12 and -1 - 1e-12 <= yR <= 1e-12
        return a, okp
    ss = np.linspace(0.01, np.pi / 2, 200001)
    feas = ss[[tri(s)[1] for s in ss]]
    s0, s1 = feas.min(), feas.max()
    ok_rng = abs(s0 - np.pi / 4) < 1e-4 and abs(s1 - 5 * np.pi / 12) < 1e-4
    I3, _ = integrate.quad(lambda s: np.sqrt(3) / 4 * tri(s)[0]**2, np.pi / 4, 5 * np.pi / 12)
    ok3 = ok_rng and abs(I3 - float(sp.sympify(md_answer(2, "3")))) < 1e-9
    report(2, ok1 and ok2 and ok3, f"L={L1:.10f} I2={I2:.12f} s0={s0:.5f} s1={s1:.5f} I3={I3:.12f}")


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
