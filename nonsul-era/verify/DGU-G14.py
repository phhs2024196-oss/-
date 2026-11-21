#!/usr/bin/env python3
"""DGU-G14 (2014학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G14"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """1000·(2/3)^10, 1000·(2/3)^20 의 첫 자리 — 정확값과 주어진 로그값(가수) 둘 다."""
    def first_digit(q):
        s = sp.floor(sp.log(q, 10))
        return int(sp.floor(q / sp.Integer(10)**s))
    v10 = 1000 * sp.Rational(2, 3)**10
    v20 = 1000 * sp.Rational(2, 3)**20
    d10, d20 = first_digit(v10), first_digit(v20)
    l2, l3 = sp.Rational(3010, 10000), sp.Rational(4771, 10000)
    m10 = (3 + 10 * (l2 - l3)) % 1
    m20 = (3 + 20 * (l2 - l3)) % 1
    def digit_from_mant(m):
        logs = {1: 0, 2: l2, 3: l3, 4: 2 * l2, 5: 1 - l2, 6: l2 + l3, 7: None, 8: 3 * l2, 9: 2 * l3}
        if m < l2: return 1
        if m < l3: return 2
        if m < 2 * l2: return 3
        return None
    ok = d10 == 1 and d20 == 3 and digit_from_mant(m10) == 1 and digit_from_mant(m20) == 3
    report(1, ok, f"10년 {float(v10):.4f}(가수 {m10}) → {d10}, 20년 {float(v20):.5f}(가수 {m20}) → {d20}")


def p02():
    """[증명-수치확인] 가수 분포가 평행이동 불변 → 균등 f=1 on [0,1), P(첫자리 2)=log3−log2."""
    u = sp.symbols("u", real=True)
    P2 = sp.integrate(1, (u, sp.log(2, 10), sp.log(3, 10)))
    ok_p = sp.simplify(P2 - sp.log(sp.Rational(3, 2), 10)) == 0
    # 균등분포는 임의 c 에 대해 (X+c) mod 1 도 균등: 구간확률 비교
    rng = np.random.default_rng(14)
    Xs = rng.random(400000)
    inv = True
    for c in rng.random(10):
        Y = (Xs + c) % 1
        for (lo, hi) in [(0, .2), (.3, .45), (.7, 1)]:
            if abs(np.mean((Y >= lo) & (Y < hi)) - (hi - lo)) > 0.005:
                inv = False
    # 비균등 밀도(2x)는 불변이 아님
    Z = np.sqrt(rng.random(400000))
    noninv = abs(np.mean(((Z + 0.5) % 1) < 0.5) - np.mean(Z < 0.5)) > 0.1
    # 단위에 의존하지 않는 자료의 예: 여러 확률변수의 곱 → 첫 자리 2 빈도
    W = np.prod(rng.lognormal(0, 1, (200000, 12)), axis=1)
    fd = (W / 10**np.floor(np.log10(W))).astype(int)
    freq2 = np.mean(fd == 2)
    ok = ok_p and inv and noninv and abs(freq2 - float(P2)) < 0.005
    report(2, ok, f"[증명-수치확인] P(첫자리 2)={P2}≈{float(P2):.4f}, 모의자료 빈도 {freq2:.4f}, 균등 불변 {inv}, 2x 밀도 불변 아님 {noninv}")


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
