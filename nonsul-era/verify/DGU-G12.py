#!/usr/bin/env python3
"""DGU-G12 (2012학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G12"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """[0,π]×[0,1] 모의 좌표 비율×π 로 ∫0^π sin x dx 추정."""
    exact = sp.integrate(sp.sin(x), (x, 0, sp.pi))
    rng = np.random.default_rng(12)
    N = 2_000_000
    X = rng.random(N) * np.pi
    Y = rng.random(N)
    est = np.mean(Y <= np.sin(X)) * np.pi
    wrong = np.mean(Y <= np.sin(X))           # 넓이 곱 누락
    ok = exact == 2 and abs(est - 2) < 0.01 and abs(wrong - 2) > 1
    report(1, ok, f"정확값 {exact}, 추정 비율×π={est:.4f} (비율만={wrong:.4f})")


def p02():
    """좌(1.5×1)·우(0.5×1)에 같은 수 살포 → 1.5p₁+0.5p₂ 가 넓이 추정, 단순 합산은 편향. (예시 렌즈 영역으로 확인)"""
    # 예시 회색 영역: 두 원호로 된 렌즈 (가로 0.15~1.85, 세로 0~1) — 원문 그림은 방정식 미제시
    def inside(X, Y):
        half = 0.5 * np.clip(1 - ((X - 1.0) / 0.85)**2, 0, None)
        return np.abs(Y - 0.5) <= half
    xx = sp.symbols("xx", real=True)
    exact = float(sp.integrate(1 - ((xx - 1) / sp.Rational(85, 100))**2, (xx, sp.Rational(15, 100), sp.Rational(185, 100))))
    rng = np.random.default_rng(1212)
    N = 1_000_000
    XL, YL = rng.random(N) * 1.5, rng.random(N)
    XR, YR = 1.5 + rng.random(N) * 0.5, rng.random(N)
    p1, p2 = np.mean(inside(XL, YL)), np.mean(inside(XR, YR))
    est = 1.5 * p1 + 0.5 * p2
    f1, f2 = p1 * N, p2 * N
    est2 = (3 * f1 + f2) / (4 * N) * 2
    naive = (f1 + f2) / (2 * N) * 2
    ok = abs(est - exact) < 0.005 and abs(est2 - est) < 1e-9 and abs(naive - exact) > 0.05
    report(2, ok, f"예시 넓이 {exact:.4f}, 1.5p₁+0.5p₂={est:.4f}, (3f₁+f₂)/(4n)×2={est2:.4f}, 단순합산 {naive:.4f}(편향)")


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
