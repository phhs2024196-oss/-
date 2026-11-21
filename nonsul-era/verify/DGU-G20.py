#!/usr/bin/env python3
"""DGU-G20 (2020학년도 동국대 수리논술 실전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import random
import re
import sys

import sympy as sp

PREFIX = "DGU-G20"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """세 꼭짓점 좌표 → 넓이 공식(sin·내적 경로로 유도) 일반식 확인 + A(1,1),B(2,2),C(1/2,5/2) 넓이 (답 1)."""
    x1, x2, x3, y1, y2, y3 = sp.symbols("x1 x2 x3 y1 y2 y3", real=True)
    u = sp.Matrix([x2 - x1, y2 - y1])
    w = sp.Matrix([x3 - x1, y3 - y1])
    # 【가】【나】【다】: S^2 = (1/4)(|u|^2|w|^2 - (u.w)^2)
    S2 = sp.Rational(1, 4) * (u.dot(u) * w.dot(w) - u.dot(w) ** 2)
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    ok1 = sp.expand(S2 - det**2 / 4) == 0
    vals = {x1: 1, y1: 1, x2: 2, y2: 2, x3: sp.Rational(1, 2), y3: sp.Rational(5, 2)}
    S_num = sp.sqrt(S2.subs(vals))
    # 독립 확인: 신발끈 공식
    P = [(1, 1), (2, 2), (sp.Rational(1, 2), sp.Rational(5, 2))]
    shoe = sp.Abs(sum(P[i][0] * P[(i + 1) % 3][1] - P[(i + 1) % 3][0] * P[i][1] for i in range(3))) / 2
    ok = ok1 and S_num == shoe == 1
    report(1, ok, f"S^2=det^2/4 항등식 {ok1}, S={S_num}(신발끈 {shoe})")


def p02():
    """[증명-수치확인] d/dx ∫_a^x f = f(x): 여러 연속함수에서 차분몫과 m≤ΔS/Δx≤M 확인."""
    import math
    random.seed(1)
    fs = [lambda u: u**2 + 1, lambda u: math.sin(3 * u) + 2, lambda u: math.exp(-u) * math.cos(u), lambda u: abs(u - 0.3)]
    worst = 0.0
    bad = 0
    for f in fs:
        for _ in range(20):
            a = -1.0
            xx = random.uniform(-0.5, 1.5)
            for h in (1e-2, -1e-2, 1e-3, -1e-3):
                # ΔS = ∫_x^{x+h} f (Simpson 고정밀)
                N = 2000
                lo, hi = (xx, xx + h) if h > 0 else (xx + h, xx)
                dx = (hi - lo) / N
                s = f(lo) + f(hi) + sum((4 if k % 2 else 2) * f(lo + k * dx) for k in range(1, N))
                dS = s * dx / 3 * (1 if h > 0 else -1)
                grid = [f(lo + k * (hi - lo) / 400) for k in range(401)]
                m_, M_ = min(grid), max(grid)
                q = dS / h
                if not (m_ - 1e-9 <= q <= M_ + 1e-9):
                    bad += 1
                if abs(h) == 1e-3:
                    worst = max(worst, abs(q - f(xx)))
    ok = bad == 0 and worst < 5e-3
    report(2, ok, f"[증명-수치확인] m≤ΔS/Δx≤M 위반 {bad}건, |ΔS/Δx−f(x)| 최대 {worst:.2e} (Δx=±1e-3)")


def p03():
    """맞물린 원반: A(반지름3) n바퀴 → B(반지름4) 3n/4바퀴 반대방향. 회색 판정 후 두 주사위 합 전수 (답 19/36)."""
    gray = [(220, 280), (340, 400), (100, 160)]  # B 고정 좌표계의 회색 부채꼴(도), 접점 방향 180°, ∠a=40°

    def is_gray(ang):
        ang %= 360
        return any(lo < ang < hi or lo < ang + 360 < hi for lo, hi in gray)

    def contact_gray(k):
        # A 반시계 k바퀴 → 접점에서 호 길이 6πk 이동 → B 시계방향 6πk/4 rad = 270k°
        # 시계방향 φ 회전 후 접점(180°)에 오는 B의 점은 원래 180°+φ
        return is_gray(180 + 270 * k)

    fav = sum(1 for i, j in itertools.product(range(1, 7), repeat=2) if contact_gray(i + j))
    P = sp.Rational(fav, 36)
    Q = sorted({i + j for i, j in itertools.product(range(1, 7), repeat=2) if contact_gray(i + j)})
    # 대표 오답: B 도 반시계로 돈다고 본 경우
    wrong = sp.Rational(sum(1 for i, j in itertools.product(range(1, 7), repeat=2) if is_gray(180 - 270 * (i + j))), 36)
    ok = P == sp.Rational(19, 36) and Q == [2, 3, 6, 7, 10, 11]
    report(3, ok, f"Q(합)={Q}, P(Q)={P}; 회전방향 착각 오답={wrong}")


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
