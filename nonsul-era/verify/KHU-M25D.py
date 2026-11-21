#!/usr/bin/env python3
"""KHU-M25D (경희대 2025학년도 모의논술 의·약학계 수학, KHU-M25D-01~02) — sympy 검산.
규칙: 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import math
import random
import re
import sys

import sympy as sp

PREFIX = "KHU-M25D"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, b, m = sp.symbols("a b m", positive=True)
    X, Y = sp.symbols("X Y", real=True)
    # (1) 교점 (x1, m x1), x1>0 와 (-x1, -m x1)
    x1 = sp.solve(sp.Eq(X**2 / a**2 + (m * X)**2 / b**2, 1), X)
    x1 = [s for s in x1 if s.is_positive][0]
    y1 = m * x1
    # 두 접선(평행) 사이 거리 = 점(-x1,-y1)에서 l1 까지 거리
    A_, B_, C_ = x1 / a**2, y1 / b**2, -1
    d = sp.Abs(A_ * (-x1) + B_ * (-y1) + C_) / sp.sqrt(A_**2 + B_**2)
    expect = 2 * a * b * sp.sqrt((a**2 * m**2 + b**2) / (a**4 * m**2 + b**4))
    ok1 = sp.simplify(sp.factor(sp.simplify(d**2 - expect**2))) == 0   # 둘 다 양수이므로 제곱 비교
    for av, bv, mv in [(3, 4, 1), (2, 1, 5), (1, 7, sp.Rational(1, 3))]:
        ok1 &= abs(float((d - expect).subs({a: av, b: bv, m: mv}))) < 1e-12
    # (2) 변환을 점에 직접 적용해 C3 를 만들고 C1 과 만나는지 판정
    def c1_pt(th):
        return (10 + 3 * math.cos(th), 20 + 4 * math.sin(th))
    def F1(px, py):
        return (px - 10)**2 / 9 + (py - 20)**2 / 16
    def to_c3(px, py, u, v):
        px, py = py, px; px, py = px + u, py + v       # 대칭 후 평행이동 → C2
        px, py = py, px; px, py = px + u, py + v       # 다시 → C3
        return px, py
    N = 20000
    cnt = 0
    pairs = []
    for u in range(0, 12):
        for v in range(0, 12):
            # C3 위 점들 중 C1 의 내부·경계(F1<=1)에 있는 것이 있으면 만난다(두 볼록 폐곡선)
            mn = min(F1(*to_c3(*c1_pt(2 * math.pi * k / N), u, v)) for k in range(N))
            if mn <= 1 + 1e-9:
                cnt += 1
                pairs.append((u, v))
    ok2 = cnt == 15 and all(u + v <= 4 for u, v in pairs)
    report(1, ok1 and ok2, f"(1) 거리식 일치, (2) 만나는 (u,v) {cnt}개 (u,v≤11 탐색), 최대 u+v={max(u + v for u, v in pairs)}")


def has_subset(res_counts, m):
    """잔여 다중집합에서 m 개를 골라 합 ≡0 (mod m) 가능한지 (DP)."""
    reach = {(0, 0)}
    for r, c in enumerate(res_counts):
        if c == 0:
            continue
        new = set()
        for (k, s) in reach:
            for j in range(0, min(c, m - k) + 1):
                new.add((k + j, (s + j * r) % m))
        reach = new
    return (m, 0) in reach


def all_ok(n, m):
    for combo in itertools.combinations_with_replacement(range(m), n):
        cnts = [combo.count(r) for r in range(m)]
        if not has_subset(cnts, m):
            return False, combo
    return True, None


def p02():
    # (1) m=3: 전수(나머지 다중집합)
    ok4, ce4 = all_ok(4, 3)
    ok5, _ = all_ok(5, 3)
    n3 = next(n for n in range(3, 10) if all_ok(n, 3)[0])
    ok1 = (not ok4) and ok5 and n3 == 5
    # (2) m=9: n=16 반례(1 여덟 개, 2 여덟 개) 확인, n=17 은 무작위 다중집합 + 극단 구조 확인
    ce = [0] * 9; ce[1] = 8; ce[2] = 8
    ok2 = not has_subset(ce, 9)
    random.seed(7)
    for _ in range(20000):
        cnts = [0] * 9
        for r in random.choices(range(9), k=17):
            cnts[r] += 1
        if not has_subset(cnts, 9):
            ok2 = False
            break
    for r1, r2 in itertools.combinations(range(9), 2):      # 두 나머지만 쓰는 17개 전부
        for k in range(18):
            cnts = [0] * 9; cnts[r1] += k; cnts[r2] += 17 - k
            ok2 &= has_subset(cnts, 9)
    report(2, ok1 and ok2, f"(1) m=3: n=4 반례 {ce4}, 최소 n={n3}; (2) m=9: n=16 반례(1×8,2×8) 확인, n=17 무작위 2만+두 나머지 구조 전부 성립 [증명-수치확인]")


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
