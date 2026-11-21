#!/usr/bin/env python3
"""KHU-M21 (경희대 2021학년도 모의논술 자연계 수학, KHU-M21-01~02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import re
import sys
from math import comb

import sympy as sp

PREFIX = "KHU-M21"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) P(√3/2, 3/4): 법선과 y축의 교점, 반지름
    px = sp.sqrt(3) / 2
    f = x**2
    py = f.subs(x, px)
    slope = sp.diff(f, x).subs(x, px)
    X = sp.Symbol("X")
    normal = py - (X - px) / slope
    b = sp.simplify(normal.subs(X, 0))
    r = sp.sqrt(px**2 + (py - b) ** 2)
    ok1 = sp.simplify(b - sp.Rational(5, 4)) == 0 and sp.simplify(r - 1) == 0
    # 대표 오답(법선 기울기 부호 누락) b=1/4 재현
    wrong_b = sp.simplify(py + (0 - px) / slope)
    # (2) 수치: 원 중심 (0,b), 반지름 = 최소거리 → 교점 수 vs r 범위 [증명-수치확인]
    ok2 = True
    bb = sp.Symbol("b", positive=True)
    g = x**2 + (x**2 - bb) ** 2
    for bv in [sp.Rational(k, 40) for k in range(1, 161)]:
        gb = g.subs(bb, bv)
        crit = [c for c in sp.solve(sp.diff(gb, x), x) if c.is_real]
        m = min(gb.subs(x, c) for c in crit)
        r_ = sp.sqrt(m)
        mins = {c for c in crit if sp.simplify(gb.subs(x, c) - m) == 0}
        npts = len(mins)
        expect = 1 if r_ <= sp.Rational(1, 2) else 2
        if npts != expect:
            ok2 = False
        if bv > sp.Rational(1, 2) and sp.simplify(m - (bv - sp.Rational(1, 4))) != 0:
            ok2 = False
        if bv <= sp.Rational(1, 2) and sp.simplify(r_ - bv) != 0:
            ok2 = False
    report(1, ok1 and ok2 and wrong_b == sp.Rational(1, 4),
           f"(1) b={b}, r={sp.simplify(r)} (오답 재현 b={wrong_b}); (2) b=1/40..4 격자 160점: r<=1/2 ⇔ 한 점, r>1/2 ⇔ 두 점 [증명-수치확인]")


def p02():
    pts = [(i, j) for i in range(4) for j in range(4)]
    A = (0, 0)

    def paths(p, q, banned=None):
        (x1, y1), (x2, y2) = p, q
        dx, dy = (1 if x2 >= x1 else -1), (1 if y2 >= y1 else -1)
        # DP over monotone lattice paths
        W = {}
        for i in range(abs(x2 - x1) + 1):
            for j in range(abs(y2 - y1) + 1):
                pt = (x1 + dx * i, y1 + dy * j)
                if banned is not None and pt == banned:
                    W[(i, j)] = 0
                elif i == 0 and j == 0:
                    W[(i, j)] = 1
                else:
                    W[(i, j)] = W.get((i - 1, j), 0) + W.get((i, j - 1), 0)
        return W[(abs(x2 - x1), abs(y2 - y1))]

    vals = sorted({paths(p, q) for p, q in itertools.combinations(pts, 2)})
    ok1 = vals == [1, 2, 3, 4, 6, 10, 20] and paths(A, (2, 2)) == 6
    rest = [p for p in pts if p != A]
    pairs = list(itertools.combinations(rest, 2))
    good = sum(1 for p, q in pairs if paths(p, q, banned=A) >= 6)
    prob = sp.Rational(good, len(pairs))
    wrong = sp.Rational(sum(1 for p, q in pairs if paths(p, q) >= 6), len(pairs))
    ok2 = prob == sp.Rational(13, 105) and len(pairs) == comb(15, 2)
    report(2, ok1 and ok2, f"(1) 경로 수 집합 {vals}; (2) 전수: {good}/{len(pairs)} = {prob} (A 무시 오답 {wrong})")


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
