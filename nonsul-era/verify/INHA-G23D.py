#!/usr/bin/env python3
"""INHA-G23D (2023학년도 인하대 자연계열 논술 오전 의예과; [문제 1]=INHA-G23A-02 라 p01 없음) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "INHA-G23D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    """(2-1) (-1)^n(2n+1) (2-2)(a) M=20, m=-16 (b) h(0)=m, h(1)=M 연속 [증명-수치확인]."""
    f = sp.pi * x * sp.sin(sp.pi * x)
    F = sp.integrate(f, x)
    ok1 = all(sp.simplify(F.subs(x, k + 1) - F.subs(x, k) - (-1) ** k * (2 * k + 1)) == 0 for k in range(-6, 7))
    Ff = sp.lambdify(x, F, "numpy")
    G = lambda a, b, c: (Ff(b) - Ff(a)) - (Ff(c) - Ff(b))
    # 정수 전수 + 실수 격자(0.05) 전수로 최대·최소
    pts = np.round(np.arange(0, 6.0001, 0.05), 10)
    A, B, C = np.meshgrid(pts, pts, pts, indexing="ij")
    mask = (A < B) & (B < C)
    vals = G(A[mask], B[mask], C[mask])
    Mx, mn = vals.max(), vals.min()
    ia, im = vals.argmax(), vals.argmin()
    arg_max = (A[mask][ia], B[mask][ia], C[mask][ia])
    arg_min = (A[mask][im], B[mask][im], C[mask][im])
    ok2 = abs(Mx - 20) < 1e-9 and abs(mn + 16) < 1e-9 and arg_max == (4, 5, 6) and arg_min == (3, 4, 5)
    # (b) h(s)=∫_{3+s}^{4+s} f - ∫_{4+s}^{5+s} f : h(0)=m, h(1)=M, 연속(미분가능) → 사잇값
    s = sp.symbols("s", real=True)
    h = (F.subs(x, 4 + s) - F.subs(x, 3 + s)) - (F.subs(x, 5 + s) - F.subs(x, 4 + s))
    ok3 = sp.simplify(h.subs(s, 0) + 16) == 0 and sp.simplify(h.subs(s, 1) - 20) == 0
    # 모든 k∈(m,M) 도달: 격자 위 h 값이 [-16,20] 을 빈틈(간격<0.5) 없이 덮음
    hv = np.sort(sp.lambdify(s, h, "numpy")(np.linspace(0, 1, 20001)))
    ok3 = ok3 and hv.min() <= -16 + 1e-9 and hv.max() >= 20 - 1e-9 and np.diff(hv).max() < 0.01
    report(2, ok1 and ok2 and ok3, f"(2-1) k=-6..6 일치 (a) M={Mx:.6f}@{arg_max}, m={mn:.6f}@{arg_min} (b) h(0)=-16,h(1)=20 [증명-수치확인]")


def _fair(f, n):
    return len({(i + f[i - 1]) % n for i in range(1, n + 1)}) == n


def p03():
    """(3-1) n! (3-2)(a) g=f+1 공평 (b) n=9, 치역 {a,b}, b-a≥5 → {6} (3-3) n 짝수 공평 일대일대응 없음 [증명-수치확인]."""
    import itertools
    import math
    ok1 = True
    cnt = {}
    for nn in range(3, 7):
        c = sum(1 for f in itertools.product(range(1, nn + 1), repeat=nn) if _fair(f, nn))
        cnt[nn] = c
        ok1 = ok1 and c == math.factorial(nn)
    ok2a = True
    for nn in range(3, 7):
        for f in itertools.product(range(1, nn), repeat=nn):
            if _fair(f, nn):
                ok2a = ok2a and _fair(tuple(v + 1 for v in f), nn)
    diffs = set()
    for a in range(1, 10):
        for b in range(a + 1, 10):
            for f in itertools.product((a, b), repeat=9):
                if set(f) == {a, b} and _fair(f, 9) and b - a >= 5:
                    diffs.add(b - a)
    ex = (3, 9, 9, 3, 9, 9, 3, 9, 9)          # 해설의 예
    ok2b = diffs == {6} and _fair(ex, 9)
    ok3 = True
    for nn in (4, 6, 8):
        ok3 = ok3 and not any(_fair(p, nn) for p in itertools.permutations(range(1, nn + 1)))
    odd_exists = all(any(_fair(p, nn) for p in itertools.permutations(range(1, nn + 1))) for nn in (3, 5, 7))
    report(3, ok1 and ok2a and ok2b and ok3 and odd_exists,
           f"(3-1) 개수 {cnt} (3-2)(a) n=3..6 전수 {ok2a} (b) b-a∈{sorted(diffs)} (3-3) n=4,6,8 없음={ok3}, 홀수 3,5,7 존재={odd_exists} [증명-수치확인]")


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
