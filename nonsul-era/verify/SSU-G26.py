#!/usr/bin/env python3
"""SSU-G26 (숭실대 2026학년도 논술고사(자연) 1~4번) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, idx=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return sp.sympify(d["subparts"][idx]["answer"]["value"])


def p01():
    """원과 직선의 교점을 직접 풀고 f/g 의 좌극한을 기호 계산."""
    m = sp.symbols("m", positive=True)
    X = sp.symbols("X", real=True)
    roots = sp.solve(sp.Eq(X**2 + (m * X + m) ** 2, 1), X)
    xp = [r for r in roots if sp.simplify(r + 1) != 0][0]       # -1 이 아닌 근(0<m<1 에서 더 큼)
    assert all(sp.N((xp - r).subs(m, sp.Rational(1, 2))) >= 0 for r in roots)
    yp = m * xp + m
    f = 1 - yp
    g = sp.Rational(1, 2) * (1 - m) * xp
    L = sp.limit(sp.simplify(f / g), m, 1, "-")
    ans = load_answer(1)
    report(1, sp.simplify(L - ans) == 0, f"P_x={sp.simplify(xp)}, f/g={sp.simplify(f/g)}, lim={L}, md={ans}")


def p02():
    """정적분 극한값을 기호 계산 + n=20000 에서 삼각형 넓이 합을 정의대로 수치 계산."""
    import math
    f = sp.Piecewise((-(2 * x - 1) * sp.exp(2 * x), x < sp.Rational(1, 2)), ((2 * x - 1) * sp.exp(2 * x), True))
    I = sp.integrate(f, (x, 0, 1)) / 3
    N = 20000
    F = lambda X: abs(2 * X - 1) * math.exp(2 * X)
    tot = 0.0
    for k in range(1, N + 1):
        xk, xk1 = (k - 1) / N, k / N
        hy = (2 * F(xk1) + 0) / 3                      # P_k G_{k+1} 를 2:1 내분
        tot += 0.5 * (xk1 - xk) * hy
    ans = load_answer(2)
    ok = sp.simplify(I - ans) == 0 and abs(tot - float(ans)) < 1e-3
    report(2, ok, f"integral/3={sp.simplify(I)}, n={N} 합={tot:.6f}, md={ans}")


def p03():
    """점화 규칙으로 a_1..a_128 을 직접 생성해 합산."""
    a = {}
    for k in range(1, 129):
        a[k] = 0 if k % 2 == 1 else a[k // 2] + k // 2
    S = sum(a.values())
    ans = load_answer(3)
    report(3, S == ans, f"S_128={S}, md={ans}")


def p04():
    """f: A→A 6^6 개 전수 조사."""
    import itertools
    cnt = 0
    for f in itertools.product(range(1, 7), repeat=6):
        if all(v <= 3 for v in f) and all(f[f[i] - 1] == f[i] for i in range(6)):
            cnt += 1
    ans = load_answer(4)
    report(4, cnt == ans, f"개수={cnt}, md={ans}")


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
