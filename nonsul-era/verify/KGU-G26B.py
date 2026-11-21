#!/usr/bin/env python3
"""KGU-G26B (경기대 2026학년도 수리 B형, 문항 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KGU-G26B"
WIDTH = 2
RESULTS = []
R = sp.Rational


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sid):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KGU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    sub = next(s for s in d["subparts"] if str(s["id"]) == str(sid))
    return sub["answer"]["value"]


def simulate(seq):
    """seq: 'A'/'B' 문자열. 처음 환자 1명 기준 남은 비율(정확한 유리수)."""
    left, usedA = R(1), False
    for t in seq:
        if t == "A":
            left *= R(1, 3) if not usedA else R(2, 3)   # 처음 A: 2/3 회복, 이후 A: 1/3 회복
            usedA = True
        else:
            left *= R(1, 2)
    return left


def p01():
    """(1) 모든 A/B 순서(n≤8)를 직접 시뮬레이션해 md 식과 비교, (2) n≤14 전수 탐색으로 최소 비용."""
    a, n, k = sp.symbols("a n k")
    expr = sp.sympify(load_answer(1, "1"), locals={"a": a, "n": n, "k": k})
    ok1 = True
    for N in range(1, 9):
        for seq in itertools.product("AB", repeat=N):
            K = seq.count("A")
            if sp.simplify(expr.subs({n: N, k: K}) - a * simulate(seq)) != 0:
                ok1 = False
    # (2) 비용 10k+15(n-k) 만 원; 비용 ≥ 10n 이므로 n ≤ 14 면 충분(답 후보보다 큰 n 배제)
    best, arg = None, None
    for N in range(1, 15):
        for seq in itertools.product("AB", repeat=N):
            if simulate(seq) <= R(1, 27):
                cost = 10 * seq.count("A") + 15 * seq.count("B")
                if best is None or cost < best:
                    best, arg = cost, "".join(seq)
    assert best is not None and best < 10 * 15   # n≥15 이면 비용 ≥150 이므로 탐색 범위 충분
    # 주어진 근삿값(log2=0.30, log3=0.48)으로도 같은 결론인지: k≥1 이면 n≥0.4k+3.8, k=0 이면 n≥4.8
    import math
    approx = min(min(10 * kk + 15 * (math.ceil(max(kk, 0.4 * kk + 3.8 - 1e-12)) - kk) for kk in range(1, 20)),
                 15 * math.ceil(4.8))
    md2 = sp.sympify(load_answer(1, "2"))
    ok2 = best == md2 and approx == md2
    report(1, ok1 and ok2, f"(1) n≤8 전 순서 시뮬레이션 일치={ok1}; (2) 전수 최소={best}만원({arg}), 근삿값 로그 계산={approx}, md={md2}")


def p02():
    """(1) 넓이 9등분 조건으로 k 재계산, (2) 경계선을 따라 P 기준 누적 넓이로 A2~A9 를 정확히 구함."""
    x, kk = sp.symbols("x k", positive=True)
    P = (0, R(2, 3))
    # 포물선 위 점 (x,x^2) 에서 P 기준 면적 미분 = (x^2+2/3)/2 dx (반시계)
    sector1 = sp.integrate((x ** 2 + R(2, 3)) / 2, (x, 0, 1))          # PA1A2
    total = 2 * kk * sp.sqrt(kk) - sp.integrate(x ** 2, (x, -sp.sqrt(kk), sp.sqrt(kk)))
    ks = [s for s in sp.solve(sp.Eq(total, 9 * sector1), kk) if s > R(2, 3)]
    md1 = sp.sympify(load_answer(2, "1"))
    ok1 = len(ks) == 1 and ks[0] == md1
    K = ks[0]
    s = sp.sqrt(K)
    unit = sector1
    # 오른쪽 포물선 x∈[0,s] → 윗변 (s,K)→(-s,K) → 왼쪽 포물선 x∈[-s,0]
    right = sp.integrate((x ** 2 + R(2, 3)) / 2, (x, 0, s))
    top = (2 * s) * (K - P[1]) / 2
    assert sp.simplify(2 * right + top - 9 * unit) == 0
    pts = []
    t = sp.symbols("t", real=True)
    for j in range(1, 9):
        target = j * unit
        if target <= right:
            sol = [r for r in sp.solve(sp.Eq((t ** 3 / 3 + R(2, 3) * t) / 2, target), t)
                   if r.is_real and 0 <= r <= s]
            pts.append((sol[0], sol[0] ** 2))
        elif target <= right + top:
            u = (target - right) / ((K - P[1]) / 2)   # 윗변을 따라 이동한 거리
            pts.append((s - u, K))
        else:
            rem = 9 * unit - target                     # 왼쪽 포물선: 대칭
            sol = [r for r in sp.solve(sp.Eq((t ** 3 / 3 + R(2, 3) * t) / 2, rem), t) if r.is_real and 0 <= r <= s]
            pts.append((-sol[0], sol[0] ** 2))
    ok_a2 = pts[0] == (1, 1)
    on_line = {p for p in pts if p[1] == K}
    md2 = sp.sympify(load_answer(2, "2"))
    md2set = {tuple(sp.nsimplify(c) for c in p) for p in md2}
    ok2 = {tuple(p) for p in on_line} == md2set
    a3 = sp.N(pts[1][0], 6)
    report(2, ok1 and ok_a2 and ok2, f"(1) k={K}, md={md1}; A2={pts[0]}; (2) y=k 위 점={sorted(on_line)}, A3 x≈{a3}(곡선 위), md 일치={ok2}")


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
