#!/usr/bin/env python3
"""KGU-G25A (경기대 2025학년도 논술우수자전형 수리 A형 문항 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "KGU-G25A"
WIDTH = 2
RESULTS = []
n, k = sp.symbols("n k", integer=True, positive=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KGU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    s = next(s for s in d["subparts"] if str(s["id"]) == str(sub))
    return sp.sympify(s["answer"]["value"], locals={"n": n})


def p01():
    """a_n=3^n, f=log_3, 삼각형 T_n 을 좌표로 직접 구성해 높이·넓이를 구하고 합을 n=2..30 및 기호합으로 비교."""
    a = lambda m: 3 ** m
    f = lambda x: sp.log(x, 3)
    ok = True
    ans1, ans2 = load_answer(1, 1), load_answer(1, 2)
    for N in range(2, 31):
        s1 = s2 = 0
        for kk in range(2, N + 1):
            P, Q, R = sp.Point(a(kk), f(a(kk))), sp.Point(a(kk - 1), 0), sp.Point(a(kk + 1), 0)
            area = abs(sp.Polygon(P, Q, R).area)
            h = 2 * area / Q.distance(R)          # 밑변 QR 에 대한 높이
            s1 += sp.nsimplify(sp.simplify(h)) ** 2
            s2 += sp.simplify(area / kk)
        ok &= sp.simplify(s1 - ans1.subs(n, N)) == 0 and sp.simplify(s2 - ans2.subs(n, N)) == 0
    sym1 = sp.summation(k ** 2, (k, 2, n))
    sym2 = sp.summation(sp.Rational(1, 2) * (3 ** (k + 1) - 3 ** (k - 1)), (k, 2, n))   # A_k/k = 밑변/2 (높이 k)
    ok &= sp.simplify(sym1 - ans1) == 0 and sp.simplify(sym2 - ans2) == 0
    report(1, ok, f"(1) {sp.factor(sym1)} (2) {sp.simplify(sym2)} | n=2..30 좌표 직접 계산 일치, md=({ans1}, {ans2})")


def p02():
    """(1) 넓이 50*|sin θ|... 실제 끼인각으로 넓이 계산, θ∈[0,4π] 해집합을 solveset 으로. (2) 시침·분침 각을 시간 함수로 두고 측도 적분."""
    th, t = sp.symbols("theta t", real=True)
    ans1, ans2 = load_answer(2, 1), load_answer(2, 2)
    area = sp.Rational(1, 2) * 10 * 20 * sp.Abs(sp.sin(th))       # 끼인각 θ 또는 2π-θ → |sin θ|
    S = sp.Interval(0, 4 * sp.pi)
    # |sin θ| = 1/2 의 해(경계 후보)를 [0,4π] 에서 nsolve 없이 asin 으로 모아 구간을 만들고 중점 판정
    base = [sp.asin(sp.Rational(1, 2)), sp.pi - sp.asin(sp.Rational(1, 2))]
    cuts = sorted({sp.Integer(0), 4 * sp.pi} | {c + j * sp.pi for c in base for j in range(4) if 0 <= c + j * sp.pi <= 4 * sp.pi},
                  key=lambda v: float(v))
    pieces = [sp.Interval(x0, x1) for x0, x1 in zip(cuts, cuts[1:]) if area.subs(th, (x0 + x1) / 2).evalf() >= 50]
    sol = sp.Union(*pieces)
    ok1 = sol == ans1
    # 수치 확인
    import random
    for _ in range(2000):
        v = random.uniform(0, float(4 * sp.pi))
        if (float(area.subs(th, v)) >= 50) != bool(ans1.contains(sp.Float(v))):
            ok1 = False
    # (2) t: 오후 1시부터 경과 시간(시). 시침 각(12시 방향 기준, 시계방향) = π/6*(1+t), 분침 = 2π t
    rel = 2 * sp.pi * t - sp.pi / 6 * (1 + t)
    w = sp.diff(rel, t)                                   # 상대 각속도(상수)
    r0, r1 = rel.subs(t, 0), rel.subs(t, 5)
    # |sin|=1/2 이 되는 상대각 jπ±π/6 들을 시간으로 바꿔 구간 경계로 쓰고, 각 구간 중점에서 조건 판정(정확한 유리수 계산)
    cuts = {sp.Integer(0), sp.Integer(5)}
    for j in range(-2, 12):
        for c in (j * sp.pi + sp.pi / 6, j * sp.pi - sp.pi / 6):
            tc = sp.nsimplify((c - r0) / w)
            if 0 < tc < 5:
                cuts.add(tc)
    cuts = sorted(cuts)
    tot = sum((b - a) for a, b in zip(cuts, cuts[1:])
              if sp.Abs(sp.sin(rel.subs(t, (a + b) / 2))).evalf(30) >= sp.Rational(1, 2))
    tot = sp.nsimplify(tot)
    ok2 = sp.simplify(tot - ans2) == 0
    report(2, ok1 and ok2, f"(1) 해집합={sol} (2) 총 시간={tot}h (상대각 {r0}→{r1}), md=({ans1}, {ans2})")


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
