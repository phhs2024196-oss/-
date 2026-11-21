#!/usr/bin/env python3
"""SSU-M26 (숭실대 2026학년도 모의논술고사(자연) 1~4번) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M26"
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
    """P, Q 를 같은 높이의 다른 교점으로 직접 풀고 PQ^2 를 a 에 대해 최소화."""
    a = sp.symbols("a", real=True)
    X = sp.symbols("X", real=True)
    c1 = X**2 + X + 1
    c2 = -X**2 + 4 * X - 3
    pa = [r for r in sp.solve(sp.Eq(c1, c1.subs(X, a)), X) if sp.simplify(r - a) != 0][0]
    qa = [r for r in sp.solve(sp.Eq(c2, c2.subs(X, a)), X) if sp.simplify(r - a) != 0][0]
    D2 = sp.expand((pa - qa) ** 2 + (c1.subs(X, a) - c2.subs(X, a)) ** 2)
    crit = [c for c in sp.solve(sp.diff(D2, a), a) if c.is_real]
    best = min(crit, key=lambda c: D2.subs(a, c))
    assert pa.subs(a, best) != best and qa.subs(a, best) != best
    ans = load_answer(1)
    report(1, sp.simplify(best - ans.rhs) == 0, f"P_x={pa}, Q_x={qa}, 임계점={crit}, 최소 a={best}, md={ans}")


def p02():
    """n=10^3..10^5 에서 정의대로 수치 계산: sqrt(n)*합 이 0 으로 감소하고 n*합 → ∫(f')^2 (= 상수) 확인."""
    import mpmath as mp
    mp.mp.dps = 30
    vals = []
    for N in (1000, 10000, 100000):
        s = mp.fsum((mp.e ** (mp.mpf(k + 1) / N) ** 2 - mp.e ** (mp.mpf(k) / N) ** 2) ** 2 for k in range(N))
        vals.append((N, mp.sqrt(N) * s, N * s))
    J = mp.quad(lambda X: (2 * X * mp.e ** (X**2)) ** 2, [0, 1])
    trend = vals[0][1] > vals[1][1] > vals[2][1] and abs(vals[2][2] - J) < 1e-3
    ans = load_answer(2)
    ok = trend and ans == 0 and vals[2][1] < 4 * mp.e**2 / mp.sqrt(100000)
    report(2, ok, f"sqrt(n)*합: " + ", ".join(f"n={N}:{mp.nstr(v,6)}" for N, v, _ in vals) + f"; n*합→{mp.nstr(J,6)}, md={ans}")


def _T_pieces():
    """f(x)=x^2-17x+30 의 부호 구간과 그 구간의 변환(선형)."""
    X = sp.symbols("X", real=True)
    f = X**2 - 17 * X + 30
    neg = sp.solve_univariate_inequality(f < 0, X, relational=False)
    nonneg = sp.Complement(sp.S.Reals, neg)
    return [(nonneg, lambda v: 2 * v, lambda v: v / 2), (neg, lambda v: v + 1, lambda v: v - 1)]


def _image(S):
    X = sp.symbols("X", real=True)
    return sp.Union(*[sp.imageset(sp.Lambda(X, g(X)), sp.Intersection(S, P)) for P, g, _ in _T_pieces()])


def _preimage(S):
    X = sp.symbols("X", real=True)
    return sp.Union(*[sp.Intersection(P, sp.imageset(sp.Lambda(X, gi(X)), S)) for P, _, gi in _T_pieces()])


def p03():
    """(1) 실수 전체의 상(像)을 구간 연산으로 (2) a_4≥32, a_3<32 에서 역상을 세 번 + 무작위 수치 확인."""
    import random
    A2 = _image(sp.S.Reals)
    ans1 = load_answer(3, 0)
    ok1 = sp.simplify(sp.Complement(A2, ans1)) == sp.S.EmptySet and sp.simplify(sp.Complement(ans1, A2)) == sp.S.EmptySet
    S3 = sp.Intersection(sp.Interval.open(-sp.oo, 32), _preimage(sp.Interval(32, sp.oo)))
    S1 = _preimage(_preimage(S3))
    ans2 = load_answer(3, 1)
    ok2 = S1 == ans2
    T = lambda v: 2 * v if (v - 2) * (v - 15) >= 0 else v + 1
    random.seed(1)
    bad = 0
    for _ in range(20000):
        v = sp.Rational(random.randint(-4000, 4000), 100)
        a2 = T(v); a3 = T(a2); a4 = T(a3)
        if (a3 < 32 <= a4) != bool(ans2.contains(v)):
            bad += 1
    for v in (sp.Integer(14), sp.Integer(15), sp.Rational(1499, 100)):   # 경계
        a3 = T(T(v)); a4 = T(a3)
        if (a3 < 32 <= a4) != bool(ans2.contains(v)):
            bad += 1
    report(3, ok1 and ok2 and bad == 0, f"(1) a2 범위={A2} md={ans1} / (2) a1 범위={S1} md={ans2}, 수치 불일치={bad}")


def p04():
    """f: A→A 4^4 개 전수 조사."""
    import itertools
    cnt = sum(1 for f in itertools.product(range(4), repeat=4) if all(f[f[f[i]]] == i for i in range(4)))
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
