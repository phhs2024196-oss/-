#!/usr/bin/env python3
"""DGU-G25 (2025학년도 동국대 논술 실전 기출) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-G25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1) cosθ=1/√((6k-1)²+4) (2) k=1/6, θ=π/3 (3) Q1,Q2=((3±√3)/6,(3±√3)/6)."""
    k = sp.symbols("k", real=True)
    X = sp.symbols("X")
    roots = sp.solve(sp.Eq(X**2 + (sp.Rational(1, 3) - 2 * k) * X + k, X), X)
    P = sp.Matrix([0, 1])
    u = sp.Matrix([roots[0], roots[0]]) - P
    w = sp.Matrix([roots[1], roots[1]]) - P
    disc = sp.discriminant(X**2 + (sp.Rational(1, 3) - 2 * k - 1) * X + k, X)
    disc_pos = sp.minimum(disc, k, sp.S.Reals) > 0
    claim = 1 / sp.sqrt((6 * k - 1)**2 + 4)
    import random
    ok1 = disc_pos
    for _ in range(40):
        kv = sp.Rational(random.randint(-3000, 3000), 997)
        cs = abs(u.dot(w)).subs(k, kv) / (u.norm() * w.norm()).subs(k, kv)
        ok1 &= abs(float(cs) - float(claim.subs(k, kv))) < 1e-12
    kstar = [c for c in sp.solve(sp.diff(claim, k), k)]
    ok2 = kstar == [sp.Rational(1, 6)] and sp.acos(claim.subs(k, kstar[0])) == sp.pi / 3
    q = sorted(sp.solve(X**2 - (sp.Rational(2, 3) + 2 * kstar[0]) * X + kstar[0], X), key=lambda r: float(r))
    ok3 = sp.simplify(q[0] - (3 - sp.sqrt(3)) / 6) == 0 and sp.simplify(q[1] - (3 + sp.sqrt(3)) / 6) == 0
    report(1, ok1 and ok2 and ok3, f"(1) 판별식 최소>0, 수치 40점 일치 (2) k={kstar}, θ=π/3 (3) Q=(x,x), x={q}")


def p02():
    """p1(과체중)=1/4, p2(비만)=1/5 유일 → z표 조회 → P(23<X≤27)=0.39."""
    p = sp.symbols("p", real=True)
    r1 = [r for r in sp.solve(3 * p * (1 - p)**2 - sp.Rational(27, 64), p) if 0 < r < 1]
    r2 = [r for r in sp.solve(3 * p * (1 - p)**2 - sp.Rational(48, 125), p) if 0 < r < 1]
    pairs = [(a, b) for a in r1 for b in r2 if a + b < sp.Rational(1, 2)]
    p1, p2 = pairs[0]
    table = {0.13: 0.05, 0.25: 0.10, 0.39: 0.15, 0.52: 0.20, 0.58: 0.22, 0.84: 0.30, 1.28: 0.40, 1.55: 0.44}
    inv = {v: z for z, v in table.items()}
    z25 = inv[round(float(sp.Rational(1, 2) - p2), 2)]          # (25-m)/σ
    z23 = inv[round(float(sp.Rational(1, 2) - p1 - p2), 2)]     # (23-m)/σ
    s_inv = (z25 - z23) / 2                                     # 1/σ
    z27 = round(z25 + 2 * s_inv, 2)
    ans = round(table[z27] - table[z23], 2)
    # 실제 정규분포로도 근사 확인
    from scipy.stats import norm
    sigma = 1 / s_inv
    m = 25 - z25 * sigma
    exact = norm.cdf(27, m, sigma) - norm.cdf(23, m, sigma)
    ok = len(pairs) == 1 and (p1, p2) == (sp.Rational(1, 4), sp.Rational(1, 5)) and abs(ans - 0.39) < 1e-9 and abs(exact - 0.39) < 0.01
    report(2, ok, f"근 r1={r1}, r2={r2}, 쌍={pairs}, z23={z23}, z25={z25}, z27={z27} → {ans} (정규 실제값 {exact:.4f})")


def p03():
    """(1) x=n,2n,6n (2) A_n=10n³ (3) S_n=(7/2+12ln(3/2))n³ (4) 40+960/7·ln(3/2)."""
    N = sp.symbols("n", positive=True)
    a, b = sp.symbols("alpha beta", positive=True)
    sol = [s for s in sp.solve([a * b / (a - b) - 20 * N, a + b - 9 * N], [a, b], dict=True) if (s[a] - s[b]).subs(N, 1) > 0]
    al, be = sol[0][a], sol[0][b]
    X = sp.symbols("X", positive=True)
    f1 = (X - al) * (X - be)
    f2 = 3 * al * be * (al - be) / (5 * X)
    xs = sorted(sp.solve(sp.Eq(f1, f2), X), key=lambda r: float(r.subs(N, 1)))
    ok1 = xs == [N, 2 * N, 6 * N]
    pts = [(xx, f2.subs(X, xx)) for xx in xs]
    A = sp.simplify(sp.Abs(sp.Matrix([[pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]], [pts[2][0] - pts[0][0], pts[2][1] - pts[0][1]]]).det()) / 2)
    ok2 = sp.simplify(A - 10 * N**3) == 0
    S = sp.integrate(f1 - f2, (X, xs[0], xs[1])) + sp.integrate(f2 - f1, (X, xs[1], xs[2]))
    # 부호 확인(수치): [n,2n]에서 f1≥f2, [2n,6n]에서 f2≥f1
    sgn = all(float((f1 - f2).subs({N: 1, X: v})) >= 0 for v in [1.1, 1.5, 1.9]) and all(float((f2 - f1).subs({N: 1, X: v})) >= 0 for v in [2.1, 4, 5.9])
    Sc = (sp.Rational(7, 2) + 12 * sp.log(sp.Rational(3, 2))) * N**3
    ok3 = sgn and sp.simplify(sp.expand_log(S - Sc, force=True)) == 0
    kk = sp.symbols("k", positive=True, integer=True)
    term = (Sc.subs(N, kk)) * (A.subs(N, 2 * kk))
    L = sp.limit(sp.summation(sp.expand(term), (kk, 1, N)) / N**7, N, sp.oo)
    Lc = 40 + sp.Rational(960, 7) * sp.log(sp.Rational(3, 2))
    ok4 = sp.simplify(sp.expand_log(L - Lc, force=True)) == 0
    report(3, ok1 and ok2 and ok3 and ok4, f"α={al}, β={be}, x={xs}, A_n={A}, S_n={sp.simplify(S)}, 극한={sp.simplify(L)}")


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
