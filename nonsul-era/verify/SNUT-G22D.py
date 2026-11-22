#!/usr/bin/env python3
"""서울과기대 2022학년도 논술전형고사 4차 (SNUT-G22D-01~03) — sympy/mpmath 검산. 정답 하드코딩 비교 없이 문제 조건에서 독립 재계산해 YAML answer 와 비교."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SNUT-G22D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, d, k = sp.symbols("a b d k", real=True)
theta, theta_k, alpha = sp.symbols("theta theta_k alpha", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "d": d, "k": k, "theta": theta,
       "theta_k": theta_k, "alpha": alpha, "pi": sp.pi, "E": sp.E}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d_ = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d_["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def same(u, v, tol=1e-9):
    """수치 동일성(상대오차 tol)."""
    return abs(complex(sp.N(u, 30)) - complex(sp.N(v, 30))) <= tol * max(1, abs(complex(sp.N(v, 30))))

def p01():
    # 1.1 x>0 에서 |x^3-2x|=x^2 의 해와 각 점에서의 접선(구간별 도함수)
    sols = []
    for expr in (-(x ** 3 - 2 * x) - x ** 2, (x ** 3 - 2 * x) - x ** 2):
        for s in sp.solve(expr, x):
            if s.is_real and s > 0 and sp.Abs(s ** 3 - 2 * s) == s ** 2:
                sols.append(s)
    sols = sorted(set(sols))
    parts = []
    for s in sols:
        sg = sp.sign(s ** 3 - 2 * s)                       # f = sg*(x^3-2x) 근방에서
        m = sg * (3 * s ** 2 - 2)
        c = s ** 2 - m * s
        txt = f"({s},{s ** 2}): y={m}x{'+' if c >= 0 else ''}{c}".replace("1x", "x") if abs(m) == 1 else f"({s},{s ** 2}): y={m}x{'+' if c >= 0 else ''}{c}"
        parts.append(txt.replace("-1x", "-x"))
    got = ", ".join(parts)
    ans = load_answer(1, 0, raw=True)
    report(1, got.replace(" ", "") == ans.replace(" ", ""), f"1.1 교점 {sols}, 접선 {got}, yaml={ans}")
    # 1.2 합이 n^2+5n 이 되도록 a_(k+1)/a_k 를 차로 정하고, 그 수열이 모든 n 에서 조건을 만족하는지 확인
    S = lambda nn: nn * nn + 5 * nn
    term = lambda kk: S(kk) - S(kk - 1)
    ratio = lambda kk: sp.Rational(kk + 1, kk + 2) * term(kk)            # a_(k+1)/a_k
    A = [None, sp.Integer(1)]
    for kk in range(1, 40):
        A.append(A[kk] * ratio(kk))
    cond = all(sum(sp.Rational(kk + 2, kk + 1) * A[kk + 1] / A[kk] for kk in range(1, nn + 1)) == S(nn) for nn in range(1, 39))
    val = A[10] / A[8]
    ans = load_answer(1, 1)
    report(1, cond and val == ans, f"1.2 a10/a8={val} (조건 n=1..38 확인), yaml={ans}")
    # 1.3
    pp = sp.symbols("pp", real=True)
    ps = [s for s in sp.solve(sp.Eq(1 - sp.Rational(1, 2) ** 4, 1 - (1 - pp) ** 2), pp) if 0 <= s <= 1]
    # 이항분포 직접 합
    pa = sum(sp.binomial(4, j) * sp.Rational(1, 2) ** 4 for j in range(1, 5))
    pb = sum(sp.binomial(2, j) * ps[0] ** j * (1 - ps[0]) ** (2 - j) for j in range(1, 3))
    ans = load_answer(1, 2)
    report(1, len(ps) == 1 and pa == pb and ps[0] == ans, f"1.3 p={ps}, 1회 이상 확률 A={pa}, B={pb}, yaml={ans}")


def p02():
    # 2.1
    f = sp.diff(sp.exp(x), x).subs(x, 0) * x + 1
    g = sp.exp(x) - f
    crit = sp.solve(sp.diff(g, x), x)
    mn = min(sp.N(g.subs(x, sp.Rational(i, 10))) for i in range(-60, 61))
    ans = load_answer(2, 0)
    report(2, f == x + 1 and crit == [0] and g.subs(x, 0) == ans and mn >= -1e-12, f"2.1 f={f}, 임계점 {crit}, 최솟값={g.subs(x, 0)}, 격자 최소={mn}, yaml={ans}")
    # 2.2
    h = 2 * x ** 2 + x + 1 - sp.exp(x)
    grid = [sp.Rational(i, 1000) for i in range(0, 1001)]
    vals = [sp.N(h.subs(x, s)) for s in grid]
    ans = load_answer(2, 1)
    report(2, abs(min(vals)) < 1e-12 and sp.N(h.subs(x, 0)) == 0 and all(v >= -1e-12 for v in vals) and ans == 0, f"2.2 [0,1] 격자 최소={min(vals)} (x=0), h''=4-e^x>0 on [0,1]: {all(sp.N(4 - sp.exp(s)) > 0 for s in grid)}")
    # 2.3 [증명-수치확인] n=1..300
    mp.mp.dps = 30
    ok = all(mp.e ** (mp.mpf(1) / m) <= 1 + mp.mpf(1) / m + mp.mpf(2) / m ** 2 for m in range(1, 301))
    ans = load_answer(2, 2, raw=True)
    report(2, ok and ans is None, "2.3 [증명-수치확인] n=1..300 에서 e^(1/n) ≤ 1+1/n+2/n^2")
    # 2.4
    L = sp.limit(n * (sp.exp(1 / n) - 1), n, sp.oo)
    ok = all(1 <= m * (mp.e ** (mp.mpf(1) / m) - 1) <= 1 + mp.mpf(2) / m for m in range(1, 301))
    ans = load_answer(2, 3)
    report(2, ok and L == ans, f"2.4 극한={L}, 조임 1≤c_n≤1+2/n (n≤300 확인), yaml={ans}")


def p03():
    import math
    C = (-1.0, 0.0); D = (3.0, 0.0)
    def Rk(th):
        P = (math.cos(th), math.sin(th))
        # 직선 C-P 와 y=-x 의 교점: C+s(P-C), y=-x
        s = -(C[0] + C[1]) / ((P[0] - C[0]) + (P[1] - C[1]))
        return (C[0] + s * (P[0] - C[0]), C[1] + s * (P[1] - C[1]))
    # 3.1
    ex = load_answer(3, 0)
    ok = True
    for th in [0.3, 0.8, 1.4, 2.0, 2.3]:
        R = Rk(th)
        Lk = math.hypot(*R)
        ok = ok and abs(R[0] + R[1]) < 1e-12 and abs(Lk - float(ex.subs(theta_k, th))) < 1e-12
    report(3, ok, "3.1 직선 CP_k 와 y=-x 의 교점 R_k 에서 OR_k = yaml 식 (θ 5개 표본)")
    # 3.2
    nn = 3
    tot = 0.0
    for kk in range(1, nn + 1):
        th = 3 * math.pi * kk / (4 * nn)
        R = Rk(th)
        L2 = R[0] ** 2 + R[1] ** 2
        M2 = (R[0] - C[0]) ** 2 + (R[1] - C[1]) ** 2
        tot += M2 - L2
    ans = load_answer(3, 1)
    report(3, abs(tot - float(ans)) < 1e-12, f"3.2 Σ(M²-L²)={tot:.3e}, yaml={ans}")
    # 3.3 삼각형 CDP_k 의 넓이를 좌표로, n 을 키워 수렴 확인 + 정적분
    def Tn_over_n(N):
        s = 0.0
        for kk in range(1, N + 1):
            th = 3 * math.pi * kk / (4 * N)
            P = (math.cos(th), math.sin(th))
            s += abs((D[0] - C[0]) * (P[1] - C[1]) - (D[1] - C[1]) * (P[0] - C[0])) / 2
        return s / N
    v1, v2 = Tn_over_n(20000), Tn_over_n(200000)
    I = sp.integrate(2 * sp.sin(3 * sp.pi * x / 4), (x, 0, 1))
    ans = load_answer(3, 2)
    report(3, abs(v2 - float(ans)) < 1e-4 and abs(v2 - float(ans)) < abs(v1 - float(ans)) + 1e-12 and sp.simplify(I - ans) == 0, f"3.3 T_n/n(n=2e5)={v2:.8f}, ∫={sp.simplify(I)}, yaml={ans}≈{sp.N(ans, 8)}")


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
