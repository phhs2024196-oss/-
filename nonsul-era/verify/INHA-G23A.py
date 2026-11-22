#!/usr/bin/env python3
"""INHA-G23A (2023학년도 인하대 자연계열 논술 오전 일반) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys
from fractions import Fraction as Fr

import numpy as np
import sympy as sp

PREFIX = "INHA-G23A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1-1) 넓이 k^3/6 (1-2) S=(m^2+51)^(3/2)/6 (1-3) 유리수 되는 m = {7,25}."""
    al, k = sp.symbols("alpha k", positive=True)
    be = al + k
    A1 = sp.simplify(-sp.integrate((x - al) * (x - be), (x, al, be)))
    ok1 = sp.simplify(A1 - k**3 / 6) == 0
    m = sp.symbols("m", positive=True)
    r = sp.solve(sp.Eq(x**2, m * x + sp.Rational(51, 4)), x)
    lo, hi = sorted(r, key=lambda e: float(e.subs(m, 1)))
    S = sp.simplify(sp.integrate(m * x + sp.Rational(51, 4) - x**2, (x, lo, hi)))
    ok2 = sp.simplify(S - (m**2 + 51) ** sp.Rational(3, 2) / 6) == 0
    # (1-3) 전수: m=1..10000 에서 S 유리수 ⇔ m^2+51 완전제곱 (큰 m 은 (m+1)^2-m^2=2m+1>51 이라 불가)
    sols = [mm for mm in range(1, 10001) if sp.nsimplify(S.subs(m, mm)).is_rational]
    ok3 = sols == [7, 25]
    report(1, ok1 and ok2 and ok3, f"(1-1) {A1} (1-2) S={S} (1-3) m={sols}")


def p02():
    """(2-1) 접선 y=0, y=-ex+2e (2-2) 16-2e^2 (2-3) t=1/2 접선, S=27√e/4-2e^2+17/e."""
    f = (x - 2) ** 2 * sp.exp(x)
    fp = sp.diff(f, x)
    ts = sp.solve(sp.Eq(-f.subs(x, t), fp.subs(x, t) * (2 - t)), t)
    lines = {sp.expand(f.subs(x, T) + fp.subs(x, T) * (x - T)) for T in ts}
    ok1 = lines == {sp.Integer(0), sp.expand(-sp.E * x + 2 * sp.E)}
    a, b = sorted(sp.solve(fp, x))
    S2 = sp.simplify(sp.integrate(6 - 3 * x - f, (x, a, b)))
    xs = np.linspace(float(a), float(b), 20001)
    ok2 = sp.simplify(S2 - (16 - 2 * sp.E**2)) == 0 and (6 - 3 * xs - (xs - 2) ** 2 * np.exp(xs)).min() >= -1e-12
    h = sp.integrate(f.subs(x, t) + fp.subs(x, t) * (x - t) - f, (x, -1, 2))
    crit = [c for c in sp.solve(sp.diff(h, t), t) if -1 <= c <= 1]
    tg = np.linspace(-1, 1, 4001)
    hv = sp.lambdify(t, h, "numpy")(tg)
    T = crit[0]
    g = f.subs(x, T) + fp.subs(x, T) * (x - T)
    Smin = sp.simplify(sp.integrate(g - f, (x, -1, 2)))
    xs = np.linspace(-1, 2, 30001)
    above = (sp.lambdify(x, g - f, "numpy")(xs)).min() >= -1e-12
    ok3 = (len(crit) == 1 and T == sp.Rational(1, 2) and abs(tg[hv.argmin()] - 0.5) < 1e-3 and above
           and sp.simplify(Smin - (sp.Rational(27, 4) * sp.sqrt(sp.E) - 2 * sp.E**2 + 17 / sp.E)) == 0
           and sp.simplify(sp.expand(g) - (-sp.Rational(3, 4) * sp.sqrt(sp.E) * x + sp.Rational(21, 8) * sp.sqrt(sp.E))) == 0)
    report(2, ok1 and ok2 and ok3, f"(2-1) {lines} (2-2) a,b={a},{b} S={S2} (2-3) t={T}, S={Smin}≈{float(Smin):.6f}")


def _f3(v):
    return -v * v + 4 * v if v >= 0 else v * v / 4


def _g3(v):          # g(t) = lim_{x→t+} f'(x)
    return 4 - 2 * v if v >= 0 else v / 2


def p03():
    """(3-1) a<-4 또는 a>4 (3-2) 0≤k<1 (3-3) (a,b)=(-4,2),(-3,1),(-2,1),(-1,1)."""
    # (3-1) 격자: 조건 ⇔ f(a) < inf_{s<a} f(s) 그리고 f(a) > sup_{t>a} f(t) (격자 근사, 경계 ±4 확인)
    grid = [Fr(i, 20) for i in range(-200, 201)]
    def cond(a):
        left = [_f3(s) for s in grid if s < a] or [float("inf")]
        right = [_f3(u) for u in grid if u > a]
        return all(_f3(a) < v for v in left) and all(_f3(a) > v for v in right)
    test = [Fr(i, 4) for i in range(-36, 37)]
    got = [a for a in test if cond(a)]
    exp_set = [a for a in test if a < -4 or a > 4]
    # 가장자리(격자 밖 극한 거동): x→-∞ f→∞, x→∞ f→-∞ 이므로 격자 [-10,10]에서 판정 충분
    ok1 = got == exp_set
    # (3-2) |g-k| 의 최솟값 존재 여부: 값의 집합 V=[-1/2,0)∪[2,4], 폐포에만 있는 점은 0
    def no_min(k):
        # V 안에서 실제로 달성 가능한 최소 거리
        cand = []
        if k < Fr(-1, 2):
            cand.append(Fr(-1, 2) - k)
        elif k < 0:
            cand.append(Fr(0))
        cand.append(0 if 2 <= k <= 4 else min(abs(k - 2), abs(k - 4)))
        inf_all = min(cand + [abs(k)])
        return min(cand) > inf_all          # 하한이 0(=폐포점)에서만 근접 → 최솟값 없음
    ks = [Fr(i, 40) for i in range(-200, 241)]
    ok2 = [k for k in ks if no_min(k)] == [k for k in ks if 0 <= k < 1]
    # (3-3) 정수 a<b 전수(-30..30): 평균변화율 = g(c) 인 c∈(a,b) 가 없는 쌍
    pairs = []
    for a in range(-30, 31):
        for b in range(a + 1, 31):
            s = Fr(_f3(Fr(b)) - _f3(Fr(a)), b - a)
            exists = False
            c = 2 * s                       # c<0 가지: c/2=s
            if c < 0 and a < c < b:
                exists = True
            c = (4 - s) / 2                 # c≥0 가지: 4-2c=s
            if c >= 0 and a < c < b:
                exists = True
            if not exists:
                pairs.append((a, b))
    ok3 = sorted(pairs) == [(-4, 2), (-3, 1), (-2, 1), (-1, 1)]
    report(3, ok1 and ok2 and ok3, f"(3-1) 격자 판정 일치={ok1} (3-2) 0≤k<1 일치={ok2} (3-3) {pairs}")


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
