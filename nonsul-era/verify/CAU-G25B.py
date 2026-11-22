#!/usr/bin/env python3
"""CAU-G25B (2025학년도 중앙대 수시 논술 자연계열Ⅱ) — sympy/전수 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys
from fractions import Fraction as Fr

import mpmath as mp
import sympy as sp

PREFIX = "CAU-G25B"
WIDTH = 2
RESULTS = []
x, y, t, n = sp.symbols("x y t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """규칙 (가)~(라)를 그대로 따라가며 제거 1회·2회의 상금 기댓값을 a 의 식으로 전수 계산
    → E(Y) ≥ E(X) 인 0<a<1 의 최댓값 (답 1/5)."""
    a = sp.symbols("a", positive=True)

    def expect(times):
        # 상태: (A의 공 수 rA, B의 파란 공 수), 확률
        states = {(1, 5): sp.Integer(1)}
        for k in range(1, times + 1):
            new = {}
            for (rA, bB), pr in states.items():
                if rA == 0:
                    opts = [((0, bB - 1), 1)]
                else:
                    opts = [((0, bB), a / k), ((rA, bB - 1), 1 - a / k)]
                for st, w in opts:
                    new[st] = new.get(st, 0) + pr * w
            states = new
        E = 0
        for (rA, bB), pr in states.items():
            E += pr * (100 * sp.Rational(rA, rA + bB)) - pr * 2 * times
        return sp.expand(E)
    EX, EY = expect(1), expect(2)
    sol = sp.solve_univariate_inequality(EY - EX >= 0, a, relational=False)
    cand = sol.intersect(sp.Interval.open(0, 1))
    mx = cand.sup
    ok = mx == sp.Rational(1, 5) and mx in cand
    report(1, ok, f"E(X)={EX}, E(Y)={EY}, 해집합∩(0,1)={cand}")


def p02():
    """(1) 삼차함수 f 를 미지수 4개로 두고 (가)(나)(다) 를 그대로 연립 (답 f'(1/3)=-70/3)
    (2) t 마다 교점을 수치로 구해 S(t) 를 적분 (답 19√3/96+√3(e^{-1/2}-e^{-2}))."""
    p, q, r = sp.symbols("p q r", real=True)
    f = x**3 + p * x**2 + q * x + r
    fp = sp.diff(f, x)
    base = sp.solve([sp.Eq(f.subs(x, 1), 1), sp.Eq(fp.subs(x, 1), 0)], [q, r], dict=True)[0]   # (가): f(1)=1 & 미분계수 0
    f1 = f.subs(base)
    T = f1.subs(x, -1) + sp.diff(f1, x).subs(x, -1) * (x + 1)
    area = sp.integrate(f1 - T, (x, -1, 0))
    res = []
    for ps in sp.solve(sp.Eq(area, sp.Rational(53, 12)), p):
        ff = f1.subs(p, ps)
        Tt = T.subs(p, ps)
        if sp.diff(ff, x, 2).subs(x, -1) <= 0:
            continue
        # 접선이 [-1,0] 에서 곡선 아래(넓이 = 적분) 수치 확인
        if not all(float((ff - Tt).subs(x, v)) >= -1e-12 for v in [-1 + i / 100 for i in range(101)]):
            continue
        res.append(sp.diff(ff, x).subs(x, sp.Rational(1, 3)))
    # (가) 극한이 실제로 0 인지
    lim_ok = all(sp.limit((sp.exp(f1.subs(p, ps)) - sp.E) / (x - 1), x, 1) == 0 for ps in sp.solve(sp.Eq(area, sp.Rational(53, 12)), p))
    ok1 = res == [sp.Rational(-70, 3)] and lim_ok

    mp.mp.dps = 30

    def S(tt):
        c1 = 1 / (tt + 1)
        c0 = mp.e ** ((tt**2 - 2) / (tt + 1))
        disc = c1**2 + 4 * c0
        a1 = (c1 + mp.sqrt(disc)) / 2
        a2 = (c1 - mp.sqrt(disc)) / 2
        PQ2 = (a1 - a2) ** 2 + (a1**2 - a2**2) ** 2
        return mp.sqrt(3) / 4 * PQ2
    val = mp.quad(S, [0, 1])
    target = 19 * mp.sqrt(3) / 96 + mp.sqrt(3) * (mp.e ** (-mp.mpf(1) / 2) - mp.e ** (-2))
    ok2 = abs(val - target) < 1e-20
    report(2, ok1 and ok2, f"(1) f'(1/3)={res}  (2) ∫S≈{mp.nstr(val, 15)}")


def p03():
    """(1) a_n=∫_1^e (ln x)^n dx 를 수치 적분해 합을 직접 계산 (답 e(1/ln(15/14)-1/ln15))
    (2) g(t)=min_x ln PA+ln PB 를 수치 최소화해 ∫2t g(t)dt (답 5ln5-ln2-27/8)."""
    mp.mp.dps = 30
    a = [None] + [mp.quad(lambda u: mp.log(u) ** k, [1, mp.e]) for k in range(1, 16)]
    tot = mp.mpf(0)
    for k in range(1, 14):
        tot += (a[k + 1] + (k + 1) * a[k]) * (mp.log(k + 1) - mp.log(k)) / (mp.log(mp.mpf(k) / 15) * mp.log(mp.mpf(k + 1) / 15))
    target1 = mp.e * (1 / mp.log(mp.mpf(15) / 14) - 1 / mp.log(15))
    ok1 = abs(tot - target1) < 1e-20

    mp.mp.dps = 20

    def g(tt):
        h = lambda xx: (mp.log(mp.sqrt((xx - 1) ** 2 + tt**2)) + mp.log(mp.sqrt((xx + 1) ** 2 + tt**2)))
        lo, hi = mp.mpf(0), mp.mpf(3)          # x≥0 에서 단봉(대칭)
        gr = (mp.sqrt(5) - 1) / 2
        for _ in range(120):
            m1 = hi - gr * (hi - lo); m2 = lo + gr * (hi - lo)
            if h(m1) < h(m2):
                hi = m2
            else:
                lo = m1
        return h((lo + hi) / 2)
    val = mp.quad(lambda tt: 2 * tt * g(tt), [0.5, 1, 2])
    target2 = 5 * mp.log(5) - mp.log(2) - mp.mpf(27) / 8
    ok2 = abs(val - target2) < 1e-12
    report(3, ok1 and ok2, f"(1) 합≈{mp.nstr(tot, 15)}  (2) 적분≈{mp.nstr(val, 15)}")


def p04():
    """(1) P(4cosθ,2sinθ) 의 접선 절편으로 QR 최소화(수치), 그때 △PRF' (답 4(√2+1))
    (2) 정수점 P 전수 × 원 위 Q 를 최대화(OP·OQ = OP·OZ + |OP|cos) (답 45+3√5)."""
    mp.mp.dps = 30
    def QR(th):
        X0, Y0 = 4 * mp.cos(th), 2 * mp.sin(th)
        qx, ry = 16 / X0, 4 / Y0
        return qx**2 + ry**2
    th = mp.findroot(lambda s: mp.diff(QR, s), 0.8)
    X0, Y0 = 4 * mp.cos(th), 2 * mp.sin(th)
    c = mp.sqrt(12)
    P = (X0, Y0); R = (0, 4 / Y0); Fp = (-c, 0)
    area = abs((R[0] - P[0]) * (Fp[1] - P[1]) - (Fp[0] - P[0]) * (R[1] - P[1])) / 2
    ok1 = abs(area - 4 * (mp.sqrt(2) + 1)) < 1e-20

    best = None
    for m in range(-10, 20):
        for nn in range(-10, 20):
            if (m - 5) ** 2 + (nn - 4) ** 2 < 9:
                # Q=(-1+cos φ, 8+sin φ): OP·OQ = -m+8n + m cos φ + n sin φ, 최댓값 = -m+8n+|OP|
                v = -m + 8 * nn + mp.sqrt(m * m + nn * nn)
                # 격자 수치 확인
                grid = max(m * (-1 + mp.cos(2 * mp.pi * i / 2000)) + nn * (8 + mp.sin(2 * mp.pi * i / 2000)) for i in range(2000))
                assert grid <= v + 1e-12 and v - grid < 1e-3
                if best is None or v > best[0]:
                    best = (v, m, nn)
    ok2 = abs(best[0] - (45 + 3 * mp.sqrt(5))) < 1e-20 and best[1:] == (3, 6)
    report(4, ok1 and ok2, f"(1) θ={mp.nstr(th, 10)}, △PRF'≈{mp.nstr(area, 15)}  (2) 최대 {mp.nstr(best[0], 15)} at P{best[1:]}")


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
