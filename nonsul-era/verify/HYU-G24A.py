#!/usr/bin/env python3
"""HYU-G24A (한양대 2024학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction
from math import comb, factorial

import sympy as sp

PREFIX = "HYU-G24A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) 등비수열: a r/(1-r^2) = -4, |a|/(1-|r|) = 20, |r|<1 — 부호 4경우 전부 풀기
    a, r = sp.symbols("a r", real=True)
    sums = set()
    for sa, sr in itertools.product((1, -1), repeat=2):
        for sol in sp.solve([a * r + 4 * (1 - r**2), sa * a - 20 * (1 - sr * r)], [a, r], dict=True):
            av, rv = sol[a], sol[r]
            if av * sa > 0 and rv * sr > 0 and abs(rv) < 1:
                sums.add(sp.nsimplify(av * rv**2 / (1 - rv**4)))
    ok1 = sums == set(sp.sympify(md_answer(1, "1")))
    # (2) f=x^2+px+q, 조건 전수: 수치로 (p,q) 풀기
    p, q = sp.symbols("p q", real=True)
    f = x**2 + p * x + q
    cands = []
    for s6 in (1, -1):        # |f| 의 x=6 근방 부호
        h0 = s6 * (sp.diff(f, x).subs(x, 6) * (0 - 6) + f.subs(x, 6))
        # h(0)=|f(0)|: q>0(α>0 근거) 가정 없이 양부호 모두 시도
        for sq in (1, -1):
            sol_q = sp.solve(sp.Eq(h0, sq * q), q)
            for qv in sol_q:
                if sq * qv <= 0:
                    continue
                al = sp.symbols("al", real=True)
                # α 는 f 의 작은 근, p = -(α+qv/α)
                pexpr = -(al + qv / al)
                ff = x**2 + pexpr * x + qv
                for av in sp.solve(sp.integrate(ff, (x, 0, al)) - sp.Rational(50, 3), al):
                    if not av.is_real or av <= 0:
                        continue
                    be = qv / av
                    if not be > av:
                        continue
                    fx = ff.subs(al, av)
                    if s6 * fx.subs(x, 6) > 0:
                        cands.append(sp.nsimplify(sp.simplify(fx.subs(x, 10))))
    ok2 = len(set(cands)) == 1 and cands[0] == sp.sympify(md_answer(1, "2"))
    # (3) g 의 극댓값(구간 ((2n-1)π/2,(2n+1)π/2) 의 최대) 수치 → 급수 합
    import mpmath as mp
    mp.mp.dps = 30
    g = lambda t: mp.e**(-t) * abs(mp.cos(t))
    maxima = []
    for nn in range(1, 30):
        lo, hi = (2 * nn - 1) * mp.pi / 2, (2 * nn + 1) * mp.pi / 2
        tm = mp.findroot(lambda t: mp.cos(t) + mp.sin(t), (lo + hi) / 2)
        maxima.append(g(tm))
    dec = all(maxima[i + 1] < maxima[i] for i in range(len(maxima) - 1)) and maxima[0] < 1
    ratio = maxima[1] / maxima[0]
    total = maxima[0] / (1 - ratio)
    md3 = sp.N(sp.sympify(md_answer(1, "3")), 30)
    ok3 = dec and abs(ratio - mp.e**(-mp.pi)) < 1e-20 and abs(total - mp.mpf(str(md3))) < 1e-20
    report(1, ok1 and ok2 and ok3, f"sum a_(4n-1)={sorted(sums)} f(10)={cands} sum b_n={mp.nstr(total, 15)}")


def p02():
    # (1),(2) 전수: k 균등 1..4, 빨간 상자 7개 중 4개 동시 추출(5가 4개, 6이 3개)
    balls = [5] * 4 + [6] * 3
    tot = Fraction(0)
    EW = Fraction(0)
    PU = Fraction(0)
    draws = list(itertools.combinations(range(7), 4))
    for k in range(1, 5):
        for d in draws:
            l = sum(1 for i in d if balls[i] == 5)
            pr = Fraction(1, 4) * Fraction(1, len(draws))
            tot += pr
            EW += pr * Fraction(k, l)
            # 최단 경로 수: 격자 DP
            dp = [[1] * (l + 1) for _ in range(k + 1)]
            for i in range(1, k + 1):
                for j in range(1, l + 1):
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
            if dp[k][l] <= 10:
                PU += pr
    ok1 = tot == 1 and sp.Rational(EW.numerator, EW.denominator) == sp.sympify(md_answer(2, "1"))
    ok2 = sp.Rational(PU.numerator, PU.denominator) == sp.sympify(md_answer(2, "2"))
    # (3) 표: P(0<=Z<=z) 로 z 찾기
    table = {sp.Rational(10, 10): sp.Rational(3413, 10000), sp.Rational(15, 10): sp.Rational(4332, 10000),
             sp.Rational(20, 10): sp.Rational(4772, 10000), sp.Rational(25, 10): sp.Rational(4938, 10000)}
    z1 = next(z for z, v in table.items() if 2 * v == sp.Rational(9876, 10000))
    z2 = next(z for z, v in table.items() if 2 * v == sp.Rational(8664, 10000))
    sig = abs(sp.Rational(-1, 2)) * 4
    # 양변 양수이므로 제곱해 정확 비교: 1 < L/sqrt(n) <=> n < L^2
    ns = [nn for nn in range(1, 10000)
          if nn < (2 * z1 * sig)**2 and (2 * z2 * sig)**2 < sp.Rational(36, 25) * nn]
    parts = dict(s.split("=") for s in md_answer(2, "3").replace(" ", "").split(";"))
    ok3 = max(ns) == int(parts["max"]) and min(ns) == int(parts["min"])
    report(2, ok1 and ok2 and ok3, f"E(W)={EW} P(U<=10)={PU} n in [{min(ns)},{max(ns)}]")


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
