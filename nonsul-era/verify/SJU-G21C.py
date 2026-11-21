#!/usr/bin/env python3
"""SJU-G21C (세종대 2021학년도 수시 논술고사[C] 자연계열) — sympy 검산. 정답 하드코딩 금지, 설정으로부터 독립 재계산."""
import re
import sys
from fractions import Fraction as Fr
from itertools import product, combinations

import sympy as sp

PREFIX = "SJU-G21C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(no: int, sub: str) -> dict:
    """'k1 = v1; k2 = v2' 형식의 정답을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def S(s: str):
    return sp.sympify(s, locals={"E": sp.E, "e": sp.E, "x": x, "t": t, "M": sp.Symbol("M", positive=True)})


def same(a, b, tol=1e-12) -> bool:
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 or abs(complex(sp.N(a - b, 30))) < tol
    except Exception:  # noqa: BLE001
        return abs(complex(sp.N(a - b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import math
    f = lambda u: u - 0.5 if u >= 1.5 else (2 * u - 2 if u > 1 else -u + 1)
    g = lambda u: (u - 1) ** 2 if u >= 1 else -u + 1
    h = lambda u: u + math.exp(u) - math.e if u >= 1 else math.e * u + 1 - math.e
    # (1) h 증가(격자) + x=1 연속
    xs = [-5 + i * 0.001 for i in range(10001)]
    ok1 = all(h(b) > h(a) for a, b in zip(xs, xs[1:])) and abs(h(1) - (math.e + 1 - math.e)) < 1e-12
    # (2) g∘f 의 좌우 미분계수가 다른 점을 격자(1/8 간격 유리점 포함)에서 탐색
    F = lambda u: g(f(u))
    hh = 1e-7
    bad = []
    for i in range(-400, 401):
        a = i / 200
        r = (F(a + hh) - F(a)) / hh
        l = (F(a) - F(a - hh)) / hh
        if abs(r - l) > 1e-3:
            bad.append(sp.Rational(i, 200))
    ok2 = set(bad) == set(S(md_answer(1, "2")))
    # (3) h^{-1}∘f 의 x=3/2 좌우 미분계수 (이분법 역함수)
    def hinv(y):
        lo, hi = -50.0, 50.0
        for _ in range(200):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if h(m) < y else (lo, m)
        return (lo + hi) / 2
    G = lambda u: hinv(f(u))
    d = 1e-6
    r = (G(1.5 + d) - G(1.5)) / d
    l = (G(1.5) - G(1.5 - d)) / d
    ok3 = abs(r - 1 / (1 + math.e)) < 1e-4 and abs(l - 2 / math.e) < 1e-4 and abs(r - l) > 0.1
    report(1, ok1 and ok2 and ok3, f"[증명-수치확인] h 증가 / A={sorted(bad)} / 우={r:.6f} 좌={l:.6f}")


def p02():
    pD, sens, fp = Fr(1, 100), Fr(99, 100), Fr(3, 100)
    pA = pD * sens + (1 - pD) * fp
    pDA = pD * sens / pA
    ans = parts(2, "1")
    ok1 = same(S(ans["P(A)"]), sp.Rational(pA.numerator, pA.denominator)) and same(S(ans["P(D|A)"]), sp.Rational(pDA.numerator, pDA.denominator))
    # (2) 정규근사 + 문제의 표(0.43 at z=1.5)
    N, p = 768, pDA
    m, v = N * p, N * p * (1 - p)
    z = (210 - m) / sp.sqrt(sp.Rational(v.numerator, v.denominator))
    table = {sp.Rational(1): 0.34, sp.Rational(3, 2): 0.43, sp.Rational(2): 0.48, sp.Rational(5, 2): 0.49}
    prob = 0.5 + table[sp.nsimplify(z)]
    ok2 = abs(prob - float(S(md_answer(2, "2")))) < 1e-12
    # (3) 비용 비교: 정수 n 을 1부터 직접 탐색
    def C(k):
        return Fr(6000 * k) if k > 40000 else Fr(10000 * k) - Fr(k * k, 10)
    nmin = next(k for k in range(1, 200001) if C(k) + pA * k * 100000 < pD * k * 1000000)
    ok3 = nmin == int(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"P(A)={pA} P(D|A)={pDA} z={z} P={prob} nmin={nmin}")


def p03():
    import mpmath as mp
    g = lambda u: mp.sin(mp.pi * u) ** 2
    f = lambda u: 1 / (1 + mp.e ** (-u))
    I1 = sp.integrate(sp.sin(sp.pi * x) ** 2, (x, 0, 1))
    ok1 = same(I1, S(md_answer(3, "1")))
    # (2) 부등식 수치 확인 n=1..30
    mp.mp.dps = 30
    ok2 = True
    an = {}
    for k in list(range(1, 31)) + [200]:
        a = mp.quad(lambda u: f(u) * g(k * u), mp.linspace(0, 1, k + 1))
        an[k] = a
        up = sum(f(mp.mpf(j) / k) for j in range(1, k + 1)) / (2 * k)
        lo = sum(f(mp.mpf(j - 1) / k) for j in range(1, k + 1)) / (2 * k)
        ok2 = ok2 and lo - 1e-25 <= a <= up + 1e-25
    lim = sp.integrate(1 / (1 + sp.exp(-x)), (x, 0, 1)) / 2
    ok3 = same(lim, S(md_answer(3, "3"))) and abs(an[200] - mp.mpf(sp.N(lim, 30))) < 1e-3
    report(3, ok1 and ok2 and ok3, f"[증명-수치확인] int g={I1} lower<=a_n<=upper (n=1..30,200) lim={sp.N(lim, 12)} a_200={mp.nstr(an[200], 12)}")


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
