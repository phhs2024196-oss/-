#!/usr/bin/env python3
"""KHU-M27D (경희대 2027학년도 모의논술 의·약학계열 수학, KHU-M27D-01~02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import random
import re
import sys
from fractions import Fraction as Fr

import mpmath as mp
import sympy as sp

PREFIX = "KHU-M27D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    mp.mp.dps = 160
    random.seed(1)
    ok = True
    checked = 0
    a_, b_ = sp.symbols("alpha beta", positive=True)
    k = sp.symbols("k", positive=True)
    nn = sp.symbols("nn", positive=True, integer=True)
    an_formula = (a_ + b_ * (a_ / b_) ** (nn / sp.Integer(3))) / (1 + (a_ / b_) ** (nn / sp.Integer(3)))
    Lsum_formula = lambda p, q, kk: (p - 2 * q) / (kk * (1 - p + q))
    while checked < 20:
        p = random.uniform(0.05, 1.99); q = random.uniform(0.001, p * p / 4)
        if not (p * p > 4 * q and p - q < 1 and p < 2):
            continue
        p, q = sp.Rational(p).limit_denominator(1000), sp.Rational(q).limit_denominator(1000)
        if not (p**2 > 4 * q and p - q < 1 and p < 2 and q > 0):
            continue
        al, be = sorted(sp.solve(x**2 - p * x + q, x), key=lambda r: float(r))
        ok &= 0 < float(al) < float(be) < 1
        for N in (1, 2, 5):
            f = al**N / (x - al) + be**N / (x - be)
            f2 = sp.diff(f, x, 2)
            num = sp.numer(sp.together(f2))
            roots = [r for r in sp.Poly(sp.expand(num), x).nroots(n=30) if abs(sp.im(r)) < 1e-20]
            an = float(an_formula.subs({a_: al, b_: be, nn: N}))
            ok &= len(roots) == 1 and abs(float(sp.re(roots[0])) - an) < 1e-12 and float(al) < an < float(be)
        # (2) 길이 합 L_n: f(x)=k 의 해를 수치로 찾고 구간 길이 계산
        kk = sp.Rational(random.choice([1, 3, 7]), random.choice([1, 2, 5]))
        tot = mp.mpf(0)
        for N in range(1, 31):
            fa, fb = mp.mpf(sp.N(al, 170)), mp.mpf(sp.N(be, 170))
            F = lambda X: fa**N / (X - fa) + fb**N / (X - fb) - mp.mpf(sp.N(kk, 170))
            r1 = mp.findroot(F, (fa + mp.mpf(10) ** -140, fb - mp.mpf(10) ** -140), solver="anderson")
            r2 = mp.findroot(F, (fb + mp.mpf(10) ** -140, fb + 10**6), solver="anderson")
            # (-inf, alpha) 에서는 f<0 이므로 해 없음: 표본 확인
            ok &= all(fa**N / (X - fa) + fb**N / (X - fb) < 0 for X in (fa - 1, fa - mp.mpf("1e-6"), -100))
            tot += (r1 - fa) + (r2 - fb)
        target = mp.mpf(sp.N(Lsum_formula(p, q, kk), 170))
        tail = (fa**31 / (1 - fa) + fb**31 / (1 - fb)) / mp.mpf(sp.N(kk, 170))
        ok &= abs(tot + tail - target) < mp.mpf("1e-60")
        checked += 1
    lim = sp.limit(an_formula.subs({a_: sp.Rational(1, 5), b_: sp.Rational(1, 2)}), nn, sp.oo)
    ok &= lim == sp.Rational(1, 5)
    report(1, ok, f"무작위 (p,q) 20쌍: a_n 유일·구간 안, lim a_n=alpha, sum L_n=(p-2q)/(k(1-p+q)) 수치 일치")


def p02():
    def rank_A(perm):
        return perm[0]

    def rank_skip(perm, m):
        ref = min(perm[:m])
        for r in perm[m:]:
            if r < ref:
                return r
        return perm[-1]
    perms = list(itertools.permutations(range(1, 6)))
    w = Fr(1, len(perms))
    dist = lambda fn: {v: sum(w for p in perms if fn(p) == v) for v in range(1, 6)}
    DX, DY, DZ = dist(rank_A), dist(lambda p: rank_skip(p, 1)), dist(lambda p: rank_skip(p, 2))
    E = lambda D: sum(k * v for k, v in D.items())
    EX, EY, EZ = E(DX), E(DY), E(DZ)
    ok = (EX, EY, EZ) == (3, Fr(21, 10), Fr(11, 5))
    ok &= [DY[i] for i in range(1, 6)] == [Fr(5, 12), Fr(4, 15), Fr(1, 6), Fr(1, 10), Fr(1, 20)]
    ok &= [DZ[i] for i in range(1, 6)] == [Fr(13, 30), Fr(7, 30), Fr(2, 15), Fr(1, 10), Fr(1, 10)]
    ok &= EY < EZ < EX
    report(2, ok, f"5! 전수: E(X)={EX}, E(Y)={EY}, E(Z)={EZ}, P(Y)={[str(DY[i]) for i in range(1,6)]}, P(Z)={[str(DZ[i]) for i in range(1,6)]}")


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
