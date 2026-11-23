import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
from fractions import Fraction
import mpmath as mp
V = Verifier("UOS-M22", "items/UOS/PAST")


def p01():
    t, xx = sp.symbols("t xx", positive=True)
    p = sp.Symbol("p", positive=True)
    sols = sp.solve(sp.Eq(-t**2 * p**2 + 1, 2 * sp.tan(t) * p), p)
    pt = [s for s in sols if sp.simplify(s.subs(t, sp.Rational(1, 5)).evalf()) > 0][0]
    V.check(1, "p(t) 꼴", pt.subs(t, sp.Rational(1, 5)).evalf(), ((-sp.tan(t) + sp.sqrt(sp.tan(t) ** 2 + t**2)) / t**2).subs(t, sp.Rational(1, 5)).evalf())
    V.check(1, "(a) lim t p(t)", sp.limit(t * pt, t, 0, "+"), V.ans(1, "1-1"))
    # A, B 를 직접 적분(기하적 정의)해서 비교: 곡선 y=1-t²x², 직선 y=2tan t·x
    def AB(tv):
        tv = mp.mpf(tv); mp.mp.dps = 30
        pv = (-mp.tan(tv) + mp.sqrt(mp.tan(tv) ** 2 + tv**2)) / tv**2
        A = mp.quad(lambda u: (1 - tv**2 * u**2) - 2 * mp.tan(tv) * u, [0, pv])   # y축·곡선(위)·직선(아래)
        B = mp.quad(lambda u: 2 * mp.tan(tv) * u, [0, pv]) + mp.quad(lambda u: 1 - tv**2 * u**2, [pv, 1 / tv])
        return A, B
    A1, B1 = AB("0.3")
    V.check(1, "A+B=2/(3t)", sp.N(A1 + B1, 20), sp.N(sp.Rational(2, 3) / sp.Rational(3, 10), 20), 1e-15)
    A2, B2 = AB("1e-9")
    V.check(1, "(b) 수치 A/B (t=1e-9)", sp.Float(str(A2 / B2), 15), V.ans(1, "1-2"), 1e-6)
    u = sp.sqrt(2) - 1
    V.check(1, "(b) 공식", 2 / (3 * u**2 + u**3 - 3 * u + 2) - 1, V.ans(1, "1-2"))


def p02():
    mp.mp.dps = 30
    def a(nv):
        L = mp.log(nv, 2)
        c = 2 - L / 2
        if c < -1 - mp.mpf("1e-25") or c > 1 + mp.mpf("1e-25"):
            return 0
        R = 2 * nv * mp.pi
        if abs(c - 1) < mp.mpf("1e-25"): th = [mp.mpf(0)]
        elif abs(c + 1) < mp.mpf("1e-25"): th = [mp.pi]
        else: th = [mp.acos(c), 2 * mp.pi - mp.acos(c)]
        cnt = set()
        for base in th:
            kmax = int(R / (2 * mp.pi)) + 2
            for k in range(-kmax - 1, kmax + 2):
                xv = base + 2 * mp.pi * k
                if abs(xv) <= R + mp.mpf("1e-20"):
                    cnt.add(int(mp.nint(xv * 10**15)))
        return len(cnt)
    tot = sum(a(nv) for nv in range(1, 2023))
    V.check(2, "Σ a_k", tot, V.ans(2, "2-1"))
    V.report(2, (a(4), a(64), a(5), a(63)) == (9, 128, 20, 252), "a_4=9, a_64=128, a_5=20, a_63=252")
    # 원식 f=g 를 부호변화로 직접 센다(접하는 n=4,64 제외)
    import numpy as np
    def direct(nv):
        L = np.log2(nv)
        xs = np.linspace(-2 * nv * np.pi, 2 * nv * np.pi, 400001)
        h = 4 * L * np.cos(xs) + 4 * (np.log(nv) / np.log(4)) ** 2 - 12 - 4 * np.sin(xs) ** 2
        return int(np.sum(h[:-1] * h[1:] < 0))
    bad = [nv for nv in (5, 10, 30) if direct(nv) != a(nv)]
    V.proof(2, "직접 부호변화 수와 일치 n=5,10,30", bad)
    L = sp.Symbol("L", nonnegative=True)
    V.check(2, "f 하한", (4 * L * (-1) + L**2 - 12) - ((L - 2) ** 2 - 16), 0)
    V.check(2, "인수분해", sp.expand((4 * sp.cos(x)**2 + 4 * L * sp.cos(x) + L**2 - 16) - 4 * (sp.cos(x) + 2 + L / 2) * (sp.cos(x) - 2 + L / 2)), 0)


def p03():
    arrows = [1, 1, 1, 1, 0, 0, 0, 0]
    pairs = list(itertools.combinations(range(8), 2))
    PA = Fraction(0); PAB = Fraction(0)
    for s1 in pairs:
        for s2 in pairs:
            a_ = arrows[:]
            for i in s1: a_[i] ^= 1
            for i in s2: a_[i] ^= 1
            w = Fraction(1, 28 * 28)
            if sum(a_) == 4:
                PA += w
                if a_[6] == 1 and a_[7] == 1: PAB += w
    f = lambda q: sp.Rational(q.numerator, q.denominator)
    V.check(3, "(a) P(A)", f(PA), V.ans(3, "3-1"))
    V.check(3, "(b) P(B|A)", f(PAB / PA), V.ans(3, "3-2"))
    V.check(3, "P(A∩B)", f(PAB), sp.Rational(9, 196))


def p04():
    xs = sp.Symbol("xs", positive=True)
    f = (2 * xs + 1) * sp.log(1 + 1 / xs)
    f1 = sp.diff(f, xs)
    f2 = sp.simplify(sp.diff(f1, xs))
    V.check(4, "f''", f2 - 1 / (xs**2 + xs) ** 2, 0)
    V.check(4, "f'", f1 - (2 * sp.log(1 + 1 / xs) - (2 * xs + 1) / (xs**2 + xs)), 0)
    V.check(4, "lim f", sp.limit(f, xs, sp.oo), 2)
    V.check(4, "lim f'", sp.limit(f1, xs, sp.oo), 0)
    mp.mp.dps = 40
    bad = [nv for nv in range(1, 3001) if (2 * nv + 1) * mp.log(1 + mp.mpf(1) / nv) < 2]
    V.proof(4, "n=1..3000 에서 e²≤(1+1/n)^(2n+1)", bad)


if __name__ == "__main__": V.run(globals())
