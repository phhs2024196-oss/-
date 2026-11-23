#!/usr/bin/env python3
"""UOS-M19 (2019학년도 모의논술 자연계열 4문항) — 정확한 유리수·열거·수치 검산. 기대값은 YAML 정답(V.ans)에서 읽는다."""
import sys, pathlib, math, itertools
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("UOS-M19", "items/UOS/PAST")
K = sp.Symbol("k")
TH = sp.Symbol("theta")


def p01():
    # [증명-수치확인] (n^n n!/(2n)!)^2 <= 2^-n, n=1..600 (정확한 유리수)
    bad = []
    for n_ in range(1, 601):
        lhs = Fraction(n_**n_ * math.factorial(n_), math.factorial(2 * n_))
        if lhs**2 * 2**n_ > 1:
            bad.append(n_)
    V.proof(1, "n=1..600 부등식", bad)
    # 해설의 짝짓기 논리: 각 쌍 <= 1/2, 홀수 n 의 가운데 항 <= 2/3 < 1/sqrt2
    bad2 = []
    for n_ in range(1, 301):
        m_ = (n_ + 1) // 2 if n_ % 2 else n_ // 2
        npairs = m_ if n_ % 2 == 0 else m_ - 1
        for kk in range(npairs):
            if Fraction(n_, 2 * n_ - kk) * Fraction(n_, n_ + 1 + kk) > Fraction(1, 2):
                bad2.append((n_, kk))
        if n_ % 2 == 1 and n_ > 1 and Fraction(n_, 2 * n_ - (m_ - 1)) > Fraction(2, 3):
            bad2.append((n_, "mid"))
        # 쌍과 가운데 항이 정확히 n! 인수 분해를 이룸
        prod = Fraction(1)
        for kk in range(npairs):
            prod *= Fraction(n_, 2 * n_ - kk) * Fraction(n_, n_ + 1 + kk)
        if n_ % 2 == 1:
            prod *= Fraction(n_, 2 * n_ - (m_ - 1)) if n_ > 1 else Fraction(1, 2)
        if prod != Fraction(n_**n_ * math.factorial(n_), math.factorial(2 * n_)):
            bad2.append((n_, "prod"))
    V.proof(1, "짝짓기 분해식(쌍≤1/2, 가운데≤2/3)", bad2)


def p02():
    # (a) 닫힌 꼴 확인 + x 격자 [증명-수치확인]
    x_, nn = sp.symbols("x_ nn", positive=True)
    for N in (1, 2, 5, 9):
        lo = 1 + sum((-x_) ** j for j in range(1, 2 * N))
        hi = 1 + sum((-x_) ** j for j in range(1, 2 * N + 1))
        V.report(2, sp.simplify(lo - (1 - x_ ** (2 * N)) / (1 + x_)) == 0 and sp.simplify(hi - (1 + x_ ** (2 * N + 1)) / (1 + x_)) == 0,
                 f"2-1 n={N} 등비합 닫힌 꼴")
    bad = []
    for N in range(1, 31):
        for i in range(0, 1001):
            xv = Fraction(i, 1000)
            lo = 1 + sum((-xv) ** j for j in range(1, 2 * N))
            hi = 1 + sum((-xv) ** j for j in range(1, 2 * N + 1))
            if not (lo <= 1 / (1 + xv) <= hi):
                bad.append((N, i))
    V.proof(2, "2-1 n=1..30, x=0..1(1/1000 격자)", bad)
    # (b) a_n 직접 합 → ln2 , 해설의 평가 ln2-1/(2n+1) <= a_n <= ln2
    mp.mp.dps = 30
    exp = mp.mpf(str(sp.N(sp.sympify(V.ans(2, "2-2")), 30)))
    bad = []
    for N in list(range(1, 200)) + [1000, 5000, 20000]:
        an = mp.fsum(mp.mpf((-1) ** (j + 1)) / j for j in range(1, 2 * N + 1))
        if not (exp - mp.mpf(1) / (2 * N + 1) - mp.mpf("1e-25") <= an <= exp + mp.mpf("1e-25")):
            bad.append(N)
    V.proof(2, "2-2 ln2-1/(2n+1)<=a_n<=ln2 (n≤200,1000,5000,20000)", bad)
    an = mp.fsum(mp.mpf((-1) ** (j + 1)) / j for j in range(1, 400001))
    V.report(2, abs(an - exp) < mp.mpf("1.3e-6"), f"2-2 a_200000={an} / ln2={exp}")


def p03():
    e1 = sp.sympify(V.ans(3, "3-1"))
    e2 = sp.sympify(V.ans(3, "3-2"))
    for kk in range(2, 7):
        cnt = 0
        for dice in itertools.product(range(1, 7), repeat=kk):
            if max(dice) - min(dice) == 3:
                cnt += 1
        V.check(3, f"3-1 k={kk} 전수열거", cnt, e1.subs(K, kk))
        num = 0
        for rest in itertools.product(range(1, 7), repeat=kk - 1):
            d = (5,) + rest
            if max(d) - min(d) == 3:
                num += 1
        V.check(3, f"3-2 k={kk} 전수열거 확률", sp.Rational(num, 6 ** (kk - 1)), e2.subs(K, kk))
    # 해설의 G1..G5 분해(포함배제)도 k=2..7 에서 열거로 확인
    for kk in range(2, 8):
        m = kk - 1
        cnt = {}
        for rest in itertools.product(range(1, 7), repeat=m):
            cnt[(min(rest), max(rest))] = cnt.get((min(rest), max(rest)), 0) + 1
        g = [cnt.get(p, 0) for p in ((2, 2), (2, 3), (2, 4), (2, 5), (3, 6))]
        form = [1, 2**m - 2, 3**m - 2 * 2**m + 1, 4**m - 2 * 3**m + 2**m, 4**m - 2 * 3**m + 2**m]
        V.report(3, g == form, f"G1..G5 개수 k={kk} 열거 {g} / 해설식 {form}")


def p04():
    f = sp.sympify(V.ans(4, "4-1"), locals={"theta": TH})
    th_ans = sp.sympify(V.ans(4, "4-2"))
    # 독립 계산: 원뿔을 펼친 부채꼴(중심각 2π·r/L)에서 BC 와 광선(각 θ r/L)의 교점까지의 거리
    r_, L_ = 2.0, 8.0
    beta = 2 * math.pi * r_ / L_
    AC = 8 * math.sqrt(3) / 3
    Bp = (L_, 0.0)
    Cp = (AC * math.cos(beta), AC * math.sin(beta))

    def bp_len(theta):
        phi = theta * r_ / L_
        d = (math.cos(phi), math.sin(phi))
        # B + s (C - B) = rho d  →  풀이
        ex, ey = Cp[0] - Bp[0], Cp[1] - Bp[1]
        det = d[0] * (-ey) - d[1] * (-ex)  # [d, -e] 행렬식
        # rho d - s e = B
        rho = (Bp[0] * (-ey) - Bp[1] * (-ex)) / det
        s = (d[0] * Bp[1] - d[1] * Bp[0]) / det
        assert 0 <= s <= 1 + 1e-12
        return s * math.hypot(ex, ey)

    bad = []
    for i in range(0, 361):
        th = 2 * math.pi * i / 360
        if abs(bp_len(th) - float(f.subs(TH, th))) > 1e-9:
            bad.append(i)
    V.proof(4, "4-1 전개도 직접 계산과 f(θ) 일치(θ=0..2π, 361점)", bad)
    # 최단거리 확인: 부채꼴 중심각 < π 이므로 직선 BC 가 최단 (모선을 따라 한 바퀴)
    V.report(4, beta < math.pi, f"중심각 β={beta:.6f} < π")
    # 4-2: 도함수 기호 계산·최솟값
    df = sp.diff(f, TH)
    V.report(4, sp.simplify(df - 1 / sp.sin(5 * sp.pi / 6 - TH / 4) ** 2) == 0, "4-2 df/dθ = 1/sin²(5π/6-θ/4)")
    grid = [2 * math.pi * i / 200000 for i in range(200001)]
    vals = [float(df.subs(TH, g)) for g in grid[::200]]
    imin = min(range(len(vals)), key=lambda i: vals[i])
    th_num = grid[::200][imin]
    V.check(4, "4-2 df/dθ 최소점(수치)", round(th_num, 3), round(float(th_ans), 3), tol=2e-2)
    V.check(4, "4-2 f'(4π/3)=1 (최솟값)", df.subs(TH, th_ans), 1)


if __name__ == "__main__":
    V.run(globals())
