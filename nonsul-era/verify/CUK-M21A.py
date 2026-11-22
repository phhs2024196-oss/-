#!/usr/bin/env python3
"""CUK-M21A (가톨릭대 2021학년도 모의논술 자연공학계열·간호학과 문항 1~3) — 문제 조건에서 독립 재계산."""
import sys, re, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
from fractions import Fraction as F
V = Verifier("CUK-M21A", "items/CUK/PAST")
mp.mp.dps = 40

def p01():
    # 조건 2 sinA cosC = sinB (B=π-A-C) 를 만족하는 (A,C) 격자: 정확히 A=C 인 경우뿐인지
    bad, hits = [], 0
    for ad in range(1, 180):
        for cd in range(1, 180 - ad):
            A_, C_ = mp.radians(ad), mp.radians(cd)
            ok = abs(2 * mp.sin(A_) * mp.cos(C_) - mp.sin(mp.pi - A_ - C_)) < 1e-25
            if ok: hits += 1
            if ok != (ad == cd): bad.append((ad, cd))
    V.report(1, not bad and hits > 0, f"1-1 조건 ⇔ A=C (격자 {hits}점, 불일치 {bad[:2]})")
    V.report(1, V.ans(1, "1").replace(" ", "") == "a=c", "1-1 YAML 답 a=c")
    # 논제 2: A=C=θ, B=π-2θ. 사인법칙으로 a=c=2R sinθ, S=½ a c sinB, AB=c
    th = sp.Symbol("th", positive=True)
    ratio = sp.Rational(1, 2) * sp.sin(sp.pi - 2 * th)                 # S/c² = ½ sin B
    best = max(sp.N(ratio.subs(th, sp.Rational(i, 1000) * sp.pi / 2), 20) for i in range(1, 1000))
    # 좌표로 직접: 꼭짓점 B, A=C 인 이등변삼각형의 S/AB² (독립 계산)
    def S_over_c2(thv):
        a_ = 1.0; Bv = mp.pi - 2 * thv
        c_ = a_ * mp.sin(thv) / mp.sin(thv)          # a=c
        return (a_ * c_ * mp.sin(Bv) / 2) / c_**2
    best2 = max(S_over_c2(mp.pi / 2 * mp.mpf(i) / 2000) for i in range(1, 2000))
    V.check(1, "1-2 M (공식)", best, V.ans(1, "2"), 1e-5)
    V.check(1, "1-2 M (직접)", sp.Float(best2, 15), V.ans(1, "2"), 1e-5)

def p02():
    want = [sp.Rational(s.strip()) for s in V.ans(2, "1").split(";")]
    pp = sp.Symbol("pp")
    sol = sp.solve(1 * pp + 2 * (1 - pp) - sp.Rational(7, 4), pp)[0]
    V.report(2, [sol, 1 - sol] == want, f"2-1 분포 {[sol, 1 - sol]} / YAML {want}")
    # 모든 가능한 공 구성(총 N≤40, 적힌 수 1/2, 색 붉/파)에서 조건 i)~iii) 만족 시 P(붉고 2) 값 집합
    vals = set()
    for N in range(1, 41):
        for r1 in range(N + 1):
            for r2 in range(N + 1 - r1):
                for b1 in range(N + 1 - r1 - r2):
                    b2 = N - r1 - r2 - b1
                    if F(r1 + r2, N) != F(3, 5): continue
                    if F(1 * (r1 + b1) + 2 * (r2 + b2), N) != F(7, 4): continue
                    if F(r1, N) != F(r1 + r2, N) * F(r1 + b1, N): continue          # A,B 독립
                    vals.add(F(r2, N))
    V.report(2, vals == {F(9, 20)}, f"2-2 P(붉은 공·2) 값 집합 {vals}")
    V.check(2, "2-2 YAML", sp.Rational(9, 20), V.ans(2, "2"))

def p03():
    bad = [N for N in range(1, 4000) if not (mp.sqrt(mp.e) < (1 + mp.mpf(1) / N)**N < mp.e)]
    V.proof(3, "3-1 명제 A (n=1..3999)", bad)
    V.report(3, V.ans(3, "1") == "참", "3-1 YAML 답 참")
    res = []
    for N in (10, 100, 1000, 10**5, 10**6):
        q = (1 + mp.mpf(1) / N)**N
        cf = lambda X: X**3 - q * X**2 + X - mp.e
        roots = mp.polyroots([1, -q, 1, -mp.e], maxsteps=200, extraprec=100)
        reals = [r.real for r in roots if abs(r.imag) < 1e-20] + [mp.sqrt(mp.e)]
        an = max(reals)
        res.append(an)
        V.report(3, q < an < mp.e, f"3-2 n={N}: (1+1/n)^n < a_n < e")
    V.check(3, "3-2 극한 근사", sp.Float(res[-1], 15), sp.N(sp.sympify(V.ans(3, "2")), 15), 1e-5)
    V.report(3, all(abs(res[i + 1] - mp.e) < abs(res[i] - mp.e) for i in range(len(res) - 1)), "3-2 a_n 이 e 로 단조 접근")

if __name__ == "__main__": V.run(globals())
