#!/usr/bin/env python3
"""CUK-M24A (가톨릭대 2024학년도 모의논술 자연공학계열·간호학과 문항 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-M24A", "items/CUK/PAST")
mp.mp.dps = 30
X0 = sp.Symbol("x")

def p01():
    R = 2
    sinB = sp.sqrt(sp.Rational(3, 4))                  # 4 sin^2 B = 3 (sin(A+C)=sin B)
    AC = 2 * R * sinB                                  # 사인법칙
    d = sp.sqrt(R**2 - (AC / 2)**2)                    # 중심에서 AC 까지 거리
    A_, C_ = sp.Point(-AC / 2, d), sp.Point(AC / 2, d)  # AC 를 y=d 위에 놓는다 (O 는 원점)
    th = sp.symbols("th", real=True)
    areaAOC = AC * d / 2
    yy = sp.Symbol("yy", real=True)
    Ls = set()
    for y0 in sp.solve((yy - d) * AC / 2 - 3 * areaAOC, yy) + sp.solve(-(yy - d) * AC / 2 - 3 * areaAOC, yy):
        if abs(y0) <= R:                               # 원 위의 점 D=(±sqrt(R^2-y0^2), y0)
            for sx in (1, -1):
                D = sp.Point(sx * sp.sqrt(R**2 - y0**2), y0)
                Ls.add(sp.nsimplify(sp.simplify(A_.distance(C_) + A_.distance(D) + C_.distance(D))))
    V.report(1, len(Ls) == 1, f"1-1 L 값 집합 {Ls}")
    V.check(1, "1-1 L", list(Ls)[0], V.ans(1, "1"))

def p02():
    p, q = sp.symbols("p q")
    # f = x^3 + p x^2 + q x (f(0)=0, 최고차 계수 1), f'(1)=1
    qq = sp.solve(3 + 2 * p + q - 1, q)[0]
    f = x**3 + p * x**2 + qq * x
    # h=f-x = x(x^2+p x+qq-1): 서로 다른 실근 2개 ⇔ (qq=1, p!=0) 또는 (판별식=0, qq!=1)
    h2 = x**2 + p * x + qq - 1
    cands = set()
    for s in sp.solve(sp.discriminant(h2, x), p):           # 이차식 중근
        if sp.simplify(qq.subs(p, s) - 1) != 0: cands.add(s)
    for s in sp.solve(qq - 1, p):                           # 이차식의 한 근이 0 (다른 근 -p != 0)
        if s != 0: cands.add(s)
    good = []
    for s in cands:
        fp = sp.diff(f.subs(p, s), x)
        if sp.discriminant(fp, x) > 0: good.append(sp.expand(f.subs(p, s)))   # 극값을 가짐
    V.report(2, len(good) == 1, f"2-1 후보 f {good}")
    fx = good[0]
    V.check(2, "2-1 f", fx, sp.sympify(V.ans(2, "1"), locals={"x": x}))
    # 논제 2: g 미분가능 ⇒ f(a)=f(a+b), f'(a)=f'(a+b)
    a, b = sp.symbols("a b", real=True)
    eq1 = sp.expand(fx.subs(x, a) - fx.subs(x, a + b))
    eq2 = sp.expand(sp.diff(fx, x).subs(x, a) - sp.diff(fx, x).subs(x, a + b))
    sol = [s for s in sp.solve([eq1, eq2], [a, b], dict=True) if sp.N(s[b]) > 0 and sp.N(s[a]) > 0]
    V.report(2, len(sol) == 1, f"2-2 해 개수 {sol}")
    V.check(2, "2-2 a", sol[0][a], V.ans(2, "2", "a"))
    V.check(2, "2-2 b", sol[0][b], V.ans(2, "2", "b"))

def p03():
    from fractions import Fraction as F
    found = set()
    half = [F(i, 2) for i in range(-24, 25)]
    for K in range(1, 31):
        for d1 in half:
            for d2 in half:
                c = lambda n: F(-20) + (n - 1) * d1 if n <= K else F(-20) + (K - 1) * d1 + (n - K) * d2
                if not (c(12) > c(11) and c(16) == c(17) + 6): continue
                S = [F(0)]
                for n in range(1, 301): S.append(S[-1] + c(n))
                if S[11] != 0: continue
                if S[19] < max(S[1:]): continue            # S_19 가 최댓값
                found.add((c(7), S[19]))
    V.report(3, len({f[0] for f in found}) == 1, f"3-1 c7 값 {sorted({f[0] for f in found})}")
    V.check(3, "3-1 c7", sp.Rational(list(found)[0][0].numerator, list(found)[0][0].denominator), V.ans(3, "1"))
    V.check(3, "3-2 S19", sp.Rational(list(found)[0][1].numerator, list(found)[0][1].denominator), V.ans(3, "2"))

if __name__ == "__main__": V.run(globals())
