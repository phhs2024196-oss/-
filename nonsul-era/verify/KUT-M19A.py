#!/usr/bin/env python3
"""KUT-M19A (2019 논술 예시문항 수학-A형 01~03) 검산 — 설정으로부터 독립 재계산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy import Rational as R, symbols, integrate, Abs, solve, Piecewise
V = Verifier("KUT-M19A", "items/KUT/PAST")

def p01():
    # 상황을 그대로 시뮬레이션: 매년 가을 열매 나무마다 2개 생존 -> 다다음해 가을 새 나무
    N = 10
    planted = {0: 1}          # 심은 연도(가을 기준) -> 그루 수
    a = []
    for yr in range(N + 1):
        trees = sum(c for y, c in planted.items() if y <= yr)
        a.append(trees)
        planted[yr + 2] = planted.get(yr + 2, 0) + 2 * trees
    ok = all(a[n + 2] == a[n + 1] + 2 * a[n] for n in range(N - 1))
    V.report(1, ok and "a_{n+2}=a_{n+1}+2a_n" in V.ans(1, "1-1").replace(" ", ""), f"관계식 확인 a={a}")
    V.check(1, "1-2 a10", a[10], V.ans(1, "1-2"))

def p02():
    xx, A, B, C = symbols("x A B C")
    g = A * xx**2 + B * xx + C
    f = xx**2 - 9
    sol = solve([g.subs(xx, 0) - f.subs(xx, 0), g.diff(xx).subs(xx, 0) - f.diff(xx).subs(xx, 0), g.subs(xx, 5)], [A, B, C], dict=True)[0]
    gg = g.subs(sol)
    V.check(2, "2-1 g", gg, V.ans(2, "1-1"))
    S1 = integrate(Abs(f), (xx, -3, 0)) if False else integrate(-f, (xx, -3, 0))   # [-3,0] 에서 f<=0
    S2 = integrate(-gg, (xx, 0, 5))                                              # (0,5] 에서 g<=0
    # 부호 확인
    assert all(f.subs(xx, R(t, 10)) <= 0 for t in range(-30, 1)) and all(gg.subs(xx, R(t, 10)) <= 0 for t in range(0, 51))
    V.check(2, "2-2 넓이", S1 + S2, V.ans(2, "1-2"))

def p03():
    A_w, A_b = 2, 3; B_w, B_b = 3, 5
    pA = R(A_w, A_w + A_b)
    PAB = pA * R(B_w + 1, B_w + B_b + 1)
    PAcB = (1 - pA) * R(B_w, B_w + B_b + 1)
    PB = PAB + PAcB
    V.check(3, "3-1 P(B)", PB, V.ans(3, "1-1"))
    V.check(3, "3-2 P(A|B)", PAB / PB, V.ans(3, "1-2"))
    # 전수 열거 확인
    from itertools import product
    A = ['w'] * 2 + ['b'] * 3; Bb = ['w'] * 3 + ['b'] * 5
    tot = nB = nAB = 0
    for i in range(5):
        Bn = Bb + [A[i]]
        for j in range(9):
            tot += 1
            if Bn[j] == 'w':
                nB += 1
                nAB += (A[i] == 'w')
    V.check(3, "3-1 열거", R(nB, tot), V.ans(3, "1-1")); V.check(3, "3-2 열거", R(nAB, nB), V.ans(3, "1-2"))

if __name__ == "__main__":
    V.run(globals())
