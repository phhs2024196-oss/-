#!/usr/bin/env python3
"""CUK-M25A (가톨릭대 2025학년도 모의논술 자연·공학 계열/간호학과 문항 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib, itertools; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
mp.mp.dps = 40
V = Verifier("CUK-M25A", "items/CUK/PAST")
X0 = sp.Symbol("x")
Sx = lambda s: sp.sympify(s, locals={"pi": sp.pi, "x": X0})

def p01():
    a, b, k = sp.symbols("a b k")
    left = x**3 + 6*x**2 + 3*x + a
    right = k*x + b**3 + 6*b**2 + (3 - k)*b + 2
    # 연속: left(b)=right(b) 가 b,k 와 무관하게 a 를 결정
    sol_a = sp.solve(sp.expand(left.subs(x, b) - right.subs(x, b)), a)
    V.check(1, "1-1 a", sol_a[0], V.ans(1, "1"))
    # 미분가능 + f'(x)=0 근 한 개: (k,b) 쌍 전수 탐색. k 는 0 아닌 정수, b 는 0 아닌 실수
    cnt = 0
    for kk in range(-60, 61):
        if kk == 0:
            continue
        for bb in sp.solve(sp.Eq(3*b**2 + 12*b + 3, kk), b):       # f'(b)=k
            if bb == 0 or sp.im(bb) != 0:
                continue
            # f'(x)=0 의 서로 다른 실근 수: x<=b 에서 3x^2+12x+3=0 의 근, x>b 에서 k=0 (k≠0 이므로 없음)
            roots = [r for r in sp.solve(3*x**2 + 12*x + 3, x) if sp.simplify(r - bb) <= 0 and float(sp.N(r - bb)) <= 1e-12]
            if len(roots) == 1:
                cnt += 1
    V.check(1, "1-2 함수 개수", cnt, V.ans(1, "2"))

def p02():
    xs = mp.findroot(lambda u: u * mp.sqrt(u) - mp.mpf(16) / 5, 1)
    V.report(2, xs < 3, f"교점 x={mp.nstr(xs, 8)} < 3")
    # 넓이 A = B 인 a>3
    A = lambda a: mp.quad(lambda u: mp.mpf(16)/5 - u*mp.sqrt(u), [0, xs]) - mp.quad(lambda u: u*mp.sqrt(u) - mp.mpf(16)/5, [xs, a])
    m = mp.findroot(A, 4.2)
    V.check(2, "2-1 m", m, Sx(V.ans(2, "1", "m")), 1e-25)
    # 둘레의 합: 각 도형의 경계를 직접 합산(곡선 길이 수치적분 + 선분)
    arc = lambda p, q: mp.quad(lambda u: mp.sqrt(1 + (mp.mpf(3)/2*mp.sqrt(u))**2), [p, q])
    h = mp.mpf(16)/5
    perA = h + xs + arc(0, xs)                                  # y축 선분 + 직선 + 곡선
    perB = arc(xs, m) + (m*mp.sqrt(m) - h) + (m - xs)           # 곡선 + 세로 선분 + 직선
    l = perA + perB
    V.check(2, "2-1 l", l, Sx(V.ans(2, "1", "l")), 1e-20)

def p03():
    # 원 P_n: 중심 (n,0), 반지름 1. 겹친 넓이 b 를 수치적분(두 원의 교집합)으로
    b = mp.quad(lambda u: 2 * mp.sqrt(1 - (u - 1)**2) if u >= 1.5 else 2 * mp.sqrt(1 - (u - 2)**2), [1, 1.5, 2])
    # u∈[1,1.5]: P2 가 바깥(왼쪽 경계는 P2), 교차 영역 = [1,2] 에서 두 원의 y 최솟값
    b = mp.quad(lambda u: 2 * min(mp.sqrt(1 - (u - 1)**2), mp.sqrt(1 - (u - 2)**2)), [1, 1.5, 2])
    V.check(3, "b", b, 2 * (sp.pi / 3 - sp.sqrt(3) / 4), 1e-25)
    # 6개 선택 전수 열거: 색칠 넓이 M = 6π - (겹침 쌍 수)·b (중심 거리 2 이상은 접할 뿐, 3중 겹침 없음) → 거리 <2 인 쌍 수 k 를 직접 센다
    from collections import Counter
    c = Counter()
    for comb in itertools.combinations(range(1, 11), 6):
        kcnt = sum(1 for p, q in itertools.combinations(comb, 2) if abs(p - q) < 2)
        c[kcnt] += 1
    vals = sorted(c, reverse=False)                              # k 작을수록 M 큼
    Ms = [6 * sp.pi - kk * 2 * (sp.pi / 3 - sp.sqrt(3) / 4) for kk in vals]
    Ms.sort(key=lambda v: -float(v))
    V.check(3, "3-1 a (4번째로 큰 M)", Ms[3], Sx(V.ans(3, "1")), 1e-20)
    k4 = sorted(c)[3]
    V.check(3, "3-2 경우의 수", c[k4], V.ans(3, "2"))

if __name__ == "__main__": V.run(globals())
