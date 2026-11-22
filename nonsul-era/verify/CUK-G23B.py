#!/usr/bin/env python3
"""CUK-G23B (가톨릭대 2023 수시 논술 의예·약학 1~4) — 문제 조건에서 독립 재계산."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
import mpmath as mp
V = Verifier("CUK-G23B", "items/CUK/PAST")


def p01():
    # A: 모든 실수 x 에서 진수>0, 밑>0, 밑≠1 인 k — k 를 촘촘히 훑어 구간 확인 (경계는 해석적으로)
    xs = np.linspace(-200, 200, 400001)
    def inA(k):
        base = k/2 - 5
        if base <= 0 or abs(base-1) < 1e-12:
            return False
        return bool(np.all(-(k-11)*xs**2 + (k-11)*xs + 2 > 0))
    ks = [round(v, 3) for v in np.arange(2.0, 14.0, 0.05)]
    got = [k for k in ks if inA(k)]
    ok = all(10 < k <= 11 for k in got) and all(k in got for k in ks if 10.0 < k <= 11.0 and abs(k-12) > 1e-9)
    V.report(1, ok and min(got) > 10 and max(got) <= 11 + 1e-9, "A={10<k<=11} 격자 확인")
    # B, C: 정수 순서쌍 전수
    lo, hi = sp.Integer(10), sp.Integer(11)
    B = [(m, n) for m in range(-60, 61) for n in range(2, 80) if lo < sp.Rational(m*m, 3) + n <= hi]
    def realroot(m, n):
        return m == 0 or n % 2 == 1 or m > 0
    C = [(m, n) for (m, n) in B if realroot(m, n)]
    # x^n=m 실수해 존재를 다항식 numpy 로도 확인
    def has_real(m, n):
        coef = [1] + [0]*(n-1) + [-m]
        return any(abs(r.imag) < 1e-7 for r in np.roots(coef))
    C2 = [(m, n) for (m, n) in B if has_real(m, n)]
    V.report(1, set(C) == set(C2), "C 판정(수치 근) 일치")
    V.check(1, "|C|", len(C), sp.Integer(int(V.ans(1, "1"))))
    V.report(1, len(B) == 11, f"|B|=11 (실제 {len(B)})")


def p02():
    mp.mp.dps = 30
    # 두 원의 공통접선 4개
    a, b = sp.symbols("a b", real=True)
    sols = []
    for s2 in (1, -1):
        sol = sp.solve([5*(a+b)**2 - (a**2+1), 2*(a+b) - s2*(-2*a+b+3)], [a, b], dict=True)
        sols += sol
    # 조건: 거리 == 반지름 (부호 경우를 모두 확인)
    lines = []
    for s in sols:
        av, bv = s[a], s[b]
        if av.is_real and bv.is_real:
            d1 = abs(av+bv)/sp.sqrt(av**2+1); d2 = abs(-2*av+bv+3)/sp.sqrt(av**2+1)
            if abs(sp.N(d1 - 1/sp.sqrt(5))) < 1e-12 and abs(sp.N(d2 - 2/sp.sqrt(5))) < 1e-12:
                lines.append((av, bv))
    lines = list({(sp.nsimplify(p), sp.nsimplify(q)) for p, q in lines})
    V.report(2, len(lines) == 4, f"공통접선 4개 (실제 {len(lines)})")
    slopes = sorted(float(p) for p, q in lines)
    l = [(p, q) for p, q in lines if abs(float(p)-slopes[-1]) < 1e-12][0]
    m = [(p, q) for (p, q) in lines if abs(float(p)-slopes[0]) < 1e-12][0]
    V.check(2, "l", l, (2, -1)); V.check(2, "m", m, (sp.Rational(1, 2), -1))
    # f, g, 곱의 최댓값 (원과 직선 교점으로 직접 각을 구함)
    C1 = (mp.mpf(1), mp.mpf(0), mp.sqrt(mp.mpf(1)/5)); C2 = (mp.mpf(-2), mp.mpf(-3), mp.sqrt(mp.mpf(4)/5))
    def tang(cx, cy, r, sl, ic):
        # 기울기 sl, 접점 구하기: 중심에서 직선에 내린 발
        return None
    def point_of_tangency(circ, sl, ic):
        cx, cy, r = circ
        # 직선 y = sl x + ic ; 중심에서의 발
        d = (sl*cx - cy + ic)/(sl**2+1)
        return (cx - sl*d, cy + d)
    A1 = point_of_tangency(C1, mp.mpf(2), mp.mpf(-1))
    A2 = point_of_tangency(C2, mp.mpf(1)/2, mp.mpf(-1))
    def inter(circ, sl, ic):
        cx, cy, r = circ
        # y=sl x+ic 와 원
        A_ = 1+sl**2; B_ = 2*(sl*(ic-cy) - cx); C_ = cx**2 + (ic-cy)**2 - r**2
        D = B_**2 - 4*A_*C_
        xs_ = [(-B_-mp.sqrt(D))/(2*A_), (-B_+mp.sqrt(D))/(2*A_)]
        return [(x_, sl*x_+ic) for x_ in xs_]
    def sinang(P, Q, A):
        v1 = (P[0]-A[0], P[1]-A[1]); v2 = (Q[0]-A[0], Q[1]-A[1])
        return abs(v1[0]*v2[1]-v1[1]*v2[0])/(mp.sqrt(v1[0]**2+v1[1]**2)*mp.sqrt(v2[0]**2+v2[1]**2))
    def f(t):   # m 을 y 방향 t 평행이동, 원 C1, 각 P1 A1 Q1
        P, Q = inter(C1, mp.mpf(1)/2, -1+t); return sinang(P, Q, A1)
    def g(t):
        P, Q = inter(C2, mp.mpf(2), -1+t); return sinang(P, Q, A2)
    V.report(2, all(abs(f(mp.mpf(i)/10) - 2*mp.sqrt(mp.mpf(i)/10 - (mp.mpf(i)/10)**2)) < 1e-15 for i in range(1, 9)), "f(t)=2sqrt(t-t^2) (t=0.9 는 A1=P1 로 각 정의 불가라 제외)")
    V.report(2, all(abs(g(mp.mpf(i)/10) - mp.sqrt(4*mp.mpf(i)/10 - (mp.mpf(i)/10)**2)/2) < 1e-15 for i in range(1, 10)), "g(t)=sqrt(4t-t^2)/2")
    h = lambda t: f(t)*g(t)
    tm = mp.findroot(lambda t: mp.diff(lambda u: h(u), t), 0.2)
    exp = sp.sympify(V.ans(2, "1").split("=")[1])
    V.check(2, "M", sp.N(tm, 25), sp.N(exp, 25), 1e-12)
    ts = [mp.mpf(i)/2000 for i in range(1, 2000)]
    V.report(2, max(ts, key=h) - tm < 1e-3 and abs(max(ts, key=h) - tm) < 1e-3, "격자 최댓값 위치 일치")


def p03():
    mp.mp.dps = 30
    f = lambda t: mp.quad(lambda x: 1/(1+x**4)**mp.mpf(0.25) - x**4/(1+x**4)**mp.mpf(1.25), [0, t])
    V.report(3, all(abs(f(mp.mpf(i)/5) - mp.mpf(i)/5/(1+(mp.mpf(i)/5)**4)**mp.mpf(0.25)) < 1e-20 for i in range(1, 6)), "f(t)=t/(1+t^4)^(1/4)")
    v = lambda t: 3*t**2*(f(t)+1)
    s = mp.quad(lambda t: abs(v(t)), [0, 0.5, 1])
    V.check(3, "s^4", sp.N(s**4, 15), sp.N(sp.sympify(V.ans(3, "1").split("=")[1]), 15), 1e-12)
    V.report(3, abs(s - mp.mpf(2)**(mp.mpf(3)/4)) < 1e-20, "s=2^(3/4)")


def p04():
    mp.mp.dps = 40
    a = lambda n: mp.mpf(n-1)**n / mp.mpf(n)**(n-1)
    V.report(4, a(1) == 0 and all(a(n) < a(n+1) for n in range(1, 60)), "a_1=0, 증가")
    bad = []
    for n in range(1, 13):
        for u in (0.0001, 0.3, 0.5, 0.9999, 1.0):
            x = a(n) + (a(n+1)-a(n))*mp.mpf(u)
            if x <= a(n):
                continue
            vals = [mp.log(mp.mpf(k)/x)/k for k in range(1, 5*n+1)]
            if abs(max(vals) - mp.log(mp.mpf(n)/x)/n) > mp.mpf(10)**-25:
                bad.append((n, u))
    V.proof(4, "최댓값이 k=n 에서 달성(전수 격자)", bad)
    # 유일성: a_n 이 이 값이어야 함 — 경계 x=a_{n+1} 에서 k=n,n+1 이 같아짐(경계 확인)
    V.report(4, all(abs(mp.log(mp.mpf(n)/a(n+1))/n - mp.log(mp.mpf(n+1)/a(n+1))/(n+1)) < mp.mpf(10)**-30 for n in range(1, 30)), "x=a_{n+1}에서 k=n,n+1 동률")
    N = 20000
    part = sum((a(n+1)/n)**(mp.mpf(1)/n) - (a(n)/n)**(mp.mpf(1)/n) for n in range(1, 400)) if False else sum(mp.mpf(1)/n - mp.mpf(1)/(n+1) for n in range(1, N+1))
    chk = all(abs((a(n+1)/n)**(mp.mpf(1)/n) - (a(n)/n)**(mp.mpf(1)/n) - (mp.mpf(1)/n - mp.mpf(1)/(n+1))) < mp.mpf(10)**-30 for n in range(1, 40))
    V.report(4, chk, "항 = 1/n - 1/(n+1)")
    V.check(4, "S", sp.nsimplify(round(float(part), 3)), sp.sympify(V.ans(4, "1").split("=")[1]))


if __name__ == "__main__":
    V.run(globals())
