#!/usr/bin/env python3
"""CUK-G26B (가톨릭대 2026 수시 논술 의예·약학) 문항 1~4 검산. 정답은 V.ans() 로 YAML 에서 읽고 문제 조건에서 독립 재계산한다."""
import sys, pathlib, itertools
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G26B", "items/CUK/PAST")


def p01():
    # 정수 d1,d2,d3(≠0), K 를 전수탐색: 조건 (가)~(라) 를 수열을 직접 만들어 확인
    N = 60
    sols = []
    for d1 in [d for d in range(-40, 41) if d]:
        for K in range(1, 46):
            a = lambda n: 35 + (n - 1)*d1
            for d2 in [d for d in range(-120, 121) if d]:
                bK = a(K)
                b = lambda n: bK + (n - K)*d2
                # x_1..x_8 로 S7=S8 (x_8=0) 먼저 걸러냄
                def xk(n, c=None):
                    if n <= K: return a(n)
                    if n <= K + 15 or c is None: return b(n)
                    return c(n)
                if xk(8) != 0:
                    continue
                for d3 in [d for d in range(-120, 121) if d]:
                    cK = b(K + 15)
                    c = lambda n: cK + (n - K - 15)*d3
                    if xk(46, c)*xk(47, c) >= 0:
                        continue
                    xs = [None] + [xk(n, c) for n in range(1, N)]
                    S = [0]
                    for n in range(1, N): S.append(S[-1] + xs[n])
                    if xs[1] != 35 or xs[46]*xs[47] >= 0 or S[7] != S[8]:
                        continue
                    if S[19]*S[17] - S[19]*S[18] + S[18]**2 - S[18]*S[17] - 25 != 0:
                        continue
                    sols.append((d1, d2, d3, K, S[46]))
    V.check(1, "해의 개수", len(sols), 1)
    d1, d2, d3, K, S46 = sols[0]
    for key, val in zip(["d1", "d2", "d3", "K", "S46"], sols[0]):
        V.check(1, key, val, V.ans(1, "1", key))


def p02():
    import numpy as np
    def f(x, a, b): return np.where(x >= 0, 2*np.maximum(x, 0)/(np.maximum(x, 0)**2 + a), b*x)
    def gmax(t, c, a, b):  # [t,t+c] 에서 f 의 최댓값 (후보: 끝점, 0, √a)
        t = np.asarray(t, float)
        cand = [f(t, a, b), f(t + c, a, b)]
        for p in (0.0, np.sqrt(a)):
            cand.append(np.where((t <= p) & (p <= t + c), f(np.full_like(t, p), a, b), -np.inf))
        return np.max(cand, axis=0)
    def jump(t0, c, a, b, h=1e-6):
        l = (gmax(t0, c, a, b) - gmax(t0 - h, c, a, b))/h
        r = (gmax(t0 + h, c, a, b) - gmax(t0, c, a, b))/h
        return float(abs(r - l))
    TS = np.arange(-8, 24 + 1e-9, 1/2000)
    def count_inter(m, c, a, b):
        d = gmax(TS, c, a, b) - (m*TS + 1/c)
        z = list(TS[np.abs(d) < 1e-11])
        idx = np.where(d[:-1]*d[1:] < 0)[0]
        z += list(TS[idx] - d[idx]*(TS[idx+1] - TS[idx])/(d[idx+1] - d[idx]))
        z.sort(); out = []
        for v in z:
            if not out or v - out[-1] > 1e-6: out.append(v)
        return len(out)
    A = [float(sp.N(sp.sympify(s.split('=')[1]), 25)) for s in V.ans(2, "1").replace(" ", "").split(";")]
    a, b, c = A
    xs = np.arange(0, 30, 0.01)
    V.report(2, bool(np.all(f(xs, a, b) <= f(np.array([c]), a, b)[0] + 1e-12)), "(ㄴ)(가) x≥0 에서 f(x)<=f(c)")
    V.report(2, jump(-0.5, c, a, b) > 1e-3, f"(ㄴ)(나) t=-1/2 에서 g 미분불가 (기울기 점프 {jump(-0.5, c, a, b):.4f})")
    V.report(2, count_inter(1/25, c, a, b) == 2, "(ㄷ)(나) m=1/25 교점 2개")
    V.report(2, all(count_inter(m, c, a, b) == 3 for m in (0.01, 0.02, 0.03, 0.039)), "(ㄷ)(다) 0<m<1/25 교점 3개(표본)")
    # 유일성(b<0): (가)로 a=c², (나) 꺾임 조건 -b/2=f(c-1/2) 로 b 를 정하고 (ㄷ) 의 기울기 조건 검사 → c=2 하나뿐
    hits = []
    for ci in range(51, 1001):
        cc = ci/100; aa = cc*cc
        bb = -2*float(f(np.array([cc - 0.5]), aa, 0)[0])
        if jump(-0.5, cc, aa, bb) < 1e-3: continue
        slope = ((1/cc) + bb/2)/0.5
        if abs(slope - 1/25) < 1e-9: hits.append(cc)
    V.report(2, hits == [c], f"b<0 유일성 c={hits}")
    # b≥0: 꺾임은 t=-c 뿐이므로 c=1/2, b 를 훑어 (ㄷ) 의 두 조건을 모두 만족하는 b 가 없음을 확인
    ok = [bv for bv in np.arange(0, 6, 0.05) if count_inter(1/25, 0.5, 0.25, bv) == 2 and count_inter(0.02, 0.5, 0.25, bv) == 3
          and jump(-0.5, 0.5, 0.25, bv) > 1e-3]
    V.report(2, not ok, f"b≥0 해 없음 {ok[:2]}")


def p03():
    import numpy as np
    for k in (1, 2, 3, 5):
        t0 = 3.3
        g = lambda t: k*(4*np.pi - t)*np.cos(k*t) + np.sin(k*t)
        tang = lambda x: k*np.cos(k*t0)*(x - t0) + np.sin(k*t0)   # 접선식에서 x=4π 의 y좌표 = g(t0)
        assert abs(tang(4*np.pi) - g(t0)) < 1e-9
        a = 4*np.pi - 5*np.pi/(2*k); b = 4*np.pi - 3*np.pi/(2*k)
        N = 400000
        T = np.linspace(a, 4*np.pi, N)
        gv = g(T)
        hmax = np.maximum.accumulate(gv[::-1])[::-1]   # [x,4π] 최대
        hmin = np.minimum.accumulate(gv[::-1])[::-1]
        H = hmax - hmin
        sel = T <= b
        C = np.trapezoid(H[sel], T[sel])
        e = float(sp.N(sp.sympify(V.ans(3, "1")), 15))
        V.report(3, abs(2*k*C - e) < 1e-4, f"k={k} 2kC 계산 {2*k*C:.6f} / YAML {e:.6f}")


def p04():
    for n, key in ((29, "E29"), (30, "E30")):
        cnt = {}
        tot = 0
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                X = min(i, j - i, n - j + 1)
                cnt[X] = cnt.get(X, 0) + 1
                tot += 1
        E = sum(Fraction(k*v, tot) for k, v in cnt.items())
        V.check(4, f"E(X) n={n}", sp.Rational(E.numerator, E.denominator), V.ans(4, "1", key))
    # P(X=k) 공식(풀이의 세 경우) 확인: 3m, 3m+1, 3m+2 각 몇 개
    for n in range(6, 40):
        cnt = {}
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                X = min(i, j - i, n - j + 1); cnt[X] = cnt.get(X, 0) + 1
        q = (n + 1)//3
        ok = max(cnt) == q and all(cnt[k] == 3*(n - 3*k + 1) for k in range(1, q)) and cnt[q] == {0: 1, 1: 3, 2: 6}[(n + 1) % 3]
        # n+1=3q 이면 1, 3q+1 이면 3, 3q+2 이면 6 (n 으로는 n=3m→6,n=3m+1→12,n=3m+2→2 는 위 개수×2 아님: 아래서 확률식 직접 비교)
        r = n % 3
        Pq = Fraction({0: 6, 1: 12, 2: 2}[r], n*(n - 1))
        okp = Fraction(cnt[q], n*(n - 1)//2) == Pq
        V.report(4, ok or okp, f"n={n} 분포") if not (ok or okp) else V.report(4, True, f"n={n} 분포")


if __name__ == "__main__": V.run(globals())
