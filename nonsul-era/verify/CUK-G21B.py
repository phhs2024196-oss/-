#!/usr/bin/env python3
"""CUK-G21B (가톨릭대 2021 수시 논술 의예과 1~4) — 문제 조건에서 독립 재계산."""
import sys, pathlib
from fractions import Fraction as Fr
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G21B", "items/CUK/PAST")
mp.mp.dps = 30


def p01():
    # 시합 규칙: 연속으로 2번 이기는 사람이 승자. 게임 단위 정확한 확률 DP, 상태 = 직전 게임 승자
    pS = {None: Fr(3, 5), "S": Fr(3, 5), "E": Fr(1, 3)}
    st = {None: Fr(1)}
    endS, endE = {}, {}
    N = 2020
    for g in range(1, N + 1):
        nst = {}
        for last, pr in st.items():
            for w, q in (("S", pS[last]), ("E", 1 - pS[last])):
                if last == w:                       # 연속 2승 → 시합 종료
                    (endS if w == "S" else endE)[g] = (endS if w == "S" else endE).get(g, 0) + pr * q
                else:
                    nst[w] = nst.get(w, 0) + pr * q
        st = nst
    p = sum(endS.values())
    q = endS[N] / (endS[N] + endE[N])
    ep, eq = (Fr(x) for x in (V.ans(1, "1", "p"), V.ans(1, "1", "q")))
    V.report(1, abs(p - ep) < Fr(1, 10**100) and q == eq,
             f"p≈{float(p):.12f} (YAML {ep}), q={q} (YAML {eq})")


def p02():
    x = mp.mpf
    # f(x) 를 n=300 에서 직접 계산하여 구간별 식과 비교
    def fn(v, n=40000):
        v = mp.mpf(v)
        return (3 * v ** (3 * n + 2) + 3 * v ** (2 * n) + v + 5) / (v ** (3 * n) + 1)
    def f(v):
        v = mp.mpf(v)
        return v + 5 if abs(v) < 1 else (6 if v == 1 else 3 * v * v)
    pts = [-3, -1.5, -1.01, -0.99, -0.5, 0, 0.3, 0.99, 1.01, 2.5]
    okf = all(abs(fn(v) - f(v)) < 1e-6 for v in pts) and fn(1) == 6
    # (ㄴ) 접선 조건: 접점 (t, t+5) 에서 기울기 1 & 값 일치 → (a,t) 를 b 마다 수치로 풀고 a=8-4b, t∈(-1,1] 확인
    a_, t_ = sp.symbols("a t", real=True)
    bb = sp.symbols("b", positive=True)
    # 접점 (t,t+5): 기울기 a/(2√(a(t+b)))=1 → √(a(t+b))=a/2 ; 값 t+5=√(a(t+b))+3 → t=a/2-2
    tt = sp.solve(sp.Eq(t_ + 5, a_ / 2 + 3), t_)[0]
    sa = sp.solve(sp.Eq(a_ * (tt + bb), (a_ / 2) ** 2), a_)
    okr = sorted(sa, key=str) == sorted([0, 8 - 4 * bb], key=str) and sp.simplify(tt.subs(a_, 8 - 4 * bb) - (2 - 2 * bb)) == 0
    # 접점 t 가 (-1,1] 이고 a>0,b>1 인 b 의 범위
    inb = [bv for bv in (1.001, 1.2, 1.499, 1.5, 1.6, 2, 3) if -1 < 2 - 2 * bv <= 1 and 8 - 4 * bv > 0]
    okr = okr and inb == [1.001, 1.2, 1.499]
    # (ㄷ) 교점 개수: b 격자마다 구간별 부호 변화로 센다
    counts = set()
    for i in range(1, 61):
        b = mp.mpf(1) + mp.mpf(i) / 121      # 1 < b < 1.5
        a = 8 - 4 * b
        h = lambda v: mp.sqrt(2 * b * (v + a)) + mp.mpf(5) / 2
        cnt = 0
        for lo, hi, g in ((-a, -1, lambda v: 3 * v * v - h(v)), (-1, 1, lambda v: v + 5 - h(v)), (1, 12, lambda v: 3 * v * v - h(v))):
            N = 4000
            vals = [g(lo + (hi - lo) * (mp.mpf(j) + (0 if j else 1e-9)) / N) for j in range(N)] + [g(hi - mp.mpf(10) ** -9)]
            cnt += sum(1 for u, w in zip(vals, vals[1:]) if u * w < 0)
        cnt += (abs(h(1) - 6) < 1e-12)       # x=1 에서 f(1)=6
        counts.add(int(cnt))
    V.report(2, okf and okr and counts == {int(V.ans(2, "1").split("=")[-1])},
             f"f 구간식 {okf}, a=8-4b {okr}, 교점 개수 집합 {counts} / YAML {V.ans(2,'1')}")


def p03():
    k1, k2, k3 = (int(V.ans(3, "1", k)) for k in ("k1", "k2", "k3"))
    ok = 0 < k2 < 10 < k3 < 50 < k1 < 100
    for n in range(21, 80):
        P = 2 ** n
        r1 = Fr(P + 61) / ((P - 2) * (1 + Fr(1, P + k1)))             # = 1 + A/m1
        r2 = r1 / (1 + Fr(1, 2 ** (n - 1) - k2))                      # = 1 + B/m2
        r3 = r2 / (1 + Fr(1, 2 ** (n - 2) + k3))                      # = 1 + C/m3
        ok &= (r1 - 1 > 0 and r2 - 1 > 0 and r3 - 1 > 0)
        ok &= (r1 - 1 == Fr(31, 2 ** (n - 1) - 1)) and (r2 - 1 == Fr(15, 2 ** (n - 2))) and (r3 - 1 == Fr(7, 2 ** (n - 3)))
    # 조건식을 정확히 만족하는지 재조립
    n = 33
    A1, B1, C1 = Fr(31, 2 ** (n - 1) - 1), Fr(15, 2 ** (n - 2)), Fr(7, 2 ** (n - 3))
    ok &= (2 ** n - 2) * (1 + A1) * (1 + Fr(1, 2 ** n + k1)) == 2 ** n + 61
    ok &= (1 + A1) == (1 + B1) * (1 + Fr(1, 2 ** (n - 1) - k2)) and (1 + B1) == (1 + C1) * (1 + Fr(1, 2 ** (n - 2) + k3))
    V.report(3, ok, f"(k1,k2,k3)=({k1},{k2},{k3}) n=21..79 정확한 분수 재계산 일치, 부등식 조건 충족")


def p04():
    f = lambda x: 2 * mp.quad(lambda th: mp.tan(th) / mp.cos(th) ** 2, [0, x]) + 1
    a = lambda n: f(mp.pi / 2 ** (n + 2)) / 4 ** n
    # 일반항 f(x)=1/cos^2 x 확인
    okf = all(abs(f(mp.mpf(v)) - 1 / mp.cos(v) ** 2) < 1e-20 for v in (0.1, 0.5, 0.78))
    S = mp.fsum(a(n) for n in range(1, 70))
    exp = sp.N(sp.sympify(V.ans(4, "1")), 25)
    V.report(4, okf and abs(S - mp.mpf(str(exp))) < 1e-18, f"S≈{mp.nstr(S, 18)} / YAML {V.ans(4,'1')}={exp}")


if __name__ == "__main__":
    V.run(globals())
