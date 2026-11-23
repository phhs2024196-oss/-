import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from fractions import Fraction
from math import comb, erf, sqrt

V = Verifier("YSM-G26M", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "t": t, "n": n, "k": k})

def win_prob(w, l, need):
    """현재 w승 l패에서 need선승제(공정한 승률 1/2)로 최종 승리할 확률 (재귀)"""
    from functools import lru_cache
    @lru_cache(None)
    def f(a, b):
        if a == need: return Fraction(1)
        if b == need: return Fraction(0)
        return (f(a + 1, b) + f(a, b + 1))/2
    return f(w, l)

def p01():
    # 1-1: 시합 수 분포 직접 열거(경기 결과 2^7 중 조기 종료 반영)
    import itertools
    dist = {}
    for seq in itertools.product((0, 1), repeat=7):
        a = b = 0
        for i, s in enumerate(seq):
            a += s; b += 1 - s
            if a == 4 or b == 4:
                m = i + 1; break
        dist[m] = dist.get(m, 0) + Fraction(1, 2**7)
    EN = sum(m*p for m, p in dist.items())
    V.check(1, "1-1 분포", [dist[m]*16 for m in (4, 5, 6, 7)], [2, 4, 5, 5])
    V.check(1, "1-1 E(N)", sp.Rational(EN.numerator, EN.denominator), E(V.ans(1, "1-1")))
    # 1-2: 마진별 최종 승리 확률 (7전 4선승제, 승률 1/2) — 도달 가능한 모든 상황 (승,패) 에서
    states = [(w, l) for w in range(0, 4) for l in range(0, 4) if w != 4 and l != 4 and w > l]
    byM = {}
    for w, l in states: byM.setdefault(w - l, []).append(win_prob(w, l, 4))
    ok = [m for m, ps in byM.items() if all(p >= Fraction(4, 5) for p in ps)]
    V.check(1, "1-2 마진(모든 상황 ≥80%)", sorted(ok), [E(V.ans(1, "1-2", "m_min")), E(V.ans(1, "1-2", "m_max"))])
    V.check(1, "1-2 3승2패", sp.Rational(win_prob(3, 2, 4).numerator, win_prob(3, 2, 4).denominator), sp.Rational(3, 4))
    V.check(1, "1-2 1승(1:0)", sp.Rational(win_prob(1, 0, 4).numerator, win_prob(1, 0, 4).denominator), sp.Rational(21, 32))
    V.check(1, "1-2 2승1패", sp.Rational(win_prob(2, 1, 4).numerator, win_prob(2, 1, 4).denominator), sp.Rational(11, 16))
    V.check(1, "1-2 2승(2:0)", sp.Rational(win_prob(2, 0, 4).numerator, win_prob(2, 0, 4).denominator), sp.Rational(13, 16))
    # 1-3: 이항 B(100,3/4) 정규근사. 표값(0.5-0.3770)과 일치, 정확한 정규값과 근사 일치
    mu, var = 100*0.75, 100*0.75*0.25
    z = (80 - mu)/sqrt(var)
    table = 0.5 - 0.3770
    V.check(1, "1-3 표값", sp.Float(table, 6), sp.Float(E(V.ans(1, "1-3")), 6), 1e-9)
    V.report(1, abs(z - 2/sqrt(3)) < 1e-12 and abs((1 - 0.5*(1 + erf(z/sqrt(2)))) - table) < 2e-3, "1-3 정규근사 z=2/√3≈1.16, 표 0.123 ≈ 1-Φ(z)")
    # 1-4
    p = sp.Rational(3, 5)
    p3 = sum(sp.binomial(2 + i, i)*p**3*(1 - p)**i for i in range(3))
    V.check(1, "1-4 5전3선승제", p3, E(V.ans(1, "1-4", "p2")))
    V.report(1, p3 > sp.Rational(V.ans(1, "1-4", "p1")), "1-4 5전3선승제 > 단판")

def p02():
    nn = sp.symbols("nn", integer=True, positive=True)
    kk = sp.symbols("kk", integer=True, positive=True)
    # 2-1
    Sn = sp.summation((2*kk - 1)/(kk*(kk + 1)*(kk + 2)), (kk, 1, nn))
    alpha = sp.limit(Sn, nn, sp.oo)
    V.check(2, "2-1 alpha", alpha, E(V.ans(2, "2-1", "alpha")))
    V.check(2, "2-1 S_n 식", sp.simplify(Sn - nn*(3*nn + 1)/(4*(nn + 1)*(nn + 2))), 0)
    V.report(2, sp.limit(Sn/alpha**nn, nn, sp.oo) == sp.oo, "2-1 일반항 S_n/alpha^n → ∞ (발산)")
    # 2-2: sin^2(Mx)-sin^2(x)=sin((M+1)x)sin((M-1)x) 로 |sin(Mx)|=sin x 의 해를 독립적으로 센다
    import math
    N = 11; M = 2**N
    xv = sp.Rational(3, 10)
    V.check(2, "2-2 항등식", sp.N(sp.sin(M*xv)**2 - sp.sin(xv)**2 - sp.sin((M + 1)*xv)*sp.sin((M - 1)*xv), 30), 0, 1e-20)
    roots = {Fraction(j, M + 1) for j in range(1, M + 1)} | {Fraction(l, M - 1) for l in range(1, M - 1)}
    roots = {r for r in roots if 0 < r < 1}          # x = rπ, 0<x<π
    V.check(2, "2-2 실근 개수", len(roots), E(V.ans(2, "2-2")))
    V.report(2, all(abs(abs(math.sin(M*math.pi*float(r))) - math.sin(math.pi*float(r))) < 1e-6 for r in roots), "2-2 모든 근이 |sin(2^11 x)|=sin x 를 만족(수치)")
    # 해설의 두 경우(①,②) 분류와 개수: ① 2047, ② 2047 (정확한 유리수 판정)
    def kind(r, sg):
        v1 = (M - sg)*r; v2 = (M + sg)*r - 1
        return (v1.denominator == 1 and v1.numerator % 2 == 0) or (v2.denominator == 1 and v2.numerator % 2 == 0)
    c1 = {r for r in roots if kind(r, 1)}; c2 = {r for r in roots if kind(r, -1)}
    V.report(2, len(c1) == M - 1 and len(c2) == M - 1 and not (c1 & c2) and len(c1 | c2) == len(roots), "2-2 ①②가 각각 2^11-1개, 서로소")
    # P(x) 식 확인
    xv = sp.Rational(7, 10)
    P = sp.prod([sp.cos(2**i*xv) for i in range(11)])
    V.check(2, "2-2 P=sin(2^11x)/(2^11 sin x)", sp.N(P - sp.sin(2**11*xv)/(2**11*sp.sin(xv)), 30), 0, 1e-20)
    # 2-3: C_n/(n^2 sin(θ/2)) 수렴값
    def ratio(nv):
        th = sp.pi/(2*nv + 1)
        C = sum((2*kv - 1)*(sp.cos(kv*th) - sp.cos((kv + 1)*th)) for kv in range(1, nv + 1))
        return sp.N(C/(nv**2*sp.sin(th/2)), 30)
    target = sp.N(E(V.ans(2, "2-3")), 30)
    errs = [abs(ratio(nv) - target) for nv in (200, 800)]
    V.report(2, errs[1] < errs[0] < 0.02 and errs[1] < 0.005, f"2-3 n=200 오차 {float(errs[0]):.4f}, n=800 오차 {float(errs[1]):.4f} (감소)")
    V.check(2, "2-3 적분", sp.simplify(8*sp.integrate(2*x*sp.sin(sp.pi*x), (x, 0, sp.Rational(1, 2)))), E(V.ans(2, "2-3")))

def p03():
    al = sp.symbols("al", positive=True)
    # 도형을 좌표로 독립 구성: 부채꼴 BOC (각 θ=2α) 에 내접하는 원
    s, c = sp.sin(al), sp.cos(al)
    g = s/(1 + s); k_ = c/(1 + s)
    G = sp.Matrix([k_, g])                         # 중심: 각이등분선 위, x축에 접
    V.report(3, all(abs(float((sp.sqrt(G.dot(G)) - (1 - g)).subs(al, av))) < 1e-12 for av in (0.2, 0.7, 1.3)), "3-1 |OG|=1-g (내접)")
    V.report(3, all(abs(float((sp.sqrt(G.dot(G) - g**2) - k_).subs(al, av))) < 1e-12 for av in (0.2, 0.7, 1.3)), "3-1 OF=OH=k")
    # 타원: 장축 AB(반장축 1), 초점 H → c=OH. 단축 반 b=OF=k
    sol = sp.solve(sp.Eq(1 - k_**2, k_**2), al)
    sa = [sp.simplify(sp.sin(v)) for v in sol if 0 < sp.N(v) < sp.pi/2]
    V.check(3, "3-1 sinα", sa[0], sp.Rational(1, 3))
    S = sp.pi*g**2
    dSdt = sp.diff(S, al)*sp.sqrt(2)/2
    V.check(3, "3-1 dS/dt", sp.simplify(dSdt.subs(al, sol[0])), E(V.ans(3, "3-1")))
    # 3-2: θ 를 스캔해 조건 JI·JO >= IJ·IF 의 만족 구간과 |OQ|^2 최댓값
    th = sp.symbols("th", positive=True)
    def cond(thv):
        a2 = thv/2
        k2 = sp.cos(a2)/(1 + sp.sin(a2))
        F = k2*sp.Matrix([sp.cos(thv), sp.sin(thv)])
        O = sp.Matrix([0, 0])
        I = sp.Matrix([F.dot(F)/F[0], 0])        # F 를 지나 OC 에 수직인 직선: (X-F)·F=0, y=0
        J = sp.Matrix([0, F.dot(F)/F[1]])
        JI, JO, IJ, IF = I - J, O - J, J - I, F - I
        return JI.dot(JO) - IJ.dot(IF)
    import numpy as np
    ths = [i*np.pi/2/400 for i in range(1, 400)]
    okt = [tv for tv in ths if float(cond(sp.Float(tv, 20))) >= -1e-12]
    V.report(3, abs(max(okt) - np.pi/4) < np.pi/2/400 + 1e-9 and abs(min(okt) - ths[0]) < 1e-9, f"3-2 조건 만족 θ ∈ (0, π/4]  (max={max(okt):.4f})")
    # Q = O 에서 AC 로 내린 수선의 발: OQ ⟂ AQ, OQ ⟂ CQ 만족, |OQ|=sinα
    Av = sp.Matrix([-1, 0]); thv = sp.pi/4; Cv = sp.Matrix([sp.cos(thv), sp.sin(thv)])
    d = Cv - Av; Q = Av + d*((-Av).dot(d))/d.dot(d)
    V.check(3, "3-2 OQ·AQ=0", sp.simplify(Q.dot(Q - Av)), 0)
    V.check(3, "3-2 OQ·CQ=0", sp.simplify(Q.dot(Q - Cv)), 0)
    V.check(3, "3-2 |OQ|^2 최댓값", sp.simplify(Q.dot(Q).subs(thv, thv)), E(V.ans(3, "3-2")))
    # 3-3: 원 O2 (중심 (f,f): OD 접, AB 접, 단위원 내접) 가 AC 에 접하는 θ 를 수치로 구해 OI 계산
    f = sp.symbols("f", positive=True)
    fv = sp.solve(sp.Eq(2*f**2, (1 - f)**2), f)
    fv = [v for v in fv if 0 < sp.N(v) < 1][0]
    L = sp.Matrix([fv, fv])
    def dist_to_AC(thv):
        Cv = sp.Matrix([sp.cos(thv), sp.sin(thv)]); d = Cv - Av
        w = L - Av
        return sp.Abs(d[0]*w[1] - d[1]*w[0])/sp.sqrt(d.dot(d))
    thv = sp.nsolve(dist_to_AC(th) - fv, th, 1.1, prec=40)
    V.report(3, 0 < thv < sp.pi/2, f"3-3 θ={float(thv):.5f} ∈ (0,π/2)")
    OI = sp.N((k_/sp.cos(2*al)).subs(al, thv/2)*1, 40)
    # OI = OH/cosθ (k=OH, 원 O1 의 H)
    kk = sp.cos(thv/2)/(1 + sp.sin(thv/2))
    OI = sp.N(kk/sp.cos(thv), 40)
    V.check(3, "3-3 OI", OI, sp.N((19 + 6*sp.sqrt(2))/21, 40), 1e-25)
    u, v, w = 19, 6, 21
    V.check(3, "3-3 u+v+w", u + v + w, E(V.ans(3, "3-3")))
    V.report(3, sp.igcd(sp.igcd(u, v), w) == 1, "3-3 gcd(u,v,w)=1")
    V.check(3, "3-3 tan(α/2)", sp.N(sp.tan(thv/4), 30), sp.N((2 - sp.sqrt(2))/2, 30), 1e-25)

if __name__ == "__main__": V.run(globals())
