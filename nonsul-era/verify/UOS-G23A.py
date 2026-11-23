import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, math
from fractions import Fraction
V = Verifier("UOS-G23A", "items/UOS/PAST")
R = sp.Rational

def dist_end():
    """주머니 공 개수 DP: (끝난 횟수, 최종 공 개수) 분포."""
    cur = {0: Fraction(1)}; ends = {}; step = 0
    while cur:
        step += 1; nxt = {}
        for s, p in cur.items():
            for d in range(1, 7):
                t = s + d; q = p/6
                if t >= 6: ends[(step, t)] = ends.get((step, t), 0) + q
                else: nxt[t] = nxt.get(t, 0) + q
        cur = nxt
    return ends

def p01():
    ends = dist_end()
    pa = sum(v for (k, t), v in ends.items() if t == 6)
    fa = sp.sympify(V.ans(1, "1-1"))
    V.check(1, "(a) P(공=6)", R(pa.numerator, pa.denominator), fa)
    V.check(1, "(a) 7^5/6^6", R(7**5, 6**6), fa)
    pb = sum(v for (k, t), v in ends.items() if k == 4)
    pab = sum(v for (k, t), v in ends.items() if k == 4 and t == 6)
    V.check(1, "P(B)=55/6^4", R(pb.numerator, pb.denominator), R(55, 6**4))
    V.check(1, "P(A∩B)=10/6^4", R(pab.numerator, pab.denominator), R(10, 6**4))
    c = pab/pb
    V.check(1, "(b) 조건부확률", R(c.numerator, c.denominator), V.ans(1, "1-2"))

def p02():
    xx = sp.symbols("xx", real=True)
    f = lambda t: -t**3 - t + 3
    fp = lambda t: -3*t**2 - 1
    # g 와 g' 를 수치로: g(y)=f^{-1}(y)
    def g(y):
        return float(sp.nsolve(-xx**3 - xx + 3 - y, xx, 1.0))
    import numpy as np
    from numpy import trapezoid as _tz
    def gp(y):
        return 1.0/float(fp(g(y)))
    # h 한 주기 [0,3) 의 연속성 확인: 3 vs 4g'(1)+4
    V.check(2, "h 연속(x=1)", 3, 4*gp(1.0) + 4, 1e-9)
    N = 20001
    xs = np.linspace(0, 3, N)
    def h(t):
        t = t % 3
        return 3*t if t < 1 else 4*gp(t) + 4
    xs = np.linspace(0, 6, 6001)
    hs = np.array([h(t) for t in xs]); hs[1000] = 3.0
    val = float(_tz(xs*hs, xs))
    V.check(2, "∫0^6 x h dx 수치", val, sp.N(sp.sympify(V.ans(2, "2-1"))), 2e-2)
    # 정확: ∫_1^3 g = ∫_0^1 f - 1
    V.check(2, "∫1^3 g = 5/4", sp.integrate(-x**3 - x + 3, (x, 0, 1)) - 1, R(5, 4))
    V.check(2, "∫1^3 x g' = -9/4", R(-9, 4), 3*0 - 1*1 - R(5, 4))
    I1 = sp.integrate(3*x**2, (x, 0, 1)) + 4*(3*0 - 1*1 - R(5, 4)) + sp.integrate(4*x, (x, 1, 3))
    V.check(2, "∫0^3 x h = 8", I1, 8)
    I2 = sp.integrate(3*x, (x, 0, 1)) + 4*(0 - 1) + 8
    V.check(2, "∫0^3 h = 11/2", I2, R(11, 2))
    V.check(2, "최종", 2*I1 + 3*I2, V.ans(2, "2-1"))

def p03():
    ans = sp.sympify(V.ans(3, "3-1"))
    def closed(N):
        # N 짝수. (a+4b)c^2 = 3*2^N, c=2^r
        tot = sum(3*2**(N - 2 - 2*r) - 1 for r in range(0, N//2))
        ac = sum(1 for r in range(0, N) if 2**(3*r) < 3*2**N and (3*2**(N - 2*r) - 2**r) % 4 == 0 and (3*2**(N - 2*r) - 2**r)//4 >= 1 and N - 2*r >= 0)
        bc = sum(1 for r in range(0, N) if 4*2**(3*r) < 3*2**N and (3*2**(N - 2*r) - 4*2**r) >= 1 and N - 2*r >= 0)
        return tot - ac - bc
    def brute(N):
        T = 3*2**N; cnt = 0
        for c in range(1, int(T**0.5) + 2):
            if T % (c*c): continue
            s = T//(c*c)  # a+4b
            for b in range(1, (s - 1)//4 + 1):
                a = s - 4*b
                if a >= 1 and len({a, b, c}) == 3: cnt += 1
        return cnt
    for N in (8, 10, 12, 14, 16, 20):
        V.check(3, f"N={N} 전수 vs 닫힌식", brute(N), closed(N))
    V.check(3, "N=100 닫힌식 = 2^100-117", closed(100), ans)
    V.check(3, "N=100 일반식 2^N-1-N/2-(ac)-(bc)", 2**100 - 51 - 32 - 34, ans)

def p04():
    lo = sp.sympify(V.ans(4, "4-1", "lo")); hi = sp.sympify(V.ans(4, "4-1", "hi"))
    import numpy as np
    # 수치: 교점 P 를 직접 구해 비 계산 (타원 x²/a²+y²/(a²-25)=1, 쌍곡선 x²/c²-y²/(25-c²)=1, 제1사분면 교점, PF*PF'=20)
    def ratio(c):
        a = math.sqrt(c*c + 20)
        # 교점: x = a*c/5, y² = (a²-25)(1 - x²/a²)
        xp = a*c/5; y2 = (a*a - 25)*(1 - xp*xp/(a*a))
        yp = math.sqrt(y2)
        PF = math.hypot(xp - 5, yp); PFp = math.hypot(xp + 5, yp)
        assert abs(PF*PFp - 20) < 1e-9 and abs(xp*xp/(c*c) - yp*yp/(25 - c*c) - 1) < 1e-9
        return PF/PFp
    vals = [ratio(c) for c in np.linspace(math.sqrt(5) + 1e-6, 5 - 1e-6, 2001)]
    V.check(4, "비의 하한(수치)", min(vals), sp.N(lo), 1e-4)
    V.check(4, "비의 상한(수치)", max(vals), sp.N(hi), 1e-4)
    V.report(4, all(a > b for a, b in zip(vals, vals[1:])), "비는 c 에 대해 감소")
    # (b): ∠FPF'=π/3 ⇒ cos = (PF²+PF'²-100)/(2*20) = 1/2
    def ang(c):
        a = math.sqrt(c*c + 20); xp = a*c/5; yp = math.sqrt((a*a - 25)*(1 - xp*xp/(a*a)))
        PF = math.hypot(xp - 5, yp); PFp = math.hypot(xp + 5, yp)
        return (PF**2 + PFp**2 - 100)/(2*PF*PFp) - 0.5
    cs = np.linspace(math.sqrt(5) + 1e-6, 5 - 1e-6, 4001); fs = [ang(c) for c in cs]
    i = next(j for j in range(len(fs) - 1) if fs[j]*fs[j + 1] < 0)
    c0 = cs[i] - fs[i]*(cs[i + 1] - cs[i])/(fs[i + 1] - fs[i])
    V.check(4, "(b) 수치", ratio(c0), sp.N(sp.sympify(V.ans(4, "4-2"))), 1e-4)
    V.check(4, "(b) 정확", (sp.sqrt(40) - sp.sqrt(20))/(sp.sqrt(40) + sp.sqrt(20)), V.ans(4, "4-2"))

if __name__ == "__main__": V.run(globals())
