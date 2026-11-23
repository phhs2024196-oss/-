import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import math
import numpy as np
from fractions import Fraction as Fr

V = Verifier("YSM-G22M", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "t": t, "E": sp.E})

def p01():
    # 1-1: i번째 사람의 생존 = 실패한 줄 수 <= i-1 (줄마다 독립, 실패 확률 1-p)
    m, i, p = 20, 18, Fr(3, 4)
    surv = sum(math.comb(m, r)*p**(m - r)*(1 - p)**r for r in range(i))
    a_ = (1 - surv)*2**40
    V.check(1, "1-1 a", sp.Integer(a_.numerator) if a_.denominator == 1 else sp.Rational(a_.numerator, a_.denominator), E(V.ans(1, "1-1")))
    # 몬테카를로로 모형(실패 줄 수 <= i-1) 자체를 점검: 소규모(m=6, 4명)에서 직접 시뮬레이션
    import random
    random.seed(1)
    def sim(mm, nn, pp, trials=40000):
        ok = 0
        for _ in range(trials):
            alive = nn; known = [False]*mm
            res = None
            for person in range(nn):
                dead = False
                for r_ in range(mm):
                    if known[r_]: continue
                    if random.random() < pp: known[r_] = True
                    else: known[r_] = True; dead = True; break
                if not dead and person == nn - 1: res = True
                if dead and person == nn - 1: res = False
            ok += bool(res)
        return ok/trials
    mm_, nn_, pp_ = 6, 4, 0.7
    theo = sum(math.comb(mm_, r)*pp_**(mm_ - r)*(1 - pp_)**r for r in range(nn_))
    V.report(1, abs(sim(mm_, nn_, pp_) - theo) < 0.01, f"1-1 모형 시뮬레이션 (m=6, 4명) 이론 {theo:.4f}")
    # 1-2: 그래프 q, 시간 부등식, 확률의 소인수분해
    q = Fr(1, 2)/100
    V.report(1, q == Fr(1, 200), "1-2 q=1/200")
    ts = [Fr(j, 100) for j in range(1, 300)]
    good = [tt for tt in ts if 20*tt + 21*2 + 15*2 <= 112 < 20*tt + 21*2 + 16*2]
    V.report(1, good and min(good) > Fr(19, 10) and max(good) == 2, f"1-2 관찰시간 범위 {min(good)}..{max(good)} (1.9<t<=2)")
    # p = 1/2 + q*ceil(10 t) : t in (1.9,2] -> 20 q
    pc = Fr(1, 2) + 20*q
    V.report(1, pc == Fr(3, 5), "1-2 p=0.6")
    prob = math.comb(20, 13)*pc**7*(1 - pc)**13
    num = prob*10**20
    assert num.denominator == 1
    f = sp.factorint(int(num.numerator))
    b = [f.get(pr, 0) for pr in (2, 3, 5, 7, 11, 13, 17, 19)]
    V.report(1, set(f) <= {2, 3, 5, 7, 11, 13, 17, 19}, f"1-2 소인수 {f}")
    V.check(1, "1-2 b1-sum", b[0] - sum(b[1:]), E(V.ans(1, "1-2")))
    # 1-3: f(t)=e^t(sin t+C), f(0)=1/2 → 적분방정식 성립(a=0), t* 존재
    C = sp.Rational(1, 2)
    ff = sp.exp(t)*(sp.sin(t) + C)
    V.check(1, "1-3 f(0)", ff.subs(t, 0), sp.Rational(1, 2))
    V.check(1, "1-3 YAML 식", sp.simplify(ff - E(V.ans(1, "1-3"))), 0)
    lhs = sp.integrate(ff, (t, 0, x)); rhs = -(sp.sin(x) + sp.cos(x))/2*sp.exp(x) + ff.subs(t, x)
    V.check(1, "1-3 적분방정식(a=0)", sp.simplify(lhs - rhs), 0)
    ts_ = sp.nsolve(ff - 1, t, 0.3)
    V.report(1, 0 < ts_ < 1 and all(float(ff.subs(t, u)) < 1 for u in np.linspace(0, float(ts_) - 1e-3, 200)), f"1-3 t*≈{float(ts_):.5f} 이전에는 f<1")

def count_h(x0, tol=1e-9):
    s, c = math.sin(math.pi*x0), math.cos(math.pi*x0)
    if s < 1e-12: return None
    g = lambda y: math.cos(math.pi*y) - c + math.pi*s*(y - x0)
    Y = x0 + (1 + c)/(math.pi*s) + 2
    ys = np.linspace(x0 + 1e-5, Y, 60001)
    gv = np.array([g(y) for y in ys])
    cnt = int(np.sum(gv[:-1]*gv[1:] < 0))
    # 접하는 근(중근) 검출: 국소 최솟값이 거의 0
    from scipy.optimize import minimize_scalar
    for i in range(1, len(ys) - 1):
        if gv[i - 1] > gv[i] < gv[i + 1] and gv[i] < 1e-3:
            r = minimize_scalar(g, bounds=(ys[i - 1], ys[i + 1]), method="bounded", options={"xatol": 1e-14})
            if abs(r.fun) < tol and gv[i - 1] > 0 and gv[i + 1] > 0: cnt += 1
    return cnt

def p02():
    xs = np.linspace(0.02, 0.98, 49)
    hv = {round(v, 4): count_h(v) for v in xs}
    V.report(2, count_h(0.5) == 0, f"2-1 h(1/2)={count_h(0.5)}")
    V.report(2, all(count_h(v) == 0 for v in np.linspace(0.51, 0.99, 25)), "2-1/2-2 1/2<x<1 에서 h=0")
    V.report(2, count_h(0.49) == 1 and count_h(0.45) >= 1, f"2-1 1/2 의 왼쪽: h(0.49)={count_h(0.49)}, h(0.45)={count_h(0.45)} (≠0)")
    V.report(2, hv[0.02] is not None and hv[0.02] > 8, f"2-1 x→0 에서 h 증가(h(0.02)={hv[0.02]})")
    # 2-2: 확률밀도함수 상수
    k_ = sp.Symbol("k_", real=True)
    I = sp.integrate(sp.exp(x)*sp.cos(sp.pi*x), (x, sp.Rational(1, 2), 1))
    V.check(2, "2-2 k", sp.simplify(1/I), E(V.ans(2, "2-2")))
    V.report(2, float(E(V.ans(2, "2-2"))) < 0 and all(float(E(V.ans(2, "2-2")))*math.exp(u)*math.cos(math.pi*u) >= -1e-12 for u in np.linspace(0.5, 1, 20)), "2-2 p(x)>=0")
    # 2-3: h(b)=8 인 b (0<b<1/2) 에서 bπ+cot(bπ)=9π/2, 접선이 c=9-b 에서 접함
    from scipy.optimize import brentq
    F = lambda b: b*math.pi + 1/math.tan(b*math.pi) - 4.5*math.pi
    cand = []
    grid = np.linspace(0.005, 0.499, 4000)
    for a_, b_ in zip(grid[:-1], grid[1:]):
        if F(a_)*F(b_) < 0: cand.append(brentq(F, a_, b_, xtol=1e-14))
    V.report(2, len(cand) >= 1, f"2-3 방정식 해 후보 {[round(c_, 6) for c_ in cand]}")
    ok8 = []
    for b0 in cand:
        c0 = 9 - b0
        tang = abs(math.sin(math.pi*c0) - math.sin(math.pi*b0)) < 1e-9 and abs(
            (math.cos(math.pi*b0) - math.cos(math.pi*c0))/(b0 - c0) + math.pi*math.sin(math.pi*b0)) < 1e-9
        ok8.append((b0, tang, count_h(b0 - 1e-7), count_h(b0 + 1e-7)))
    V.report(2, any(t_[1] for t_ in ok8), f"2-3 c=9-b 에서 접함 확인 {ok8}")
    b0 = cand[0]
    V.check(2, "2-3 값", sp.nsimplify(b0*math.pi + 1/math.tan(b0*math.pi), [sp.pi], tolerance=1e-9), E(V.ans(2, "2-3")))

if __name__ == "__main__": V.run(globals())
