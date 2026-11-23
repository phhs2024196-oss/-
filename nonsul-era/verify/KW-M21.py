import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, math
V = Verifier("KW-M21", "items/KW/PAST")
R_ = sp.Rational

def p01():
    # [1](1) 내분점이 제1사분면: 독립 계산 (m 격자 + 기호)
    m = sp.symbols("m", real=True)
    P = ((5*m - (3-m))/3, (3*m - 3*(3-m))/3)
    sol = sp.solve_univariate_inequality(P[0] > 0, m) 
    cond = sp.And(P[0] > 0, P[1] > 0, m > 0, 3 - m > 0)
    rng = sp.solve_univariate_inequality(sp.Min(P[0], P[1], m, 3 - m) > 0, m, relational=False) if False else None
    lo = sp.Rational(3, 2); hi = 3
    good = all(bool((P[0].subs(m, v) > 0) and (P[1].subs(m, v) > 0) and v > 0 and 3 - v > 0) == (lo < v < hi) for v in [R_(i, 20) for i in range(-10, 90)])
    V.report(1, good, "1-1 독립 계산: 제1사분면 ⇔ 3/2<m<3")
    off_lo, off_hi = sp.sympify(V.ans(1, "1-1", "mmin")), sp.sympify(V.ans(1, "1-1", "mmax"))
    # 공식 모범답안(2/3<m<3)과 다름을 확인하고 보고 대상으로 표시(수정하지 않음)
    V.check(1, "1-1 mmin (공식 2/3 오기 정정)", lo, off_lo); V.check(1, "1-1 mmax", hi, off_hi)
    # [1](2)
    xx, yy, k_ = sp.symbols("xx yy k_", real=True)
    d = (xx+1)**2 + (yy+3)**2 - (xx-5)**2 - (yy-3)**2
    V.check(1, "1-2 AC^2-BC^2", sp.expand(d), 12*xx - 24 + 12*yy)
    vals = [float(d.subs({xx: 2 + math.cos(t_), yy: math.sin(t_)})) for t_ in [i*2*math.pi/20000 for i in range(20000)]]
    V.check(1, "1-2 최솟값", min(vals), sp.N(sp.sympify(V.ans(1, "1-2", "lo"))), tol=1e-3)
    V.check(1, "1-2 최댓값", max(vals), sp.N(sp.sympify(V.ans(1, "1-2", "hi"))), tol=1e-3)
    tmin = min(range(20000), key=lambda i: vals[i])*2*math.pi/20000
    V.check(1, "1-2 y좌표", math.sin(tmin), sp.N(sp.sympify(V.ans(1, "1-2", "y"))), tol=1e-3)
    # [2]
    a = sp.symbols("a0:11")
    poly = sp.expand(sum(a[k]*(x-1)**k for k in range(11)) - (x**10 - 1))
    sol = sp.solve(sp.Poly(poly, x).coeffs(), a, dict=True)[0]
    V.check(1, "2 a4+a6+a8+a10", sum(sol[a[k]] for k in (4, 6, 8, 10)), V.ans(1, "2"))
    # [3] 대합 함수 개수
    def count(nn):
        return sum(1 for f in itertools.product(range(nn), repeat=nn) if all(f[f[i]] == i for i in range(nn)))
    cs = {nn: count(nn) for nn in range(1, 7)}
    V.check(1, "3-1 a1", cs[1], V.ans(1, "3-1", "a1")); V.check(1, "3-1 a2", cs[2], V.ans(1, "3-1", "a2"))
    # (2) 일대일: 모든 대합 함수가 단사
    V.report(1, all(len(set(f)) == len(f) for nn in range(1, 6) for f in itertools.product(range(nn), repeat=nn) if all(f[f[i]] == i for i in range(nn))), "3-2 조건 p 를 만족하는 f 는 모두 일대일(n≤5 전수)")
    V.check(1, "3-3 a3", cs[3], V.ans(1, "3-3", "a3")); V.check(1, "3-3 a4", cs[4], V.ans(1, "3-3", "a4"))
    V.report(1, all(cs[nn] == cs[nn-1] + (nn-1)*cs[nn-2] for nn in range(3, 7)) and "(n-1)*a_(n-2)" in V.ans(1, "3-4").replace(" ", "").replace("a_(n-1)+", "a_(n-1)+"), "3-4 a_n=a_(n-1)+(n-1)a_(n-2) (n=3..6)")

def p02():
    f = sp.cos(sp.sin(x))
    V.check(2, "1-1 f(x+pi)-f(x)", sp.simplify(f.subs(x, x+sp.pi) - f), 0)
    # 주기 p<2pi 는 pi 뿐: 수치(0<p<2pi 중 f(x+p)=f(x) 이 모든 x 에서)
    import math
    fn = lambda v: math.cos(math.sin(v))
    ps = [p_ for p_ in [i*math.pi/200 for i in range(1, 400)] if all(abs(fn(t_ + p_) - fn(t_)) < 1e-9 for t_ in [j*0.37 for j in range(40)])]
    V.check(2, "1-1 p", ps, [float(sp.sympify(V.ans(2, "1-1")))], tol=1e-6)
    # 1-2: f' 의 부호 변화: 극값 위치
    d = lambda v: -math.sin(math.sin(v))*math.cos(v)
    pts = [i*0.001 for i in range(1, 8000)]
    ext = [(pts[i], 'min' if d(pts[i-1]) < 0 < d(pts[i+1]) else 'max') for i in range(1, len(pts)-1) if d(pts[i-1])*d(pts[i+1]) < 0]
    V.check(2, "1-2 첫 극값 위치", ext[0][0], sp.N(sp.sympify(V.ans(2, "1-2"), locals={"pi": sp.pi})), tol=2e-3)
    mins = []
    for e in ext:
        if e[1] == 'min' and (not mins or e[0] - mins[-1] > 0.05): mins.append(e[0])
    V.check(2, "1-3 극솟값 x_k=(2k-1)pi/2", mins[:3], [(2*kk-1)*math.pi/2 for kk in (1, 2, 3)], tol=3e-3)
    # 1-3 급수: 문제 그대로(분자 pi^2) → 2. 공식 답과 비교해 불일치 보고
    kk = sp.symbols("kk", integer=True, positive=True)
    xk = lambda j: (2*j-1)*sp.pi/2
    term = sp.simplify(sp.pi**2/(xk(kk)*xk(kk+1)))
    s_ = sp.limit(sp.summation(term, (kk, 1, sp.Symbol("N", integer=True, positive=True))), sp.Symbol("N", integer=True, positive=True), sp.oo)
    V.check(2, "1-3 문제 그대로의 급수 합", s_, 2)
    V.check(2, "1-3 답(공식 1 오기 정정)", s_, V.ans(2, "1-3"))
    # [2](1) 미분 불가능한 점
    a_ = sp.Rational(1, 2)
    fp = lambda v: (v**2 - a_**2) if abs(v) > a_ else sp.sqrt(a_ - abs(v))
    def slopes(t0):
        h = sp.Rational(1, 10**8)
        return float((fp(t0+h) - fp(t0))/h), float((fp(t0) - fp(t0-h))/h)
    nd = []
    for t0 in [-a_, 0, a_, sp.Rational(1, 4), -sp.Rational(1, 4), 1, -1]:
        r_, l_ = slopes(t0)
        if not (abs(r_ - l_) < 1e-3 and abs(r_) < 1e3): nd.append(t0)
    ans = [s.strip() for s in V.ans(2, "2-1", "t").split(",")]
    V.check(2, "2-1 미분 불가능한 점", sorted(nd), sorted([-a_, 0, a_]))
    V.report(2, set(ans) == {"-a", "0", "a"}, "2-1 YAML 답 -a,0,a")
    # [2](2) 교점 4개 범위
    def ncross(av):
        pts = set()
        for r in sp.solve(sp.Eq(x+1, x**2 - av**2), x):
            if sp.im(r) == 0 and abs(r) > av: pts.add(round(float(r), 9))
        for sgn in (1, -1):   # x>=0: sqrt(a-x), x<0: sqrt(a+x)
            # y=x+1 = sqrt(a - sgn*x)  (x*sgn>=0 영역)
            for r in sp.solve(sp.Eq((x+1)**2, av - sgn*x), x):
                r = sp.re(r) if sp.im(r) == 0 else None
                if r is not None and abs(r) <= av and sgn*r >= 0 and r+1 >= 0: pts.add(round(float(r), 9))
        return len(pts)
    lo, hi = sp.sympify(V.ans(2, "2-2", "lo")), 1
    ok = True
    for i in range(10, 250):
        av = R_(i, 100)
        ok &= ((ncross(av) == 4) == (lo < av < hi))
    V.report(2, ok, "2-2 교점 4개 ⇔ 3/4<a<1 (a=0.10..2.49 격자)")
    # [2](3) 기울기 범위: 영역 점 (x,y) 격자
    av = R_(1, 2); sa = math.sqrt(0.5)
    ts = []
    for i in range(-499, 500):
        xv = i/1000
        fy = math.sqrt(0.5 - abs(xv))
        for j in range(0, 51):
            yv = fy*j/50
            ts.append((yv - sa)/(xv - 0.5))
    V.report(2, min(ts) >= -1e-12 and abs(min(ts)) < 1e-9 and max(ts) > 100, f"2-3 기울기 범위 [0,∞): 최소 {min(ts):.3g}, 최대 {max(ts):.3g}")

if __name__ == "__main__": V.run(globals())
