#!/usr/bin/env python3
"""S05 초월함수의 미분과 최대·최소·부등식 — 검산 스크립트"""
import sys
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.optimize import brentq

x, a, k, t, n = sp.symbols('x a k t n', real=True)
mp.mp.dps = 30
results = []


def check(pid, cond, msg):
    results.append(bool(cond))
    print(f"{pid} {'PASS' if cond else 'FAIL'}  {msg}")


def same(u, v, tol=1e-12):
    return abs(complex(sp.N(u - v, 30))) < tol


def count_roots(func, lo, hi, N=400001):
    xs = np.linspace(lo, hi, N)
    ys = func(xs)
    s = np.sign(ys)
    cnt = int(np.sum(s[:-1] * s[1:] < 0)) + int(np.sum(ys == 0))
    return cnt


# S05-01: y=e^{-x} sin x (x>=0) 극댓값들의 합
def p01():
    f = sp.exp(-x) * sp.sin(x)
    fp = sp.simplify(sp.diff(f, x))
    f2 = sp.diff(f, x, 2)
    # 임계점 pi/4 + m*pi : 짝수 m 극대, 홀수 m 극소
    ok = all(same(fp.subs(x, sp.pi / 4 + m * sp.pi), 0) for m in range(6))
    ok &= all((f2.subs(x, sp.pi / 4 + m * sp.pi) < 0) == (m % 2 == 0) for m in range(6))
    S = sp.summation(f.subs(x, sp.pi / 4 + 2 * n * sp.pi), (n, 0, sp.oo))
    ans = sp.sqrt(2) * sp.exp(7 * sp.pi / 4) / (2 * (sp.exp(2 * sp.pi) - 1))
    wrong = sp.summation(sp.Abs(f.subs(x, sp.pi / 4 + n * sp.pi)), (n, 0, 60))
    ok &= same(S, ans)
    check("S05-01", ok, f"극대 x=pi/4+2n*pi, 합={sp.N(ans, 12)} (오답: 공비 e^-pi 로 합 {sp.N(wrong, 8)})")


# S05-02: M(a)=max(a ln x - x^2/2), M(a)의 최솟값
def p02():
    ap = sp.symbols('ap', positive=True)
    xp = sp.symbols('xp', positive=True)
    g = ap * sp.log(xp) - xp ** 2 / 2
    crit = sp.solve(sp.diff(g, xp), xp)
    M = sp.simplify(g.subs(xp, crit[0]))
    ac = sp.solve(sp.diff(M, ap), ap)
    val = sp.simplify(M.subs(ap, ac[0]))
    ok = len(crit) == 1 and ac == [1] and val == sp.Rational(-1, 2)
    ok &= sp.simplify(sp.diff(M, ap, 2).subs(ap, 1)) > 0
    check("S05-02", ok, f"M(a)={M}, 최소 a={ac[0]}, 최솟값={val}")


# S05-03: e^x = k x^3 의 실근 개수 N(k)
def p03():
    h = sp.exp(x) / x ** 3
    crit = sp.solve(sp.diff(h, x), x)
    m = h.subs(x, 3)
    ok = crit == [3] and sp.simplify(m - sp.exp(3) / 27) == 0
    c = float(sp.exp(3) / 27)

    def N(kv):
        # x<0, x>0 각각
        F = lambda xs: np.exp(xs) - kv * xs ** 3
        return count_roots(F, -40, -1e-9) + count_roots(F, 1e-9, 40)
    table = {-2: 1, -0.01: 1, 0: 0, 0.3: 0, c * 0.999: 0, c * 1.001: 2, 2: 2}
    for kv, exp_ in table.items():
        ok &= N(kv) == exp_
    # k=c 일 때 접하는 근 x=3 하나 (중근)
    ok &= abs(np.exp(3) - c * 27) < 1e-9 and N(c * 0.9999) == 0
    check("S05-03", ok, f"x>0 최솟값 e^3/27={c:.6f}, N(k): k<0→1, 0<=k<e^3/27→0, k=e^3/27→1, 크면 2; 불연속 k=0, e^3/27")


# S05-04: ln(1+x) >= x/(1+ax) (x>0) 의 a 최솟값
def p04():
    g = sp.log(1 + x) - x / (1 + a * x)
    gp = sp.together(sp.diff(g, x))
    num = sp.factor(sp.numer(gp))
    ok = sp.expand(num - x * (a ** 2 * x + 2 * a - 1)) == 0
    xs = np.linspace(1e-6, 200, 200001)
    g_half = np.log1p(xs) - xs / (1 + 0.5 * xs)
    ok &= g_half.min() >= -1e-15
    g_bad = np.log1p(xs) - xs / (1 + 0.49 * xs)
    ok &= g_bad.min() < 0
    check("S05-04", ok, f"g'분자={num}, a=1/2에서 g>=0, a=0.49에서 음수 존재 → 최솟값 1/2")


# S05-05: f=e^x+e^{-x}-a x^2 가 극댓값을 갖는 a 범위, 극솟값 개수
def p05():
    def extrema(av):
        fp = lambda xs: np.exp(xs) - np.exp(-xs) - 2 * av * xs
        xs = np.linspace(-20, 20, 400001)
        ys = fp(xs)
        idx = np.where(np.sign(ys[:-1]) * np.sign(ys[1:]) < 0)[0]
        z = np.where(ys == 0)[0]
        pts = sorted(list(xs[idx]) + list(xs[z]))
        # 부호변화 판정 (+→- 극대, -→+ 극소)
        mx = mn = 0
        for p in pts:
            l, r = fp(p - 1e-4), fp(p + 1e-4)
            if l > 0 > r:
                mx += 1
            elif l < 0 < r:
                mn += 1
        return mx, mn
    ok = extrema(0.5) == (0, 1) and extrema(1.0) == (0, 1)
    ok &= extrema(1.001) == (1, 2) and extrema(3) == (1, 2)
    f = sp.exp(x) + sp.exp(-x) - a * x ** 2
    ok &= sp.simplify(sp.diff(f, x, 2).subs(x, 0) - (2 - 2 * a)) == 0
    check("S05-05", ok, "a<=1: 극소 1개·극대 없음, a>1: 극대 1(x=0)·극소 2 → 답 a>1, 극솟값 2개")


# S05-06: x>=0 에서 e^x >= k sin x 의 k 범위
def p06():
    q = sp.exp(x) / sp.sin(x)
    crit = sp.solve(sp.diff(q, x), x)
    lo = -sp.sqrt(2) * sp.exp(5 * sp.pi / 4)
    hi = sp.sqrt(2) * sp.exp(sp.pi / 4)
    ok = same(q.subs(x, sp.pi / 4), hi) and same(q.subs(x, 5 * sp.pi / 4), lo)
    xs = np.linspace(0, 30, 3000001)
    holds = lambda kv: np.all(np.exp(xs) - kv * np.sin(xs) >= -1e-9)
    H, L = float(hi), float(lo)
    ok &= holds(H) and holds(L) and not holds(H * 1.001) and not holds(L * 1.001)
    prod = sp.simplify(hi * lo)
    ok &= sp.simplify(prod + 2 * sp.exp(3 * sp.pi / 2)) == 0
    check("S05-06", ok, f"{sp.N(lo, 8)}<=k<={sp.N(hi, 8)}, 최댓값×최솟값={prod}")


# S05-07: y=ln(x^2+a) 변곡점 접선과 x축이 만드는 삼각형 넓이 최대
def p07():
    ap = sp.symbols('ap', positive=True)
    y = sp.log(x ** 2 + ap)
    infl = sp.solve(sp.diff(y, x, 2), x)
    r = sp.sqrt(ap)
    ok = set(infl) == {r, -r}
    slope = sp.diff(y, x).subs(x, r)
    tan = y.subs(x, r) + slope * (x - r)
    yint = sp.simplify(tan.subs(x, 0))
    xint = sp.solve(tan, x)[0]
    area = sp.simplify(xint * (-yint))  # 밑변 2*xint, 높이 -yint, /2
    tt = sp.symbols('tt', real=True)
    A_t = sp.simplify(area.subs(ap, sp.exp(tt) / 2))  # t = ln 2a
    ts = sp.solve(sp.diff(A_t, tt), tt)
    ts = [s for s in ts if s < 1]
    best = sp.simplify(A_t.subs(tt, ts[0]))
    ans = 8 * sp.sqrt(2) * sp.exp(sp.Rational(-3, 2))
    ok &= len(ts) == 1 and ts[0] == -3 and same(best, ans)
    # 수치 최대 확인
    av = np.linspace(1e-6, np.e / 2 - 1e-6, 200001)
    num = np.sqrt(av) * (1 - np.log(2 * av)) ** 2
    ok &= abs(num.max() - float(ans)) < 1e-6
    check("S05-07", ok, f"y절편 {yint}, 넓이 sqrt(a)(1-ln2a)^2, 최대 a=1/(2e^3), 최댓값 {sp.N(ans, 10)}")


# S05-08: M(a)=max_{0<=x<=1} x e^{-ax}, ∫_0^2 M(a) da
def p08():
    I = sp.integrate(sp.exp(-a), (a, 0, 1)) + sp.integrate(1 / (a * sp.E), (a, 1, 2))
    ans = 1 - sp.exp(-1) + sp.log(2) / sp.E
    ok = sp.simplify(I - ans) == 0
    xs = np.linspace(0, 1, 20001)
    Mnum = lambda av: np.max(xs * np.exp(-av * xs))
    from scipy.integrate import quad
    v, _ = quad(Mnum, 0, 2, points=[1], limit=200)
    ok &= abs(v - float(ans)) < 1e-6
    wrong = sp.integrate(sp.exp(-a), (a, 0, 2))
    check("S05-08", ok, f"적분={sp.N(ans, 12)} (수치 {v:.8f}); 오답(경우 누락) {sp.N(wrong, 8)}")


# S05-09: f(0)=1, f'+f <= x e^{-x} (x>=0) 일 때 f(2) 최댓값
def p09():
    f = (1 + x ** 2 / 2) * sp.exp(-x)
    ok = sp.simplify(sp.diff(f, x) + f - x * sp.exp(-x)) == 0 and f.subs(x, 0) == 1
    ok &= sp.simplify(f.subs(x, 2) - 3 * sp.exp(-2)) == 0
    # 조건을 만족하는 다른 함수들은 f(2) 가 더 작음 (반례 탐색)
    rng = np.random.default_rng(0)
    for _ in range(200):
        c1, c2 = rng.random(2) * 3
        h = (1 + x ** 2 / 2 - c1 * x ** 3 - c2 * (1 - sp.exp(-x))) * sp.exp(-x)
        lhs = sp.lambdify(x, sp.diff(h, x) + h - x * sp.exp(-x))
        xs = np.linspace(0, 2, 401)
        if np.all(lhs(xs) <= 1e-12) and h.subs(x, 0) == 1:
            ok &= float(h.subs(x, 2)) <= float(3 * np.exp(-2)) + 1e-12
    check("S05-09", ok, "g=e^x f - x^2/2 감소 → f(2)<=3/e^2, 등호 f=(1+x^2/2)e^{-x} [증명-수치확인]")


# S05-10: e^x = 1 + x + k x^2 의 실근이 x=0 뿐인 k
def p10():
    N = (x - 2) * sp.exp(x) + x + 2
    h = (sp.exp(x) - 1 - x) / x ** 2
    hp = sp.simplify(sp.diff(h, x) - N / x ** 3)
    ok = hp == 0
    ok &= sp.limit(h, x, 0) == sp.Rational(1, 2)
    xs = np.concatenate([np.linspace(-30, -1e-3, 100000), np.linspace(1e-3, 30, 100000)])
    hv = (np.expm1(xs) - xs) / xs ** 2
    ok &= np.all(np.diff(hv[:100000]) > 0) and np.all(np.diff(hv[100000:]) > 0)

    def nonzero_roots(kv):
        F = lambda z: np.expm1(z) - z - kv * z ** 2
        return count_roots(F, -60, -1e-6) + count_roots(F, 1e-6, 30)
    for kv, e in [(-1, 0), (0, 0), (0.2, 1), (0.5, 0), (0.7, 1), (3, 1)]:
        ok &= nonzero_roots(kv) == e
    check("S05-10", ok, "h=(e^x-1-x)/x^2 증가, 치역 (0,1/2)∪(1/2,∞) → k<=0 또는 k=1/2 [증명-수치확인]")


for p in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        p()
    except Exception as e:  # noqa
        results.append(False)
        print(f"{p.__name__} FAIL  exception {e!r}")

npass = sum(results)
print(f"SUMMARY T05: {npass}/10 PASS")
if npass != 10:
    sys.exit(1)
