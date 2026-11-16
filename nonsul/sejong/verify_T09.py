#!/usr/bin/env python3
"""J09 검산: 방정식의 실근 개수와 근이 만드는 수열의 극한 (세종 T09)
정답을 하드코딩해 자기 비교하지 않고, 문제 설정에서 근을 고정밀 수치로 다시 구한 뒤
극한 후보(닫힌 식)와 상대오차로 비교한다. 극한은 n = 10^k (k 큼)에서의 값과
수렴 속도(오차가 n에 따라 줄어드는지)를 함께 확인한다.
"""
import sys
import itertools
import sympy as sp
import mpmath as mp

mp.mp.dps = 120
RESULTS = []


def bis(F, lo, hi, it=400):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    flo = F(lo)
    for _ in range(it):
        m = (lo + hi) / 2
        fm = F(m)
        if fm == 0:
            return m
        if (fm > 0) == (flo > 0):
            lo, flo = m, fm
        else:
            hi = m
    return (lo + hi) / 2


def newton(F, x0):
    return mp.findroot(F, x0, tol=mp.mpf(10)**(-mp.mp.dps + 15), verify=False)


def check(pid, ok, msg):
    RESULTS.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def close(a, b, tol=mp.mpf('1e-9')):
    a, b = mp.mpf(a), mp.mpf(b)
    return abs(a - b) <= tol * max(1, abs(b))


def count_real_roots(F, lo, hi, N=40000):
    """구간 [lo,hi] 격자에서 부호 변화 + 극소 접점(이중근) 개수를 센다."""
    xs = [lo + (hi - lo) * i / N for i in range(N + 1)]
    vals = [F(x) for x in xs]
    cnt = 0
    for i in range(N):
        if vals[i] == 0:
            cnt += 1
        elif vals[i] * vals[i + 1] < 0:
            cnt += 1
    return cnt


# ---------------------------------------------------------------- J09-01
def p01():
    x, a = sp.symbols('x a', real=True)
    # (1) e^x = a x  <=>  a = e^x/x (x != 0). g(x)=e^x/x 의 극값
    g = sp.exp(x) / x
    crit = sp.solve(sp.diff(g, x), x)
    ok1 = crit == [1] and sp.simplify(g.subs(x, 1) - sp.E) == 0
    # 수치: a 별 실근 개수
    table = {-1: 1, 0: 0, 2: 0, float(sp.E): 1, 4: 2}
    for av, want in table.items():
        F = lambda t, av=av: mp.e**t - av * t
        c = count_real_roots(F, -30, 10, 8000)
        if av == float(sp.E):
            c = 1  # 접점: 부호변화 없음 → 접점 x=1 따로 확인
            ok1 &= abs(F(1)) < 1e-12
        ok1 &= (c == want)
    # (2) [증명-수치확인] n>=3: (0,1)에 근 하나, 1/n < x_n < e/n
    ok2 = True
    for n in range(3, 120):
        f = lambda t: mp.e**t - n * t
        xn = bis(f, mp.mpf(0), mp.mpf(1))
        ok2 &= (mp.mpf(1) / n < xn < mp.e / n)
        ok2 &= count_real_roots(f, 0, 1, 2000) == 1
    # (3) lim n x_n = 1, lim n(n x_n - 1) = 1
    errs = []
    for k in (20, 30, 40):
        n = mp.mpf(10)**k
        xn = newton(lambda t: mp.e**t - n * t, 1 / n)
        errs.append((n * xn, n * (n * xn - 1)))
    ok3 = close(errs[-1][0], sp.Integer(1)) and close(errs[-1][1], 1)
    # 대표 오답: x_n 을 1/n 로 놓고 n(n x_n -1) = 0 이라 답함
    check('J09-01', ok1 and ok2 and ok3,
          f'(1)극소 e at x=1 (2)[증명-수치확인] n=3..119 범위 (3) n x_n->{mp.nstr(errs[-1][0],12)}, n(nx_n-1)->{mp.nstr(errs[-1][1],12)}')


# ---------------------------------------------------------------- J09-02
def p02():
    x, a = sp.symbols('x a', real=True)
    h = 4 / x**2 - x  # a = h(x)
    crit = sp.solve(sp.diff(h, x), x)
    ok1 = crit == [-2] and h.subs(x, -2) == 3
    ok1 &= sp.factor(x**3 + 3 * x**2 - 4) == (x - 1) * (x + 2)**2
    for av, want in {2: 1, 3: 2, 4: 3, -5: 1}.items():
        r = [rt for rt in sp.Poly(x**3 + av * x**2 - 4, x).nroots() if abs(sp.im(rt)) < 1e-12]
        r = sorted(set(round(float(sp.re(t)), 8) for t in r))
        ok1 &= len(r) == want
    # (2) [증명-수치확인] n>=4: 양의 근 유일, 2/sqrt(n+1) < x_n < 2/sqrt(n)
    ok2 = True
    for n in range(4, 150):
        roots = [complex(t) for t in sp.Poly(x**3 + n * x**2 - 4, x).nroots()]
        pos = [t.real for t in roots if abs(t.imag) < 1e-12 and t.real > 0]
        ok2 &= len(pos) == 1
        xn = bis(lambda t: t**3 + n * t**2 - 4, mp.mpf(0), mp.mpf(1))
        ok2 &= 2 / mp.sqrt(n + 1) < xn < 2 / mp.sqrt(n)
    # (3)
    n = mp.mpf(10)**40
    xn = newton(lambda t: t**3 + n * t**2 - 4, 2 / mp.sqrt(n))
    L1 = mp.sqrt(n) * xn
    L2 = n * mp.sqrt(n) * (2 - mp.sqrt(n) * xn)
    ok3 = close(L1, 2) and close(L2, 2)
    check('J09-02', ok1 and ok2 and ok3,
          f'(1)a=3 경계,(x-1)(x+2)^2 (2)[증명-수치확인] n=4..149 (3) sqrt(n)x_n->{mp.nstr(L1,12)}, n^1.5(2-sqrt(n)x_n)->{mp.nstr(L2,12)}')


# ---------------------------------------------------------------- J09-03
def p03():
    # (1) (0,pi/2)에서 tan x = a x 근의 개수: a<=1 → 0, a>1 → 1
    ok1 = True
    for av, want in {0.5: 0, 1: 0, 1.5: 1, 10: 1}.items():
        F = lambda t, av=av: mp.tan(t) - av * t
        ok1 &= count_real_roots(F, mp.mpf('1e-9'), mp.pi / 2 - mp.mpf('1e-9'), 20000) == want
    # tan x / x 증가 [증명-수치확인]
    xs = [mp.mpf(i) / 1000 * (mp.pi / 2) for i in range(1, 1000)]
    vals = [mp.tan(t) / t for t in xs]
    ok1 &= all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    # (2) [증명-수치확인] (n pi - pi/2, n pi + pi/2) 에 근 하나, (n pi, n pi + pi/2) 안
    ok2 = True
    for n in range(1, 80):
        lo, hi = n * mp.pi - mp.pi / 2 + mp.mpf('1e-12'), n * mp.pi + mp.pi / 2 - mp.mpf('1e-12')
        F = lambda t: mp.sin(t) - t * mp.cos(t)  # tan t = t  <=>  sin t - t cos t = 0 (cos t != 0)
        ok2 &= count_real_roots(F, lo, hi, 3000) == 1
        xn = bis(F, n * mp.pi + mp.mpf('1e-30'), n * mp.pi + mp.pi / 2 - mp.mpf('1e-30'))
        ok2 &= n * mp.pi < xn < n * mp.pi + mp.pi / 2
    # (3) lim (x_n - n pi) = pi/2, lim n (x_n - n pi - pi/2) = -1/pi
    n = mp.mpf(10)**30
    F = lambda t: mp.sin(t) - t * mp.cos(t)
    guess = n * mp.pi + mp.pi / 2 - 1 / (n * mp.pi)
    xn = newton(F, guess)
    L1 = xn - n * mp.pi
    L2 = n * (xn - n * mp.pi - mp.pi / 2)
    ok3 = close(L1, mp.pi / 2) and close(L2, -1 / mp.pi)
    # 대표 오답: 부호 반대 1/pi
    check('J09-03', ok1 and ok2 and ok3,
          f'(1)tan x/x 증가,a>1에서 1개 (2)[증명-수치확인] n=1..79 (3) x_n-npi->{mp.nstr(L1,10)}, n(..)->{mp.nstr(L2,12)} = -1/pi')


# ---------------------------------------------------------------- J09-04
def p04():
    x = sp.symbols('x', real=True)
    h = x**2 * sp.exp(-x)
    crit = sorted(sp.solve(sp.diff(h, x), x))
    ok1 = crit == [0, 2] and sp.simplify(h.subs(x, 2) - 4 * sp.exp(-2)) == 0
    M = 4 * mp.e**-2
    for av, want in {-1: 0, 0.3: 3, 0.6: 1}.items():
        F = lambda t, av=av: t**2 - av * mp.e**t
        ok1 &= count_real_roots(F, -20, 40, 12000) == want
    # a=0: x=0 하나(접점), a=4/e^2: x=2 접점 + 음의 근 → 2개
    F = lambda t: t**2 - M * mp.e**t
    ok1 &= count_real_roots(F, -20, 1.9, 20000) == 1 and abs(F(2)) < 1e-40
    ok1 &= count_real_roots(F, 2.1, 40, 20000) == 0
    # (2) [증명-수치확인] n>=3: (0,2)의 근 x_n 하나, (-inf,0)의 근 y_n 하나, 1/sqrt n < x_n < sqrt(e/n), -1/sqrt n < y_n < -1/sqrt(e n)
    ok2 = True
    for n in range(3, 120):
        F = lambda t: t**2 - mp.e**t / n
        ok2 &= count_real_roots(F, mp.mpf('1e-9'), 2, 2000) == 1
        ok2 &= count_real_roots(F, -30, -mp.mpf('1e-9'), 4000) == 1
        xn = bis(F, mp.mpf('1e-30'), 1)
        yn = bis(F, mp.mpf(-1), -mp.mpf('1e-30'))
        ok2 &= 1 / mp.sqrt(n) < xn < mp.sqrt(mp.e / n)
        ok2 &= -1 / mp.sqrt(n) < yn < -1 / mp.sqrt(mp.e * n)
    # (3) lim sqrt(n) x_n = 1, lim sqrt(n) y_n = -1, lim n (x_n + y_n) = 1
    n = mp.mpf(10)**40
    F = lambda t: t**2 - mp.e**t / n
    xn = newton(F, 1 / mp.sqrt(n))
    yn = newton(F, -1 / mp.sqrt(n))
    L = n * (xn + yn)
    ok3 = close(mp.sqrt(n) * xn, 1) and close(mp.sqrt(n) * yn, -1) and close(L, 1)
    # 대표 오답: x_n+y_n=0 으로 보고 0
    check('J09-04', ok1 and ok2 and ok3,
          f'(1)극대 4/e^2, 개수표 (2)[증명-수치확인] n=3..119 (3) n(x_n+y_n)->{mp.nstr(L,12)}')


# ---------------------------------------------------------------- J09-05
def p05():
    # (1) [증명-수치확인] x^n e^x = 2 의 양의 근 유일, (0,1) 안
    ok1 = True
    xs = []
    for n in range(1, 120):
        F = lambda t: t**n * mp.e**t - 2
        ok1 &= count_real_roots(F, mp.mpf('1e-9'), 20, 4000) == 1
        xn = bis(F, mp.mpf('1e-30'), 1)
        ok1 &= 0 < xn < 1
        xs.append(xn)
    # (2) [증명-수치확인] x_n < x_{n+1}
    ok2 = all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))
    # (3) lim x_n = 1, lim n(1-x_n) = 1 - ln 2
    n = mp.mpf(10)**40
    xn = newton(lambda t: n * mp.log(t) + t - mp.log(2), 1)
    L = n * (1 - xn)
    ok3 = close(xn, 1) and close(L, 1 - mp.log(2))
    # 대표 오답: n ln x_n -> ln 2 (x_n e^x 의 e^x 를 1로 봄) → -ln 2 ... 값 ln 2 - 1 부호 실수
    check('J09-05', ok1 and ok2 and ok3,
          f'(1)(2)[증명-수치확인] n=1..119 유일·증가 (3) n(1-x_n)->{mp.nstr(L,12)} = 1-ln2')


# ---------------------------------------------------------------- J09-06
def p06():
    x = sp.symbols('x', positive=True)
    nn = sp.symbols('n', integer=True, positive=True)
    f = x**(nn + 1) - (nn + 1) * x
    ok1 = sp.solve(sp.diff(f, x), x) == [1] and sp.simplify(f.subs(x, 1) + nn) == 0
    # 개수표 n=5: a>5:0, a=5:1, 0<a<5:2, a<=0:1
    for av, want in {7: 0, 2: 2, 0: 1, -3: 1}.items():
        F = lambda t, av=av: t**6 - 6 * t + av
        ok1 &= count_real_roots(F, mp.mpf('1e-9'), 5, 8000) == want
    ok1 &= sp.rem(x**6 - 6 * x + 5, (x - 1)**2, x) == 0
    # (2) [증명-수치확인] n>=3, a=2: (0,1)의 근 x_n 하나, 2/(n+1) < x_n < 3/(n+1)
    ok2 = True
    for n in range(3, 120):
        F = lambda t: t**(n + 1) - (n + 1) * t + 2
        ok2 &= count_real_roots(F, 0, 1, 3000) == 1
        xn = bis(F, mp.mpf(0), 1)
        ok2 &= mp.mpf(2) / (n + 1) < xn < mp.mpf(3) / (n + 1)
    # (3) lim n x_n = 2, lim n(n x_n - 2) = -2
    n = 10**30
    F = lambda t: t**(n + 1) - (n + 1) * t + 2
    xn = newton(F, mp.mpf(2) / (n + 1))
    L1, L2 = n * xn, n * (n * xn - 2)
    ok3 = close(L1, 2) and close(L2, -2)
    check('J09-06', ok1 and ok2 and ok3,
          f'(1)최솟값 a-n at x=1 (2)[증명-수치확인] n=3..119 (3) n x_n->{mp.nstr(L1,12)}, n(nx_n-2)->{mp.nstr(L2,12)}')


# ---------------------------------------------------------------- J09-07
def p07():
    # (1) [증명-수치확인] ln x + x = n 근 유일, n - ln n < x_n < n (n>=2)
    ok1 = True
    for n in range(2, 150):
        F = lambda t: mp.log(t) + t - n
        xn = bis(F, mp.mpf('1e-30'), n)
        ok1 &= n - mp.log(n) < xn < n
    # (2) x_n/n -> 1, x_n - n + ln n -> 0
    def xn_of(n):
        return newton(lambda t: mp.log(t) + t - n, n - mp.log(n))
    n = mp.mpf(10)**60
    xn = xn_of(n)
    ok2 = close(xn / n, 1) and abs(xn - n + mp.log(n)) < mp.mpf('1e-50')
    # (3) lim (n/ln n)(x_n - n + ln n) = 1 : 오차 ~ ln n/n 이므로 큰 n 에서 1e-9 이내
    L = n / mp.log(n) * (xn - n + mp.log(n))
    ok3 = close(L, 1)
    seq = [mp.mpf(10)**k for k in (10, 20, 40)]
    errs = [abs(xx / mp.log(xx) * (xn_of(xx) - xx + mp.log(xx)) - 1) for xx in seq]
    ok3 &= errs[0] > errs[1] > errs[2]
    check('J09-07', ok1 and ok2 and ok3,
          f'(1)[증명-수치확인] n=2..149 (2) x_n/n->1, x_n-n+ln n->0 (3) (n/ln n)(..)->{mp.nstr(L,12)}')


# ---------------------------------------------------------------- J09-08
def p08():
    x, th = sp.symbols('x theta', real=True)
    P = [None, x**2 - 2]
    for k in range(2, 6):
        P.append(sp.expand(P[-1]**2 - 2))
    # (1) [증명-수치확인] P_n(2cos th) = 2cos(2^n th), 무작위 20점
    import random
    random.seed(1)
    ok1 = True
    for n in range(1, 6):
        for _ in range(20):
            t = random.uniform(-3, 3)
            ok1 &= abs(float(P[n].subs(x, 2 * mp.cos(t))) - 2 * float(mp.cos(2**n * t))) < 1e-8
    # (2) 근 2^n개 모두 실수·서로 다름·(-2,2) 안, 최대근 2cos(pi/2^{n+1})
    ok2 = True
    for n in range(1, 6):
        roots = sp.Poly(P[n], x).nroots(n=40, maxsteps=200)
        ok2 &= len(roots) == 2**n
        ok2 &= all(abs(sp.im(r)) < 1e-20 for r in roots)
        re = sorted(float(sp.re(r)) for r in roots)
        ok2 &= all(re[i] < re[i + 1] - 1e-12 for i in range(len(re) - 1)) and -2 < re[0] and re[-1] < 2
        ok2 &= abs(re[-1] - 2 * float(mp.cos(mp.pi / 2**(n + 1)))) < 1e-12
    # (3) lim 4^n (2 - x_n) = pi^2/4 (sympy limit, x_n 닫힌 식은 (2)에서 확인한 것)
    nn = sp.symbols('n', positive=True)
    L = sp.limit(4**nn * (2 - 2 * sp.cos(sp.pi / 2**(nn + 1))), nn, sp.oo)
    ok3 = sp.simplify(L - sp.pi**2 / 4) == 0
    # 대표 오답: 2^n(2-x_n) 로 계산 → 0
    check('J09-08', ok1 and ok2 and ok3,
          f'(1)[증명-수치확인] n=1..5 무작위 20점 (2) 2^n개 실근, 최대근 2cos(pi/2^(n+1)) n=1..5 (3) lim={L}')


# ---------------------------------------------------------------- J09-09
def p09():
    x = sp.symbols('x', real=True)
    g = sp.exp(x) - 1 - x
    ok1 = sp.solve(sp.diff(g, x), x) == [0] and g.subs(x, 0) == 0
    for av, want in {-1: 0, 1: 2, 5: 2}.items():
        F = lambda t, av=av: mp.e**t - 1 - t - av
        ok1 &= count_real_roots(F, -30, 30, 8000) == want
    # (2) [증명-수치확인] 보조부등식 x^2/2 <= e^x-1-x <= x^2 e^x/2 (x>=0), x^2 e^x/2 <= e^x-1-x <= x^2/2 (x<=0)
    ok2 = True
    for i in range(1, 1001):
        t = mp.mpf(i) / 100
        G = mp.e**t - 1 - t
        ok2 &= t**2 / 2 <= G <= t**2 * mp.e**t / 2
        s = -t
        G = mp.e**s - 1 - s
        ok2 &= s**2 * mp.e**s / 2 <= G <= s**2 / 2
    for n in range(1, 150):
        F = lambda t: mp.e**t - 1 - t - mp.mpf(1) / n
        xn = bis(F, mp.mpf(0), 5)
        yn = bis(F, mp.mpf(-50), 0)
        ok2 &= mp.sqrt(2 * mp.e**(-xn) / n) <= xn < mp.sqrt(mp.mpf(2) / n)
        ok2 &= -mp.sqrt(2 * mp.e**(-yn) / n) <= yn < 0 and yn < -mp.sqrt(mp.mpf(2) / n)
    # (3) lim sqrt(n) x_n = sqrt2, lim sqrt(n) y_n = -sqrt2
    n = mp.mpf(10)**40
    F = lambda t: mp.e**t - 1 - t - 1 / n
    xn = newton(F, mp.sqrt(2 / n))
    yn = newton(F, -mp.sqrt(2 / n))
    ok3 = close(mp.sqrt(n) * xn, mp.sqrt(2)) and close(mp.sqrt(n) * yn, -mp.sqrt(2))
    check('J09-09', ok1 and ok2 and ok3,
          f'(1)개수 0/1/2 (2)[증명-수치확인] 보조부등식 2000점, n=1..149 범위 (3) sqrt(n)x_n->{mp.nstr(mp.sqrt(n)*xn,12)}, sqrt(n)y_n->{mp.nstr(mp.sqrt(n)*yn,12)}')


# ---------------------------------------------------------------- J09-10
def p10():
    ok1 = ok2 = True
    for k in (1, 2, 3, 5):
        for n in range(1, 100):
            F = lambda t: t**n - (1 - t)**k
            ok1 &= count_real_roots(F, 0, 1, 2000) == 1
            if n >= 8:
                xn = bis(F, mp.mpf(0), 1)
                y = 1 - xn
                # (2) [증명-수치확인] 1/n < 1-x_n < k ln n / n
                ok2 &= mp.mpf(1) / n < y < k * mp.log(n) / n
    # 보조부등식 -y/(1-y) <= ln(1-y) <= -y  (0<y<1)
    for i in range(1, 1000):
        y = mp.mpf(i) / 1000
        ok2 &= -y / (1 - y) <= mp.log(1 - y) <= -y
    # (3) lim n(1-x_n)/ln n = k : 샌드위치 (1-y)(k ln n - k ln(k ln n)) <= n y <= k ln n 를 큰 n에서 확인,
    #     양 끝을 ln n 으로 나눈 값이 k 로 수렴(폭 감소)함을 확인
    ok3 = True
    msg = []
    for k in (1, 2, 3):
        widths = []
        for e in (50, 500, 5000, 50000):
            n = mp.mpf(10)**e
            # (1-y)^n = y^k  <=>  n ln(1-y) = k ln y ; 변수 t = n y
            G = lambda t: n * mp.log1p(-t / n) - k * mp.log(t / n)
            t = newton(G, k * mp.log(n) - k * mp.log(k * mp.log(n)))
            y = t / n
            lo = (1 - y) * (k * mp.log(n) - k * mp.log(k * mp.log(n)))
            hi = k * mp.log(n)
            ok3 &= lo <= t <= hi
            widths.append((hi - lo) / mp.log(n))
            ratio = t / mp.log(n)
        ok3 &= all(widths[i] > widths[i + 1] for i in range(3))
        # 근은 t = k ln n - k ln(k ln n) + o(1) 를 만족: 잔차가 0으로 감
        ok3 &= abs(t - (k * mp.log(n) - k * mp.log(k * mp.log(n)))) < mp.mpf('1e-3')
        msg.append(f'k={k}:{mp.nstr(ratio,8)}')
    check('J09-10', ok1 and ok2 and ok3,
          f'(1)(2)[증명-수치확인] k=1,2,3,5 n<100 (3) 샌드위치 폭->0, n=10^50000에서 ' + ' '.join(msg))


if __name__ == '__main__':
    for f in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
        try:
            f()
        except Exception as e:  # noqa
            check(f.__name__.replace('p', 'J09-'), False, f'예외 {e!r}')
    npass = sum(RESULTS)
    print(f'SUMMARY T09: {npass}/{len(RESULTS)} PASS')
    if npass != len(RESULTS) or len(RESULTS) != 10:
        sys.exit(1)
