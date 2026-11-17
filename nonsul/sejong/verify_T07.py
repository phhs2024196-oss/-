#!/usr/bin/env python3
"""J07. 부등식 증명과 급수·정적분의 평가 — 검산 스크립트 (세종 J07-01 ~ J07-10)"""
import sys, math
import numpy as np
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
x, t, n, k, m = sp.symbols('x t n k m', positive=True)
RESULTS = []


def report(pid, ok, msg):
    RESULTS.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def grid(a, b, N=1000):
    return np.linspace(a, b, N + 2)[1:-1]


# J07-01  ln(1+x) 양쪽 부등식 -> lim prod (1+k^2/n^3) = e^{1/3}
def p01():
    xs = grid(1e-4, 50, 2000)
    ok1 = all(xx - xx**2 / 2 < math.log1p(xx) < xx for xx in xs)  # [증명-수치확인]
    s2 = sp.summation(k**2, (k, 1, n)) / n**3
    lim2 = sp.limit(s2, n, sp.oo)
    ok2 = lim2 == sp.Rational(1, 3)
    # 0 < sum k^4/n^6 <= 1/n
    ok2b = all(0 < sum(kk**4 for kk in range(1, N + 1)) / N**6 <= 1 / N for N in range(1, 201))
    # (3) 하계·상계의 극한
    s4 = sp.summation(k**4, (k, 1, n)) / n**6
    low = sp.limit(s2 - s4 / 2, n, sp.oo)
    up = sp.limit(s2, n, sp.oo)
    ok3 = low == up == sp.Rational(1, 3)
    N = 10**6
    kk = np.arange(1, N + 1, dtype=np.float64)
    lnP = np.sum(np.log1p(kk**2 / N**3))
    ok3n = abs(math.exp(lnP) - math.exp(1 / 3)) < 1e-5
    # 대표 오답: 상계만 쓰고 n^3 대신 n^2로 정규화 → 발산 착각 / Σk^2/n^3 극한을 1/6으로 오산 → e^{1/6}
    report('J07-01', ok1 and ok2 and ok2b and ok3 and ok3n,
           f'[증명-수치확인] 부등식 2000점; (2) lim={lim2}; (3) 하·상계 극한 {low},{up} -> e^(1/3), 수치 P(10^6)={math.exp(lnP):.7f}')


# J07-02  n! 의 적분 평가 -> lim ((mn)!/(n! n^{(m-1)n}))^{1/n} = m^m/e^{m-1}
def p02():
    ok1 = True
    for N in range(2, 301):
        lf = math.lgamma(N + 1)
        ok1 &= (N * math.log(N) - N + 1 < lf < (N + 1) * math.log(N + 1) - N)  # [증명-수치확인]
    lim2 = sp.limit(sp.factorial(n)**(1 / n) / n, n, sp.oo)
    ok2 = sp.simplify(lim2 - sp.exp(-1)) == 0
    ok3 = True
    vals = []
    for M in (2, 3, 4):
        # 하계·상계(문항 (1)의 부등식으로 만든 것)의 n제곱근 극한을 sympy로 계산
        lowexpr = ((M * n)**(M * n) * sp.exp(1 - M * n) / ((n + 1)**(n + 1) * sp.exp(-n) * n**((M - 1) * n)))**(1 / n)
        upexpr = ((M * n + 1)**(M * n + 1) * sp.exp(-M * n) / (n**n * sp.exp(1 - n) * n**((M - 1) * n)))**(1 / n)
        lo = sp.limit(sp.log(lowexpr), n, sp.oo)
        hi = sp.limit(sp.log(upexpr), n, sp.oo)
        target = M * sp.log(M) - (M - 1)
        ok3 &= sp.simplify(lo - target) == 0 and sp.simplify(hi - target) == 0
        # 수치: n = 10^7
        Nn = 10**7
        val = math.exp((math.lgamma(M * Nn + 1) - math.lgamma(Nn + 1) - (M - 1) * Nn * math.log(Nn)) / Nn)
        vals.append(val)
        ok3 &= abs(val - M**M / math.e**(M - 1)) / (M**M / math.e**(M - 1)) < 1e-5
    report('J07-02', ok1 and ok2 and ok3,
           f'[증명-수치확인] n=2..300; (2) lim={lim2}; (3) m=2,3,4 하·상계 극한 = m^m/e^(m-1), m=2 -> 4/e={4/math.e:.6f} (수치 {vals[0]:.6f})')


# J07-03  I_n = ∫_0^1 x^n e^x dx
def p03():
    # a_1=1, a_{n+1}=a_n+1/a_n 직접 반복 (고정밀)
    a = [None, mp.mpf(1)]
    for j in range(1, 1300):
        a.append(a[j] + 1 / a[j])
    ok1 = all(a[j] ** 2 >= 2 * j - 1 - mp.mpf(10) ** -30 for j in range(1, 1300))  # [증명-수치확인]
    # 앞 몇 항은 정확값으로도
    q = [None, sp.Integer(1)]
    for j in range(1, 8):
        q.append(q[j] + 1 / q[j])
    ok1 &= all(q[j] ** 2 >= 2 * j - 1 for j in range(1, 9))
    ok2 = all(a[j] ** 2 < 2 * j + mp.log(2 * j - 3) / 2 for j in range(3, 1300))  # [증명-수치확인]
    # 적분 비교 1/(2k-1) < ∫_{k-1}^{k} dx/(2x-1)
    kk = sp.Symbol('kk', positive=True, integer=True)
    ok2 &= all(sp.Rational(1, 2 * j - 1) < sp.integrate(1 / (2 * x - 1), (x, j - 1, j)) for j in range(2, 60))
    ok2 &= sp.simplify(sp.integrate(1 / (2 * x - 1), (x, 1, n - 1)) - sp.log(2 * n - 3) / 2) == 0
    # (3) 극한: 상·하계의 극한을 sympy 로
    lo = sp.limit(sp.sqrt(2 - 1 / n), n, sp.oo)
    hi = sp.limit(sp.sqrt(2 + sp.log(2 * n - 3) / (2 * n)), n, sp.oo)
    r = a[1248]
    ok3 = lo == sp.sqrt(2) and hi == sp.sqrt(2) and int(mp.floor(r)) == 49
    ok3 &= 2496 + math.log(2493) / 2 < 2500 and math.log(2493) < 8 and 2.7 ** 8 > 2493
    ratio = a[1299] / mp.sqrt(1299)
    ok3 &= abs(ratio - mp.sqrt(2)) < 0.002
    # 대표 오답: 1/(2k-1)<=1/k 상계 2n+ln(n-1) 은 50^2 를 넘어 판정 실패
    wrong_hi = 2496 + math.log(1247)
    report('J07-03', ok1 and ok2 and ok3 and wrong_hi > 2500,
           f'[증명-수치확인] a_n^2>=2n-1, a_n^2<2n+ln(2n-3)/2 (n<1300); a_1248={float(r):.6f} -> [ ]=49; a_n/sqrt n->{lo}; 오답 상계 {wrong_hi:.2f}>2500')


def p04():
    ts = grid(0, 1, 2000)
    ok1 = all(1 + tt + tt**2 / 2 < math.exp(tt) for tt in ts)  # [증명-수치확인]
    ok2 = all(math.exp(tt) < 1 + tt + (math.e - 2) * tt**2 for tt in ts)  # [증명-수치확인]
    low = sp.integrate(1 + x**2 + x**4 / 2, (x, 0, 1))
    up = sp.integrate(1 + x**2 + (sp.E - 2) * x**4, (x, 0, 1))
    upmax = sp.Rational(4, 3) + (sp.Rational(272, 100) - 2) / 5
    Ival = sp.N(sp.integrate(sp.exp(x**2), (x, 0, 1)), 30)
    ok3 = (low == sp.Rational(43, 30)) and (sp.floor(10 * low) == 14) and (upmax < sp.Rational(3, 2)) \
        and (low < Ival < up) and (sp.floor(10 * Ival) == 14)
    # 대표 오답: 상계를 현(1+(e-1)t)로 잡으면 (e+2)/3 ≈ 1.573 -> 정수부분 15 로 착각
    bad = sp.N((sp.E + 2) / 3)
    report('J07-04', ok1 and ok2 and ok3,
           f'[증명-수치확인] 두 부등식 2000점; 하계 {low}, 상계 4/3+(e-2)/5≈{float(up):.4f}(<1.5), I≈{float(Ival):.6f} -> 14; 오답 상계 {float(bad):.4f}')


# J07-05  H_n 중점 평가 -> floor(H_1200) = 7
def p05():
    ts = grid(0, 1, 2000)
    ok1 = all(math.log((1 + tt) / (1 - tt)) > 2 * tt for tt in ts)  # [증명-수치확인]
    ok2 = True
    for N in range(1, 2001):
        H = math.fsum(1.0 / j for j in range(1, N + 1))
        ok2 &= math.log(N + 1) < H < math.log(2 * N + 1)
    H1200 = sp.Add(*[sp.Rational(1, j) for j in range(1, 1201)])
    ok3 = sp.floor(H1200) == 7
    # 문제의 단서 1096<e^7<1097, 2980<e^8<2981 확인 및 1201>e^7, 2401<e^8
    ok3b = 1096 < math.exp(7) < 1097 and 2980 < math.exp(8) < 2981 and 1201 > 1097 and 2401 < 2980
    bad = 1 + math.log(1200)  # 대표 오답: 상계 1+ln n 사용 -> 8.09 로 판정 불가
    report('J07-05', ok1 and ok2 and ok3 and ok3b,
           f'[증명-수치확인] (1) 2000점, (2) n=1..2000; H_1200≈{float(H1200):.4f} -> 7; 오답 상계 1+ln1200≈{bad:.3f}')


# J07-06  lim n(e-(1+1/n)^n) = e/2
def p06():
    xs = grid(1e-4, 30, 2000)
    ok1 = all(2 * xx / (2 + xx) < math.log1p(xx) < xx - xx**2 / 2 + xx**3 / 3 for xx in xs)  # [증명-수치확인]
    ok2 = True
    for N in range(1, 501):
        a = mp.mpf(1) + mp.mpf(1) / N
        ok2 &= a**(N + mp.mpf(1) / 2) > mp.e
        ok2 &= a**N < mp.e**(1 - mp.mpf(1) / (2 * N) + mp.mpf(1) / (3 * N * N))
    L = sp.limit(n * (sp.E - (1 + 1 / n)**n), n, sp.oo)
    lo = sp.limit(n * sp.E * (1 - sp.exp(-1 / (2 * n) + 1 / (3 * n**2))), n, sp.oo)
    hi = sp.limit(n * sp.E * (1 - (1 + 1 / n)**sp.Rational(-1, 2)), n, sp.oo)
    ok3 = sp.simplify(L - sp.E / 2) == 0 and sp.simplify(lo - sp.E / 2) == 0 and sp.simplify(hi - sp.E / 2) == 0
    report('J07-06', ok1 and ok2 and ok3,
           f'[증명-수치확인] (1) 2000점, (2) n=1..500; (3) sympy 극한 {L}, 하계극한 {lo}, 상계극한 {hi}')


# J07-07  cos 다항 부등식 + ln(1-v) -> lim prod cos(√k/n) = e^{-1/4}
def p07():
    xs = [mp.mpf(j) / 400 for j in range(1, 2001)]  # (0, 5]
    ok1 = all(1 - xx**2 / 2 < mp.cos(xx) < 1 - xx**2 / 2 + xx**4 / 24 for xx in xs)  # [증명-수치확인]
    vs = [mp.mpf(j) / 4000 for j in range(1, 2000)]  # (0, 1/2)
    ok2 = all(-vv - vv**2 < mp.log(1 - vv) < -vv for vv in vs)  # [증명-수치확인]
    ok2b = (1 - math.cos(1)) <= 0.5
    sv_hi = sp.limit(sp.summation(k, (k, 1, n)) / (2 * n**2), n, sp.oo)
    sv_lo = sp.limit(sp.summation(k, (k, 1, n)) / (2 * n**2) - sp.summation(k**2, (k, 1, n)) / (24 * n**4), n, sp.oo)
    sq = sp.limit(sp.summation(k**2, (k, 1, n)) / (4 * n**4), n, sp.oo)
    ok3 = sv_hi == sv_lo == sp.Rational(1, 4) and sq == 0
    N = 10**6
    kk = np.arange(1, N + 1, dtype=np.float64)
    lnP = np.sum(np.log(np.cos(np.sqrt(kk) / N)))
    ok3n = abs(lnP + 0.25) < 1e-5
    report('J07-07', ok1 and ok2 and ok2b and ok3 and ok3n,
           f'[증명-수치확인] cos·ln 부등식 각 2000점; Σv 극한 {sv_lo}~{sv_hi}, Σv^2->0; ln P(10^6)={lnP:.7f} -> e^(-1/4)')


# J07-08  S_n = Σ_{k=n+1}^{2n} 1/k^3, lim n^p S_n 분류
def p08():
    ok1 = all(1 / (kk + 1)**3 < 0.5 * (1 / kk**2 - 1 / (kk + 1)**2) < 1 / kk**3 for kk in range(1, 5001))  # [증명-수치확인]
    low = sp.Rational(1, 2) * (1 / (n + 1)**2 - 1 / (2 * n + 1)**2)
    up = sp.Rational(1, 2) * (1 / n**2 - 1 / (2 * n)**2)
    ok2 = sp.simplify(up - sp.Rational(3, 8) / n**2) == 0
    ok2b = all(float(low.subs(n, N)) < float(mp.zeta(3, N + 1) - mp.zeta(3, 2 * N + 1)) < float(up.subs(n, N)) for N in range(1, 300))
    L2lo = sp.limit(n**2 * low, n, sp.oo)
    L2hi = sp.limit(n**2 * up, n, sp.oo)
    ok3 = L2lo == L2hi == sp.Rational(3, 8)
    # p<2, p>2
    ok3b = sp.limit(n**sp.Rational(3, 2) * up, n, sp.oo) == 0 and sp.limit(n**sp.Rational(5, 2) * low, n, sp.oo) == sp.oo
    N = 10**6
    val = N**2 * (mp.zeta(3, N + 1) - mp.zeta(3, 2 * N + 1))
    ok3n = abs(val - mp.mpf(3) / 8) < 2.0 / N
    report('J07-08', ok1 and ok2 and ok2b and ok3 and ok3b and ok3n,
           f'[증명-수치확인] k=1..5000; 상계 3/(8n^2); n^2 S_n 하·상계 극한 {L2lo},{L2hi}; 수치 {float(val):.7f}; p<2->0, p>2->∞')


# J07-09  Wallis: I_n=∫_0^{π/2} sin^n x dx, lim √n I_n = √(π/2)
def p09():
    I = [sp.integrate(sp.sin(x)**j, (x, 0, sp.pi / 2)) for j in range(0, 32)]
    ok1 = all(sp.simplify(I[j + 2] - sp.Rational(j + 1, j + 2) * I[j]) == 0 for j in range(0, 30))
    ok2 = all(sp.simplify((j + 1) * I[j + 1] * I[j] - sp.pi / 2) == 0 for j in range(0, 31))
    ok2b = all(I[j + 1] < I[j] for j in range(0, 31))
    nn = sp.symbols('nn', positive=True, integer=True)
    In = sp.sqrt(sp.pi) * sp.gamma((nn + 1) / 2) / (2 * sp.gamma(nn / 2 + 1))
    lim3 = sp.limit(sp.sqrt(nn) * In, nn, sp.oo)
    ok3 = sp.simplify(lim3 - sp.sqrt(sp.pi / 2)) == 0
    # 샌드위치: n/(n+1)·π/2 < n I_n^2 < π/2 (n=1..30 정확값)
    ok3b = all(sp.Rational(j, j + 1) * sp.pi / 2 < j * I[j]**2 < sp.pi / 2 for j in range(1, 31))
    report('J07-09', ok1 and ok2 and ok2b and ok3 and ok3b,
           f'[증명-수치확인] 점화식 n=0..29, (n+1)I_(n+1)I_n=π/2 (n=0..30), 단조감소; lim √n I_n = {lim3}')


# J07-10  S=Σ_{k=1}^{1000} k^{-1/3}, floor = 149
def p10():
    f = lambda u: u**(-1 / 3)
    F = lambda u: 1.5 * u**(2 / 3)
    ok1 = True
    for N in range(2, 1001):
        S = math.fsum(f(j) for j in range(1, N + 1))
        ok1 &= 1.5 * ((N + 1)**(2 / 3) - 1) < S < 1.5 * N**(2 / 3) - 0.5  # [증명-수치확인]
    ks = grid(1, 3000, 3000)
    ok2 = all((f(kk) + f(kk + 1)) / 2 > F(kk + 1) - F(kk) for kk in ks)  # [증명-수치확인]
    Sx = mp.fsum(mp.power(j, mp.mpf(-1) / 3) for j in range(1, 1001))
    lower = sp.Rational(3, 2) * (100 - 1) + (1 + sp.Rational(1, 10)) / 2
    upper = sp.Rational(3, 2) * 100 - sp.Rational(1, 2)
    ok3 = lower == sp.Rational(2981, 20) and upper == sp.Rational(299, 2) and lower < Sx < upper and int(mp.floor(Sx)) == 149
    bad = 1.5 * (1001**(2 / 3) - 1)  # 대표 오답: (1) 하계만 사용 -> 148.6 -> 148
    report('J07-10', ok1 and ok2 and ok3,
           f'[증명-수치확인] (1) n=2..1000, (2) 3000점; 하계 {lower}=149.05, 상계 {upper}, S≈{float(Sx):.5f} -> 149; 오답 하계 {bad:.3f}')


if __name__ == '__main__':
    for fn in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
        try:
            fn()
        except Exception as e:  # noqa
            report(fn.__name__.replace('p', 'J07-'), False, f'예외 {e!r}')
    npass = sum(RESULTS)
    print(f'SUMMARY T07: {npass}/{len(RESULTS)} PASS')
    if npass != len(RESULTS):
        sys.exit(1)
