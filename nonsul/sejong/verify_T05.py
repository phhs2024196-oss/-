#!/usr/bin/env python3
"""J05 귀납적으로 정의된 수열과 극한 — 검산 스크립트 (sympy/mpmath)."""
import sys
import math
import sympy as sp
import mpmath as mp

mp.mp.dps = 60
n = sp.symbols('n', positive=True, integer=True)
results = []


def report(pid, ok, msg):
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def close(a, b, tol=mp.mpf('1e-9')):
    a, b = mp.mpf(a), mp.mpf(b)
    return abs(a - b) <= tol * max(1, abs(b))


# ---------------------------------------------------------------- J05-01
def p01():
    # a1=0, a_{n+1}=sqrt(2a_n+3)
    a = [mp.mpf(0)]
    for _ in range(300):
        a.append(mp.sqrt(2 * a[-1] + 3))
    ok = True
    # (1) 0<=a_n<3  [증명-수치확인]
    ok &= all(0 <= x < 3 for x in a[:100])
    # (2) 0<3-a_{n+1}<=(2/3)(3-a_n)  [증명-수치확인]
    ok &= all(0 < 3 - a[k + 1] <= mp.mpf(2) / 3 * (3 - a[k]) for k in range(80))
    # 고정점: 독립 계산
    alpha = [s for s in sp.solve(sp.Eq(sp.Symbol('x') ** 2, 2 * sp.Symbol('x') + 3)) if s >= 0][0]
    ok &= alpha == 3 and close(a[-1], alpha)
    # (3) a_1+...+a_n > 3n-9  [증명-수치확인]
    s = mp.mpf(0)
    for k in range(1, 201):
        s += a[k - 1]
        ok &= s > 3 * k - 9
    # 대표 오답: 0<=a_n 을 안 쓰고 분모를 3으로 잡지 않고 2로 잡으면 r=1 → 수렴 미증명
    report("J05-01", ok, f"lim a_n = {alpha}, (1)(2)(3) 부등식 n<=200 [증명-수치확인]")


# ---------------------------------------------------------------- J05-02
def p02():
    a = [sp.Integer(0)]
    for _ in range(40):
        a.append((3 * a[-1] + 2) / (a[-1] + 2))
    ok = True
    b = [(x - 2) / (x + 1) for x in a]
    ok &= all(sp.simplify(b[k + 1] - b[k] / 4) == 0 for k in range(39))  # (1) [증명-수치확인]
    # (2) md 일반항 a_n=(2*4^n-8)/(4^n+8)
    gen = lambda k: sp.Rational(2 * 4 ** k - 8, 4 ** k + 8)
    ok &= all(a[k - 1] == gen(k) for k in range(1, 41))
    # (3) lim 4^n(2-a_n) : sympy limit
    L = sp.limit(4 ** n * (2 - (2 * 4 ** n - 8) / (4 ** n + 8)), n, sp.oo)
    ok &= L == 24
    # 대표 오답: b_1 을 (0-2)/(0+1)=-2 대신 b_n=-2(1/4)^n 로 지수 하나 밀림 → 6
    wrong = sp.limit(4 ** n * (-3 * (-2 * sp.Rational(1, 4) ** n) / (1 + 2 * sp.Rational(1, 4) ** n)), n, sp.oo)
    report("J05-02", ok, f"b_(n+1)=b_n/4 n<=39, 일반항 n<=40 일치, 극한={L} (오답 재현 {wrong})")


# ---------------------------------------------------------------- J05-03
def p03():
    a = [mp.mpf(3) / 2]
    for _ in range(400):
        a.append(mp.power(2, a[-1] - 1))
    ok = True
    ok &= all(1 < a[k + 1] < a[k] <= mp.mpf(3) / 2 for k in range(200))  # (1)
    r = 2 * mp.sqrt(2) - 2
    ok &= all(0 < a[k + 1] - 1 <= r * (a[k] - 1) for k in range(200))  # (2)
    # 보조 부등식 2^t-1 <= r t (0<=t<=1/2) 격자 1000점 [증명-수치확인]
    ok &= all(mp.power(2, t) - 1 <= r * t + mp.mpf('1e-40') for t in [mp.mpf(i) / 2000 for i in range(1001)])
    # (3) 극한값: 반복으로 얻은 비율 vs sympy limit
    t = sp.symbols('t')
    L = sp.limit((2 ** t - 1) / t, t, 0)
    ok &= sp.simplify(L - sp.log(2)) == 0
    ratio = (a[60] - 1) / (a[59] - 1)
    ok &= close(ratio, mp.log(2), mp.mpf('1e-6'))
    report("J05-03", ok, f"단조·유계, 축소율 2√2-2 n<=200, lim 비율 = {L} (반복값 {mp.nstr(ratio, 10)})")


# ---------------------------------------------------------------- J05-04
def p04():
    def seq(c, N=400):
        a = [mp.mpf(c)]
        for _ in range(N):
            if abs(a[-1]) > mp.mpf(10) ** 30:
                break
            a.append((a[-1] ** 3 + 6) / 7)
        return a
    ok = True
    x = sp.Symbol('x')
    fixed = sorted(sp.solve(sp.Eq((x ** 3 + 6) / 7, x), x))
    ok &= fixed == [-3, 1, 2]
    # (1)(2): 1<c<2 에서 단조 감소, 축소율 (c^2+c+1)/7
    for c in [mp.mpf(i) / 100 for i in range(101, 200)]:
        a = seq(c, 300)
        rr = (c ** 2 + c + 1) / 7
        ok &= all(1 < a[k + 1] < a[k] for k in range(100))
        ok &= all(a[k + 1] - 1 <= rr * (a[k] - 1) * (1 + mp.mpf('1e-30')) for k in range(100))
    # (3) 분류: c in [0,2) -> 1, c=2 -> 2, c>2 -> 발산
    lim_ok = True
    for c in [mp.mpf(i) / 50 for i in range(0, 100)]:
        lim_ok &= close(seq(c)[-1], 1, mp.mpf('1e-8'))
    lim_ok &= seq(2)[-1] == 2
    for c in [mp.mpf(2) + mp.mpf(1) / 10 ** j for j in range(1, 6)] + [3, 5]:
        lim_ok &= seq(c, 2000)[-1] > 10 ** 20
    # 증가 하한 a_{n+1}-2 >= (12/7)(a_n-2) for c>2 [증명-수치확인]
    a = seq(mp.mpf('2.01'), 50)
    ok &= all(a[k + 1] - 2 >= mp.mpf(12) / 7 * (a[k] - 2) for k in range(len(a) - 1))
    ok &= lim_ok
    report("J05-04", ok, "고정점 -3,1,2; 0<=c<2→1, c=2→2, c>2→발산 격자 확인 [증명-수치확인]")


# ---------------------------------------------------------------- J05-05
def p05():
    a = [mp.mpf(1)]
    for _ in range(80):
        a.append(2 * mp.sqrt(a[-1]))
    ok = True
    b = [mp.log(x, 2) for x in a]
    ok &= all(close(b[k + 1] - 2, (b[k] - 2) / 2, mp.mpf('1e-40')) for k in range(70))  # (1)
    gen = lambda k: mp.power(2, 2 - mp.power(2, 2 - k))  # md 일반항
    ok &= all(close(a[k - 1], gen(k), mp.mpf('1e-40')) for k in range(1, 70))
    # (3) sympy limit
    L = sp.limit(2 ** n * (4 - 2 ** (2 - 2 ** (2 - n))), n, sp.oo)
    ok &= sp.simplify(L - 16 * sp.log(2)) == 0
    # 대표 오답: a_n = 2^{2-2^{1-n}} (지수 밀림) → 8 ln 2
    wrong = sp.limit(2 ** n * (4 - 2 ** (2 - 2 ** (1 - n))), n, sp.oo)
    report("J05-05", ok, f"log 치환, 일반항 n<=69, lim 2^n(4-a_n)={L} (오답 {wrong})")


# ---------------------------------------------------------------- J05-06
def p06():
    a = [sp.Rational(5, 2)]
    for _ in range(9):
        a.append(a[-1] ** 2 - 2)
    ok = True
    gen = lambda k: sp.Integer(2) ** (2 ** (k - 1)) + sp.Rational(1, 2) ** (2 ** (k - 1))
    ok &= all(a[k - 1] == gen(k) for k in range(1, 10))  # (1)
    P = sp.Integer(1)
    for k in range(1, 9):
        P *= a[k - 1]
        ok &= sp.Rational(3, 2) * P == sp.Integer(2) ** (2 ** k) - sp.Rational(1, 2) ** (2 ** k)  # (2)
    # (3) 닫힌 식 극한 (x=2^{2^n} → oo)
    X = sp.symbols('X', positive=True)
    L = sp.limit((X - 1 / X) / sp.Rational(3, 2) / (X + 1 / X), X, sp.oo)
    ok &= L == sp.Rational(2, 3)
    ratio = mp.mpf(sp.Rational(P, a[8]))
    ok &= close(ratio, mp.mpf(2) / 3, mp.mpf('1e-30'))
    report("J05-06", ok, f"일반항 n<=9 정확, 곱 항등식 n<=8 정확, 극한={L}")


# ---------------------------------------------------------------- J05-07
def p07():
    a = [sp.Integer(1)]
    for _ in range(60):
        a.append(sp.Rational(6, 1) / (a[-1] + 1))
    ok = True
    ok &= all(1 <= x <= 3 for x in a)  # (1)
    ok &= all(abs(a[k + 2] - 2) <= sp.Rational(1, 2) * abs(a[k] - 2) for k in range(58))  # (2)
    # 1단계 축소율은 1 을 넘을 수 있음(a_1=1 → |a_2-2|=|a_1-2|)
    ok &= abs(a[1] - 2) == abs(a[0] - 2)
    x = sp.Symbol('x')
    two = sp.simplify(6 / (6 / (x + 1) + 1))
    ok &= sp.simplify(two - 2 - 4 * (x - 2) / (x + 7)) == 0
    ok &= all(x_ != 2 for x_ in a)
    L = sp.limit((two - 2) / (x - 2), x, 2)
    ok &= L == sp.Rational(4, 9)
    ok &= close(mp.mpf((a[51] - 2) / (a[49] - 2)), mp.mpf(4) / 9, mp.mpf('1e-9'))
    ok &= close(mp.mpf(a[60]), 2)
    report("J05-07", ok, f"1<=a_n<=3, 2단계 축소 1/2 n<=58, lim a_n=2, 비율극한={L}")


# ---------------------------------------------------------------- J05-08
def p08():
    a = [sp.Rational(3, 2)]
    for _ in range(14):
        a.append(a[-1] ** 2 - a[-1] + 1)
    ok = True
    ok &= all(a[k - 1] >= sp.Rational(k + 5, 4) for k in range(1, 15))  # (1)
    ok &= all(1 / a[k] == 1 / (a[k] - 1) - 1 / (a[k + 1] - 1) for k in range(13))  # (2)
    S = sum(mp.mpf(1) / mp.mpf(x) for x in a[:12])
    # 꼬리 = 1/(a_13 - 1) 로 정확 → 부분합 + 꼬리 = 2
    S_exact = sum(1 / x for x in a[:12]) + 1 / (a[12] - 1)
    ok &= S_exact == 2
    ok &= close(S, 2, mp.mpf('1e-9'))
    # 대표 오답: 1/(a_1-1) 대신 1/a_1 로 시작 → 2/3? (계산 실수 재현: 첫 항 누락) 2 - 2/3 = 4/3
    report("J05-08", ok, f"하한 n<=14, 망원 항등식 n<=13 정확, 합 = 2 (부분합 {mp.nstr(S, 15)})")


# ---------------------------------------------------------------- J05-09
def p09():
    a = [mp.mpf(1) / 2]
    for _ in range(60):
        a.append(mp.sqrt((1 + a[-1]) / 2))
    ok = True
    ok &= all(mp.mpf(1) / 2 <= x < 1 for x in a[:50])  # (1)
    ok &= all(1 - a[k + 1] <= (1 - a[k]) / 3 for k in range(50))
    gen = lambda k: mp.cos(mp.pi / (3 * mp.power(2, k - 1)))  # (2) md 일반항
    ok &= all(close(a[k - 1], gen(k), mp.mpf('1e-45')) for k in range(1, 40))
    th = sp.pi / 3
    k = sp.symbols('k', positive=True, integer=True)
    closed = sp.sin(2 * th) / (2 ** n * sp.sin(th / 2 ** (n - 1)))
    L = sp.limit(closed, n, sp.oo)
    ok &= sp.simplify(L - 3 * sp.sqrt(3) / (4 * sp.pi)) == 0
    P = mp.mpf(1)
    for x in a[:45]:
        P *= x
    ok &= close(P, 3 * mp.sqrt(3) / (4 * mp.pi), mp.mpf('1e-20'))
    report("J05-09", ok, f"a_n=cos(π/(3·2^(n-1))) n<=39, 곱 극한={L} (수치 {mp.nstr(P, 12)})")


# ---------------------------------------------------------------- J05-10
def p10():
    N = 200000
    a = 1.0
    ok = True
    from fractions import Fraction
    # 정확 계산(작은 n)으로 (1)(2) 부등식 확인 [증명-수치확인]
    af = Fraction(1)
    H = 0.0
    for k in range(1, 40):
        ok &= 0 < af <= Fraction(2, k + 1)
        ok &= float(1 / af) <= (k + 1) / 2 + math.log(k) + 1e-12
        af = af - af * af / 2
    # 부동소수로 큰 n 까지
    x = mp.mpf(1)
    mp.mp.dps = 30
    vals = {}
    for k in range(1, 5001):
        ok &= 1 / x <= (k + 1) / mp.mpf(2) + mp.log(k) + mp.mpf('1e-20')
        ok &= x <= mp.mpf(2) / (k + 1)
        x = x - x * x / 2
    mp.mp.dps = 60
    xf = 1.0
    for k in range(1, N + 1):
        if k == N:
            vals[k] = k * xf
        xf = xf - xf * xf / 2
    # (3) 샌드위치 하한 n/((n+1)/2+ln n) 의 극한 = 2 (sympy)
    lo = sp.limit(n / ((n + 1) / 2 + sp.log(n)), n, sp.oo)
    hi = sp.limit(2 * n / (n + 1), n, sp.oo)
    ok &= lo == 2 and hi == 2
    ok &= abs(vals[N] - 2) < 2e-4 * 1e1
    report("J05-10", ok, f"(1)(2) 부등식 n<=5000, 샌드위치 양끝 극한 {lo},{hi}, n a_n(n=2e5)={mp.nstr(vals[N], 8)}")


for f in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        f()
    except Exception as e:  # noqa
        report(f.__name__, False, f"예외 {e!r}")

passed = sum(results)
print(f"SUMMARY T05: {passed}/10 PASS")
if passed != 10:
    sys.exit(1)
