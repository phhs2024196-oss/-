# -*- coding: utf-8 -*-
"""D10. 정적분과 부등식·존재성 증명 — sympy 검산
정답을 하드코딩해 자기 자신과 비교하지 않고 문제 설정에서 다시 계산한다.
증명 소문항은 [증명-수치확인] 으로 격자·무작위 반례 탐색."""
import sys
import random
from sympy import (symbols, Rational, integrate, diff, solve, sqrt, exp, log, sin, cos, pi, E,
                   simplify, factor, expand, Poly, real_roots, binomial, factorial, Abs, N, floor,
                   summation, oo, limit, nsolve, lambdify)
import mpmath as mp

mp.mp.dps = 40
x, t, k, c, a, b = symbols('x t k c a b', real=True)
RESULTS = []
random.seed(2027)


def report(pid, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


# D10-01  f=4x^3-3x^2-2x+1, F=int_0^x f
def p01():
    f = 4 * x**3 - 3 * x**2 - 2 * x + 1
    F = integrate(f, (x, 0, x))
    fac = factor(F)
    roots = solve(f, x)
    in01 = [r for r in roots if 0 < float(r) < 1]
    inm10 = [r for r in roots if -1 < float(r) < 0]
    area = integrate(Abs(F), (x, -1, 1))
    area2 = -integrate(F, (x, -1, 0)) + integrate(F, (x, 0, 1))
    ok_sign = all(float(F.subs(x, Rational(j, 100))) < 0 for j in range(-99, 0)) and \
        all(float(F.subs(x, Rational(j, 100))) > 0 for j in range(1, 100))
    wrong = integrate(F, (x, -1, 1))  # 절댓값 없이 적분
    ok = (expand(fac - x * (x - 1)**2 * (x + 1)) == 0 and set(roots) == {1, (-1 + sqrt(17)) / 8, (-1 - sqrt(17)) / 8}
          and in01 == [(-1 + sqrt(17)) / 8] and inm10 == [(-1 - sqrt(17)) / 8] and area2 == Rational(1, 2)
          and simplify(area - Rational(1, 2)) == 0 and ok_sign)
    report('D10-01', ok, f"F={fac}, 근 {roots}, 넓이={area2} (오답 {wrong})")


# D10-02  1/x 의 볼록성, ln2 범위
def p02():
    lo = lambda A, B: (B - A) * 2 / (A + B)
    hi = lambda A, B: (B - A) * (A + B) / (2 * A * B)
    # (1) 0<a<b 무작위 500쌍 [증명-수치확인]
    ok1 = True
    for _ in range(500):
        A = mp.mpf(random.uniform(0.01, 10)); B = A + mp.mpf(random.uniform(0.001, 10))
        L = mp.log(B / A)
        if not (lo(A, B) < L < hi(A, B)):
            ok1 = False
    r2 = (lo(Rational(1), Rational(2)), hi(Rational(1), Rational(2)))
    r3 = (lo(1, Rational(3, 2)) + lo(Rational(3, 2), 2), hi(1, Rational(3, 2)) + hi(Rational(3, 2), 2))
    l2 = mp.log(2)
    ok3 = r3[0] < l2 < r3[1]
    # (4) 대소: 24 ln2 vs 17, 35 ln2 vs 24  (r3 로 결정, r2 로는 결정 불가)
    dec_a = 24 * r3[1] <= 17  # ln2 < 17/24 (엄격) 이므로 24ln2 < 17
    dec_b = 35 * r3[0] >= 24  # ln2 > 24/35 (엄격)
    undec_r2 = not (24 * r2[1] < 17) and not (35 * r2[0] > 24)
    real_a = 24 * l2 < 17 and 35 * l2 > 24
    ok = ok1 and r2 == (Rational(2, 3), Rational(3, 4)) and r3 == (Rational(24, 35), Rational(17, 24)) and ok3 \
        and dec_a and dec_b and undec_r2 and real_a
    report('D10-02', ok, f"(2) {r2}, (3) {r3}, 2^24<e^17, e^24<2^35 [증명-수치확인 500쌍]")


# D10-03  a_n = sum_{k=n+1}^{2n} 1/k
def p03():
    ok1 = all(Rational(1, kk + 1) < log(Rational(kk + 1, kk)) < Rational(1, kk) for kk in range(1, 300))
    ok2 = True
    for n in range(1, 400):
        an = mp.fsum(mp.mpf(1) / j for j in range(n + 1, 2 * n + 1))
        if not (mp.log(mp.mpf(2 * n + 1) / (n + 1)) < an < mp.log(2)):
            ok2 = False
    nn = symbols('nn', positive=True)
    lim3 = limit(log((2 * nn + 1) / (nn + 1)), nn, oo)
    lim4 = limit(log((6 * nn + 1) / (2 * nn + 1)), nn, oo)
    bn = mp.fsum(mp.mpf(1) / j for j in range(2 * 20000 + 1, 6 * 20000 + 1))
    ok = ok1 and ok2 and lim3 == log(2) and simplify(lim4 - log(3)) == 0 and abs(bn - mp.log(3)) < 1e-4
    report('D10-03', ok, f"샌드위치 n=1..399 확인 [증명-수치확인], lim a_n={lim3}, lim b_n={simplify(lim4)}")


# D10-04  최소제곱
def p04():
    g = 6 * x - 2
    c1, c2 = integrate(g, (x, 0, 1)), integrate(x * g, (x, 0, 1))
    # g 가 조건을 만족, g = p + q x 꼴 중 유일
    p_, q_ = symbols('p_ q_')
    sol = solve([integrate(p_ + q_ * x, (x, 0, 1)) - 1, integrate(x * (p_ + q_ * x), (x, 0, 1)) - 1], [p_, q_])
    g2 = integrate(g**2, (x, 0, 1))
    # (2) f=x 가 등식 만족, 다른 다항식 무작위로는 깨짐
    f2 = x
    ok2 = integrate(f2**2, (x, 0, 1)) == 2 * integrate(x * f2, (x, 0, 1)) - Rational(1, 3)
    # (3)(4) 조건 만족 함수 무작위 생성: f = g + h, h 는 int h = int x h = 0 인 다항식
    ok4 = True
    for _ in range(200):
        cs = [Rational(random.randint(-9, 9), random.randint(1, 5)) for _ in range(4)]
        h = sum(cs[i] * x**(i + 2) for i in range(2)) + cs[2] * x + cs[3]
        # 1, x 와 직교하도록 cs[2], cs[3] 재설정
        A_, B_ = symbols('A_ B_')
        hh = cs[0] * x**2 + cs[1] * x**3 + A_ * x + B_
        s = solve([integrate(hh, (x, 0, 1)), integrate(x * hh, (x, 0, 1))], [A_, B_])
        hh = hh.subs(s)
        f = g + hh
        if not (integrate(f, (x, 0, 1)) == 1 and integrate(x * f, (x, 0, 1)) == 1):
            ok4 = False
        if integrate(f * g, (x, 0, 1)) != 4:
            ok4 = False
        val = integrate(f**2, (x, 0, 1))
        if val < 4 or (val == 4 and expand(hh) != 0):
            ok4 = False
    ok = c1 == 1 and c2 == 1 and sol == {p_: -2, q_: 6} and g2 == 4 and ok2 and ok4
    report('D10-04', ok, f"g=6x-2, int fg=4, int g^2={g2}, 최솟값 4 [증명-수치확인 200함수]")


# D10-05  e^x = 1+2x, 2.71<e<2.72
def p05():
    F = lambda X: mp.e**X - 1 - 2 * X
    r = mp.findroot(F, 1.25)
    # 근이 0, r 두 개뿐: F 는 아래로 볼록, F(ln2)<0
    ok1 = F(mp.log(2)) < 0 and abs(F(0)) < 1e-30
    # 보장된 범위로 판정 (e 의 상하한만 이용)
    ok2 = (Rational(271, 100)**3 > 16) and (Rational(272, 100)**5 < Rational(7, 2)**4) and Rational(5, 4) < r < Rational(3, 2)
    rr = symbols('rr', positive=True)
    area = simplify((integrate(1 + 2 * x - exp(x), (x, 0, rr))).subs(exp(rr), 1 + 2 * rr))
    area_num = mp.quad(lambda X: 1 + 2 * X - mp.e**X, [0, r])
    V = (pi * integrate((1 + 2 * x)**2 - exp(2 * x), (x, 0, rr)))
    u = symbols('u')
    Vu = simplify(V.subs(exp(2 * rr), (1 + 2 * rr)**2).subs(rr, (u - 1) / 2))
    Vclaim = pi * (u**3 - 3 * u**2 + 2) / 6
    Vnum = mp.pi * mp.quad(lambda X: (1 + 2 * X)**2 - mp.e**(2 * X), [0, r])
    lowV, highV = Vclaim.subs(u, Rational(7, 2)), Vclaim.subs(u, 4)
    ok = (ok1 and ok2 and expand(area - (rr**2 - rr)) == 0 and abs(area_num - (r**2 - r)) < 1e-25
          and simplify(Vu - Vclaim) == 0 and simplify(lowV - 65 * pi / 48) == 0 and simplify(highV - 3 * pi) == 0
          and float(lowV) < Vnum < float(highV) and Rational(5, 16) < r**2 - r < Rational(3, 4))
    report('D10-05', ok, f"r≈{mp.nstr(r, 8)}, 넓이 r^2-r≈{mp.nstr(r**2 - r, 6)}, V∈(65π/48,3π), V≈{mp.nstr(Vnum, 8)}")


# D10-06  n! 범위와 (n!)^{1/n}/n -> 1/e
def p06():
    nn = symbols('nn', positive=True)
    I1 = integrate(log(x), (x, 1, nn))
    ok1 = simplify(I1 - (nn * log(nn) - nn + 1)) == 0
    ok2 = all((n * mp.log(n) - n + 1) < mp.log(mp.factorial(n)) < ((n + 1) * mp.log(n + 1) - n) for n in range(2, 500))
    ok3 = all(mp.log(X) < mp.sqrt(X) for X in [mp.mpf(1) + j / mp.mpf(7) for j in range(1, 5000)])
    hmin = (sqrt(x) - log(x)).subs(x, 4)
    L = limit(((nn + 1) * log(nn + 1) - nn) / nn - log(nn), nn, oo)
    L2 = limit((nn * log(nn) - nn + 1) / nn - log(nn), nn, oo)
    val = mp.e**(mp.log(mp.factorial(10**6)) / 10**6) / 10**6
    ok = ok1 and ok2 and ok3 and float(hmin) > 0 and L == -1 and L2 == -1 and abs(val - 1 / mp.e) < 1e-4
    report('D10-06', ok, f"부등식 n=2..499, ln x<√x [증명-수치확인], 극한 e^-1 (n=10^6 값 {mp.nstr(val, 6)})")


# D10-07  ln(1+x) >= x - k x^2 (x>=0) iff k>=1/2
def p07():
    def holds(K):
        return all(mp.log(1 + X) >= X - K * X**2 - mp.mpf(10)**-35
                   for X in [mp.mpf(j) / 1000 for j in range(0, 3001)] + [mp.mpf(10)**(-j) for j in range(1, 12)])
    ok2 = holds(mp.mpf(1) / 2) and holds(1) and not holds(mp.mpf('0.49')) and not holds(0) and not holds(-1)
    ok1 = all(mp.log(1 + X) <= X for X in [mp.mpf(j) / 100 for j in range(0, 2000)])
    lo = integrate(x**2 - x**4 / 2, (x, 0, 1))
    hi = integrate(x**2, (x, 0, 1))
    I = mp.quad(lambda X: mp.log(1 + X**2), [0, 1])
    nn = symbols('nn', positive=True, integer=True)
    j = symbols('j', positive=True, integer=True)
    up = limit(summation(j / nn**2, (j, 1, nn)), nn, oo)
    dn = limit(summation(j / nn**2 - j**2 / (2 * nn**4), (j, 1, nn)), nn, oo)
    P = mp.e**mp.fsum(mp.log(1 + mp.mpf(jj) / 4000**2) for jj in range(1, 4001))
    ok = ok1 and ok2 and lo == Rational(7, 30) and hi == Rational(1, 3) and lo < I < hi and up == Rational(1, 2) \
        and dn == Rational(1, 2) and abs(P - mp.sqrt(mp.e)) < 1e-3
    report('D10-07', ok, f"k>=1/2 [증명-수치확인], 7/30<I={mp.nstr(I, 6)}<1/3, 곱 극한 √e (n=4000: {mp.nstr(P, 6)})")


# D10-08  르장드르형 직교 조건
def p08():
    p2 = x**2 + a * x + b
    s2 = solve([integrate(p2, (x, 0, 1)), integrate(x * p2, (x, 0, 1))], [a, b])
    q2 = p2.subs(s2)
    r2 = solve(q2, x)
    p3 = x**3 + a * x**2 + b * x + c
    s3 = solve([integrate(x**j * p3, (x, 0, 1)) for j in range(3)], [a, b, c])
    q3 = p3.subs(s3)
    r3 = real_roots(Poly(q3 * 20, x))
    ok_r3 = len(set(r3)) == 3 and all(0 < float(r) < 1 for r in r3)
    # [증명-수치확인] 무작위 연속함수(다항식) 중 1,x,x^2 와 직교하는 것은 (0,1)에서 부호변화 >=3
    ok_gen = True
    for _ in range(150):
        cs = [random.randint(-9, 9) for _ in range(3)]
        A_, B_, C_ = symbols('A_ B_ C_')
        h = cs[0] * x**3 + cs[1] * x**4 + cs[2] * x**5 + A_ * x**2 + B_ * x + C_
        sol = solve([integrate(x**j * h, (x, 0, 1)) for j in range(3)], [A_, B_, C_])
        h = expand(h.subs(sol))
        if h == 0:
            continue
        # (0,1) 안의 홀수 중복도 근(= 부호가 바뀌는 점) 개수
        mult = {}
        for r in real_roots(Poly(h, x)):
            mult[r] = mult.get(r, 0) + 1
        changes = sum(1 for r, m in mult.items() if 0 < r < 1 and m % 2 == 1)
        if changes < 3:
            ok_gen = False
    ok = (s2 == {a: -1, b: Rational(1, 6)} and set(r2) == {(3 - sqrt(3)) / 6, (3 + sqrt(3)) / 6}
          and s3 == {a: Rational(-3, 2), b: Rational(3, 5), c: Rational(-1, 20)} and ok_r3 and ok_gen)
    report('D10-08', ok, f"q2={q2}, q3={q3}, 근 3개 (0,1) 안, 직교 다항식 150개 부호변화>=3 [증명-수치확인]")


# D10-09  f'(c) = k f(c)
def p09():
    f1 = x * (1 - x)
    c1 = [r for r in solve(diff(f1, x) - 2 * f1, x) if 0 < float(r) < 1]
    # (2) [증명-수치확인] 무작위 f(0)=f(1)=0 다항식과 k 에서 해 존재
    ok2 = True
    for _ in range(200):
        cs = [random.randint(-5, 5) for _ in range(3)]
        f = x * (1 - x) * (cs[0] + cs[1] * x + cs[2] * x**2)
        if expand(f) == 0:
            continue
        K = random.uniform(-6, 6)
        h = lambdify(x, diff(f, x) - K * f)
        vals = [float(h(i / 1000)) for i in range(1, 1000)]
        if not (any(v == 0 for v in vals) or any(vals[i] * vals[i + 1] < 0 for i in range(len(vals) - 1))):
            ok2 = False
    # (3) g = pi cos(pi x), k = pi
    g = pi * cos(pi * x)
    G = integrate(g.subs(x, t), (t, 0, x))
    cands = [(mp.mpf(i) + mp.mpf(1) / 3) / 4000 for i in range(0, 4000)]
    hfun = lambda X: mp.pi * mp.cos(mp.pi * X) - mp.pi * mp.sin(mp.pi * X)
    signch = [i for i in range(len(cands) - 1) if hfun(cands[i]) * hfun(cands[i + 1]) < 0]
    ok3 = integrate(g, (x, 0, 1)) == 0 and simplify(G - sin(pi * x)) == 0 and simplify(g.subs(x, Rational(1, 4)) - pi * G.subs(x, Rational(1, 4))) == 0 \
        and len(signch) == 1
    # (4) f = x(1-x)e^x, 모든 k 에서 (0,1) 의 해가 유일, k=3 이면 1-1/√2
    f4 = x * (1 - x) * exp(x)
    quad = simplify((diff(f4, x) - k * f4) / exp(x))
    ok4 = True
    for K in [Rational(i, 4) for i in range(-40, 41)]:
        rs = [r for r in real_roots(Poly(expand(quad.subs(k, K) * 4), x)) if 0 < r < 1]
        if len(set(rs)) != 1:
            ok4 = False
    c4 = [r for r in solve(quad.subs(k, 3), x) if 0 < float(r) < 1]
    ok = (c1 == [1 - sqrt(2) / 2] and ok2 and ok3 and ok4 and c4 == [1 - sqrt(2) / 2])
    report('D10-09', ok, f"(1) c={c1[0]}, (3) c=1/4, (4) k=-10..10 에서 유일, k=3 → c={c4[0]} [증명-수치확인]")


# D10-10  S = sum_{k=1}^{100} 1/sqrt(k)
def p10():
    S = mp.fsum(1 / mp.sqrt(j) for j in range(1, 101))
    lo1, hi1 = 2 * mp.sqrt(101) - 2, mp.mpf(19)
    lo2 = integrate(1 / sqrt(x), (x, 1, 100)) + (1 + Rational(1, 10)) / 2
    hi3_exact = 1 + 1 / mp.sqrt(2) + 2 * (mp.sqrt(mp.mpf('100.5')) - mp.sqrt(mp.mpf('2.5')))
    hi3_bound = 1 + Rational(7072, 10000) + 2 * (Rational(10025, 1000) - Rational(1581, 1000))
    ok_given = (Rational(7072, 10000)**2 * 2 > 1) and Rational(10025, 1000)**2 > Rational(1005, 10) and Rational(1581, 1000)**2 < Rational(5, 2)
    # 개별 부등식 확인 [증명-수치확인]
    okt = all((1 / mp.sqrt(j) + 1 / mp.sqrt(j + 1)) / 2 > 2 * (mp.sqrt(j + 1) - mp.sqrt(j)) for j in range(1, 100))
    okm = all(1 / mp.sqrt(j) < 2 * (mp.sqrt(j + 0.5) - mp.sqrt(j - 0.5)) for j in range(2, 101))
    ans = floor(10 * N(S, 30))
    wrong_hi = 1 + 2 * (mp.sqrt(mp.mpf('100.5')) - mp.sqrt(mp.mpf('1.5')))
    assert wrong_hi > mp.mpf('18.6')
    ok = (lo1 < S < hi1 and floor(lo1) == 18 and lo2 == Rational(371, 20) and S > lo2 and okt and okm and
          S < hi3_exact < hi3_bound < Rational(186, 10) and ok_given and ans == 185)
    report('D10-10', ok, f"S≈{mp.nstr(S, 10)}, 하한 {lo2}=18.55, 상한 {float(hi3_bound)}, [10S]={ans}")


if __name__ == '__main__':
    for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
        try:
            fn()
        except Exception as e:  # noqa
            report(fn.__name__.replace('p', 'D10-'), False, f'예외 {e!r}')
    n = sum(RESULTS)
    print(f"SUMMARY T10: {n}/{len(RESULTS)} PASS")
    if n != len(RESULTS) or len(RESULTS) != 10:
        sys.exit(1)
