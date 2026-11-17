# -*- coding: utf-8 -*-
"""D09. 정적분의 계산과 넓이·부피 종합 — sympy 검산
정답을 하드코딩해 자기 자신과 비교하지 않고, 문제 설정에서 독립적으로 다시 계산한다."""
import sys
import random
from sympy import (symbols, Rational, integrate, diff, solve, sqrt, exp, log, sin, cos,
                   pi, E, simplify, nsimplify, Abs, binomial, factorial, Symbol, summation,
                   oo, limit, N, factor, expand, Poly, nsolve)
import mpmath as mp

mp.mp.dps = 40
x, t, y, s, a, u = symbols('x t y s a u', real=True)
RESULTS = []


def report(pid, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def close(v, w, tol=1e-9):
    v, w = float(N(v, 30)), float(N(w, 30))
    return abs(v - w) <= tol * max(1.0, abs(w))


# D09-01  f(x)=x^3+x, g = f^{-1}
def p01():
    f = x**3 + x
    # (1) g(10), g'(10)
    r = [z for z in solve(f - 10, x) if z.is_real]
    g10 = r[0]
    gp = 1 / diff(f, x).subs(x, g10)
    # (2) int_0^10 g : 수치 적분(역함수를 직접 풀어서)과 비교
    ginv = lambda X: mp.findroot(lambda T: T**3 + T - X, 1)
    I2 = mp.quad(ginv, [0, 10])
    I2_formula = 10 * g10 - integrate(f, (x, 0, g10))
    # (3) g 와 y=x/5 사이 넓이 ; 교점 확인
    inter = solve(f / 5 - x, x)  # x=f(t) 일 때 t=f(t)/5
    A3 = mp.quad(lambda X: ginv(X) - X / 5, [0, 10])
    ok_sign = all(ginv(X) >= X / 5 - 1e-30 for X in [mp.mpf(k) / 10 for k in range(1, 100)])
    # (4) 단면 원판(반지름 g)에서 반지름 x/5 원판을 뺀 고리 -> 부피
    V4 = mp.pi * mp.quad(lambda X: ginv(X)**2 - (X / 5)**2, [0, 10])
    V4_sym = pi * (integrate(t**2 * diff(f.subs(x, t), t), (t, 0, 2)) - integrate(x**2 / 25, (x, 0, 10)))
    # 대표 오답: 직사각형을 빼먹고 int_0^2 f = 6
    wrong = integrate(f, (x, 0, 2))
    ok = (g10 == 2 and gp == Rational(1, 13) and I2_formula == 14 and abs(I2 - 14) < 1e-20
          and sorted(inter) == [-2, 0, 2] and abs(A3 - 4) < 1e-20 and ok_sign
          and simplify(V4_sym - 128 * pi / 15) == 0 and abs(V4 - 128 * mp.pi / 15) < 1e-20)
    report('D09-01', ok, f"g'(10)={gp}, int g=14, 넓이=4, 부피={V4_sym} (오답 {wrong})")


# D09-02  ln x on [1,3], 접선과의 넓이 S(t)
def p02():
    I1 = integrate(log(x), (x, 1, 3))
    tt = symbols('tt', positive=True)
    tang = log(tt) + (x - tt) / tt
    S = simplify(integrate(tang - log(x), (x, 1, 3)))
    S_claim = 4 / tt + 2 * log(tt) - 3 * log(3)
    crit = solve(diff(S, tt), tt)
    Smin = simplify(S.subs(tt, 2))
    # 경계값 비교 (1<=t<=3)
    ok_min = float(S.subs(tt, 1)) > float(Smin) and float(S.subs(tt, 3)) > float(Smin)
    # (4) ln x 위에 있는 직선 m x + n 들을 무작위 생성 -> 적분 >= Smin, 등호는 t=2 접선뿐
    random.seed(1)
    ok4 = True
    for _ in range(2000):
        m = random.uniform(0, 1.5)
        # 직선 기울기 m 일 때 ln x 위에 놓이게 하는 최소 n 보다 크게
        xs = [1 + 2 * k / 400 for k in range(401)]
        nmin = max(mp.log(X) - m * X for X in xs)
        n = nmin + random.uniform(0, 0.3)
        val = mp.quad(lambda X: m * X + n - mp.log(X), [1, 3])
        if val < float(Smin) - 1e-9:
            ok4 = False
    # 정적분 = 2*l(2) 항등식 확인
    m_, n_ = symbols('m_ n_')
    ok_id = simplify(integrate(m_ * x + n_, (x, 1, 3)) - 2 * (2 * m_ + n_)) == 0
    wrong = simplify(S.subs(tt, Rational(3, 2) + Rational(1, 2)) - 0)  # 정답 재현용
    ok = (simplify(I1 - (3 * log(3) - 2)) == 0 and simplify(S - S_claim) == 0 and crit == [2]
          and simplify(Smin - (2 + 2 * log(2) - 3 * log(3))) == 0 and ok_min and ok4 and ok_id
          and float(Smin) > 0)
    report('D09-02', ok, f"S(t)={S_claim}, 최소 t=2, 최솟값={Smin}≈{float(Smin):.5f} [증명-수치확인 2000직선]")


# D09-03  y=sin x (0<=x<=pi) 아래 영역, 높이 y 단면이 고리(안 g(y), 밖 pi-g(y))
def p03():
    # (1) 대칭: sin(pi-x)=sin x
    ok1 = simplify(sin(pi - x) - sin(x)) == 0
    # (2) int_0^1 g(y)dy, 치환 y=sin x
    I2 = integrate(x * cos(x), (x, 0, pi / 2))
    I2num = mp.quad(lambda Y: mp.asin(Y), [0, 1])
    # (3)(4) 단면 넓이 pi((pi-g)^2-g^2) 적분
    V = mp.quad(lambda Y: mp.pi * ((mp.pi - mp.asin(Y))**2 - mp.asin(Y)**2), [0, 1])
    Vsym = pi * (pi**2 - 2 * pi * I2)
    wrong = pi * integrate(sin(x)**2, (x, 0, pi))  # x축 둘레로 착각
    # (3) int g^2, int (pi-g)^2
    G2 = integrate(x**2 * cos(x), (x, 0, pi / 2))
    H2 = integrate((pi - x)**2 * cos(x), (x, 0, pi / 2))
    G2n = mp.quad(lambda Y: mp.asin(Y)**2, [0, 1])
    ok3 = (simplify(G2 - (pi**2 / 4 - 2)) == 0 and simplify(H2 - (pi**2 / 4 + 2 * pi - 2)) == 0
           and abs(G2n - (mp.pi**2 / 4 - 2)) < 1e-25 and simplify(pi * (H2 - G2) - 2 * pi**2) == 0)
    ok = (ok1 and ok3 and simplify(I2 - (pi / 2 - 1)) == 0 and abs(I2num - (mp.pi / 2 - 1)) < 1e-25
          and simplify(Vsym - 2 * pi**2) == 0 and abs(V - 2 * mp.pi**2) < 1e-25)
    report('D09-03', ok, f"int g = {I2}, V = {simplify(Vsym)} (오답 {wrong})")


# D09-04  S(a)=int_0^1 |e^x - a| dx
def p04():
    ap = symbols('ap', positive=True)
    Smid = integrate(ap - exp(x), (x, 0, log(ap))) + integrate(exp(x) - ap, (x, log(ap), 1))
    Smid = simplify(Smid)
    claim = 2 * ap * log(ap) - 3 * ap + 1 + E
    crit = solve(diff(claim, ap), ap)
    Smin = simplify(claim.subs(ap, sqrt(E)))
    # 전 구간 수치 최소 (a in [-2, 5])
    Sf = lambda A: mp.quad(lambda X: abs(mp.e**X - A), [0, mp.log(A)] if 1 < A < mp.e else [0, 1]) + \
        (mp.quad(lambda X: abs(mp.e**X - A), [mp.log(A), 1]) if 1 < A < mp.e else 0)
    grid = [mp.mpf(-2) + k * mp.mpf(7) / 3000 for k in range(3001)]
    best = min(Sf(A) for A in grid)
    ok_lt1 = simplify(integrate(exp(x) - ap, (x, 0, 1)) - (E - 1 - ap)) == 0
    # (4) U(b)=int_0^2 |e^{x/2}-b| dx = 2 S(b)
    U = lambda B: mp.quad(lambda X: abs(mp.e**(X / 2) - B), [0, 2 * mp.log(B), 2]) if 1 < B < mp.e else \
        mp.quad(lambda X: abs(mp.e**(X / 2) - B), [0, 2])
    Umin = min(U(B) for B in [mp.mpf(1) + k * (mp.e - 1) / 2000 for k in range(1, 2000)])
    Uat = U(mp.sqrt(mp.e))
    target = 2 * (mp.sqrt(mp.e) - 1)**2
    ok = (simplify(Smid - claim) == 0 and crit == [sqrt(E)] and simplify(Smin - (sqrt(E) - 1)**2) == 0
          and abs(best - (mp.sqrt(mp.e) - 1)**2) < 1e-6 and best >= (mp.sqrt(mp.e) - 1)**2 - 1e-20
          and ok_lt1 and abs(Uat - target) < 1e-25 and Umin >= target - 1e-25)
    report('D09-04', ok, f"S(a)={claim}, 최소 (√e-1)^2, U 최소 2(√e-1)^2 [격자 3000점 확인]")


# D09-05  y=x^3, P1=(1,1), P_{n+1} 의 접선이 P_n 을 지남
def p05():
    p = symbols('p')
    # 접선 at s 가 (p,p^3)을 지날 조건
    cond = factor(p**3 - s**3 - 3 * s**2 * (p - s))
    sols_s = solve(cond, s)
    area = integrate(x**3 - (3 * s**2 * x - 2 * s**3), (x, -2 * s, s))  # s>0 기준 부호 확인
    area_abs = simplify(abs(area.subs(s, 1)))
    # 수열 생성
    seq = [Rational(1)]
    for _ in range(40):
        cands = [z.subs(p, seq[-1]) for z in sols_s if z.subs(p, seq[-1]) != seq[-1]]
        seq.append(cands[0])
    areas = []
    for n in range(40):
        sn1, sn = seq[n + 1], seq[n]
        lo, hi = sorted([sn1, sn])
        tl = 3 * sn1**2 * x - 2 * sn1**3
        areas.append(Abs(integrate(x**3 - tl, (x, lo, hi))))
    tot = sum(areas)
    nn = symbols('nn', integer=True, positive=True)
    tot_sym = summation(Rational(27, 4) * Rational(1, 16)**nn, (nn, 1, oo))
    wrong = summation(Rational(27, 4) * Rational(1, 16)**nn, (nn, 0, oo))
    ok = (set(sols_s) == {p, -p / 2} and area_abs == Rational(27, 4) and seq[1] == Rational(-1, 2)
          and areas[0] == Rational(27, 64) and tot_sym == Rational(9, 20) and abs(float(tot) - 0.45) < 1e-12)
    report('D09-05', ok, f"s_(n+1)=-s_n/2, S_1=27/64, 합={tot_sym} (오답 {wrong})")


# D09-06  y=1/x 위 (t,1/t) 의 접선
def p06():
    tp = symbols('tp', positive=True)
    tang = -x / tp**2 + 2 / tp
    xint = solve(tang, x)
    A = simplify(integrate(1 / x - tang, (x, tp, 2 * tp)))
    ok_pos = all((1 / X - (-X / 4 + 1)) >= 0 for X in [Rational(2) + Rational(k, 50) for k in range(51)])
    V = simplify(pi * integrate(1 / x**2 - tang**2, (x, tp, 2 * tp)))
    k = symbols('k', integer=True, positive=True)
    tot = summation(V.subs(tp, 2**(k - 1)), (k, 1, oo))
    wrong = summation(V.subs(tp, 2**(k - 1)), (k, 1, 10))
    # (4) sum_{k<=n} A > 1 의 최소 n (ln2 의 범위 0.69~0.70 에서도 동일)
    nmin = [min(n for n in range(1, 100) if n * (L - 0.5) > 1) for L in (0.6901, 0.6931471805599453, 0.6999)]
    ok = (nmin == [6, 6, 6] and xint == [2 * tp] and simplify(A - (log(2) - Rational(1, 2))) == 0 and ok_pos
          and simplify(V - pi / (6 * tp)) == 0 and simplify(tot - pi / 3) == 0)
    report('D09-06', ok, f"넓이={A} (t 무관), V={V}, 합={tot}")


# D09-07  Wallis
def p07():
    I = {n: integrate(sin(x)**n, (x, 0, pi / 2)) for n in range(0, 8)}
    ok_rec = all(simplify(I[n] - Rational(n - 1, n) * I[n - 2]) == 0 for n in range(2, 8))
    J = integrate(x * sin(x)**5, (x, 0, pi))
    wrong = pi / 2 * I[5]  # int_0^pi sin^5 = 2 I5 의 2배를 빠뜨림
    ok = (I[0] == pi / 2 and I[1] == 1 and I[2] == pi / 4 and ok_rec and I[4] == 3 * pi / 16 and I[5] == Rational(8, 15)
          and simplify(J - 8 * pi / 15) == 0)
    report('D09-07', ok, f"I4={I[4]}, I5={I[5]}, int x sin^5 = {simplify(J)} (오답 {wrong})")


# D09-08  F(x)=int_0^x e^t (t-1)(t-3) dt, x>=0
def p08():
    F = integrate(exp(t) * (t - 1) * (t - 3), (t, 0, x))
    ok1 = simplify(F - (exp(x) * (x - 3)**2 - 9)) == 0
    crit = solve(diff(F, x), x)
    Fmax, Fmin = simplify(F.subs(x, 1)), simplify(F.subs(x, 3))
    Ff = lambda X: mp.e**X * (X - 3)**2 - 9

    def count(k):
        # x>=0 에서 F(x)=k 의 근 수: 구간별 단조성을 이용한 이분 + 경계
        c = 0
        segs = [(0, 1), (1, 3), (3, 20)]
        for lo, hi in segs:
            flo, fhi = Ff(lo) - k, Ff(hi) - k
            if lo == 0 and abs(flo) < 1e-30:
                c += 1
                continue
            if abs(fhi) < 1e-30:
                c += 1  # 우끝 근 (다음 구간 시작과 중복 방지 위해 구간끝만 셈)
                continue
            if flo * fhi < 0:
                c += 1
        return c
    M = 4 * mp.e - 9
    tests = {-10: 0, -9: 1, -5: 2, 0: 3, 1: 3, M: 2, 2: 1, 50: 1}
    ok3 = all(count(k) == v for k, v in tests.items())
    A = integrate(F + 9, (x, 0, 3))
    ok = (ok1 and sorted(crit) == [1, 3] and simplify(Fmax - (4 * E - 9)) == 0 and Fmin == -9
          and ok3 and simplify(A - (2 * exp(3) - 17)) == 0)
    report('D09-08', ok, f"F=e^x(x-3)^2-9, 극대 4e-9, 극소 -9, 근수 분류 확인, 넓이={simplify(A)}")


# D09-09  S(a)=int_0^{pi/2} |sin x - a cos x| dx
def p09():
    def Snum(A):
        A = mp.mpf(A)
        if A > 0:
            th = mp.atan(A)
            return mp.quad(lambda X: abs(mp.sin(X) - A * mp.cos(X)), [0, th, mp.pi / 2])
        return mp.quad(lambda X: abs(mp.sin(X) - A * mp.cos(X)), [0, mp.pi / 2])
    ap = symbols('ap', positive=True)
    th = symbols('th')
    # a=tan th 로 놓고 기호 계산
    Sth = integrate(sin(x) * 0 + (a * cos(x) - sin(x)), (x, 0, th)) + integrate(sin(x) - a * cos(x), (x, th, pi / 2))
    Sa = simplify(Sth.subs({th: symbols('TH')}))
    # 공식 2sqrt(1+a^2)-1-a 와 수치 비교
    ok2 = all(abs(Snum(A) - (2 * mp.sqrt(1 + A**2) - 1 - A)) < 1e-25 for A in [mp.mpf('0.1'), mp.mpf('0.5'), 1, 2, 7])
    ok_neg = all(abs(Snum(A) - (1 - A)) < 1e-25 for A in [0, mp.mpf('-0.5'), -3])
    claim = 2 * sqrt(1 + ap**2) - 1 - ap
    crit = solve(diff(claim, ap), ap)
    Smin = simplify(claim.subs(ap, crit[0]))
    S1 = simplify(claim.subs(ap, 1))
    roots = solve(claim - 1, ap)
    grid_min = min(Snum(mp.mpf(-3) + k * mp.mpf(8) / 4000) for k in range(4001))
    ok = (ok2 and ok_neg and crit == [sqrt(3) / 3] and simplify(Smin - (sqrt(3) - 1)) == 0
          and simplify(S1 - (2 * sqrt(2) - 2)) == 0 and roots == [Rational(4, 3)]
          and grid_min >= mp.sqrt(3) - 1 - 1e-20 and abs(grid_min - (mp.sqrt(3) - 1)) < 1e-6)
    report('D09-09', ok, f"S(1)=2√2-2, S(a)={claim}, 최소 a=1/√3 값 √3-1, S<1 ⇔ 0<a<4/3")


# D09-10  I(m,n)=int_0^1 x^m (1-x)^n dx
def p10():
    # f(x)=[log_2 x] 의 계단 영역을 직접 구간별 적분으로 계산
    def f_on(k):  # [2^k, 2^{k+1}) 에서 f=k
        return k
    def S(n):
        return sum(integrate(f_on(k), (x, 2**k, 2**(k + 1))) for k in range(n))
    def V(n):
        return pi * sum(integrate(f_on(k)**2, (x, 2**k, 2**(k + 1))) for k in range(n))
    def Lg(n):
        return integrate(log(x) / log(2), (x, 1, 2**n))
    ok1 = S(3) == 10 and simplify(Lg(3) - (24 - 7 / log(2))) == 0
    # (2) 가로 띠 넓이 합과 비교, 닫힌식
    ok2 = all(S(n) == sum(2**n - 2**j for j in range(1, n)) == (n - 2) * 2**n + 2 for n in range(2, 25))
    # (3) T_n = ∫log_2 x - S_n
    ok3 = all(simplify(Lg(n) - S(n) - (2**n - 1) * (2 - 1 / log(2))) == 0 for n in range(1, 12))
    # 수치 교차검산: T_n 을 mpmath 로 구간별 적분
    for n in (3, 6):
        Tn = sum(mp.quad(lambda z: mp.log(z, 2) - k, [2**k, 2**(k + 1)]) for k in range(n))
        ok3 &= abs(Tn - (2**n - 1) * (2 - 1 / mp.log(2))) < mp.mpf(10)**-25
    # (4) V_n, 가로 띠 분해 Σ(2j-1)(2^n-2^j)
    ok4 = all(V(n) == pi * ((n**2 - 4 * n + 6) * 2**n - 6) for n in range(1, 20))
    ok4 &= all(sum(k**2 * 2**k for k in range(n)) == sum((2 * j - 1) * (2**n - 2**j) for j in range(1, n)) for n in range(1, 30))
    wrong = [pi * ((n - 2) * 2**n + 2) for n in (4,)]  # 띠 높이 1로 둔 오답
    ok = ok1 and ok2 and ok3 and ok4 and wrong[0] != V(4)
    report('D09-10', ok, f"S_3={S(3)}, ∫_1^8 log2={Lg(3)}, Σk2^k 두 방법 일치 n<25, T_n=(2^n-1)(2-1/ln2), V_n=π((n²-4n+6)2^n-6) n<20 (V_4={V(4)}, 오답 {wrong[0]})")


if __name__ == '__main__':
    for f in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
        try:
            f()
        except Exception as e:  # noqa
            report(f.__name__.replace('p', 'D09-'), False, f'예외 {e!r}')
    n = sum(RESULTS)
    print(f"SUMMARY T09: {n}/{len(RESULTS)} PASS")
    if n != len(RESULTS) or len(RESULTS) != 10:
        sys.exit(1)
