# -*- coding: utf-8 -*-
"""S08 수열의 극한·급수·정적분과 급수의 관계 — sympy/mpmath 검산.
문제 설정으로부터 limit/summation/integrate 또는 고정밀 수치로 독립 계산하여 md의 정답과 비교한다."""
import sys
from sympy import (Rational as R, symbols, sqrt, limit, oo, summation, integrate, log, exp,
                   simplify, factorial, floor, nsimplify, Abs, solve, diff, E, Piecewise)
import mpmath

RESULTS = []
n, k, x, m = symbols('n k x m', positive=True)


def report(pid, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


# ---------------- S08-01 ----------------
def p01():
    mpmath.mp.dps = 40
    s2 = mpmath.sqrt(2)
    vals = []
    for N in (10 ** 3, 10 ** 4, 10 ** 5):
        S = sum(int(mpmath.floor(j * s2)) for j in range(1, N + 1))
        vals.append(mpmath.mpf(S) / N ** 2)
    target = mpmath.sqrt(2) / 2
    # 오차가 1/N 정도로 줄어드는지
    ok = abs(vals[-1] - target) < 2e-5 and abs(vals[-1] - target) < abs(vals[0] - target)
    # 기호: 상·하계의 극한
    lo = limit((sqrt(2) * n * (n + 1) / 2 - n) / n ** 2, n, oo)
    hi = limit((sqrt(2) * n * (n + 1) / 2) / n ** 2, n, oo)
    ok &= (lo == sqrt(2) / 2) and (hi == sqrt(2) / 2)
    report("S08-01", ok, f"N=1e5 수치 {mpmath.nstr(vals[-1], 8)}, 샌드위치 상·하계 극한 모두 sqrt(2)/2")


# ---------------- S08-02 ----------------
def p02():
    a, b = symbols('a b', real=True)
    # 수렴 조건: (a-2b)=0, 극한 -b^2/2 = -2
    expr = n * (sqrt(n ** 2 + a * n) - n - b)
    sols = []
    for bb in (-3, -2, -1, 0, 1, 2, 3):
        for aa in range(-8, 9):
            L = limit(expr.subs({a: aa, b: bb}), n, oo)
            if L == -2:
                sols.append((aa, bb))
    prods = {aa * bb for aa, bb in sols}
    ok = sorted(sols) == [(-4, -2), (4, 2)] and prods == {8}
    report("S08-02", ok, f"정수 격자 탐색 해 {sorted(sols)}, ab={prods}")


# ---------------- S08-03 ----------------
def p03():
    ok = True
    for mm in range(1, 8):
        term = 1
        for j in range(0, mm + 1):
            term = term * (n + j)
        T = summation(1 / term, (n, 1, oo))
        ok &= simplify(T - R(1, mm * factorial(mm))) == 0
    T3 = summation(1 / (n * (n + 1) * (n + 2) * (n + 3)), (n, 1, oo))
    mmin = min(mm for mm in range(1, 20) if R(1, mm * factorial(mm)) < R(1, 1000))
    ok &= T3 == R(1, 18) and mmin == 6
    report("S08-03", ok, f"T_m=1/(m*m!) (m=1..7) summation 일치, T_3={T3}, 최소 m={mmin}")


# ---------------- S08-04 ----------------
def p04():
    def f(xv):
        xv = R(xv)
        seq = []
        for N in (400, 401, 800, 801):
            den = xv ** N + 3 ** N
            if den == 0:
                return None
            seq.append((xv ** (N + 1) + 2 * 3 ** N) / den)
        if max(abs(float(s - seq[0])) for s in seq) > 1e-9:
            return None  # 발산(진동)
        return seq[-1]
    ok = f(-3) is None
    sol = []
    grid = [R(j, 4) for j in range(-40, 41)]
    for xv in grid:
        fv = f(xv)
        if fv is not None and abs(float(fv - xv)) < 1e-9:
            sol.append(xv)
    expected = [xv for xv in grid if xv == 2 or abs(xv) > 3]
    ok &= sol == expected and abs(float(f(3)) - 2.5) < 1e-12
    report("S08-04", ok, f"x=-3 미정의, f(3)=5/2, 격자(0.25 간격)에서 f(x)=x ⇔ x=2 또는 |x|>3")


# ---------------- S08-05 ----------------
def p05():
    # C=(0,0), A=(4,0), B=(0,3). 빗변 AB 위에 한 변을 둔 정사각형
    s, t = symbols('s t', positive=True)
    h = R(12, 5)  # C에서 AB까지 거리
    # 정사각형 윗변은 AB와 평행, C로부터 거리 h-s, 길이 = 5*(h-s)/h = s
    s1 = solve(sp_eq(5 * (h - s) / h, s), s)[0]
    r = (h - s1) / h
    total = summation(s1 ** 2 * r ** (2 * (n - 1)), (n, 1, oo))
    # 좌표로 직접 확인: 윗변 양 끝이 두 직각변 위에 있는지
    # AB 단위법선(C쪽) 방향
    import sympy as sp
    A = sp.Matrix([4, 0]); B = sp.Matrix([0, 3])
    u = (B - A) / 5
    nrm = sp.Matrix([-3, -4]) / 5  # AB에서 C로 향하는 단위법선
    foot = A + u * ((sp.Matrix([0, 0]) - A).dot(u))
    P = foot + nrm * 0  # 기준
    # 윗변 중심선: AB로부터 s1 거리, 윗변 끝점이 x=0, y=0 직선 위
    line_pt = A + nrm * s1
    # 윗변: line_pt + u*t, x=0 과 y=0 교점 사이 길이
    t1 = sp.solve(line_pt[0] + u[0] * t, t)[0]
    t2 = sp.solve(line_pt[1] + u[1] * t, t)[0]
    ok = sp.simplify(abs(t1 - t2) - s1) == 0
    ok &= s1 == R(60, 37) and r == R(12, 37) and total == R(144, 49)
    wrong = s1 ** 2 / (1 - r)
    report("S08-05", ok, f"s1={s1}, 닮음비 {r}, 넓이 합={total} (좌표 검증), 오답(비 r 사용)={wrong}")


def sp_eq(a, b):
    import sympy as sp
    return sp.Eq(a, b)


# ---------------- S08-06 ----------------
def p06():
    I = integrate(log(1 + x), (x, 0, 2))
    L = simplify(exp(I))
    mpmath.mp.dps = 30
    N = 20000
    lg = sum(mpmath.log(1 + mpmath.mpf(j) / N) for j in range(1, 2 * N + 1)) / N
    num = mpmath.e ** lg
    ok = simplify(L - 27 / E ** 2) == 0 and abs(num - 27 / mpmath.e ** 2) < 1e-3
    report("S08-06", ok, f"∫_0^2 ln(1+x)dx={simplify(I)}, 극한 {L}, n=20000 수치 {mpmath.nstr(num, 8)}")


# ---------------- S08-07 ----------------
def p07():
    # y=x^2, P_k=(3k/n, 9k^2/n^2), 삼각형 O P_k P_{k+1} (k=1..n-1) 넓이 합
    area_k = R(1, 2) * ((3 * k / n) * (9 * (k + 1) ** 2 / n ** 2) - (3 * (k + 1) / n) * (9 * k ** 2 / n ** 2))
    S = summation(simplify(area_k), (k, 1, n - 1))
    L = limit(S, n, oo)
    I = integrate(3 * x - x ** 2, (x, 0, 3))
    ok = L == R(9, 2) and I == R(9, 2)
    wrong = integrate(x ** 2, (x, 0, 3))
    report("S08-07", ok, f"S_n={simplify(S)}, 극한 {L} = ∫(3x-x^2)={I}, 오답(곡선 아래 넓이)={wrong}")


# ---------------- S08-08 ----------------
def p08():
    xs = symbols('xs', positive=True)
    r = 1 - log(xs)
    # 수렴 범위: |r|<1 ⇔ 1<x<e^2 (수치 확인)
    import math
    conv_ok = all((abs(1 - math.log(v)) < 1) == (1 < v < math.e ** 2) for v in [0.05 * j for j in range(1, 200)])
    S = simplify(xs / (1 - r))
    crit = solve(diff(S, xs), xs)
    mval = S.subs(xs, crit[0])
    # 끝점 근방에서 S가 크다는 것 확인
    edge = [float(S.subs(xs, v)) for v in (1.0001, 7.389)]
    ok = conv_ok and S == xs / log(xs) and crit == [E] and mval == E and min(edge) > float(E)
    # 직접 합 수치
    v = mpmath.e
    direct = mpmath.nsum(lambda j: v * (1 - mpmath.log(v)) ** j, [0, mpmath.inf])
    ok &= abs(direct - mpmath.e) < 1e-20
    report("S08-08", ok, f"S(x)={S}, 수렴 1<x<e^2, 임계점 x={crit}, 최솟값 {mval}")


# ---------------- S08-09 ----------------
def p09():
    T = summation(n * (n + 1) / 2 ** n, (n, 1, oo))
    N = symbols('N', positive=True, integer=True)
    SN = summation(k * (k + 1) / 2 ** k, (k, 1, N))
    closed_ok = simplify(SN - (8 - (N ** 2 + 5 * N + 8) / 2 ** N)) == 0
    # [증명-수치확인] 2^n > n^3 (10<=n<=300), 망원 항등식
    ind_ok = all(2 ** j > j ** 3 for j in range(10, 301)) and not (2 ** 9 > 9 ** 3)
    g = lambda j: R(j * j + 5 * j + 8, 2 ** j)
    tel_ok = all(g(j - 1) - g(j) == R(j * (j + 1), 2 ** j) for j in range(1, 200))
    ok = T == 8 and closed_ok and ind_ok and tel_ok
    report("S08-09", ok, f"[증명-수치확인] Σ=8, S_N=8-(N^2+5N+8)/2^N, 2^n>n^3 (n>=10), 망원 항등식 n<=199")


# ---------------- S08-10 ----------------
def p10():
    mpmath.mp.dps = 30
    S = mpmath.fsum(mpmath.mpf(j) ** (-mpmath.mpf(1) / 3) for j in range(1, 1001))
    I = integrate(x ** R(-1, 3), (x, 1, 1000))
    lower = I + R(1, 2) * (1 + R(1, 10))
    upper = 1 + I
    ok = I == R(297, 2) and lower == R(14905, 100) and upper == R(299, 2)
    ok &= lower < S < upper and int(mpmath.floor(S)) == 149
    # [증명-수치확인] 볼록성: 각 구간에서 ∫ <= 사다리꼴, f(k) <= ∫_{k-1}^k
    ok &= all(mpmath.quad(lambda t: t ** (-mpmath.mpf(1) / 3), [j, j + 1]) <=
              (mpmath.mpf(j) ** (-mpmath.mpf(1) / 3) + mpmath.mpf(j + 1) ** (-mpmath.mpf(1) / 3)) / 2
              for j in range(1, 1000, 37))
    report("S08-10", ok, f"[증명-수치확인] Σ={mpmath.nstr(S, 10)}, 하계 {float(lower)}, 상계 {float(upper)} -> 149")


if __name__ == "__main__":
    for f in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
        try:
            f()
        except Exception as e:  # noqa
            report(f.__name__, False, f"예외 {e!r}")
    npass = sum(RESULTS)
    print(f"SUMMARY T08: {npass}/{len(RESULTS)} PASS")
    if npass != len(RESULTS):
        sys.exit(1)
