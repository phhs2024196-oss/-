"""verify_T08.py — 동국 D08 (급수와 수열의 극한) 검산.
도형은 좌표/재귀 구성으로 직접 만들고, 극한은 sympy limit·integrate 또는 mpmath 고정밀 수치로 독립 계산한다."""
import sys
from fractions import Fraction as F
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
results = []
n, k, x = sp.symbols('n k x', positive=True)


def report(tag, ok, msg):
    results.append(bool(ok))
    print(f"{tag} {'PASS' if ok else 'FAIL'}  {msg}")


# ---------------------------------------------------------------- D08-01
def p01():
    # 정삼각형(한 변 2)을 좌표로 두고 실제로 중점 삼각형을 칠하는 재귀 구성
    s3 = sp.sqrt(3)
    def area(t):
        (a, b), (c, d), (e, f) = t
        return sp.Abs((c - a) * (f - b) - (e - a) * (d - b)) / 2
    def per(t):
        P = [sp.Matrix(p) for p in t]
        return sum(sp.sqrt(((P[i] - P[(i + 1) % 3]).T * (P[i] - P[(i + 1) % 3]))[0]) for i in range(3))
    T = [((0, 0), (2, 0), (1, s3))]
    S, L, Ss, Ls = 0, 0, [], []
    for step in range(1, 6):
        newT = []
        for t in T:
            A, B, C = [sp.Matrix(p) for p in t]
            mAB, mBC, mCA = (A + B) / 2, (B + C) / 2, (C + A) / 2
            mid = (tuple(mAB), tuple(mBC), tuple(mCA))
            S += area(mid); L += per(mid)
            newT += [(tuple(A), tuple(mAB), tuple(mCA)), (tuple(mAB), tuple(B), tuple(mBC)), (tuple(mCA), tuple(mBC), tuple(C))]
        T = newT
        Ss.append(sp.nsimplify(sp.simplify(S))); Ls.append(sp.simplify(L))
    ok = Ss[0] == s3 / 4 and sp.simplify(Ss[1] - 7 * s3 / 16) == 0
    for i, st in enumerate(range(1, 6)):
        ok &= sp.simplify(Ss[i] - s3 * (1 - sp.Rational(3, 4) ** st)) == 0
        ok &= sp.simplify(Ls[i] - 6 * (sp.Rational(3, 2) ** st - 1)) == 0
    ok &= sp.limit(s3 * (1 - sp.Rational(3, 4) ** n), n, sp.oo) == s3
    n1 = next(m for m in range(1, 50) if F(3, 4) ** m < F(1, 10))
    n2 = next(m for m in range(1, 50) if 6 * (F(3, 2) ** m - 1) > 100)
    ok &= (n1, n2) == (9, 8)
    report("D08-01", ok, f"S1=√3/4,S2=7√3/16, S_n=√3(1-(3/4)^n), L_n=6((3/2)^n-1) (좌표 재귀 5단계), 최소 n={n1}, {n2}")


# ---------------------------------------------------------------- D08-02
def p02():
    a = 1 / (k * (k + 1) * (k + 2))
    ok = sp.simplify(a - (1 / (k * (k + 1)) - 1 / ((k + 1) * (k + 2))) / 2) == 0
    b = sp.summation(a, (k, 1, n))
    ok &= sp.simplify(b - (sp.Rational(1, 4) - 1 / (2 * (n + 1) * (n + 2)))) == 0
    ok &= sp.limit(b, n, sp.oo) == sp.Rational(1, 4)
    s3 = sp.summation((k + 3) / (k * (k + 1) * (k + 2)), (k, 1, sp.oo))
    l3 = sp.limit(n ** 2 * (sp.Rational(1, 4) - b), n, sp.oo)
    ok &= sp.simplify(s3 - sp.Rational(5, 4)) == 0 and l3 == sp.Rational(1, 2)
    report("D08-02", ok, f"b_n=1/4-1/(2(n+1)(n+2)), Σ(k+3)/(k(k+1)(k+2))={s3}, lim n^2(1/4-b_n)={l3}")


# ---------------------------------------------------------------- D08-03
def p03():
    P = [sp.Matrix([0, 0]), sp.Matrix([2, 0])]
    v = sp.Matrix([2, 0])
    R = sp.Matrix([[0, -1], [1, 0]])
    for _ in range(80):
        v = R * v / 2
        P.append(P[-1] + v)
    ok = [tuple(p) for p in P[2:5]] == [(2, 1), (sp.Rational(3, 2), 1), (sp.Rational(3, 2), sp.Rational(3, 4))]
    Q = sp.Matrix([sp.Rational(8, 5), sp.Rational(4, 5)])
    ok &= max(abs(P[-1][i] - Q[i]) for i in range(2)) < sp.Rational(1, 10 ** 20)
    # |P_n Q| = OQ (1/2)^n 정확히 확인
    OQ = sp.sqrt(Q.dot(Q))
    for m in range(0, 20):
        d = sp.sqrt((Q - P[m]).dot(Q - P[m]))
        ok &= sp.simplify(d - OQ * sp.Rational(1, 2) ** m) == 0
    tot = sum(sp.N(sp.sqrt((Q - P[m]).dot(Q - P[m])), 30) for m in range(0, 80))
    ok &= abs(tot - sp.N(8 * sp.sqrt(5) / 5, 30)) < 1e-20 and sp.simplify(OQ - 4 * sp.sqrt(5) / 5) == 0
    report("D08-03", ok, "P2..P4 일치, Q=(8/5,4/5), |P_nQ|=(4√5/5)(1/2)^n (n<20 정확), Σ|P_nQ|=8√5/5 (80항 수치)")


# ---------------------------------------------------------------- D08-04
def p04():
    a, b = sp.symbols('a b', real=True)
    def f_num(xv, av, bv, N=4000):
        xv = mp.mpf(xv)
        return (xv ** (2 * N + 1) + av * xv ** 2 + bv) / (xv ** (2 * N) + 1)
    ok = True
    # 구간별 극한식 확인 (a=3/2,b=-1/2)
    av, bv = mp.mpf(3) / 2, mp.mpf(-1) / 2
    for xv in ['-1.7', '-0.4', '0.3', '0.9', '1.3', '2']:
        xx = mp.mpf(xv)
        expect = xx if abs(xx) > 1 else av * xx ** 2 + bv
        ok &= abs(f_num(xv, av, bv) - expect) < 1e-12
    ok &= abs(f_num(1, av, bv) - (1 + av + bv) / 2) < 1e-30 and abs(f_num(-1, av, bv) - (-1 + av + bv) / 2) < 1e-30
    # 연속 조건: x=1 에서 a+b=1, x=-1 에서 a+b=-1 → 동시 불가
    c1 = sp.solve([sp.Eq(a + b, 1), sp.Eq((1 + a + b) / 2, 1)], [a, b], dict=True)
    c2 = sp.solve([sp.Eq(a + b, 1), sp.Eq(a + b, -1)], [a, b], dict=True)
    ok &= c2 == [] and len(c1) == 1 and sp.simplify(list(c1[0].values())[0] + list(c1[0].keys())[0]) is not None
    # a+b=1 에서 ∫_{-2}^{2} f = 0
    expr = sp.integrate(x, (x, -2, -1)) + sp.integrate(a * x ** 2 + (1 - a), (x, -1, 1)) + sp.integrate(x, (x, 1, 2))
    sol = sp.solve(sp.Eq(expr, 0), a)
    ok &= sol == [sp.Rational(3, 2)]
    # 수치 적분으로도 확인 (큰 N 근사함수)
    num = mp.quad(lambda t: f_num(t, av, bv, N=400), [-2, -1, 1, 2])
    ok &= abs(num) < 1e-3
    report("D08-04", ok, f"구간별 극한식 확인, 두 점 동시 연속 불가, ∫=2-4a/3 → a={sol[0]}, b=-1/2 (수치적분 {mp.nstr(num, 3)})")


# ---------------------------------------------------------------- D08-05
def p05():
    t = sp.symbols('t', positive=True)
    y = x ** 3
    tang = sp.diff(y, x).subs(x, t) * (x - t) + t ** 3
    root = sp.solve(sp.Eq(tang, 0), x)[0]
    ok = sp.simplify(root - 2 * t / 3) == 0
    seq = [sp.Integer(1)]
    tri = []
    for _ in range(60):
        an = seq[-1]
        an1 = root.subs(t, an)
        tri.append(sp.Rational(1, 2) * (an - an1) * an ** 3)
        seq.append(an1)
    ok &= seq[1] == sp.Rational(2, 3) and seq[2] == sp.Rational(4, 9)
    ok &= all(sp.simplify(tri[i] - sp.Rational(1, 6) * sp.Rational(16, 81) ** i) == 0 for i in range(10))
    total = sp.summation(sp.Rational(1, 6) * sp.Rational(16, 81) ** k, (k, 0, sp.oo))
    ok &= total == sp.Rational(27, 130) and abs(sum(tri) - total) < sp.Rational(1, 10 ** 30)
    # 삼각형이 곡선 아래에 있음: 접선 <= x^3 (x>0) 를 격자에서 확인
    ok &= all(float(tang.subs({t: tv, x: xv})) <= xv ** 3 + 1e-15 for tv in [0.1 * i for i in range(1, 11)] for xv in [0.01 * j for j in range(0, 101)])
    rest = sp.integrate(y, (x, 0, 1)) - total
    ok &= rest == sp.Rational(11, 260)
    report("D08-05", ok, f"a_(n+1)=2a_n/3, T_n=(1/6)(16/81)^(n-1), ΣT_n={total}, 남은 넓이={rest}, 접선<=곡선 [증명-수치확인]")


# ---------------------------------------------------------------- D08-06
def p06():
    U2 = sp.summation((k / n) ** 2, (k, 1, n)) / n
    L2 = sp.summation((k / n) ** 2, (k, 0, n - 1)) / n
    ok = sp.simplify(U2 - (n + 1) * (2 * n + 1) / (6 * n ** 2)) == 0 and sp.simplify(L2 - (n - 1) * (2 * n - 1) / (6 * n ** 2)) == 0
    # n(U-L)=f(1)-f(0) 를 여러 증가함수에서 확인
    for f in (x ** 2, x ** 3, sp.exp(x), sp.sqrt(x + 1)):
        for m in (3, 7, 20):
            U = sum(f.subs(x, sp.Rational(j, m)) for j in range(1, m + 1)) / m
            L = sum(f.subs(x, sp.Rational(j, m)) for j in range(0, m)) / m
            ok &= sp.simplify(m * (U - L) - (f.subs(x, 1) - f.subs(x, 0))) == 0
    ok &= sp.limit(U2, n, sp.oo) == sp.Rational(1, 3) == sp.limit(L2, n, sp.oo)
    U3 = sp.summation((k / n) ** 3, (k, 1, n)) / n
    L3 = sp.summation((k / n) ** 3, (k, 0, n - 1)) / n
    l1 = sp.limit(n * (U3 - sp.Rational(1, 4)), n, sp.oo)
    l2 = sp.limit(n * (sp.Rational(1, 4) - L3), n, sp.oo)
    ok &= l1 == sp.Rational(1, 2) and l2 == sp.Rational(1, 2)
    report("D08-06", ok, f"U_n=(n+1)(2n+1)/(6n^2), L_n=(n-1)(2n-1)/(6n^2), n(U-L)=f(1)-f(0) (4개 함수), x^3: 극한 {l1}, {l2}")


# ---------------------------------------------------------------- D08-07
def p07():
    # a_1=6, a_{n+1}=(a_n^2+4)/(2a_n) 를 정확 유리수로 직접 반복
    a = [None, sp.Integer(6)]
    for j in range(1, 9):
        a.append((a[j] ** 2 + 4) / (2 * a[j]))
    ok = all(a[j] > 0 for j in range(1, 10))
    b = [None] + [(a[j] - 2) / (a[j] + 2) for j in range(1, 10)]
    ok &= all(sp.simplify(b[j + 1] - b[j] ** 2) == 0 for j in range(1, 9))
    ok &= all(b[j] == sp.Rational(1, 2 ** (2 ** (j - 1))) for j in range(1, 10))
    ok &= all(a[j] == 2 * (2 ** (2 ** (j - 1)) + 1) / sp.Integer(2 ** (2 ** (j - 1)) - 1) for j in range(1, 10))
    # (2) 극한: 일반항으로 sympy limit
    N_ = sp.Symbol('N', positive=True)
    B = 2 ** (-(2 ** (N_ - 1)))
    A = 2 * (1 + B) / (1 - B)
    lim_a = sp.limit(A, N_, sp.oo)
    ratio = [(a[j + 1] - 2) / (a[j] - 2) ** 2 for j in range(1, 9)]
    ok &= lim_a == 2 and all(sp.simplify(ratio[j - 1] - 1 / (2 * a[j])) == 0 for j in range(1, 9))
    ok &= 0 < sp.Rational(1, 4) - ratio[-1] < sp.Rational(1, 10 ** 30)
    # (3) log2(a_n-2)/2^n -> -1/2 (고정밀), 최소 n
    lg = [sp.N(sp.log(a[j] - 2, 2) / 2 ** j, 50) for j in (6, 7, 8, 9)]
    ok &= abs(lg[-1] + sp.Rational(1, 2)) < sp.Rational(1, 2 ** 7) and abs(lg[-1] + sp.Rational(1, 2)) < abs(lg[0] + sp.Rational(1, 2))
    nmin = next(j for j in range(1, 10) if a[j] - 2 < sp.Rational(1, 10 ** 10))
    ok &= nmin == 7 and all(a[j] - 2 >= sp.Rational(1, 10 ** 10) for j in range(1, 7))
    # 대표 오답: b_n=2^{-2^n} 로 밀면 극한 -1, 최소 n=6
    Aw = [2 * (1 + sp.Rational(1, 2 ** (2 ** j))) / (1 - sp.Rational(1, 2 ** (2 ** j))) for j in range(1, 9)]
    wn = next(j for j in range(1, 9) if Aw[j - 1] - 2 < sp.Rational(1, 10 ** 10))
    report("D08-07", ok and wn == 6, f"b_(n+1)=b_n^2, b_n=2^(-2^(n-1)) (n<=9 정확), lim a_n={lim_a}, 비 1/(2a_n)->1/4, log2(a_9-2)/2^9={sp.N(lg[-1],8)}->-1/2, 최소 n={nmin} (오답 {wn})")


# ---------------------------------------------------------------- D08-08
def p08():
    ok = True
    for m in range(2, 15):
        th = [sp.pi * j / m for j in range(1, m + 1)]
        s = sum((sp.cos(t) - 1) ** 2 + sp.sin(t) ** 2 for t in th) / m
        ok &= abs(sp.N(s - 2 - sp.Rational(2, m), 30)) < 1e-25
    th = sp.symbols('theta')
    l2 = sp.integrate(2 * sp.sin(th / 2), (th, 0, sp.pi)) / sp.pi
    l3 = sp.integrate(4 + 2 * (2 * sp.sin(th / 2)) * (2 * sp.cos(th / 2)), (th, 0, sp.pi)) / sp.pi
    ok &= sp.simplify(l2 - 4 / sp.pi) == 0 and sp.simplify(l3 - (4 + 8 / sp.pi)) == 0
    # 실제 좌표로 리만합 (n=20000)
    N = 4000
    s2 = s3 = mp.mpf(0)
    for j in range(1, N + 1):
        t = mp.pi * j / N
        P = (mp.cos(t), mp.sin(t))
        AP = mp.sqrt((P[0] - 1) ** 2 + P[1] ** 2); BP = mp.sqrt((P[0] + 1) ** 2 + P[1] ** 2)
        s2 += AP; s3 += (AP + BP) ** 2
    ok &= abs(s2 / N - 4 / mp.pi) < 1e-3 and abs(s3 / N - (4 + 8 / mp.pi)) < 1e-3
    report("D08-08", ok, f"평균 AP^2=2+2/n (n<15), lim 평균 AP={l2}, lim 평균 (AP+BP)^2={l3} (좌표 리만합 n=4000)")


# ---------------------------------------------------------------- D08-09
def p09():
    l1 = sp.integrate(sp.sqrt(x), (x, 0, 1))
    ok = l1 == sp.Rational(2, 3)
    # [증명-수치확인] 2√(n+1)-2 < S_n <= 2√n - 1  (n<=10^5, numpy 누적합)
    import numpy as np
    m = np.arange(1, 10 ** 5 + 1, dtype=np.float64)
    S = np.cumsum(1 / np.sqrt(m))
    ok &= bool(np.all(2 * np.sqrt(m + 1) - 2 < S)) and bool(np.all(S <= 2 * np.sqrt(m) - 1 + 1e-9))
    ok &= sp.limit((2 * sp.sqrt(n + 1) - 2) / sp.sqrt(n), n, sp.oo) == 2 == sp.limit((2 * sp.sqrt(n) - 1) / sp.sqrt(n), n, sp.oo)
    # (3) 직접 합 (n=10^7)
    big = 10 ** 7
    j = np.arange(1, 4 * big + 1, dtype=np.float64)
    inv = 1 / np.sqrt(j)
    Sb, S4b = np.sum(inv[:big]), np.sum(inv)
    Tb = np.sum(np.sqrt(j[:big]))
    v1 = (S4b - Sb) / np.sqrt(big)
    v2 = Tb * Sb / big ** 2
    v0 = Tb / big ** 1.5
    ok &= abs(v1 - 2) < 1e-3 and abs(v2 - 4 / 3) < 2e-3 and abs(v0 - 2 / 3) < 1e-3
    report("D08-09", ok, f"lim Σ√k/n^1.5={l1}, 적분 비교 부등식 n<=10^5 [증명-수치확인], (S_4n-S_n)/√n→{v1:.6f}, 곱→{v2:.6f} (n=10^7 직접합)")


# ---------------------------------------------------------------- D08-10
def p10():
    ok = True
    for xv in (2, 4):
        lhs = (x ** (k + 1) / sp.factorial(k + 1) - x ** (k + 2) / sp.factorial(k + 2)).subs(x, xv)
        rhs = (xv ** (k + 1) * (k + 2 - xv) / sp.factorial(k + 2))
        ok &= all(sp.simplify((lhs - rhs).subs(k, kv)) == 0 for kv in range(1, 15))
    # lim 4^n/n! = 0 의 비교 부등식: n>=8 에서 4^n/n! <= (4^8/8!)(1/2)^(n-8)
    ok &= all(F(4 ** m, sp.factorial(m)) <= F(4 ** 8, sp.factorial(8)) * F(1, 2) ** (m - 8) for m in range(8, 80))
    s1 = mp.nsum(lambda kk: kk * 2 ** kk / mp.factorial(kk + 2), [1, mp.inf])
    s2 = mp.nsum(lambda kk: (kk - 2) * 4 ** kk / mp.factorial(kk + 2), [1, mp.inf])
    ok &= abs(s1 - 1) < 1e-30 and abs(s2 - 2) < 1e-30
    # 정확한 부분합(망원식)과 비교
    for m in (5, 10, 30):
        ps = sum(F(kk * 2 ** kk, sp.factorial(kk + 2)) for kk in range(1, m + 1))
        ok &= ps == 1 - F(2 ** (m + 1), sp.factorial(m + 2))
        ps2 = sum(F((kk - 2) * 4 ** kk, sp.factorial(kk + 2)) for kk in range(1, m + 1))
        ok &= ps2 == 2 - F(4 ** (m + 1), sp.factorial(m + 2))
    report("D08-10", ok, f"망원 항등식(x=2,4), 4^n/n! 비교 부등식 n<80, Σk2^k/(k+2)!={mp.nstr(s1,15)}, Σ(k-2)4^k/(k+2)!={mp.nstr(s2,15)}")


for f in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
    try:
        f()
    except Exception as ex:  # noqa
        report(f.__name__, False, f"예외 {ex!r}")

npass = sum(results)
print(f"SUMMARY T08: {npass}/10 PASS")
if npass != 10:
    sys.exit(1)
