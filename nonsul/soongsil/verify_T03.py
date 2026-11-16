#!/usr/bin/env python3
"""S03. 접선·도형과 극한 — 검산 스크립트 (숭실 T03)
각 문항의 설정에서 출발해 sympy/mpmath 로 정답을 독립적으로 다시 계산한다."""
import sys
import sympy as sp
import mpmath as mp

mp.mp.dps = 50
results = []


def report(pid, ok, msg):
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def bisect(g, lo, hi, it=400):
    for _ in range(it):
        mid = (lo + hi)/2
        if g(lo)*g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi)/2


def tri_area(P, Q, R):
    return sp.Abs((Q[0]-P[0])*(R[1]-P[1]) - (R[0]-P[0])*(Q[1]-P[1]))/2


def dist_pt_line(pt, a, b, c):
    """점 pt 와 직선 ax+by+c=0 사이 거리"""
    return sp.Abs(a*pt[0] + b*pt[1] + c)/sp.sqrt(a**2 + b**2)


# ---------------------------------------------------------------- S03-01
def p01():
    th = sp.symbols('theta', positive=True)
    P = (sp.cos(th), sp.sin(th))
    # 접선 x cos + y sin = 1
    A = (1/sp.cos(th), 0)
    B = (sp.Integer(1), sp.Integer(0))
    n = sp.Matrix([sp.cos(th), sp.sin(th)])
    # B 에서 접선에 내린 수선의 발 H = B + (1 - B·n) n
    s = 1 - (B[0]*n[0] + B[1]*n[1])
    H = (B[0] + s*n[0], B[1] + s*n[1])
    S = sp.simplify(sp.Abs(sp.simplify((A[0]-B[0])*(H[1]-B[1]) - (H[0]-B[0])*(A[1]-B[1])))/2)
    expr = ((1/sp.cos(th) - 1)*(H[1]) )/2  # 밑변 BA(x축 위), 높이 = H 의 y좌표
    # 두 방식 일치 확인(수치)
    ok0 = abs(sp.N((S - expr).subs(th, sp.Rational(1, 7)))) < 1e-30
    L = sp.limit(expr/th**5, th, 0, '+')
    # 대표 오답: BH 를 1-cos 대신 1/cos-1 로 착각 → 직각삼각형을 못 보고 넓이 = BA*BH/2 로 계산
    wrong = sp.limit((1/sp.cos(th)-1)*(1-sp.cos(th))/2/th**5, th, 0, '+')
    report('S03-01', ok0 and L == sp.Rational(1, 8),
           f'lim S/θ^5 = {L} (=1/8); 오답(직각을 B로 보고 BA·BH/2)/θ^5 → {wrong}')


# ---------------------------------------------------------------- S03-02
def p02():
    t, x = sp.symbols('t x', positive=True)
    xs = sp.solve(sp.Eq(x**2, x + t), x)
    xs = sp.solve(x**2 - x - t, sp.Symbol('X'))  # dummy
    X = sp.Symbol('X', real=True)
    r = sp.solve(X**2 - X - t, X)
    x1, x2 = sorted(r, key=lambda e: sp.N(e.subs(t, 1)))
    A = (x1, x1 + t); B = (x2, x2 + t)
    S = sp.simplify(sp.Abs(A[0]*B[1] - B[0]*A[1])/2)
    R = sp.integrate((X + t) - X**2, (X, x1, x2))
    L = sp.limit(sp.simplify(S/R), t, sp.oo)
    wrong = sp.limit(sp.simplify(S/sp.integrate((X+t) - X**2, (X, 0, x2))), t, sp.oo)
    report('S03-02', L == sp.Rational(3, 4), f'lim S/R = {L} (=3/4); 오답(넓이 구간 0~x2) → {wrong}')


# ---------------------------------------------------------------- S03-03
def p03():
    t = sp.symbols('t', positive=True)
    a = sp.symbols('a', real=True)
    X = sp.Symbol('X')
    # 접선: y = 3t^2 X - 2t^3, 곡선과의 교점
    roots = sp.roots(sp.Poly(X**3 - (3*t**2*X - 2*t**3), X))
    xq = [r for r in roots if sp.simplify(r - t) != 0][0]
    P = (t, t**3); Q = (xq, xq**3); A = (a, 0)
    S = sp.Abs(sp.expand((P[0]-A[0])*(Q[1]-A[1]) - (Q[0]-A[0])*(P[1]-A[1])))/2
    ok = True
    for av in [sp.Integer(2), sp.Rational(-3, 5), sp.Integer(7)]:
        L = sp.limit(S.subs(a, av)/t**3, t, 0, '+')
        ok &= (L == sp.Rational(9, 2)*abs(av))
        ok &= sp.limit(S.subs(a, av)/t**2, t, 0, '+') == 0
        ok &= sp.limit(S.subs(a, av)/t**4, t, 0, '+') == sp.oo
    L0 = sp.limit(S.subs(a, 0)/t**4, t, 0, '+')
    ok &= (L0 == 3) and sp.limit(S.subs(a, 0)/t**3, t, 0, '+') == 0
    report('S03-03', bool(ok) and xq == -2*t,
           f'x_Q={xq}; a≠0: n=3, 극한 9|a|/2 (a=2,-3/5,7 확인); a=0: n=4, 극한 {L0}')


# ---------------------------------------------------------------- S03-04
def p04():
    t = sp.symbols('t', positive=True)
    u = sp.symbols('u', nonnegative=True)   # u = x^2
    D2 = u + (u/2 - t)**2                    # 거리의 제곱
    # t > 1: 임계점 u = 2t-2 > 0
    uc = sp.solve(sp.diff(D2, u), u)[0]
    r2 = sp.simplify(D2.subs(u, uc))
    ok = sp.simplify(uc - (2*t - 2)) == 0 and sp.simplify(r2 - (2*t - 1)) == 0
    # t<=1 이면 최소 u=0, r=t (수치)
    for tv in [0.3, 0.8, 1.0]:
        vals = [float(D2.subs({t: tv, u: k/1000})) for k in range(0, 5000)]
        ok &= abs(min(vals) - tv**2) < 1e-9
    for tv in [3.0, 10.0]:
        vals = [float(D2.subs({t: tv, u: k/200})) for k in range(0, 8000)]
        ok &= abs(min(vals) - (2*tv - 1)) < 1e-3
    L = sp.limit(sp.sqrt(t)*(sp.sqrt(2*t) - sp.sqrt(r2)), t, sp.oo)
    # 대표 오답: u 대신 x 로 미분하면서 x=0 을 놓치지 않았지만 거리제곱 대신 거리 비교 혼동 → r^2=2t-2
    wrong = sp.limit(sp.sqrt(t)*(sp.sqrt(2*t) - sp.sqrt(2*t - 2)), t, sp.oo)
    report('S03-04', bool(ok) and sp.simplify(L - sp.sqrt(2)/4) == 0,
           f'r(t)^2 = {r2} (t>1), r=t (t≤1); 극한 = {L}; 오답(r^2=2t-2) → {wrong}')


# ---------------------------------------------------------------- S03-05
def p05():
    m = sp.symbols('m', positive=True)
    X = sp.Symbol('X', real=True)
    r = sp.solve(X**2 - m*X - 1, X)
    x1 = [e for e in r if sp.N(e.subs(m, 1)) < 0][0]
    x2 = [e for e in r if sp.N(e.subs(m, 1)) > 0][0]
    A = (0, 1)
    AP = sp.sqrt((x1 - 0)**2 + (m*x1 + 1 - 1)**2)
    AQ = sp.sqrt((x2 - 0)**2 + (m*x2 + 1 - 1)**2)
    expr = AQ - AP - m**2
    L = sp.limit(expr, m, sp.oo)
    Ln = mp.mpf(0)
    mv = mp.mpf(10)**6
    f = lambda mm: (lambda d: (mp.sqrt(1+mm**2)*((mm+d)/2) - mp.sqrt(1+mm**2)*((d-mm)/2) - mm**2))(mp.sqrt(mm**2+4))
    num = f(mv)
    # 대표 오답: AQ-AP 를 |x2-x1|√(1+m²) 로 → 발산
    wrong = sp.limit(sp.sqrt(1+m**2)*sp.sqrt(m**2+4) - m**2, m, sp.oo)
    report('S03-05', L == sp.Rational(1, 2) and abs(num - mp.mpf(1)/2) < 1e-9,
           f'lim(AQ-AP-m^2) = {L}, m=1e6 수치 {mp.nstr(num, 12)}; 오답(AQ+AP 꼴) → {wrong}')


# ---------------------------------------------------------------- S03-06
def p06():
    x = sp.symbols('x', positive=True)
    f = x**2 - 3
    # 접선의 x절편
    g = sp.simplify(x - f/sp.diff(f, x))
    ok = sp.simplify(g - (x**2 + 3)/(2*x)) == 0
    ok &= sp.simplify((g - sp.sqrt(3)) - (x - sp.sqrt(3))**2/(2*x)) == 0
    seq = [mp.mpf(2)]
    for _ in range(6):
        xn = seq[-1]
        seq.append((xn**2 + 3)/(2*xn))
    mp.mp.dps = 400
    seq = [mp.mpf(2)]
    for _ in range(7):
        xn = seq[-1]
        seq.append((xn**2 + 3)/(2*xn))
    s3 = mp.sqrt(3)
    ratios = [(seq[i+1]-s3)/(seq[i]-s3)**2 for i in range(len(seq)-1)]
    mono = all(seq[i] > seq[i+1] > s3 for i in range(len(seq)-1))
    L = sp.limit(1/(2*x), x, sp.sqrt(3))
    ok &= mono and abs(ratios[-1] - s3/6) < mp.mpf(10)**-50 and sp.simplify(L - sp.sqrt(3)/6) == 0
    mp.mp.dps = 50
    report('S03-06', bool(ok), f'x_(n+1)-√3=(x_n-√3)^2/(2x_n), 감소·하계 √3 확인, 비 → √3/6 (n=7 수치 {mp.nstr(ratios[-1], 15)})')


# ---------------------------------------------------------------- S03-07
def p07():
    n = sp.symbols('n', positive=True, integer=True)
    N = sp.symbols('N', positive=True)
    # 외접(변이 원의 접선) 정n각형: 중심각 2π/n, 한 삼각형 넓이 tan(π/n)
    T = N*sp.tan(sp.pi/N)
    I = N*sp.sin(2*sp.pi/N)/2
    L = sp.limit(N**2*(T - I), N, sp.oo)
    ok = sp.simplify(L - sp.pi**3) == 0
    # 정사각형(n=4) 검사: 외접 정사각형 넓이 4, 내접 2
    ok &= sp.simplify(T.subs(N, 4) - 4) == 0 and sp.simplify(I.subs(N, 4) - 2) == 0
    wrong = sp.limit(N**2*(T - N*sp.sin(sp.pi/N)), N, sp.oo)  # 내접 넓이를 n sin(π/n) 로 착각
    report('S03-07', bool(ok), f'lim n^2(T_n-I_n) = {L}; 오답(I_n=n sin(π/n)) → {wrong}')


# ---------------------------------------------------------------- S03-08
def p08():
    t = sp.symbols('t', positive=True)
    k = sp.symbols('k', real=True)
    mm = sp.symbols('mm', real=True)
    # P(0,t) 에서 단위원에 그은 접선 y = m x + t : t/√(1+m^2)=1
    ms = sp.solve(sp.Eq(t**2, 1 + mm**2), mm)
    C = (-t/ms[0], 0); D = (-t/ms[1], 0); P = (0, t)
    S = sp.simplify(tri_area(P, C, D).subs(t, t))
    # 접점: 직선 OP 위 y=1/t
    A = (sp.sqrt(1 - 1/t**2), 1/t); B = (-sp.sqrt(1 - 1/t**2), 1/t)
    T = sp.simplify(tri_area(P, A, B))
    ok = True
    for tv in [sp.Rational(3, 2), sp.Integer(2), sp.Rational(11, 10)]:
        ok &= abs(sp.N(S.subs(t, tv) - tv**2/sp.sqrt(tv**2 - 1))) < 1e-30
        ok &= abs(sp.N(T.subs(t, tv) - (tv**2 - 1)**sp.Rational(3, 2)/tv**2)) < 1e-30
    Sf = t**2/sp.sqrt(t**2 - 1); Tf = (t**2 - 1)**sp.Rational(3, 2)/t**2
    L3 = sp.limit(Sf**3*Tf, t, 1, '+')
    L2 = sp.limit(Sf**2*Tf, t, 1, '+'); L4 = sp.limit(Sf**4*Tf, t, 1, '+')
    ok &= (L3 == 1) and L2 == 0 and L4 == sp.oo
    report('S03-08', bool(ok), f'S={sp.simplify(Sf)}, T={Tf}; k=3 → {L3}, k=2 → {L2}, k=4 → {L4}')


# ---------------------------------------------------------------- S03-09
def p09():
    u = sp.symbols('u', positive=True)
    a = u + sp.exp(2*u)                    # 최근접점 x=u 의 조건 (u-a)+e^{2u}=0
    d = sp.sqrt((a - u)**2 + sp.exp(2*u))
    L = sp.limit((a - d)/sp.log(a), u, sp.oo)
    # 수치: 주어진 a 에 대해 직접 최소화
    ok = (L == sp.Rational(1, 2))
    for av in [mp.mpf(5), mp.mpf(50)]:
        h = lambda x: (x - av)**2 + mp.e**(2*x)
        uu = bisect(lambda x: 2*(x - av) + 2*mp.e**(2*x), mp.mpf(-10), av)
        # 격자 탐색으로 전역 최소 확인
        grid = min(h(mp.mpf(i)/1000) for i in range(-5000, 5000))
        ok &= h(uu) <= grid + mp.mpf(10)**-9
    # 수렴 경향: a = 1e40 에서 (a-d)/ln a
    av = mp.mpf(10)**40
    uu = bisect(lambda x: x + mp.e**(2*x) - av, mp.mpf(0), mp.mpf(100))
    val = (av - mp.sqrt((av - uu)**2 + mp.e**(2*uu)))/mp.log(av)
    report('S03-09', bool(ok), f'lim (a-d(a))/ln a = {L}; a=1e40 수치 {mp.nstr(val, 8)} (1/2 - 1/(2ln a) 꼴로 접근)')


# ---------------------------------------------------------------- S03-10
def p10():
    p = sp.symbols('p', nonzero=True, real=True)
    X = sp.Symbol('X')
    # 법선과 포물선의 교점
    sol = sp.solve(sp.Eq(X**2 - p**2, -(X - p)/(2*p)), X)
    xnext = [s for s in sol if sp.simplify(s - p) != 0][0]
    ok = sp.simplify(xnext - (-p - 1/(2*p))) == 0
    # s_n = x_n^2 점화, 부등식 n <= s_n <= n + sqrt(n)/2 [증명-수치확인]
    s = mp.mpf(1)
    Nmax = 200000
    bound_ok = True
    for n in range(1, Nmax + 1):
        if not (n - 1e-12 <= s <= n + mp.sqrt(n)/2 + 1e-12):
            bound_ok = False
            break
        if n == Nmax:
            sN = s
        s = s + 1 + 1/(4*s)
    # P_nP_{n+1}^2 = (2x+1/(2x))^2 + (1+1/(4x^2))^2, s=x^2
    sv = sp.symbols('s', positive=True)
    len2 = 4*sv + 2 + 1/(4*sv) + (1 + 1/(4*sv))**2
    nn = sp.symbols('n', positive=True)
    Lsym = sp.limit(sp.sqrt(len2.subs(sv, nn + sp.sqrt(nn)/2))/sp.sqrt(nn), nn, sp.oo)
    Lsym2 = sp.limit(sp.sqrt(len2.subs(sv, nn))/sp.sqrt(nn), nn, sp.oo)
    num = mp.sqrt(4*sN + 2 + 1/(4*sN) + (1 + 1/(4*sN))**2)/mp.sqrt(Nmax)
    ok &= bound_ok and Lsym == 2 and Lsym2 == 2 and abs(num - 2) < 1e-4
    report('S03-10', bool(ok), f'x_(n+1)=-x_n-1/(2x_n); n≤s_n≤n+√n/2 (n≤{Nmax}) [증명-수치확인]; 극한 2 (n={Nmax} 수치 {mp.nstr(num, 10)})')


if __name__ == '__main__':
    for f in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
        try:
            f()
        except Exception as e:  # noqa
            report(f.__name__.replace('p', 'S03-'), False, f'예외 {e!r}')
    n_ok = sum(results)
    print(f'SUMMARY T03: {n_ok}/{len(results)} PASS')
    sys.exit(0 if n_ok == len(results) == 10 else 1)
