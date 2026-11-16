#!/usr/bin/env python3
"""J08. 삼각함수와 도형의 최적화·극한 — 검산 스크립트 (세종 J08-01 ~ J08-10)"""
import sys, math
import numpy as np
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
th, x, al, ph, kk = sp.symbols('theta x alpha phi k', positive=True)
RESULTS = []


def report(pid, ok, msg):
    RESULTS.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def grid(a, b, N=1000):
    return np.linspace(a, b, N + 2)[1:-1]


def tri_area(P, Q, R):
    return abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2


# J08-01  A(1,0), 접선 x=1 과 OP 의 교점 T, 삼각형 APT 넓이 f, 역함수 극한
def p01():
    f = sp.tan(th) * (1 - sp.cos(th)) / 2
    ok1 = True
    for t in grid(0.01, 1.5, 300):
        A, P, T = (1, 0), (math.cos(t), math.sin(t)), (1, math.tan(t))
        ok1 &= abs(tri_area(A, P, T) - float(f.subs(th, t))) < 1e-12
    fp = sp.lambdify(th, sp.diff(f, th))
    ok2 = all(fp(t) > 0 for t in grid(1e-3, math.pi / 2 - 1e-3, 1000))  # [증명-수치확인]
    L = sp.limit(th**3 / f, th, 0, '+')
    ff = sp.lambdify(th, f, 'mpmath')
    vals = []
    for kv in (mp.mpf('1e-6'), mp.mpf('1e-9'), mp.mpf('1e-12')):
        g = mp.findroot(lambda u: ff(u) - kv, (4 * kv)**(mp.mpf(1) / 3))
        vals.append(g**3 / kv)
    ok3 = L == 4 and abs(vals[-1] - 4) < 1e-6
    report('J08-01', ok1 and ok2 and ok3,
           f'(1) 좌표 넓이 일치; [증명-수치확인] f\'>0 1000점; (3) lim θ^3/f = {L}, 수치 g(k)^3/k = {float(vals[-1]):.8f}')


# J08-02  반원, PA+PH 최대, 삼각형 PBH 넓이 변화율
def p02():
    ok1 = True
    for t in grid(0.01, 1.55, 200):
        A, B = (-1, 0), (1, 0)
        # ∠PAB = t: P 는 A 에서 각 t 방향으로 2cos t 만큼
        P = (-1 + 2 * math.cos(t) * math.cos(t), 2 * math.cos(t) * math.sin(t))
        ok1 &= abs(P[0]**2 + P[1]**2 - 1) < 1e-12
        PA = math.dist(P, A); PB = math.dist(P, B); PH = P[1]
        ok1 &= abs(PA - 2 * math.cos(t)) < 1e-12 and abs(PB - 2 * math.sin(t)) < 1e-12 and abs(PH - math.sin(2 * t)) < 1e-12
    f = 2 * sp.cos(th) + sp.sin(2 * th)
    crit = sp.solve(sp.diff(f, th), th)
    crit = [c for c in crit if 0 < float(c) < float(sp.pi / 2)]
    fmax = sp.simplify(f.subs(th, crit[0]))
    fl = sp.lambdify(th, f)
    ok2 = len(crit) == 1 and crit[0] == sp.pi / 6 and sp.simplify(fmax - 3 * sp.sqrt(3) / 2) == 0 \
        and all(fl(t) <= float(fmax) + 1e-12 for t in grid(0, math.pi / 2, 1000))  # [증명-수치확인]
    S = sp.Rational(1, 2) * (2 * sp.sin(th)**2) * sp.sin(2 * th)
    dS = sp.simplify(sp.diff(S, th).subs(th, sp.pi / 6))
    bad = sp.simplify(sp.diff(sp.sin(2 * th) * sp.cos(th)**2, th).subs(th, sp.pi / 6))
    report('J08-02', ok1 and ok2 and dS == 1,
           f'(1) 좌표 확인; (2) 임계점 {crit}, 최댓값 {fmax}; (3) dS/dθ(π/6) = {dS} (오답: 삼각형 PAH 로 계산 시 {bad})')


# J08-03  렌즈 넓이, dA/dx = -sqrt(4-x^2), ∫_0^2 A dx = 8/3
def p03():
    A_th = 2 * th - sp.sin(2 * th)
    # 수치: 두 단위원 겹친 넓이 공식과 비교
    def lens(d):
        return 2 * math.acos(d / 2) - d / 2 * math.sqrt(4 - d * d)
    ok1 = all(abs(lens(2 * math.cos(t)) - float(A_th.subs(th, t))) < 1e-12 for t in grid(0.01, 1.56, 200))
    dAdx = sp.simplify(sp.diff(A_th, th) / sp.diff(2 * sp.cos(th), th))
    ok2 = all(abs(float(dAdx.subs(th, t)) + math.sqrt(4 - (2 * math.cos(t))**2)) < 1e-12 for t in grid(0.01, 1.56, 1000))  # [증명-수치확인]
    integ = sp.integrate(x * sp.sqrt(4 - x**2), (x, 0, 2))
    num = mp.quad(lambda d: 2 * mp.acos(d / 2) - d / 2 * mp.sqrt(4 - d * d), [0, 2])
    ok3 = integ == sp.Rational(8, 3) and abs(num - mp.mpf(8) / 3) < 1e-20
    report('J08-03', ok1 and ok2 and ok3,
           f'(1) 렌즈 공식 일치; [증명-수치확인] dA/dx=-√(4-x^2) 1000점; (3) 부분적분 {integ}, 직접 수치적분 {float(num):.12f}')


# J08-04  각 안의 원열: 공비, 총넓이, lim θS = π/4
def p04():
    s = sp.sin(th)
    q = (1 - s) / (1 + s)
    ok1 = True
    for t in grid(0.05, 1.5, 100):
        r1 = 1.0
        d1 = r1 / math.sin(t)
        # 다음 원: 중심거리 d2 = r2/sin t, d1 - d2 = r1 + r2 를 수치로 풀기
        r2 = mp.findroot(lambda r: d1 - r / mp.sin(t) - (r1 + r), 0.5)
        ok1 &= abs(float(r2) - float(q.subs(th, t))) < 1e-12
    S = sp.pi / (1 - q**2)
    ok2 = sp.simplify(S - sp.pi * (1 + s)**2 / (4 * s)) == 0
    L = sp.limit(th * S, th, 0, '+')
    report('J08-04', ok1 and ok2 and L == sp.pi / 4,
           f'(1) 접원 조건 수치해 = 공비 (1-sinθ)/(1+sinθ); (2) S=π(1+sinθ)^2/(4sinθ); (3) lim θS = {L}')


# J08-05  종이접기 a=4: p=2/sin^2θ, 접는 선 최소 3√3, dT/dθ = -9/2
def p05():
    ok1 = True
    for t in grid(math.pi / 4 + 1e-3, math.pi / 2 - 1e-3, 200):
        p = 2 / math.sin(t)**2
        q = p * math.tan(t)
        # O(0,0) 을 직선 PQ 에 대해 대칭이동
        nx, ny = q, p  # 직선 x/p + y/q = 1 의 법선 (1/p,1/q) ∝ (q,p)
        c = p * q
        dd = (0 * nx + 0 * ny - c) / (nx * nx + ny * ny)
        Ox, Oy = -2 * dd * nx, -2 * dd * ny
        ok1 &= abs(Ox - 4) < 1e-9 and Oy > 0 and p <= 4 + 1e-12
    L = 2 / (sp.sin(th)**2 * sp.cos(th))
    crit = [c for c in sp.solve(sp.diff(L, th), th) if 0 < float(c) < float(sp.pi / 2)]
    t0 = crit[0]
    Lmin = sp.nsimplify(sp.simplify(L.subs(th, t0)))
    Ll = sp.lambdify(th, L)
    ok2 = len(crit) == 1 and abs(sp.N(sp.tan(t0) - sp.sqrt(2), 50)) < 1e-40 and float(t0) > math.pi / 4 \
        and abs(sp.N(L.subs(th, t0) - 3 * sp.sqrt(3), 50)) < 1e-40 \
        and all(Ll(t) >= float(Lmin) - 1e-12 for t in grid(math.pi / 4, math.pi / 2, 1000))  # [증명-수치확인]
    T = sp.Rational(1, 2) * (2 / sp.sin(th)**2)**2 * sp.tan(th)
    dT = sp.nsimplify(sp.simplify(sp.diff(T, th).subs(th, t0)))
    report('J08-05', ok1 and ok2 and dT == sp.Rational(-9, 2),
           f'(1) 대칭점이 x=4 위, p≤4 ⇔ θ≥π/4; (2) tanθ0=√2, 최소 {Lmin}; (3) dT/dθ = {dT}')


# J08-06  직각삼각형 PHA 내접원 반지름, 분류
def p06():
    r = sp.sin(th / 2) * (sp.sin(th / 2) + sp.cos(th / 2) - 1)
    ok1 = True
    for t in grid(0.01, 1.56, 200):
        P = (math.cos(t), math.sin(t)); H = (math.cos(t), 0); A = (1, 0)
        a, b, c = math.dist(P, H), math.dist(H, A), math.dist(A, P)
        ok1 &= abs((a + b - c) / 2 - float(r.subs(th, t))) < 1e-12
    L2 = sp.limit(r / th**2, th, 0, '+')
    Ls = {p: sp.limit(r / (1 - sp.cos(th))**p, th, 0, '+') for p in (sp.Rational(1, 2), 1, sp.Rational(3, 2))}
    ok3 = Ls[sp.Rational(1, 2)] == 0 and Ls[1] == sp.Rational(1, 2) and Ls[sp.Rational(3, 2)] == sp.oo
    report('J08-06', ok1 and L2 == sp.Rational(1, 4) and ok3,
           f'(1) 내접원 반지름 공식 200점 일치; (2) lim r/θ^2 = {L2}; (3) p=1/2,1,3/2 -> {list(Ls.values())}')


# J08-07  반원 위 오각형 ABCDE, 최대 √2, 해가 두 개인 k 범위
def p07():
    S = (3 * sp.sin(2 * th) + sp.sin(6 * th)) / 2
    ok1 = True
    for t in grid(0.01, math.pi / 6 - 0.01, 100):
        pts = [(-1, 0), (1, 0)] + [(math.cos(2 * j * t), math.sin(2 * j * t)) for j in (1, 2, 3)]
        poly = [pts[0], pts[1], pts[2], pts[3], pts[4]]  # A,B,C,D,E
        area = 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % 5][1] - poly[(i + 1) % 5][0] * poly[i][1] for i in range(5)))
        ok1 &= abs(area - float(S.subs(th, t))) < 1e-12
    crit = [c for c in sp.solve(sp.diff(S, th), th) if c.is_real and 0 < float(c) < float(sp.pi / 6)]
    Smax = sp.simplify(S.subs(th, crit[0]))
    ok2 = crit == [sp.pi / 8] and sp.simplify(Smax - sp.sqrt(2)) == 0  # [증명-수치확인] 아래 개수 세기 포함
    Send = sp.limit(S, th, sp.pi / 6, '-')
    Sl = sp.lambdify(th, S)
    ts = np.linspace(1e-7, math.pi / 6 - 1e-7, 200001)
    vals = Sl(ts)
    def count(kv):
        sgn = np.sign(vals - kv)
        return int(np.sum(sgn[:-1] * sgn[1:] < 0))
    lo, hi = float(Send), math.sqrt(2)
    ok3 = sp.simplify(Send - 3 * sp.sqrt(3) / 4) == 0 and count((lo + hi) / 2) == 2 and count(lo - 0.05) == 1 \
        and count(lo + 1e-4) == 2 and count(hi - 1e-4) == 2 and count(0.3) == 1
    report('J08-07', ok1 and ok2 and ok3,
           f'(1) 신발끈 넓이 일치; (2) 임계점 {crit}, 최댓값 {Smax}; (3) 끝값 {Send}, k∈(3√3/4, √2) 에서 해 2개 확인')


# J08-08  아르키메데스 배가 점화식
def p08():
    n = sp.symbols('n', positive=True, integer=True)
    P = lambda N: 2 * N * sp.sin(sp.pi / N)
    Q = lambda N: 2 * N * sp.tan(sp.pi / N)
    ok1 = all(float(P(N)) < 2 * math.pi < float(Q(N)) for N in range(3, 200))  # [증명-수치확인]
    ok2 = True
    for N in range(3, 60):
        ok2 &= abs(float(Q(2 * N)) - float(2 * P(N) * Q(N) / (P(N) + Q(N)))) < 1e-10
        ok2 &= abs(float(P(2 * N)) - math.sqrt(float(P(N) * Q(2 * N)))) < 1e-10  # [증명-수치확인]
    P6, Q6 = sp.Integer(6), 4 * sp.sqrt(3)
    Q12 = sp.radsimp(2 * P6 * Q6 / (P6 + Q6))
    P12 = sp.sqrtdenest(sp.sqrt(sp.expand(P6 * Q12)))
    ok3 = sp.simplify(Q12 - (48 - 24 * sp.sqrt(3))) == 0 and sp.simplify(P12 - (6 * sp.sqrt(6) - 6 * sp.sqrt(2))) == 0 \
        and sp.simplify(P12 - P(12)) == 0 and sp.simplify(Q12 - Q(12)) == 0 \
        and 3.10 < float(P12) / 2 < math.pi < float(Q12) / 2 < 3.22
    report('J08-08', ok1 and ok2 and ok3,
           f'[증명-수치확인] P_n<2π<Q_n (n=3..199), 배가식 n=3..59; (3) Q12={Q12}, P12={P12}, {float(P12)/2:.4f}<π<{float(Q12)/2:.4f}')


# J08-09  활꼴/삼각형 넓이비 -> 4/3 (sin 다항 평가)
def p09():
    S = th - sp.sin(th) * sp.cos(th)
    T = sp.sin(th) * (1 - sp.cos(th))
    ok1 = True
    for t in grid(0.01, 1.56, 200):
        A, B, M = (math.cos(t), -math.sin(t)), (math.cos(t), math.sin(t)), (1, 0)
        seg = t - math.sin(2 * t) / 2  # 중심각 2t 활꼴
        ok1 &= abs(seg - float(S.subs(th, t))) < 1e-12 and abs(tri_area(A, B, M) - float(T.subs(th, t))) < 1e-12
    xs = [mp.mpf(j) / 200 for j in range(1, 1001)]  # (0,5]
    okb = all(u**3 / 6 - u**5 / 120 < u - mp.sin(u) < u**3 / 6 for u in xs)  # [증명-수치확인]
    L2 = sp.limit((x - sp.sin(x)) / x**3, x, 0, '+')
    L3 = sp.limit(S / T, th, 0, '+')
    report('J08-09', ok1 and okb and L2 == sp.Rational(1, 6) and L3 == sp.Rational(4, 3),
           f'(1) 좌표 일치; [증명-수치확인] x^3/6-x^5/120<x-sinx<x^3/6 1000점; (2) {L2}; (3) lim S/T = {L3}')


# J08-10  부채꼴 내접 직사각형 2단 최적화, 비율의 범위 (1/2, 2/π]
def p10():
    area = 2 * sp.sin(ph) * (sp.cos(ph) - sp.sin(ph) * sp.cos(al) / sp.sin(al))
    ok1 = sp.simplify(area - (sp.cos(2 * ph - al) - sp.cos(al)) / sp.sin(al)) == 0
    ok2 = True
    for a in grid(0.05, math.pi / 2, 60):
        fa = sp.lambdify(ph, area.subs(al, a))
        ps = np.linspace(1e-6, a - 1e-6, 20001)
        v = fa(ps)
        ok2 &= abs(ps[np.argmax(v)] - a / 2) < 1e-3 and abs(v.max() - math.tan(a / 2)) < 1e-8  # [증명-수치확인]
    M = sp.simplify(area.subs(ph, al / 2))
    ok2 &= all(abs(sp.N((M - sp.tan(al / 2)).subs(al, a), 50)) < 1e-40 for a in (sp.Rational(1, 7), sp.Rational(1, 2), sp.pi / 3, sp.Rational(3, 2)))
    R = sp.tan(al / 2) / al
    Rl = sp.lambdify(al, R)
    rs = Rl(np.linspace(1e-4, math.pi / 2, 5000))
    mono = bool(np.all(np.diff(rs) > 0))  # [증명-수치확인]
    lo = sp.limit(R, al, 0, '+')
    hi = sp.simplify(R.subs(al, sp.pi / 2))
    ok3 = mono and lo == sp.Rational(1, 2) and sp.simplify(hi - 2 / sp.pi) == 0
    report('J08-10', ok1 and ok2 and ok3,
           f'(1) 넓이 식 항등; (2) 최대 φ=α/2, M(α)={M}; (3) R 증가(5000점), 범위 ({lo}, {hi}]')


if __name__ == '__main__':
    for i, fn in enumerate((p01, p02, p03, p04, p05, p06, p07, p08, p09, p10), 1):
        try:
            fn()
        except Exception as e:  # noqa
            report(f'J08-{i:02d}', False, f'예외 {e!r}')
    npass = sum(RESULTS)
    print(f'SUMMARY T08: {npass}/{len(RESULTS)} PASS')
    if npass != len(RESULTS):
        sys.exit(1)
