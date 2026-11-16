#!/usr/bin/env python3
"""J10 검산: 매개변수로 나타낸 곡선 — 접선·속도·길이·넓이 (세종 T10)
문제의 매개변수 식(또는 운동 서술에서 다시 세운 식)으로부터 sympy 로 길이·넓이·극한을
독립 계산하고, 넓이는 추가로 곡선을 잘게 나눈 다각형의 신발끈 공식(고정밀 수치)으로 교차 확인한다.
"""
import sys
import random
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
t, s, u = sp.symbols('t s u', real=True)
RESULTS = []


def check(pid, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def eq(a, b):
    return sp.simplify(sp.nsimplify(a) - b) == 0 or abs(sp.N(a - b, 40)) < 1e-25


def shoelace(pts):
    """닫힌 다각형 넓이(부호 없음)."""
    A = mp.mpf(0)
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        A += x1 * y2 - x2 * y1
    return abs(A) / 2


def sample(X, Y, a, b, N=4000):
    fx = sp.lambdify(t, X, 'mpmath')
    fy = sp.lambdify(t, Y, 'mpmath')
    a, b = mp.mpf(sp.N(a, 50)), mp.mpf(sp.N(b, 50))
    return [(fx(a + (b - a) * i / N), fy(a + (b - a) * i / N)) for i in range(N + 1)]


def seg(P, Q, N=200):
    return [(P[0] + (Q[0] - P[0]) * i / N, P[1] + (Q[1] - P[1]) * i / N) for i in range(N + 1)]


def speed(X, Y):
    return sp.sqrt(sp.diff(X, t)**2 + sp.diff(Y, t)**2)


def num_close(a, b, tol=1e-6):
    return abs(mp.mpf(a) - mp.mpf(sp.N(b, 40))) < tol * max(1, abs(mp.mpf(sp.N(b, 40))))


# ---------------------------------------------------------------- J10-01
def p01():
    X, Y = t**2, t - t**3 / 3
    # (1) 자기교차: x(s)=x(t), y(s)=y(t), s != t
    sol = sp.solve([X.subs(t, s) - X, Y.subs(t, s) - Y, ], [s, t], dict=True)
    cross = {(sp.simplify(X.subs(d)), sp.simplify(Y.subs(d))) for d in sol if sp.simplify(d[s] - d[t]) != 0}
    ok1 = cross == {(3, 0)}
    slope = sp.diff(Y, t) / sp.diff(X, t)
    sl = {sp.simplify(slope.subs(t, sp.sqrt(3))), sp.simplify(slope.subs(t, -sp.sqrt(3)))}
    ok1 &= sl == {-1 / sp.sqrt(3), 1 / sp.sqrt(3)}
    # (2) 길이
    sp_ = sp.simplify(speed(X, Y)**2 - (1 + t**2)**2) == 0
    L = sp.integrate(1 + t**2, (t, -sp.sqrt(3), sp.sqrt(3)))
    ok2 = sp_ and eq(L, 4 * sp.sqrt(3))
    # (3) 넓이: 위쪽 절반(0<=t<=sqrt3) 2배
    S = 2 * sp.integrate(Y * sp.diff(X, t), (t, 0, sp.sqrt(3)))
    Snum = shoelace(sample(X, Y, -sp.sqrt(3), sp.sqrt(3))[:-1])
    ok3 = eq(S, 8 * sp.sqrt(3) / 5) and num_close(Snum, S) and eq(S / L, sp.Rational(2, 5))
    check('J10-01', ok1 and ok2 and ok3, f'(1)교차점 (3,0), 기울기 ±1/sqrt3 (2)L={L} (3)S={S}, S/L={sp.simplify(S/L)} (신발끈 {mp.nstr(Snum,10)})')


# ---------------------------------------------------------------- J10-02
def p02():
    X, Y = 3 * t**2, 2 * t**3
    slope = sp.simplify(sp.diff(Y, t) / sp.diff(X, t))
    ok1 = slope == t
    # (1) 접선과 곡선의 교점: 2s^3 - 2t^3 = t(3s^2 - 3t^2)
    tt = sp.symbols('tt', nonzero=True)
    eqn = sp.factor(2 * s**3 - 2 * tt**3 - tt * (3 * s**2 - 3 * tt**2))
    roots = sp.solve(eqn, s)
    ok1 &= set(roots) == {tt, -tt / 2}
    # (2) 길이
    L = sp.integrate(sp.simplify(speed(X, Y).subs(t, u)), (u, 0, sp.sqrt(3)))
    L2 = sp.integrate(6 * t * sp.sqrt(1 + t**2), (t, 0, sp.sqrt(3)))
    ok2 = eq(L2, 14) and abs(sp.N(L) - 14) < 1e-12
    # (3) t=2 접선 y=2x-8, 교점 s=-1 ; 닫힌 경로(곡선 -1->2, 선분 P->Q) 넓이
    P = (X.subs(t, 2), Y.subs(t, 2)); Q = (X.subs(t, -1), Y.subs(t, -1))
    xx = sp.symbols('xx')
    line_int = sp.integrate(2 * xx - 8, (xx, P[0], Q[0]))
    S = sp.Abs(sp.integrate(Y * sp.diff(X, t), (t, -1, 2)) + line_int)
    pts = sample(X, Y, -1, 2)[:-1] + seg((mp.mpf(12), mp.mpf(16)), (mp.mpf(3), mp.mpf(-2)))[:-1]
    ok3 = P == (12, 16) and Q == (3, -2) and eq(S, sp.Rational(81, 5)) and num_close(shoelace(pts), S)
    check('J10-02', ok1 and ok2 and ok3, f'(1)dy/dx=t, 재교점 s=-t/2 (2)L=14 (3)S={S} (신발끈 {mp.nstr(shoelace(pts),10)})')


# ---------------------------------------------------------------- J10-03
def p03():
    # 운동에서 식 세우기: A=(2cos t,2sin t), AP 는 각 -2t 방향 길이 1
    X = 2 * sp.cos(t) + sp.cos(-2 * t)
    Y = 2 * sp.sin(t) + sp.sin(-2 * t)
    ok1 = (X.subs(t, 0), Y.subs(t, 0)) == (3, 0)
    # [증명-수치확인] 속력 = 4|sin(3t/2)| 무작위 20점
    random.seed(3)
    v = sp.lambdify(t, speed(X, Y), 'mpmath')
    ok1 &= all(abs(v(r) - 4 * abs(mp.sin(1.5 * r))) < 1e-20 for r in [random.uniform(0, 7) for _ in range(20)])
    # (2) 길이, 넓이
    L = 3 * sp.integrate(4 * sp.sin(3 * t / 2), (t, 0, 2 * sp.pi / 3))
    S = sp.Abs(sp.integrate(sp.expand(sp.expand_trig(Y * sp.diff(X, t))), (t, 0, 2 * sp.pi)))
    Snum = shoelace(sample(X, Y, 0, 2 * sp.pi)[:-1])
    Lnum = mp.quad(lambda r: v(r), [0, 2 * mp.pi / 3, 4 * mp.pi / 3, 2 * mp.pi])
    ok2 = eq(L, 16) and num_close(Lnum, 16, 1e-15) and eq(S, 2 * sp.pi) and num_close(Snum, S)
    # (3) |OP|^2 = 5+4cos3t >= 1, 내접원 반지름 1, 넓이 차 pi
    r2 = sp.simplify(sp.expand_trig(X**2 + Y**2) - (5 + 4 * sp.cos(3 * t)))
    ok3 = sp.simplify(sp.expand(sp.expand_trig(X**2 + Y**2 - 5 - 4 * sp.cos(3 * t)))) == 0
    rmin = min(mp.sqrt(5 + 4 * mp.cos(3 * mp.mpf(i) / 1000)) for i in range(0, 6284))
    ok3 &= abs(rmin - 1) < 1e-5 and eq(S - sp.pi, sp.pi)
    check('J10-03', ok1 and ok2 and ok3, f'(1)P(0)=(3,0), [증명-수치확인] 속력식 20점 (2)L={L}, S={S} (3)|OP|min=1, S-pi={sp.simplify(S-sp.pi)}')


# ---------------------------------------------------------------- J10-04
def p04():
    X, Y = t - sp.sin(t), 1 - sp.cos(t)
    # (1) [증명-수치확인] 접선이 T(t,2)를 지남: (T-P) x (P') = 0
    cross = sp.simplify((t - X) * sp.diff(Y, t) - (2 - Y) * sp.diff(X, t))
    ok1 = cross == 0
    # (2) s(t) + 2|PT| = 4 (0<t<=pi) [증명-수치확인] 격자 1000점
    sfun = lambda T: mp.quad(lambda r: 2 * mp.sin(r / 2), [0, T])
    ok2 = True
    for i in range(1, 1001):
        T = mp.pi * i / 1000
        PT = mp.sqrt(mp.sin(T)**2 + (1 + mp.cos(T))**2)
        ok2 &= abs(sfun(T) + 2 * PT - 4) < 1e-20
    # (3) s = 2 PT  <=> PT = 1
    T0 = sp.solve(sp.Eq(4 * (1 - sp.cos(t / 2)), 2 * 2 * sp.cos(t / 2)), t)
    T0 = [r for r in T0 if 0 < r <= sp.pi]
    S = sp.integrate(sp.expand(Y * sp.diff(X, t)), (t, 0, 2 * sp.pi / 3))
    x0 = X.subs(t, 2 * sp.pi / 3)
    pts = sample(X, Y, 0, 2 * sp.pi / 3)
    pts = pts + [(mp.mpf(sp.N(x0, 40)), mp.mpf(0))]
    ok3 = T0 == [2 * sp.pi / 3] and eq(S, sp.pi - 9 * sp.sqrt(3) / 8) and num_close(shoelace(pts), S)
    check('J10-04', ok1 and ok2 and ok3, f'(1)접선이 꼭대기점 통과 (2)[증명-수치확인] s+2PT=4 1000점 (3)t0={T0}, S={S}')


# ---------------------------------------------------------------- J10-05
def p05():
    # 굴림 관계에서 식 세우기: 중심 M=(cos t, sin t), MP 방향각 = t - 2t
    d = sp.symbols('d', positive=True)
    Mx, My = sp.cos(t), sp.sin(t)
    Px, Py = Mx + sp.cos(t - 2 * t), My + sp.sin(t - 2 * t)
    ok1 = sp.simplify(Px - 2 * sp.cos(t)) == 0 and sp.simplify(Py) == 0
    # 미끄러지지 않음: 고정원 호 2t = 작은 원 호 1*(2t) ; 접점 K 에서 P 의 속도 0 (순간중심) 확인
    Kx, Ky = 2 * sp.cos(t), 2 * sp.sin(t)
    Qx, Qy = Mx + d * sp.cos(-t), My + d * sp.sin(-t)
    ok1 &= sp.simplify(Qx - (1 + d) * sp.cos(t)) == 0 and sp.simplify(Qy - (1 - d) * sp.sin(t)) == 0
    # (2) 타원 방정식, 넓이, 속력
    ell = sp.simplify(Qx**2 / (1 + d)**2 + Qy**2 / (1 - d)**2)
    S = sp.Abs(sp.integrate(sp.expand(Qy * sp.diff(Qx, t)), (t, 0, 2 * sp.pi)))
    v2 = sp.simplify(sp.diff(Qx, t)**2 + sp.diff(Qy, t)**2 - (1 + d**2 - 2 * d * sp.cos(2 * t)))
    ok2 = ell == 1 and sp.simplify(S - sp.pi * (1 - d**2)) == 0 and sp.simplify(sp.expand_trig(v2)) == 0
    # (3) 넓이 = (3/4) pi -> d ; 최대 속력
    dsol = [r for r in sp.solve(sp.Eq(sp.pi * (1 - d**2), sp.Rational(3, 4) * sp.pi), d) if r > 0]
    vmax = max(mp.sqrt(1 + mp.mpf(1) / 4 - mp.cos(2 * mp.pi * i / 2000)) for i in range(2001))
    ok3 = dsol == [sp.Rational(1, 2)] and abs(vmax - mp.mpf(3) / 2) < 1e-12
    Qhalf = (sp.simplify(Qx.subs({d: sp.Rational(1, 2), t: sp.pi / 2})), sp.simplify(Qy.subs({d: sp.Rational(1, 2), t: sp.pi / 2})))
    ok3 &= Qhalf == (0, sp.Rational(1, 2))
    check('J10-05', ok1 and ok2 and ok3, f'(1)P=(2cos t,0) (2)타원, S=pi(1-d^2), |v|^2=1+d^2-2d cos2t (3)d={dsol}, 최대속력 3/2 at {Qhalf}')


# ---------------------------------------------------------------- J10-06
def p06():
    X, Y = sp.sin(t), sp.sin(2 * t)
    xx = sp.symbols('xx')
    ok1 = sp.simplify(sp.expand_trig(Y**2 - 4 * X**2 * (1 - X**2))) == 0
    sl = sp.diff(Y, t) / sp.diff(X, t)
    ok1 &= (sl.subs(t, 0), sl.subs(t, sp.pi)) == (2, -2)
    # (2) 직선 y=mx 와의 교점 개수 (서로 다른 점) — 수치 전수
    def count(m):
        pts = set()
        # sin t (2cos t - m) = 0
        cand = [0, mp.pi]
        if abs(m) <= 2:
            a = mp.acos(mp.mpf(m) / 2)
            cand += [a, 2 * mp.pi - a]
        for c in cand:
            pts.add((round(float(mp.sin(c)), 9) + 0.0, round(float(mp.sin(2 * c)), 9) + 0.0))
        return len(pts)
    ok2 = all(count(m) == 3 for m in (-1.9, -1, 0, 0.5, 1.99)) and all(count(m) == 1 for m in (-3, -2, 2, 5))
    # (3) m=1: 교점 t=pi/3, 넓이
    t0 = [r for r in sp.solve(sp.Eq(2 * sp.cos(t), 1), t) if 0 < r < sp.pi / 2]
    A = sp.integrate(sp.expand(Y * sp.diff(X, t)), (t, 0, sp.pi / 3)) - sp.Rational(1, 2) * X.subs(t, sp.pi / 3) * Y.subs(t, sp.pi / 3)
    pts = sample(X, Y, 0, sp.pi / 3)
    ok3 = t0 == [sp.pi / 3] and eq(A, sp.Rational(5, 24)) and num_close(shoelace(pts), A)
    check('J10-06', ok1 and ok2 and ok3, f'(1)y^2=4x^2(1-x^2), 기울기 ±2 (2)|m|<2:3, |m|>=2:1 (3)A={sp.nsimplify(A)}')


# ---------------------------------------------------------------- J10-07
def p07():
    X, Y = sp.cos(t)**3, sp.sin(t)**3
    xx, yy = sp.symbols('xx yy')
    sl = sp.simplify(sp.diff(Y, t) / sp.diff(X, t))
    tan_line = yy - Y - sl * (xx - X)
    A = sp.simplify(sp.solve(tan_line.subs(yy, 0), xx)[0])
    B = sp.simplify(sp.solve(tan_line.subs(xx, 0), yy)[0])
    ok1 = sp.simplify(A - sp.cos(t)) == 0 and sp.simplify(B - sp.sin(t)) == 0
    # (2) 삼각형 넓이 최대
    tri = A * B / 2
    crit = [r for r in sp.solve(sp.diff(tri, t), t) if 0 < r < sp.pi / 2]
    Q = sp.integrate(sp.expand(Y * sp.diff(X, t)), (t, sp.pi / 2, 0))
    ok2 = crit == [sp.pi / 4] and eq(tri.subs(t, sp.pi / 4), sp.Rational(1, 4)) and eq(Q, 3 * sp.pi / 32)
    pts = sample(X, Y, 0, sp.pi / 2) + [(mp.mpf(0), mp.mpf(0))]
    ok2 &= num_close(shoelace(pts), Q)
    # (3) 볼록성(접선이 곡선 아래) [증명-수치확인], 넓이
    d2 = sp.simplify(sp.diff(sl, t) / sp.diff(X, t))
    f2 = sp.lambdify(t, d2, 'mpmath')
    ok3 = all(f2(mp.pi / 2 * i / 500) > 0 for i in range(1, 500))
    ans = Q - sp.Rational(1, 4)
    ok3 &= eq(ans, (3 * sp.pi - 8) / 32) and ans > 0
    check('J10-07', ok1 and ok2 and ok3, f'(1)A=(cos t,0),B=(0,sin t),AB=1 (2)t=pi/4, 1/4, 사분면넓이 {Q} (3)답 {sp.simplify(ans)}')


# ---------------------------------------------------------------- J10-08
def p08():
    X, Y = sp.exp(-t) * sp.cos(t), sp.exp(-t) * sp.sin(t)
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    ok1 = sp.simplify(vx**2 + vy**2 - 2 * sp.exp(-2 * t)) == 0
    cosang = sp.simplify((X * vx + Y * vy) / (sp.sqrt(X**2 + Y**2) * sp.sqrt(vx**2 + vy**2)))
    ok1 &= sp.simplify(cosang + 1 / sp.sqrt(2)) == 0
    # (2) A_n = -(-1)^n ∫ y x' ... 부호 없는 넓이 = |∫_{nπ}^{(n+1)π} y x' dt|
    def A(n):
        return sp.simplify(sp.Abs(sp.integrate(sp.expand(Y * vx), (t, n * sp.pi, (n + 1) * sp.pi))))
    As = [A(n) for n in range(4)]
    ok2 = eq(As[0], (1 - sp.exp(-2 * sp.pi)) / 4) and all(eq(As[i + 1] / As[i], sp.exp(-2 * sp.pi)) for i in range(3))
    pts = sample(X, Y, 0, sp.pi, 6000)
    ok2 &= num_close(shoelace(pts), As[0])
    # (3) 합과 전체 이동 거리
    n = sp.symbols('n', integer=True, nonnegative=True)
    tot = sp.summation(As[0] * sp.exp(-2 * sp.pi * n), (n, 0, sp.oo))
    Ltot = sp.integrate(sp.sqrt(2) * sp.exp(-t), (t, 0, sp.oo))
    ok3 = eq(tot, sp.Rational(1, 4)) and eq(Ltot, sp.sqrt(2)) and eq(tot / Ltot**2, sp.Rational(1, 8))
    check('J10-08', ok1 and ok2 and ok3, f'(1)속력 sqrt2 e^-t, 각 3pi/4 (2)A0={As[0]}, 공비 e^(-2pi) (3)합={sp.simplify(tot)}, 거리={Ltot}')


# ---------------------------------------------------------------- J10-09
def p09():
    # 식 세우기: Q=(cos t, sin t), QP = t*(sin t, -cos t)
    X = sp.cos(t) + t * sp.sin(t)
    Y = sp.sin(t) - t * sp.cos(t)
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    ok1 = sp.simplify(vx * (X - sp.cos(t)) + vy * (Y - sp.sin(t))) == 0
    ok1 &= sp.simplify(vx**2 + vy**2 - t**2) == 0
    # QP 가 원에 접함: OQ · QP = 0, |QP| = t
    ok1 &= sp.simplify(sp.cos(t) * (X - sp.cos(t)) + sp.sin(t) * (Y - sp.sin(t))) == 0
    L = sp.integrate(t, (t, 0, 2 * sp.pi))
    ok2 = eq(L, 2 * sp.pi**2)
    # (3) 닫힌 경로: 곡선 0->2π, 선분 (1,-2π)->(1,0) (dx=0)
    I = sp.integrate(sp.expand(Y * vx), (t, 0, 2 * sp.pi))
    S = sp.Abs(I)
    pts = sample(X, Y, 0, 2 * sp.pi, 20000)[:-1] + seg((mp.mpf(1), -2 * mp.pi), (mp.mpf(1), mp.mpf(0)))[:-1]
    # 선분 x=1 이 곡선과 끝점 외에서 만나지 않음 [수치확인]
    fx = sp.lambdify(t, X, 'mpmath')
    simple = all(fx(2 * mp.pi * i / 4000) != 1 or i in (0, 4000) for i in range(4001))
    xs = [fx(2 * mp.pi * i / 4000) for i in range(1, 4000)]
    hits = sum(1 for i in range(len(xs) - 1) if (xs[i] - 1) * (xs[i + 1] - 1) < 0)
    ok3 = eq(S, sp.pi + 4 * sp.pi**3 / 3) and num_close(shoelace(pts), S, 1e-5) and hits == 0
    check('J10-09', ok1 and ok2 and ok3, f'(1)속도⊥QP, 속력 t (2)L={L} (3)S={sp.simplify(S)} (신발끈 {mp.nstr(shoelace(pts),10)}), x=1 재교차 {hits}')


# ---------------------------------------------------------------- J10-10
def p10():
    ok1 = ok2 = ok3 = True
    msg = []
    for k in range(1, 7):
        # 운동에서 식 세우기: 중심 C=((k+1)cos t,(k+1)sin t), CP 방향각 = π + t + k t
        X = (k + 1) * sp.cos(t) + sp.cos(sp.pi + (k + 1) * t)
        Y = (k + 1) * sp.sin(t) + sp.sin(sp.pi + (k + 1) * t)
        ok1 &= (sp.simplify(X.subs(t, 0)), sp.simplify(Y.subs(t, 0))) == (k, 0)
        # 굴림 조건: 접점 K=(k cos t,k sin t) 에서 P 속도가 순간중심 K 에 대해 수직 [수치]
        vx, vy = sp.diff(X, t), sp.diff(Y, t)
        f = sp.lambdify(t, vx * (X - k * sp.cos(t)) + vy * (Y - k * sp.sin(t)), 'mpmath')
        ok1 &= all(abs(f(r)) < 1e-25 for r in (0.3, 1.1, 2.7, 4.4))
        sp2 = sp.lambdify(t, sp.sqrt(vx**2 + vy**2), 'mpmath')
        ok1 &= all(abs(sp2(r) - 2 * (k + 1) * abs(mp.sin(k * r / 2))) < 1e-25 for r in (0.3, 1.1, 2.7, 4.4))
        L = mp.quad(sp2, [2 * mp.pi * j / k for j in range(k + 1)])
        ok2 &= abs(L - 8 * (k + 1)) < 1e-20
        S = sp.Abs(sp.integrate(sp.expand(sp.expand_trig(Y * vx)), (t, 0, 2 * sp.pi)))
        ok2 &= eq(S, sp.pi * (k + 1) * (k + 2))
        Snum = shoelace(sample(X, Y, 0, 2 * sp.pi, 8000)[:-1])
        ok2 &= num_close(Snum, S, 1e-5)
        D = sp.simplify(S - sp.pi * k**2)
        ok3 &= eq(D, sp.pi * (3 * k + 2))
        msg.append(f'k={k}:L={mp.nstr(L,6)},S={S}')
    kk = sp.symbols('k', positive=True)
    lim = sp.limit(sp.pi * (3 * kk + 2) / (8 * (kk + 1)), kk, sp.oo)
    ok3 &= eq(lim, 3 * sp.pi / 8)
    check('J10-10', ok1 and ok2 and ok3, f'k=1..6 ' + ' '.join(msg[:3]) + f' ... (3) D_k=pi(3k+2), lim={lim}')


if __name__ == '__main__':
    for f in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
        try:
            f()
        except Exception as e:  # noqa
            check(f.__name__.replace('p', 'J10-'), False, f'예외 {e!r}')
    npass = sum(RESULTS)
    print(f'SUMMARY T10: {npass}/{len(RESULTS)} PASS')
    if npass != len(RESULTS) or len(RESULTS) != 10:
        sys.exit(1)
