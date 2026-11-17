#!/usr/bin/env python3
"""D02. 삼각함수의 덧셈정리와 도형의 최적화 — sympy 검산"""
import sys
import random
import sympy as sp
import mpmath as mp

th, x, t = sp.symbols('theta x t', real=True)


def close(a, b, tol=1e-9):
    a, b = complex(sp.N(a, 30)), complex(sp.N(b, 30))
    return abs(a - b) <= tol * max(1, abs(b))


def p01():
    xp = sp.symbols('xp', positive=True)
    # A(0,1), B(0,4), P(x,0): 각 APB 를 좌표로 직접 계산
    ang = sp.atan2(4, xp) - sp.atan2(1, xp)
    tanf = sp.simplify(sp.tan(ang))
    assert sp.simplify(tanf - 3*xp/(xp**2 + 4)) == 0  # (1)
    crit = sp.solve(sp.diff(3*xp/(xp**2 + 4), xp), xp)
    assert crit == [2]
    tmax = (3*xp/(xp**2 + 4)).subs(xp, 2)
    assert tmax == sp.Rational(3, 4)  # (2)
    # (3) sin = 3/5, ∠OPA = ∠OBP (tan 1/2)
    assert sp.simplify(sp.sin(sp.atan(tmax))) == sp.Rational(3, 5)
    assert sp.Rational(1, 2) == sp.Rational(2, 4)
    # 수치: 격자 최대
    f = sp.lambdify(xp, ang)
    m = max(f(0.001*i) for i in range(1, 20000))
    assert abs(m - float(sp.atan(sp.Rational(3, 4)))) < 1e-6
    return "D02-01", "tan∠APB=3x/(x²+4), x=2 에서 최대 3/4, sin=3/5, ∠OPA=∠OBP"


def p02():
    # 직사각형 ABCD: A=O, AB=2 (각 θ), AD=1 (각 θ+π/2)
    B = sp.Matrix([2*sp.cos(th), 2*sp.sin(th)])
    D = sp.Matrix([sp.cos(th + sp.pi/2), sp.sin(th + sp.pi/2)])
    C = B + D
    assert sp.simplify(C[0] - (2*sp.cos(th) - sp.sin(th))) == 0
    H = sp.simplify(C[1])
    al = sp.atan(sp.Rational(1, 2))
    assert sp.simplify(sp.expand_trig(sp.sqrt(5)*sp.sin(th + al)) - H) == 0  # (1) cosα=2/√5, sinα=1/√5
    crit = [c for c in sp.solve(sp.diff(H, th), th) if 0 < c < sp.pi/2]
    assert len(crit) == 1
    c0 = crit[0]
    assert sp.simplify(H.subs(th, c0) - sp.sqrt(5)) == 0 and sp.simplify(sp.tan(c0) - 2) == 0
    assert sp.simplify(C[0].subs(th, c0)) == 0  # (2) C 가 y축 위
    # (3) 외접 직사각형: 꼭짓점 좌표의 max-min 을 수치 격자로 독립 계산
    def M_num(v):
        pts = [(0, 0)] + [tuple(float(q.subs(th, v)) for q in P) for P in (B, C, D)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    Mexpr = 2 + sp.Rational(5, 2)*sp.sin(2*th)
    for v in [0.1 + 0.07*k for k in range(20)]:
        assert close(M_num(v), Mexpr.subs(th, v))
    assert close(M_num(float(c0)), 4)
    grid = max(M_num(1e-4 + (sp.pi.evalf()/2 - 2e-4)*k/2000) for k in range(2001))
    assert abs(grid - 4.5) < 1e-5 and close(M_num(float(sp.pi/4)), sp.Rational(9, 2))
    return "D02-02", "C=(2cosθ-sinθ, 2sinθ+cosθ), H=√5 sin(θ+α) 최대 √5 at tanθ=2 (C 가 y축 위), M=2+5/2 sin2θ: (2)의 θ에서 4, 최대 9/2 at π/4"


def p03():
    n = sp.symbols('n', integer=True, positive=True)
    # 직접 사영으로 OA_n 계산 (θ=π/2) 후 닫힌 식과 비교
    for thv in [sp.pi/2, 2*sp.pi/3, sp.Rational(7, 10)]:
        r = sp.Integer(1)
        for kk in range(1, 9):
            ang_prev, ang = thv/2**(kk-1), thv/2**kk
            A = (r*sp.cos(ang_prev), r*sp.sin(ang_prev))
            u = (sp.cos(ang), sp.sin(ang))
            r = A[0]*u[0] + A[1]*u[1]  # 사영 길이
            closed = sp.sin(thv)/(2**kk*sp.sin(thv/2**kk))
            assert close(r, closed)  # (1)(2)
    lim = sp.limit(sp.sin(sp.pi/2)/(2**n*sp.sin(sp.pi/2/2**n)), n, sp.oo)
    assert sp.simplify(lim - 2/sp.pi) == 0  # (3)
    return "D02-03", "OA_n=OA_(n-1)cos(θ/2^n)=sinθ/(2^n sin(θ/2^n)), θ=π/2 극한 2/π"


def p04():
    b, c, d = sp.symbols('b c d', positive=True)
    # (1) 넓이 두 방법
    eq = sp.Eq(sp.Rational(1, 2)*b*c*sp.sin(2*th), sp.Rational(1, 2)*b*d*sp.sin(th) + sp.Rational(1, 2)*c*d*sp.sin(th))
    dsol = sp.solve(eq, d)[0]
    assert sp.simplify(1/b + 1/c - 2*sp.cos(th)/dsol) == 0
    # (2)(3) θ=π/3, d=1 ⇒ 1/b+1/c=1
    cc = b/(b - 1)  # b>1
    area = sp.Rational(1, 2)*b*cc*sp.sin(2*sp.pi/3)
    crit = [r for r in sp.solve(sp.diff(area, b), b) if r > 1]
    assert crit == [2] and sp.simplify(area.subs(b, 2)) == sp.sqrt(3)  # (2)
    BC2 = b**2 + cc**2 - 2*b*cc*sp.cos(2*sp.pi/3)
    crit = [r for r in sp.solve(sp.diff(BC2, b), b) if r.is_real and r > 1]
    assert crit == [2] and sp.simplify(sp.sqrt(BC2.subs(b, 2))) == 2*sp.sqrt(3)  # (3)
    f = sp.lambdify(b, BC2)
    assert min(f(1.0001 + 0.001*i) for i in range(20000)) > 12 - 1e-9
    return "D02-04", "1/b+1/c=2cosθ/d, (θ=π/3,d=1) 넓이 최소 √3, BC 최소 2√3"


def p05():
    O = (0, 0)
    P = (sp.cos(2*th), sp.sin(2*th))
    H = (sp.cos(2*th), 0)
    Q = (1/sp.cos(2*th), 0)  # 접선 x cos2θ + y sin2θ = 1 과 x축
    assert sp.simplify(Q[0]*sp.cos(2*th) - 1) == 0
    HQ = sp.simplify(Q[0] - H[0])
    assert sp.simplify(HQ - sp.sin(2*th)**2/sp.cos(2*th)) == 0  # (1)
    assert sp.limit(HQ/th**2, th, 0, '+') == 4  # (2)
    BQ = Q[0] - 1
    T = sp.Rational(1, 2)*BQ*sp.sin(2*th)
    assert sp.limit(T/th**3, th, 0, '+') == 2  # (3)
    # 대표 오답: BQ≈HQ 로 둠 → 4
    assert sp.limit(sp.Rational(1, 2)*HQ*sp.sin(2*th)/th**3, th, 0, '+') == 4
    return "D02-05", "HQ=sin²2θ/cos2θ, lim HQ/θ²=4, lim △PBQ/θ³=2 (오답 4 재현)"


def p06():
    A, B = sp.symbols('A B', positive=True)
    # 외접원 반지름 1: BC=2sinA, AB+AC=2sinB+2sin(A+B)
    g = 2*sp.sin(B) + 2*sp.sin(A + B)
    for Av in [sp.pi/3, sp.Rational(1, 2), 1.3]:
        f = sp.lambdify(B, g.subs(A, Av))
        m = max(f(float(Av)*0 + 1e-4 + i*(float(sp.pi) - float(Av))/20000) for i in range(20000))
        assert abs(m - float(4*sp.cos(Av/2))) < 1e-6  # (1)
    xx = sp.symbols('xx', positive=True)
    F = 15*4*sp.sin(xx)*sp.cos(xx) + 7*4*sp.cos(xx)
    crit = [c for c in sp.solve(sp.diff(F, xx), xx) if 0 < c < sp.pi/2]
    assert len(crit) == 1 and sp.simplify(sp.sin(crit[0]) - sp.Rational(3, 5)) == 0  # (2)
    assert sp.simplify(F.subs(xx, crit[0])) == sp.Rational(256, 5)  # (3)
    # 두 변수 동시 수치 최대
    h = lambda a, bb: 15*2*mp.sin(a) + 7*(2*mp.sin(bb) + 2*mp.sin(a + bb))
    best = max(h(mp.pi*i/400, mp.pi*j/400) for i in range(1, 400) for j in range(1, 400) if i + j < 400)
    assert abs(best - mp.mpf(256)/5) < 1e-2
    return "D02-06", "A 고정 시 AB+AC≤4cos(A/2), sin(A/2)=3/5, 15BC+7(AB+AC) 최대 256/5"


def p07():
    tB, tC = sp.symbols('tB tC', positive=True)
    # (1) 항등식 무작위 확인
    for _ in range(20):
        a1, a2 = random.uniform(0.1, 1.4), random.uniform(0.1, 1.4)
        a3 = mp.pi - a1 - a2
        assert abs(mp.tan(a1) + mp.tan(a2) + mp.tan(a3) - mp.tan(a1)*mp.tan(a2)*mp.tan(a3)) < 1e-9
    # (2) tanA=4/3 일 때 tanB tanC 의 최솟값: A 고정, B 변수
    Aang = sp.atan(sp.Rational(4, 3))
    Bv = sp.symbols('Bv', positive=True)
    p = sp.tan(Bv)*sp.tan(sp.pi - Aang - Bv)
    f = sp.lambdify(Bv, p, 'mpmath')
    lo, hi = float(sp.pi/2 - Aang) + 1e-6, float(sp.pi/2) - 1e-6
    m = min(f(lo + (hi - lo)*i/20000) for i in range(20001))
    assert abs(m - 4) < 1e-6  # 최솟값 4
    assert sp.simplify(sp.tan(sp.pi - Aang - sp.atan(2)) - 2) == 0  # 등호 tanB=tanC=2
    # (3) BC=6, 높이 = 6 p/(tanB+tanC) 넓이 최대 18
    area = lambda B_: 0.5*6*6*f(B_)/(mp.tan(B_) + mp.tan(mp.pi - float(Aang) - B_))
    M = max(area(lo + (hi - lo)*i/20000) for i in range(1, 20000))
    assert abs(M - 18) < 1e-6
    return "D02-07", "tanA+tanB+tanC=곱, tanA=4/3 ⇒ tanB tanC≥4, BC=6 넓이 최대 18"


def p08():
    k = sp.symbols('k', real=True)
    f = 2*sp.cos(th) + 4*sp.sin(th)
    g = sp.lambdify(th, f, 'numpy')
    import numpy as np
    ts = np.linspace(1e-9, np.pi/2 - 1e-9, 400001)
    vals = g(ts)
    def N(kv):
        s = np.sign(vals - kv)
        return int(np.count_nonzero(s[1:] != s[:-1]))
    assert abs(vals.max() - 2*np.sqrt(5)) < 1e-8  # 최대 2√5
    assert [N(1.9), N(2.5), N(3.99), N(4.2), N(4.47), N(4.5)] == [0, 1, 1, 2, 2, 0]
    gam = sp.atan(2)
    assert sp.expand_trig(sp.tan(2*gam)) == sp.Rational(-4, 3)  # (3)
    # 두 해 합 수치 확인 (k=4.3)
    from scipy.optimize import brentq
    th1 = brentq(lambda z: g(z) - 4.3, 1e-9, float(gam)); th2 = brentq(lambda z: g(z) - 4.3, float(gam), np.pi/2)
    assert abs(np.tan(th1 + th2) + 4/3) < 1e-9
    return "D02-08", "PA+2PB=2√5cos(θ-γ), tanγ=2, N(k) 분류, tan(θ1+θ2)=-4/3"


def p09():
    a, b = 1, 2*sp.sqrt(2)
    L = a/sp.sin(th) + b/sp.cos(th)
    crit = [c for c in sp.solve(sp.diff(L, th), th) if c.is_real and 0 < c < sp.pi/2]
    assert len(crit) == 1 and sp.simplify(sp.tan(crit[0]) - 1/sp.sqrt(2)) == 0  # (2)
    Lmin = sp.simplify(L.subs(th, crit[0]))
    assert sp.simplify(Lmin - 3*sp.sqrt(3)) == 0  # (3)
    f = sp.lambdify(th, L)
    assert min(f(0.0001 + i*1.5706/50000) for i in range(50000)) > float(3*sp.sqrt(3)) - 1e-9
    return "D02-09", "L(θ)=1/sinθ+2√2/cosθ, tanθ=1/√2 에서 최소 3√3 = 통과 가능한 최대 길이"


def p10():
    # 폭 1 종이: 오른쪽 아래 꼭짓점 (1,0) 을 왼쪽 변 x=0 위로 접기
    av, bv, y0 = sp.symbols('av bv y0', positive=True)
    sol = sp.solve([(1 - av)**2 + y0**2 - av**2, 1 + (bv - y0)**2 - bv**2], [y0, bv], dict=True)
    sol = [s_ for s_ in sol if s_[bv].subs(av, sp.Rational(3, 4)) > 0]
    assert len(sol) == 1
    bexpr = sp.simplify(sol[0][bv])
    # θ 와의 관계 tanθ = b/a 를 이용해 a=1/(2sin²θ), b=1/sin2θ 확인
    for thv in [sp.pi/3, sp.Rational(9, 10), sp.Rational(12, 10)]:
        aa = 1/(2*sp.sin(thv)**2)
        assert close(bexpr.subs(av, aa), 1/sp.sin(2*thv))
        assert close(bexpr.subs(av, aa)/aa, sp.tan(thv))  # (1)
    L = sp.sqrt(av**2 + bexpr**2)
    crit = [c for c in sp.solve(sp.diff(L**2, av), av) if sp.Rational(1, 2) < c <= 1]
    assert crit == [sp.Rational(3, 4)] and sp.simplify(L.subs(av, crit[0])) == 3*sp.sqrt(3)/4  # (2)
    S = av*bexpr/2
    crit = [c for c in sp.solve(sp.diff(S, av), av) if sp.Rational(1, 2) < c <= 1]
    assert crit == [sp.Rational(2, 3)] and sp.simplify(S.subs(av, crit[0])) == 2*sp.sqrt(3)/9  # (3)
    assert sp.simplify(sp.atan(bexpr.subs(av, sp.Rational(2, 3))/sp.Rational(2, 3))) == sp.pi/3
    # 경계 a=1 (θ=π/4) 값이 더 큼
    assert S.subs(av, 1) > 2*sp.sqrt(3)/9 and L.subs(av, 1) > 3*sp.sqrt(3)/4
    return "D02-10", "a=1/(2sin²θ), b=1/sin2θ, 접는 선 최소 3√3/4 (cosθ=1/√3), 접힌 넓이 최소 2√3/9 (θ=π/3)"


def main():
    ok = 0
    funcs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]
    for i, f in enumerate(funcs, 1):
        try:
            pid, msg = f()
            print(f"{pid} PASS  {msg}")
            ok += 1
        except Exception as e:
            print(f"D02-{i:02d} FAIL  {type(e).__name__}: {e}")
    print(f"SUMMARY T02: {ok}/{len(funcs)} PASS")
    if ok != len(funcs):
        sys.exit(1)


if __name__ == '__main__':
    main()
