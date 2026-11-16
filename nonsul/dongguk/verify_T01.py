#!/usr/bin/env python3
"""D01. 평면 위의 운동: 속도·가속도·이동 거리 — sympy 검산"""
import sys
import sympy as sp
import mpmath as mp

t, s, k, T = sp.symbols('t s k T', real=True)
RESULTS = []


def speed(X, Y):
    return sp.sqrt(sp.diff(X, t)**2 + sp.diff(Y, t)**2)


def close(a, b, tol=1e-9):
    a, b = complex(sp.N(a, 30)), complex(sp.N(b, 30))
    return abs(a - b) <= tol * max(1, abs(b))


def p01():
    X = (sp.exp(t) + sp.exp(-t)) / 2
    Y = t
    sp2 = sp.simplify(sp.diff(X, t)**2 + sp.diff(Y, t)**2 - X**2)
    assert sp2 == 0  # (1) 속력 = (e^t+e^-t)/2
    # 속력 최솟값 1 (t=0)
    assert sp.simplify(X.subs(t, 0)) == 1
    dist = sp.integrate(X, (t, 0, T))
    assert sp.simplify(dist - (sp.exp(T) - sp.exp(-T)) / 2) == 0  # (2)
    sol = sp.solve(sp.Eq(dist, sp.Rational(3, 4)), T)
    sol = [x for x in sol if x.is_real]
    assert len(sol) == 1 and sp.simplify(sol[0] - sp.log(2)) == 0  # (3)
    pos = (sp.simplify(X.subs(t, sp.log(2))), sp.log(2))
    assert pos[0] == sp.Rational(5, 4)
    # 대표 오답: 거리를 x(T)-x(0) 로 착각 -> (e^T+e^-T)/2 -1 = 3/4 -> e^T = 2+...
    wrong = sp.solve(sp.Eq((sp.exp(T) + sp.exp(-T)) / 2 - 1, sp.Rational(3, 4)), T)
    return "D01-01", "속력=cosh형, 최소속력 1, 거리 (e^T-e^-T)/2, T=ln2, P=(5/4, ln2)"


def p02():
    q = sp.symbols('q', positive=True)
    poly = sp.expand((t**2 - 4)**2 + q**2 * t**2)
    # 완전제곱 조건: (t^2+ct+d)^2 와 계수 비교
    c, d = sp.symbols('c d', real=True)
    eqs = sp.Poly(sp.expand((t**2 + c*t + d)**2) - poly, t).all_coeffs()
    sols = sp.solve(eqs, [c, d, q], dict=True)
    qs = sorted({x[q] for x in sols})
    assert qs == [4]  # (1)
    vx, vy = t**2 - 4, 4*t
    X = sp.integrate(vx, (t, 0, t)); Y = sp.integrate(vy, (t, 0, t))
    roots = [r for r in sp.solve(sp.Eq(X, 0), t) if r.is_positive]
    assert roots == [2*sp.sqrt(3)]  # (2)
    assert sp.simplify(Y.subs(t, 2*sp.sqrt(3))) == 24
    assert sp.factor(vx**2 + vy**2) == (t**2 + 4)**2
    dist = sp.integrate(t**2 + 4, (t, 0, 2*sp.sqrt(3)))
    assert sp.simplify(dist - 16*sp.sqrt(3)) == 0  # (3)
    # 대표 오답: 변위 크기 24
    return "D01-02", "q=4, T=2√3, P(T)=(0,24), 이동거리 16√3 (변위 24와 구별)"


def p03():
    X = sp.cos(t) + t*sp.sin(t)
    Y = sp.sin(t) - t*sp.cos(t)
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    ax, ay = sp.diff(vx, t), sp.diff(vy, t)
    tp = sp.symbols('tp', positive=True)
    sp_ = sp.simplify(sp.sqrt(vx**2 + vy**2).subs(t, tp))
    assert sp.simplify(sp_ - tp) == 0  # (1) 속력 t
    assert sp.simplify(ax**2 + ay**2 - (1 + t**2)) == 0  # |a|=sqrt(1+t^2)
    dot = sp.simplify(vx*ax + vy*ay)
    assert sp.simplify(dot - t) == 0
    cosphi = sp.simplify((dot / (sp_ * sp.sqrt(1 + tp**2))).subs(t, tp))
    assert sp.simplify(cosphi - 1/sp.sqrt(1 + tp**2)) == 0  # (2)
    t1 = sp.solve(sp.Eq(1/sp.sqrt(1 + tp**2), sp.cos(sp.pi/4)), tp)
    t2 = sp.solve(sp.Eq(1/sp.sqrt(1 + tp**2), sp.cos(sp.pi/3)), tp)
    assert t1 == [1] and t2 == [sp.sqrt(3)]
    dist = sp.integrate(tp, (tp, 1, sp.sqrt(3)))
    assert dist == 1  # (3)
    return "D01-03", "속력 t, cosφ=1/√(1+t²), φ∈[π/4,π/3] ⇔ t∈[1,√3], 이동거리 1"


def p04():
    X = sp.exp(-t)*sp.cos(t); Y = sp.exp(-t)*sp.sin(t)
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    spd2 = sp.simplify(vx**2 + vy**2)
    assert sp.simplify(spd2 - 2*sp.exp(-2*t)) == 0  # (1)
    cosang = sp.simplify((X*vx + Y*vy) / (sp.exp(-t) * sp.sqrt(2)*sp.exp(-t)))
    assert sp.simplify(cosang + 1/sp.sqrt(2)) == 0  # 각 3π/4
    n = sp.symbols('n', integer=True, positive=True)
    ln = sp.integrate(sp.sqrt(2)*sp.exp(-t), (t, 2*(n-1)*sp.pi, 2*n*sp.pi))
    assert sp.simplify(ln - sp.sqrt(2)*sp.exp(-2*(n-1)*sp.pi)*(1 - sp.exp(-2*sp.pi))) == 0
    assert sp.simplify(ln.subs(n, n+1) / ln - sp.exp(-2*sp.pi)) == 0  # (2)
    tot = sp.summation(sp.simplify(ln), (n, 1, sp.oo))
    assert sp.simplify(tot - sp.sqrt(2)) == 0  # (3)
    return "D01-04", "속력 √2e^-t, OP와 v 각 3π/4, l_(n+1)=e^(-2π)l_n, Σl_n=√2"


def p05():
    X = 2*sp.cos(t); Y = sp.sin(t)
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    ax, ay = sp.diff(vx, t), sp.diff(vy, t)
    assert sp.simplify(ax + X) == 0 and sp.simplify(ay + Y) == 0  # (1) a=-OP
    spd2 = sp.simplify(vx**2 + vy**2)
    vals = [spd2.subs(t, v) for v in sp.solve(sp.diff(spd2, t), t)] + [spd2.subs(t, 0)]
    assert min(vals) == 1 and max(vals) == 4
    u = sp.symbols('u', real=True)
    cosphi = (vx*ax + vy*ay) / sp.sqrt((vx**2 + vy**2)*(ax**2 + ay**2))
    target = 3*u/sp.sqrt(16 + 9*u**2)
    import random
    for _ in range(20):  # (2) 항등식 무작위 확인
        tv = random.uniform(0, 2*3.14159)
        assert close(cosphi.subs(t, tv), target.subs(u, sp.sin(2*tv)))
    # (3) cosφ 최댓값 3/5 (u=1)
    assert sp.simplify(sp.diff(target, u) - 48/(16 + 9*u**2)**sp.Rational(3, 2)) == 0
    assert sp.simplify(target.subs(u, 1)) == sp.Rational(3, 5)
    f = sp.lambdify(t, cosphi, 'mpmath')
    m = max(f(mp.mpf(i)*2*mp.pi/4000) for i in range(4000))
    assert abs(m - mp.mpf(3)/5) < 1e-6
    return "D01-05", "a=-OP, 속력 1~2, cosφ=3u/√(16+9u²), 최소각 cos=3/5 (t=π/4), tan=4/3"


def p06():
    kk = sp.symbols('kk', positive=True)
    vx = sp.integrate(-sp.cos(t), (t, 0, t)) + 0
    vy = sp.integrate(-sp.sin(t), (t, 0, t)) + kk
    X = 1 + sp.integrate(vx, (t, 0, t)); Y = sp.integrate(vy, (t, 0, t))
    assert sp.simplify(X - sp.cos(t)) == 0
    assert sp.simplify(Y - (sp.sin(t) + (kk - 1)*t)) == 0  # (1)
    spd2 = sp.expand(sp.simplify(vx**2 + vy**2))
    assert sp.simplify(spd2 - (1 + (kk-1)**2 + 2*(kk-1)*sp.cos(t))) == 0
    # (2)(3) 비 = 3 인 k 를 수치 탐색으로 독립 확인
    found = []
    import numpy as np
    def ratio(kv):
        ts = np.linspace(0, 2*np.pi, 20001)
        sv = np.sqrt(1 + (kv-1)**2 + 2*(kv-1)*np.cos(ts))
        return sv.max() / sv.min() if sv.min() > 1e-12 else np.inf
    ks = np.linspace(0.0137, 10.0137, 99901)  # 격자가 근에 정확히 걸리지 않게 이동
    r = np.array([ratio(x) for x in ks[::50]])
    g = r - 3
    ks2 = ks[::50]
    for i in range(len(g)-1):
        if np.isfinite(g[i]) and np.isfinite(g[i+1]) and g[i]*g[i+1] < 0:
            from scipy.optimize import brentq
            found.append(round(brentq(lambda x: ratio(x) - 3, ks2[i], ks2[i+1]), 3))
    assert found == [0.5, 1.5, 3.0], found
    # 기호 확인: 최대 1+|k-1|, 최소 |1-|k-1||
    for kv, mx, mn in [(sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(1, 2)),
                       (sp.Rational(3, 2), sp.Rational(3, 2), sp.Rational(1, 2)),
                       (3, 3, 1)]:
        e = sp.sqrt(spd2.subs(kk, kv))
        assert sp.simplify(e.subs(t, 0) - max(mx, mn)) == 0 or sp.simplify(e.subs(t, sp.pi) - max(mx, mn)) == 0
        assert sp.nsimplify(max(e.subs(t, 0), e.subs(t, sp.pi)) / min(e.subs(t, 0), e.subs(t, sp.pi))) == 3
    return "D01-06", "속력²=1+(k-1)²+2(k-1)cos t, 최대/최소=3 ⇔ k∈{1/2, 3/2, 3}"


def p07():
    X = 2*sp.sin(t); Y = sp.sin(2*t)
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    vert = [r for r in sp.solveset(sp.Eq(vx, 0), t, sp.Interval.Ropen(0, 2*sp.pi)) if vy.subs(t, r) != 0]
    hor = [r for r in sp.solveset(sp.Eq(vy, 0), t, sp.Interval.Ropen(0, 2*sp.pi)) if vx.subs(t, r) != 0]
    assert len(vert) == 2 and len(hor) == 4  # (1)
    c = sp.symbols('c', real=True)
    spd2 = sp.simplify(vx**2 + vy**2)
    g = 4*(4*c**2 - 3*c + 1)
    import random
    for _ in range(20):
        tv = random.uniform(0, 6.28)
        assert close(spd2.subs(t, tv), g.subs(c, sp.cos(tv)**2))
    cm = sp.solve(sp.diff(g, c), c)[0]
    assert cm == sp.Rational(3, 8) and sp.sqrt(g.subs(c, cm)) == sp.sqrt(7)/2  # (2) 최소
    assert sp.sqrt(g.subs(c, 1)) == 2*sp.sqrt(2)
    # (3) √2 ≤ 속력 ≤ 2 인 시간 총 길이 = π (수치 적분)
    f = sp.lambdify(t, sp.sqrt(spd2), 'numpy')
    import numpy as np
    N = 2_000_000
    ts = (np.arange(N) + 0.5) * 2*np.pi / N
    sv = f(ts)
    meas = np.count_nonzero((sv >= np.sqrt(2)) & (sv <= 2)) * 2*np.pi / N
    assert abs(meas - np.pi) < 1e-4, meas
    # 기호 확인: 구간 [π/6,π/4] 등
    # 기호 확인: 2 ≤ 속력² ≤ 4 ⇔ c∈[0,1/4]∪[1/2,3/4]
    S = sp.Intersection(sp.solveset(g >= 2, c, sp.Interval(0, 1)), sp.solveset(g <= 4, c, sp.Interval(0, 1)))
    assert S == sp.Union(sp.Interval(0, sp.Rational(1, 4)), sp.Interval(sp.Rational(1, 2), sp.Rational(3, 4)))
    return "D01-07", "수직 2회·수평 4회, 속력 최소 √7/2·최대 2√2, √2≤속력≤2 인 시간 총합 π"


def p08():
    P = (sp.cos(t), sp.sin(t)); Q = (2*sp.cos(2*t), 2*sp.sin(2*t))
    d2 = sp.simplify(sp.expand_trig((Q[0]-P[0])**2 + (Q[1]-P[1])**2))
    assert sp.simplify(d2 - (5 - 4*sp.cos(t))) == 0  # (1)
    fp = sp.diff(sp.sqrt(5 - 4*sp.cos(t)), t)
    assert sp.simplify(fp - 2*sp.sin(t)/sp.sqrt(5 - 4*sp.cos(t))) == 0
    crit = [r for r in sp.solveset(sp.diff(fp, t), t, sp.Interval.open(0, sp.pi))]
    assert crit == [sp.pi/3], crit
    assert sp.simplify(fp.subs(t, sp.pi/3)) == 1  # (2) 최댓값 1
    # (3) t=π/3 에서 ∠OPQ = π/2: PO·PQ = 0
    Pv = sp.Matrix([sp.cos(sp.pi/3), sp.sin(sp.pi/3)]); Qv = sp.Matrix([2*sp.cos(2*sp.pi/3), 2*sp.sin(2*sp.pi/3)])
    assert sp.simplify((-Pv).dot(Qv - Pv)) == 0
    return "D01-08", "|PQ|²=5-4cos t, (d/dt)|PQ| 최대 1 (t=π/3), 그때 ∠OPQ=π/2"


def p09():
    a, b = sp.symbols('a b', real=True)
    ax, ay = 6*t, -4
    def pos(p0, v0):
        vx = v0[0] + sp.integrate(ax, (t, 0, t)); vy = v0[1] + sp.integrate(ay, (t, 0, t))
        return (p0[0] + sp.integrate(vx, (t, 0, t)), p0[1] + sp.integrate(vy, (t, 0, t)))
    P = pos((0, 0), (0, 6))
    assert sp.simplify(P[0] - t**3) == 0 and sp.simplify(P[1] - (6*t - 2*t**2)) == 0  # (1)
    Q = pos((a, b), (2, 2))
    D = (sp.expand(P[0] - Q[0]), sp.expand(P[1] - Q[1]))
    assert sp.degree(D[0], t) == 1 and sp.degree(D[1], t) == 1  # 차가 일차식
    sol = sp.solve([D[0], D[1]], [a, t], dict=True)
    assert sol == [{a: -b/2, t: b/4}]  # (2) b=-2a, a<0
    Q2 = [x.subs({a: -4, b: 3}) for x in Q]
    dist2 = sp.expand((P[0]-Q2[0])**2 + (P[1]-Q2[1])**2)
    assert dist2 == 20*t**2 - 40*t + 25
    tm = sp.solve(sp.diff(dist2, t), t)
    assert tm == [1] and sp.sqrt(dist2.subs(t, 1)) == sp.sqrt(5)  # (3)
    # 만나지 않음 확인
    assert sp.solve([D[0].subs({a: -4, b: 3}), D[1].subs({a: -4, b: 3})], t) == []
    return "D01-09", "P=(t³,6t-2t²), P-Q 일차식 ⇒ 만남 ⇔ b=-2a(a<0), (a,b)=(-4,3)이면 최소거리 √5 (t=1)"


def p10():
    n = sp.symbols('n', integer=True, positive=True)
    def curve(nv):
        return ((nv-1)*sp.cos(t) + sp.cos((nv-1)*t), (nv-1)*sp.sin(t) - sp.sin((nv-1)*t))
    for nv in range(3, 9):
        X, Y = curve(nv)
        spd2 = sp.diff(X, t)**2 + sp.diff(Y, t)**2
        target = 4*(nv-1)**2*sp.sin(nv*t/2)**2
        import random
        for _ in range(10):
            tv = random.uniform(0, 6.28)
            assert close(spd2.subs(t, tv), target.subs(t, tv))  # (1)(2)
        zeros = sp.solveset(sp.sin(nv*t/2), t, sp.Interval.open(0, 2*sp.pi))
        assert len(list(zeros)) == nv - 1
        L = mp.quad(lambda x: 2*(nv-1)*abs(mp.sin(nv*x/2)), [mp.mpf(0)] + [2*mp.pi*j/nv for j in range(1, nv+1)])
        assert abs(L - 8*(nv-1)) < 1e-9  # (3)
    nmin = min(m for m in range(3, 100) if 8*(m-1) > 50)
    assert nmin == 8
    return "D01-10", "속력 2(n-1)|sin(nt/2)|, 정지 n-1회, L(n)=8(n-1) (n=3..8 수치적분), L(n)>50 최소 n=8"


def main():
    ok = 0
    funcs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]
    for i, f in enumerate(funcs, 1):
        try:
            pid, msg = f()
            print(f"{pid} PASS  {msg}")
            ok += 1
        except Exception as e:
            print(f"D01-{i:02d} FAIL  {type(e).__name__}: {e}")
    print(f"SUMMARY T01: {ok}/{len(funcs)} PASS")
    if ok != len(funcs):
        sys.exit(1)


if __name__ == '__main__':
    main()
