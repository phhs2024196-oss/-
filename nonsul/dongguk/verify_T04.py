#!/usr/bin/env python3
"""D04 역함수·음함수·매개변수로 나타낸 함수의 미분과 접선 — sympy 검산.
각 문항의 설정으로부터 독립적으로 다시 계산해 md의 정답과 비교한다."""
import sys
import numpy as np
import sympy as sp
import mpmath as mp

x, y, t, s, u, a, k = sp.symbols('x y t s u a k', real=True)
E, pi = sp.E, sp.pi
results = []


def report(pid, ok, msg):
    ok = bool(ok)
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def inv_num(f, yv, lo=-50, hi=50):
    """증가함수 f의 역함수 값을 이분법으로(고정밀)."""
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    for _ in range(int(mp.mp.prec) + 20):
        mid = (lo + hi)/2
        if f(mid) < yv:
            lo = mid
        else:
            hi = mid
    return (lo + hi)/2


# ---------------- D04-01 ----------------
def p01():
    f = lambda v: v + mp.e**v
    g = lambda v: inv_num(f, v)
    mp.mp.dps = 40
    g1 = g(1)
    gp = mp.diff(g, 1)
    gpp = mp.diff(g, 1, 2)
    ok1 = abs(g1) < 1e-30 and abs(gp - mp.mpf(1)/2) < 1e-15
    ok2 = abs(gpp + mp.mpf(1)/8) < 1e-10
    # g''<0 전 구간 (위로 볼록) → 접선 위에 곡선 없음
    ok3 = all(g(xx) <= (xx - 1)/2 + 1e-30 for xx in [mp.mpf(v) for v in np.linspace(-5, 30, 71)])
    # 접선 y=(x-1)/2, 법선 y=-2(x-1), y축과 삼각형
    tan_b = sp.Rational(-1, 2); nor_b = 2
    area = sp.Rational(1, 2)*(nor_b - tan_b)*1
    ok3 &= area == sp.Rational(5, 4)
    report("D04-01", ok1 and ok2 and ok3, f"g'(1)={mp.nstr(gp,12)}, g''(1)={mp.nstr(gpp,10)}, 곡선≤접선, 넓이 {area}")


# ---------------- D04-02 ----------------
def p02():
    X = t**2 - 1; Y = t**3 - 4*t
    dydx = sp.simplify(sp.diff(Y, t)/sp.diff(X, t))
    d2 = sp.simplify(sp.diff(dydx, t)/sp.diff(X, t))
    ok1 = sp.simplify(dydx - (3*t**2 - 4)/(2*t)) == 0 and sp.simplify(d2 - (3*t**2 + 4)/(4*t**3)) == 0
    # x(t1)=x(t2), t1≠t2 ⇒ t2=-t1, 그때 y(t1)=y(-t1)
    t1 = sp.symbols('t1', real=True)
    cand = sp.solve(sp.Eq(Y.subs(t, t1), Y.subs(t, -t1)), t1)
    cand = [c for c in cand if c != 0]
    pts = {(X.subs(t, c), Y.subs(t, c)) for c in cand}
    # 전수 수치 확인: 격자에서 서로 다른 매개변수로 같은 점이 되는 경우
    ts = np.linspace(-4, 4, 1601)
    P = np.stack([ts**2 - 1, ts**3 - 4*ts], 1)
    dup = set()
    for i in range(len(ts)):
        dd = np.abs(P[i+1:] - P[i]).sum(1)
        for j in np.where(dd < 1e-9)[0]:
            dup.add((round(P[i][0], 6), round(P[i][1], 6)))
    pts_ok = dup == {(3.0, 0.0)}
    ok2 = pts_ok and pts == {(3, 0)} and {dydx.subs(t, 2), dydx.subs(t, -2)} == {2, -2}
    # 두 접선 y=±2(x-3) 과 y축
    b1 = (2*(x - 3)).subs(x, 0); b2 = (-2*(x - 3)).subs(x, 0)
    area = sp.Rational(1, 2)*abs(b1 - b2)*3
    ok3 = area == 18
    report("D04-02", ok1 and ok2 and ok3, f"dy/dx, d²y/dx² 확인; 자기교점 {pts}, 기울기 ±2; 넓이 {area}")


# ---------------- D04-03 ----------------
def p03():
    X = 2*sp.cos(t); Y = sp.sin(t)
    dX, dY = sp.diff(X, t), sp.diff(Y, t)
    # 법선: dX(x-X)+dY(y-Y)=0
    nl = dX*(x - X) + dY*(y - Y)
    n3 = sp.expand(nl.subs(t, pi/3))
    target = sp.expand(-sp.sqrt(3)*(4*sp.sqrt(3)*x - 2*y - 3*sp.sqrt(3))/4)  # 비례 확인
    ratio = sp.simplify(n3/(y - 2*sp.sqrt(3)*x + 3*sp.sqrt(3)/2))
    ok1 = ratio.free_symbols == set()
    d = sp.Abs(nl.subs({x: 0, y: 0}))/sp.sqrt(dX**2 + dY**2)
    dform = 3*sp.sin(t)*sp.cos(t)/sp.sqrt(1 + 3*sp.sin(t)**2)
    ok2 = all(abs(float((d - dform).subs(t, v))) < 1e-13 for v in [0.1, 0.5, 1.0, 1.4])
    mp.mp.dps = 30
    fd = sp.lambdify(t, dform, 'mpmath')
    tmax = mp.findroot(lambda v: mp.diff(fd, v), 0.6)
    ok3 = abs(fd(tmax) - 1) < 1e-20 and abs(mp.sin(tmax)**2 - mp.mpf(1)/3) < 1e-20
    ok3 &= max(fd(v) for v in mp.linspace(0.001, mp.pi/2 - 0.001, 2000)) <= 1 + 1e-20
    P = (2*mp.cos(tmax), mp.sin(tmax))
    ok3 &= abs(P[0] - 2*mp.sqrt(6)/3) < 1e-20 and abs(P[1] - mp.sqrt(3)/3) < 1e-20
    report("D04-03", ok1 and ok2 and ok3, f"t=π/3 법선 y=2√3x-3√3/2, d(t) 식 확인, 최댓값 1 (sin²t=1/3, P=(2√6/3,√3/3))")


# ---------------- D04-04 ----------------
def p04():
    tp, sp_ = sp.symbols('tp sp_', positive=True)
    # y=ln x 위 (tp, ln tp)의 접선, y=x^2+a 위 (s, s^2+a)의 접선 일치
    sol = sp.solve([sp.Eq(1/tp, 2*s), sp.Eq(sp.log(tp) - 1, -s**2 + a)], [s, a], dict=True)[0]
    phi = sp.simplify(sol[a])
    ok1 = sp.simplify(phi - (1/(4*tp**2) + sp.log(tp) - 1)) == 0
    crit = sp.solve(sp.diff(phi, tp), tp)
    amin = sp.simplify(phi.subs(tp, crit[0]))
    ok2 = crit == [1/sp.sqrt(2)] and sp.simplify(amin + (1 + sp.log(2))/2) == 0

    def ncommon(av):
        f = sp.lambdify(tp, phi - av)
        xs = np.geomspace(1e-4, 1e4, 200001)
        v = f(xs)
        return int(np.sum(np.sign(v[1:]) != np.sign(v[:-1])))
    am = float(amin)
    ok2 &= ncommon(am - 0.1) == 0 and ncommon(am + 0.1) == 2 and ncommon(0) == 2 and ncommon(-5) == 0
    # (3) a=amin: 두 곡선이 x=1/√2에서 접함
    h = x**2 + amin - sp.log(x)
    ok3 = sp.simplify(h.subs(x, 1/sp.sqrt(2))) == 0 and sp.simplify(sp.diff(h, x).subs(x, 1/sp.sqrt(2))) == 0
    ok3 &= sp.simplify(sp.log(1/sp.sqrt(2)) + sp.log(2)/2) == 0
    line = sp.sqrt(2)*x - 1 - sp.log(2)/2
    ok3 &= sp.simplify(line.subs(x, 1/sp.sqrt(2)) - sp.log(1/sp.sqrt(2))) == 0
    report("D04-04", ok1 and ok2 and ok3, f"a=1/(4t²)+ln t-1, 최솟값 {amin}; 공통접선 2/1/0개; 접점 (1/√2,-ln2/2)")


# ---------------- D04-05 ----------------
def p05():
    f = lambda v: 2*v + mp.sin(v)
    g = lambda v: inv_num(f, v, -100, 100)
    mp.mp.dps = 40
    ok1 = abs(g(2*mp.pi) - mp.pi) < 1e-30 and abs(mp.diff(g, 2*mp.pi) - 1) < 1e-15
    gpp = lambda v: mp.sin(g(v))/(2 + mp.cos(g(v)))**3
    ok2 = abs(mp.diff(g, 2*mp.pi, 2)) < 1e-8 and abs(mp.diff(g, 5, 2) - gpp(5)) < 1e-8
    ok2 &= gpp(2*mp.pi - 0.5) > 0 and gpp(2*mp.pi + 0.5) < 0
    area = mp.quad(lambda v: g(v) - (v - mp.pi), [0, mp.pi, 2*mp.pi])
    ok3 = abs(area - (mp.pi**2 - 2)) < 1e-15
    ok3 &= all(g(v) >= v - mp.pi - 1e-30 for v in mp.linspace(0, 2*mp.pi, 200))
    # 기호: ∫_0^{2π} g = 2π·π - ∫_0^π f
    I = 2*pi*pi - sp.integrate(2*y + sp.sin(y), (y, 0, pi))
    ok3 &= sp.simplify(I - (pi**2 - 2)) == 0
    report("D04-05", ok1 and ok2 and ok3, f"g'(2π)=1, 변곡점 (2π,π), 넓이={mp.nstr(area,15)} = π²-2")


# ---------------- D04-06 ----------------
def p06():
    F = x**2 - x*y + y**2 - 3
    yp = sp.simplify(-sp.diff(F, x)/sp.diff(F, y))
    ok1 = sp.simplify(yp - (2*x - y)/(x - 2*y)) == 0
    hor = sp.solve([F, 2*x - y], [x, y], dict=True)
    ok1 &= {(d[x], d[y]) for d in hor} == {(1, 2), (-1, -2)}
    x0, y0 = sp.symbols('x0 y0', real=True)
    tl = (2*x0 - y0)*(x - x0) + (2*y0 - x0)*(y - y0)
    pts = sp.solve([F.subs({x: x0, y: y0}), tl.subs({x: 2, y: 2})], [x0, y0], dict=True)
    P = {(d[x0], d[y0]) for d in pts}
    ok2 = P == {(1, 2), (2, 1)}
    lines = {sp.factor(sp.expand(tl.subs({x0: d[x0], y0: d[y0]}))) for d in pts}
    ok2 &= lines == {sp.factor(3*y - 6), sp.factor(3*x - 6)}
    # (3) 법선이 원점을 지나는 점: 접선 방향 (x-2y, -(2x-y)) ⟂ (x,y)
    cond = x*(x - 2*y) + y*(2*x - y)  # x + y*y' = 0 의 분모 정리
    nn = sp.solve([F, cond], [x, y], dict=True)
    NP = {(sp.nsimplify(d[x]), sp.nsimplify(d[y])) for d in nn}
    r3 = sp.sqrt(3)
    ok3 = NP == {(r3, r3), (-r3, -r3), (1, -1), (-1, 1)}
    dmax = max(sp.sqrt(p[0]**2 + p[1]**2) for p in NP)
    # 수치: 곡선 위 원점까지 최대거리
    th = np.linspace(0, 2*np.pi, 200001)
    rr = np.sqrt(3/(1 - np.cos(th)*np.sin(th)))
    ok3 &= sp.simplify(dmax - sp.sqrt(6)) == 0 and abs(rr.max() - np.sqrt(6)) < 1e-9
    report("D04-06", ok1 and ok2 and ok3, f"수평접선 (1,2),(-1,-2); (2,2)에서 접선 y=2, x=2; 법선 원점 통과 4점, 최대거리 {dmax}")


# ---------------- D04-07 ----------------
def p07():
    X = t - sp.sin(t); Y = 1 - sp.cos(t)
    dydx = sp.diff(Y, t)/sp.diff(X, t)
    d2 = sp.diff(dydx, t)/sp.diff(X, t)
    ok1 = all(abs(float((dydx - sp.cos(t/2)/sp.sin(t/2)).subs(t, v))) < 1e-12 for v in [0.3, 1, 2, 3, 5])
    ok1 &= all(abs(float((d2 + 1/(1 - sp.cos(t))**2).subs(t, v))) < 1e-10 for v in [0.3, 1, 2, 3, 5])
    # 법선이 (t,0), 접선이 (t,2) 지남
    nl = sp.diff(X, t)*(x - X) + sp.diff(Y, t)*(y - Y)
    tl = sp.diff(Y, t)*(x - X) - sp.diff(X, t)*(y - Y)
    ok2 = sp.simplify(nl.subs({x: t, y: 0})) == 0 and sp.simplify(tl.subs({x: t, y: 2})) == 0
    # (3) t=2π/3: 접선·법선·x축 삼각형
    tv = 2*pi/3
    Px, Py = X.subs(t, tv), Y.subs(t, tv)
    m = dydx.subs(t, tv)
    T = sp.nsimplify(sp.simplify(Px - Py/m))
    N = tv
    area = sp.simplify(sp.Rational(1, 2)*sp.Abs(N - T)*Py)
    ok3 = sp.simplify(area - 3*sp.sqrt(3)/2) == 0 and sp.simplify(T - (tv - 2*sp.tan(tv/2))) == 0
    report("D04-07", ok1 and ok2 and ok3, f"dy/dx=cot(t/2), d²y/dx²=-1/(1-cos t)²; 법선 (t,0)·접선 (t,2) 통과; 넓이 {area}")


# ---------------- D04-08 ----------------
def p08():
    ok = True
    msgs = []
    for n in range(1, 9):
        f = lambda v, n=n: v**(2*n + 1) + v
        g = lambda v, f=f: inv_num(f, v, -10, 10)
        mp.mp.dps = 40
        gp = mp.diff(g, 2)
        ok &= abs(gp - mp.mpf(1)/(2*n + 2)) < 1e-15
        I = mp.quad(g, [0, 2])
        ok &= abs(I - (mp.mpf(3)/2 - mp.mpf(1)/(2*n + 2))) < 1e-15
        S = mp.quad(lambda v: 1 + (v - 2)/(2*n + 2) - g(v), [0, 2])
        ok &= abs(S - mp.mpf(n)/(2*(n + 1))) < 1e-15
        ok &= all(g(v) <= 1 + (v - 2)/(2*n + 2) + 1e-30 for v in mp.linspace(0, 2, 41))
    # 기호 일반 n
    n = sp.symbols('n', positive=True, integer=True)
    Ig = 2 - sp.integrate(y**(2*n + 1) + y, (y, 0, 1))
    Sn = sp.simplify(sp.integrate(1 + (x - 2)/(2*n + 2), (x, 0, 2)) - Ig)
    ok &= sp.simplify(Sn - n/(2*(n + 1))) == 0 and sp.limit(Sn, n, sp.oo) == sp.Rational(1, 2)
    report("D04-08", ok, f"n=1..8: g_n'(2)=1/(2n+2), ∫g_n=3/2-1/(2n+2), S_n={Sn}, 극한 1/2")


# ---------------- D04-09 ----------------
def p09():
    kp = sp.symbols('kp', positive=True)
    p = sp.symbols('p', positive=True)
    # 공통점 (p,p), 기울기 일치: k e^{kp} = 1/(kp), e^{kp}=p
    sol = sp.solve([sp.exp(kp*p) - p, kp*sp.exp(kp*p) - 1/(kp*p)], [kp, p], dict=True)
    ok2 = any(sp.simplify(d[kp] - 1/E) == 0 and sp.simplify(d[p] - E) == 0 for d in sol)
    # k=1/e 에서 e^{x/e} >= x, 등호 x=e 하나
    h = sp.lambdify(x, sp.exp(x/E) - x)
    xs = np.linspace(-5, 40, 90001)
    hv = h(xs)
    ok2 &= hv.min() > -1e-12 and abs(xs[hv.argmin()] - np.e) < 1e-3
    # (1) 증가함수 f에 대해 f(p)=q, f(q)=p ⇒ p=q : k 여러 값에서 교점 수치 확인
    ok1 = True
    for kv in [0.1, 0.2, 0.3, 1/np.e]:
        F = lambda v: np.exp(kv*v) - np.log(v)/kv
        vv = np.linspace(1e-3, 200, 400001)
        Fv = F(vv)
        roots = vv[np.where(np.sign(Fv[1:]) != np.sign(Fv[:-1]))[0]]
        ok1 &= all(abs(np.exp(kv*r) - r) < 1e-2 for r in roots)
    # (3) 넓이 = ∫_0^e e^{x/e} dx - ∫_1^e e ln x dx
    A = sp.integrate(sp.exp(x/E), (x, 0, E)) - sp.integrate(E*sp.log(x), (x, 1, E))
    ok3 = sp.simplify(A - (E**2 - 2*E)) == 0
    # 영역 직접 수치: 격자에서 y ≤ f(x) 이고 (x≤1 또는 y ≥ g(x))
    mp.mp.dps = 20
    Anum = mp.quad(lambda X: mp.e**(X/mp.e) - (mp.e*mp.log(X) if X > 1 else 0), [0, 1, mp.e])
    ok3 &= abs(Anum - (mp.e**2 - 2*mp.e)) < 1e-12
    report("D04-09", ok1 and ok2 and ok3, f"교점은 y=x 위; 접할 조건 k=1/e, 접점 (e,e); 넓이 {sp.simplify(A)}")


# ---------------- D04-10 ----------------
def p10():
    up = sp.symbols('up', positive=True)
    # y=e^x 의 (s,e^s) 접선과 y=ln x 의 (up, ln up) 접선 일치
    sol = sp.solve([sp.Eq(sp.exp(s), 1/up), sp.Eq(sp.exp(s)*(1 - s), sp.log(up) - 1)], [up], dict=True)
    phi = (s - 1)*sp.exp(s) - s - 1
    cond = sp.simplify((sp.exp(s)*(1 - s) - (sp.log(sp.exp(-s)) - 1)))
    ok1 = sp.simplify(cond + phi) == 0
    ok2 = sp.simplify(phi.subs(s, -s) - sp.exp(-s)*phi) == 0
    ok2 &= sp.N(phi.subs(s, -2)) > 0 and phi.subs(s, 0) == -2 and sp.N(phi.subs(s, 2)) > 0
    f = sp.lambdify(s, phi)
    ss = np.linspace(-40, 40, 800001)
    v = f(ss)
    nroot = int(np.sum(np.sign(v[1:]) != np.sign(v[:-1])))
    ok2 &= nroot == 2
    mp.mp.dps = 30
    r = mp.findroot(lambda q: (q - 1)*mp.e**q - q - 1, 1.5)
    ok3 = abs((-r - 1)*mp.e**(-r) + r - 1) < 1e-25  # -r 도 근
    slopes = (mp.e**r, mp.e**(-r))
    ok3 &= abs(slopes[0]*slopes[1] - 1) < 1e-25
    # 두 접선 교점이 y=x 위
    l1 = lambda X: mp.e**r*X + mp.e**r*(1 - r)
    l2 = lambda X: mp.e**(-r)*X + mp.e**(-r)*(1 + r)
    Xi = (mp.e**(-r)*(1 + r) - mp.e**r*(1 - r))/(mp.e**r - mp.e**(-r))
    ok3 &= abs(l1(Xi) - Xi) < 1e-25 and abs(l2(Xi) - Xi) < 1e-25
    report("D04-10", ok1 and ok2 and ok3, f"조건 φ(s)=(s-1)e^s-s-1=0, φ(-s)=e^(-s)φ(s), 근 2개(±{mp.nstr(r,10)}); 기울기 곱 1, 교점 y=x 위")


for p in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        p()
    except Exception as e:  # noqa
        report(p.__name__, False, f"예외 {e!r}")

n = sum(results)
print(f"SUMMARY T04: {n}/{len(results)} PASS")
sys.exit(0 if n == len(results) == 10 else 1)
