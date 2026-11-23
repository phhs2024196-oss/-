import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G25A", "items/YSM/PAST")
th, tt = sp.symbols("theta tt", real=True)
E = lambda v: sp.sympify(v, locals={"x": x, "t": t, "n": n, "k": k})

def p01():
    # 1-1: 5등분, 그림1(오른쪽 끝점 k=1..5), 그림2(왼쪽 끝점 k=0..4)
    f = lambda z: z**2
    A1 = sum(sp.Rational(1, 5)*f(sp.Rational(i, 5)) for i in range(1, 6))
    A2 = sum(sp.Rational(1, 5)*f(sp.Rational(i, 5)) for i in range(0, 5))
    V.check(1, "1-1 A1", A1, E(V.ans(1, "1-1", "A1")))
    V.check(1, "1-1 A2", A2, E(V.ans(1, "1-1", "A2")))
    # 1-2: n=100 을 직접 합산하고, 공식 n(n+1)(2n+1)/(6n^3) 과도 일치 확인
    N = 100
    B1 = sum(sp.Rational(1, N)*sp.Rational(i, N)**2 for i in range(1, N + 1))
    B2 = sum(sp.Rational(1, N)*sp.Rational(i, N)**2 for i in range(0, N))
    V.check(1, "1-2 A1", B1, E(V.ans(1, "1-2", "A1")))
    V.check(1, "1-2 A2", B2, E(V.ans(1, "1-2", "A2")))
    V.check(1, "1-2 A1 공식", sp.Rational(N*(N+1)*(2*N+1), 6*N**3), B1)
    V.check(1, "1-2 A2 공식", sp.Rational((N-1)*N*(2*N-1), 6*N**3), B2)
    # 두 합은 S=1/3 로 수렴(제시문): n 크게
    nn = sp.symbols("nn", positive=True)
    V.check(1, "1-2 극한 1/3", sp.limit(nn*(nn+1)*(2*nn+1)/(6*nn**3), nn, sp.oo), sp.Rational(1, 3))
    # 1-3
    V.check(1, "1-3", sp.integrate(x*sp.cos(x**2), (x, 0, sp.sqrt(sp.pi/2))), E(V.ans(1, "1-3")))
    V.report(1, all(float(xv*sp.cos(xv**2)) >= -1e-12 for xv in [i*float(sp.sqrt(sp.pi/2))/200 for i in range(201)]),
             "1-3 구간에서 y>=0(넓이=정적분)")

def p02():
    # 2-1: 켤레근 → a,b 를 연립으로 구한다
    a, b, x1 = sp.symbols("a b x1", real=True)
    poly = sp.expand((x - (1 + sp.I))*(x - (1 - sp.I))*(x - x1))
    eqs = [sp.Eq(poly.coeff(x, 2), a), sp.Eq(poly.coeff(x, 1), b), sp.Eq(poly.coeff(x, 0), -4)]
    sol = sp.solve(eqs, [a, b, x1], dict=True)[0]
    f = x**3 + sol[a]*x**2 + sol[b]*x - 4
    V.report(2, sp.simplify(f.subs(x, 1 + sp.I)) == 0, f"2-1 f(1+i)=0 (a={sol[a]}, b={sol[b]})")
    V.check(2, "2-1 실근", sol[x1], 2)
    rts = sp.Poly(f, x).nroots()
    V.report(2, sum(1 for r in rts if abs(sp.im(r)) < 1e-9) == 1, "2-1 실근 1개")
    V.report(2, all(f.subs(x, sp.Rational(i, 10)) <= 0 for i in range(0, 21)), "2-1 [0,2]에서 f<=0")
    V.check(2, "2-1 넓이", sp.integrate(-f, (x, 0, 2)), E(V.ans(2, "2-1")))
    # 2-2: f'=0 이 서로 다른 두 실근 ⇔ D>0, 양의 정수 c 최솟값
    c = sp.symbols("c", real=True)
    D = sp.discriminant(sp.diff(x**3 + c*x**2 + 100*x + 100, x), x)
    cmin = next(cv for cv in range(1, 100) if D.subs(c, cv) > 0)
    V.check(2, "2-2 c 최솟값", cmin, E(V.ans(2, "2-2")))
    V.report(2, D.subs(c, cmin - 1) <= 0, "2-2 c-1 은 조건 불만족")

def p03():
    th = sp.symbols("th", real=True)
    x1 = sp.integrate(th*sp.sin(th), (th, 0, t)); y1 = sp.integrate(th*sp.cos(th), (th, 0, t))
    V.check(3, "3-1 x1", x1, -t*sp.cos(t) + sp.sin(t))
    V.check(3, "3-1 y1", y1, t*sp.sin(t) + sp.cos(t) - 1)
    s2 = sp.expand(x1**2 + y1**2)
    s = sp.sqrt(s2)
    V.check(3, "3-1 거리", sp.simplify(s2 - sp.sympify(V.ans(3, "3-1"), locals={"t": t})**2), 0)
    # 3-2
    ds = sp.diff(s, t)
    tn = E(V.ans(3, "3-2", "t_n")); t0 = E(V.ans(3, "3-2", "t_0"))
    V.check(3, "3-2 (s^2)' = 2t(1-cos t)", sp.diff(s2, t), 2*t*(1 - sp.cos(t)))
    V.report(3, all(abs(float(ds.subs(t, tn.subs(n, i)))) < 1e-9 for i in range(1, 6)), "3-2 t=2nπ 에서 s'=0 (n=1..5)")
    V.report(3, abs(float(sp.limit(ds, t, 0, "+"))) < 1e-9 and t0 == 0, "3-2 t=0 에서 s'→0")
    xs = [v for v in [i*0.001 for i in range(1, 13000)] if abs(float(sp.diff(s2, t).subs(t, v))) < 1e-5 and v > 0.1]
    V.report(3, all(min(abs(v - 2*sp.pi.evalf()*j) for j in range(1, 3)) < 0.05 for v in xs[:50]) or not xs, "3-2 (0,13] 에서 다른 영점 없음")
    # 3-3
    dx, dy = sp.diff(x1, t), sp.diff(y1, t)
    V.check(3, "3-3 dx/dt", sp.simplify(dx - t*sp.sin(t)), 0)
    V.check(3, "3-3 dy/dt", sp.simplify(dy - t*sp.cos(t)), 0)
    V.check(3, "3-3 이동거리", sp.integrate(sp.sqrt(sp.simplify(dx**2 + dy**2)).subs(sp.sqrt(t**2), t), (t, 0, 2*sp.pi)), E(V.ans(3, "3-3")))
    V.check(3, "3-3 수치적분", sp.N(sp.Integral(sp.sqrt(sp.expand(dx**2 + dy**2)), (t, 0, 2*sp.pi)), 15), sp.N(E(V.ans(3, "3-3")), 15), 1e-8)
    # 3-4
    x2, y2 = -t*sp.cos(t), t*sp.sin(t)
    d2 = sp.simplify((x1 - x2)**2 + (y1 - y2)**2)
    V.check(3, "3-4 d^2", d2, 2 - 2*sp.cos(t))
    d = sp.sqrt(2 - 2*sp.cos(t))
    grid = [(i*2*sp.pi/2000).evalf() for i in range(0, 2001)]
    vals = [float(d.subs(t, g)) for g in grid]
    tmax = grid[vals.index(max(vals))]
    V.check(3, "3-4 최대 시각", sp.N(tmax, 6), sp.N(E(V.ans(3, "3-4", "t")), 6), 1e-3)
    V.check(3, "3-4 최대 거리", sp.simplify(d.subs(t, sp.pi)), E(V.ans(3, "3-4", "d")))
    V.report(3, sp.simplify(sp.diff(sp.sin(t)/sp.sqrt(2 - 2*sp.cos(t)), t).subs(t, sp.pi)) == sp.Rational(-1, 2), "3-4 이계도함수 -1/2")

if __name__ == "__main__": V.run(globals())
