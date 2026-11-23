import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-M23", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "n": n, "k": k})

def p01():
    f = lambda z: 3*sp.exp(-z)
    def tri(kv):
        slope = sp.diff(f(x), x).subs(x, kv)
        px = sp.solve(sp.Eq(f(kv) + slope*(x - kv), 0), x)[0]      # 접선의 x절편
        return px, sp.Rational(1, 2)*abs(px - kv)*f(kv)
    P1, A1 = tri(1)
    V.check(1, "1-1 P_x=2", P1, 2); V.check(1, "1-1 넓이", A1, E(V.ans(1, "1-1")))
    # 1-2: 각 k 에 대해 P_k=k+1, A_k 를 직접 구해 합산(n=7) 후 닫힌 식과 비교
    N = 7
    ok = all(sp.simplify(tri(kv)[0] - (kv + 1)) == 0 for kv in range(1, N + 1))
    V.report(1, ok, "1-2 P_k=(k+1,0) (k=1..7)")
    S = sum(tri(kv)[1] for kv in range(1, N + 1))
    V.check(1, "1-2 S_7", S, E(V.ans(1, "1-2")).subs(n, N))
    # 1-3: 극한
    nn = sp.Symbol("nn", positive=True)
    V.check(1, "1-3 합", sp.limit(E(V.ans(1, "1-2")).subs(n, nn), nn, sp.oo), E(V.ans(1, "1-3")))
    V.check(1, "1-3 급수 직접", sp.summation(sp.Rational(3, 2)*sp.exp(-k), (k, 1, sp.oo)), E(V.ans(1, "1-3")))

def p02():
    a = sp.Symbol("a", real=True)
    f = 2*sp.sin(x) + 2
    d = lambda av: f - 2*sp.cos(x - av)
    # 접함: f-g_a 의 최솟값 = 0 (f>=g_a 가 한 점 이상에서 접촉). 합성 진폭 R(a)=2*sqrt(2-2 sin a)
    # 수치로 접촉 확인: min(f-g_a)=0
    import numpy as np
    xs = np.linspace(-np.pi, 3*np.pi, 400001)
    for av in (sp.pi/6, 5*sp.pi/6):
        mn = float(np.min(sp.lambdify(x, d(av), "numpy")(xs)))
        V.report(2, abs(mn) < 1e-9, f"2-1 a={av} 에서 min(f-g)={mn:.2e} (접촉)")
    # 그림 조건: g 의 최솟점 a+pi 가 f 의 최댓점 pi/2 보다 pi 미만 오른쪽(0<a<pi/2) → pi/6
    V.check(2, "2-1 a(그림)", sp.pi/6, E(V.ans(2, "2-1")))
    av = E(V.ans(2, "2-1"))
    # 2-2: 접점(f-g=0, 구간 [-pi,0])
    x0 = [s for s in sp.solve(sp.Eq(sp.expand(sp.expand_trig(d(av))), 0), x) if -sp.pi <= s <= 0]
    V.check(2, "2-2 접점 x0", x0[0], -sp.pi/6)
    tang = sp.diff(f, x).subs(x, x0[0])*(x - x0[0]) + f.subs(x, x0[0])
    V.check(2, "2-2 접선", sp.simplify(tang - E(V.ans(2, "2-2"))), 0)
    # 접선이 g_a 에도 접하는지(같은 점, 같은 기울기)
    g = 2*sp.cos(x - av)
    V.report(2, sp.simplify(g.subs(x, x0[0]) - f.subs(x, x0[0])) == 0 and sp.simplify(sp.diff(g, x).subs(x, x0[0]) - sp.diff(f, x).subs(x, x0[0])) == 0, "2-2 접점에서 g 와 값·기울기 일치")
    # 2-3: 접점 사이 한 주기 적분, 수치 적분과 비교
    A = sp.integrate(sp.expand(sp.expand_trig(d(av))), (x, x0[0], x0[0] + 2*sp.pi))
    V.check(2, "2-3 넓이", sp.simplify(A), E(V.ans(2, "2-3")))
    V.report(2, float(np.min(sp.lambdify(x, d(av), "numpy")(np.linspace(float(x0[0]), float(x0[0]) + 2*np.pi, 20001)))) > -1e-9, "2-3 구간에서 f>=g")

def p03():
    c = sp.Symbol("c", positive=True)
    cc = sp.solve(sp.Eq(-c*16 + 4, 0), c)[0]
    y = 4 - cc*x**2
    V.check(3, "3-1 포물선 계수", cc, sp.Rational(1, 4))
    T = sp.integrate(y, (x, 0, 4)); V.check(3, "3-1 넓이", T, E(V.ans(3, "3-1")))
    p = sp.Symbol("p", positive=True)
    Aa = sp.integrate(y - y.subs(x, p), (x, 0, p))
    ps = [s for s in sp.solve(sp.Eq(Aa, T/2), p) if s.is_real and 0 < s < 4]
    V.check(3, "3-2 p", ps[0], E(V.ans(3, "3-2", "p")))
    V.check(3, "3-2 y", y.subs(x, ps[0]), E(V.ans(3, "3-2", "y")))
    # B 를 직접(직사각형 + 포물선 아래)으로 계산해 A=B 확인
    Bb = ps[0]*y.subs(x, ps[0]) + sp.integrate(y, (x, ps[0], 4))
    V.check(3, "3-2 A=B", sp.simplify(Aa.subs(p, ps[0]) - Bb), 0)

if __name__ == "__main__": V.run(globals())
