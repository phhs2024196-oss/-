import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DDW-G25", "items/DDW/PAST")
x, a, n = sp.symbols("x a n", real=True)


def p01():
    # 1-1: x^2=a/(x+3a) 가 서로 다른 두 실근(양수 a) 인 a 를 판별식 없이 직접 탐색
    f = x**3 + 3*a*x**2 - a
    cands = [s for s in sp.solve(sp.Eq(f.subs(x, -2*a), 0), a) if s.is_real and s > 0]
    V.check(1, "1-1 극댓값=0 → a", cands[0], sp.sympify(V.ans(1, "1-1")))
    av = sp.sympify(V.ans(1, "1-1"))
    fa = sp.factor(f.subs(a, av))
    V.check(1, "1-1 서로 다른 실근 2개", len(set(sp.real_roots(sp.Poly(f.subs(a, av), x)))), 2)
    V.check(1, "1-1 인수분해 (x+1)^2(x-1/2)", sp.expand(fa - (x + 1)**2*(x - sp.Rational(1, 2))), 0)
    # 1-2: 넓이
    g = av/(x + 3*av)
    A = sp.integrate(g - x**2, (x, -1, sp.Rational(1, 2)))
    V.check(1, "1-2 넓이", A, sp.sympify(V.ans(1, "1-2")))
    # 구간 안에서 g >= x^2 인지 수치 확인
    V.check(1, "1-2 위아래 관계", all(g.subs(x, t) >= t**2 for t in (-1, -0.5, 0, 0.25, 0.5)), True)
    V.check(1, "1-2 수치적분", sp.N(sp.Integral(sp.Abs(g - x**2), (x, -1, sp.Rational(1, 2)))), sp.N(A), 1e-8)


def p02():
    k = sp.symbols("k", integer=True, positive=True)
    da = sp.solve(sp.Eq(5*(2*3 + 4*sp.Symbol("d"))/2, -15), sp.Symbol("d"))[0]
    db = sp.solve(sp.Eq(2*1 + 4*sp.Symbol("d"), 18), sp.Symbol("d"))[0]
    V.check(2, "공차 (da,db)", [da, db], [-3, 4])
    ak = lambda m: 3 + (m - 1)*da
    bk = lambda m: 1 + (m - 1)*db
    dist = sp.sqrt((ak(2) - ak(1))**2 + (bk(2) - bk(1))**2)
    V.check(2, "중심거리", dist, 5)
    # T_n: 반지름 r 인 원 위 점 P 로 삼각형 넓이 최대 — 매개변수 탐색(수치)
    def Tn(m):
        r = sp.Rational(m, 2) + 1
        O1 = sp.Matrix([ak(m), bk(m)]); O2 = sp.Matrix([ak(m + 1), bk(m + 1)])
        th = sp.symbols("th", real=True)
        P = O1 + r*sp.Matrix([sp.cos(th), sp.sin(th)])
        area = sp.Abs((O2 - O1)[0]*(P - O1)[1] - (O2 - O1)[1]*(P - O1)[0])/2
        best = max(sp.N(area.subs(th, sp.pi*j/180), 20) for j in range(0, 360))
        return best
    for m in (1, 2, 5):
        V.check(2, f"T_{m}", Tn(m), sp.N(sp.Rational(5, 4)*m + sp.Rational(5, 2)), 1e-3)
    T = sp.Rational(5, 4)*k + sp.Rational(5, 2)
    S = sp.simplify(sp.summation(T, (k, 1, n)))
    V.check(2, "ΣT_k 식", S, sp.Rational(5, 8)*n*(n + 5))
    ns = [m for m in range(1, 100) if S.subs(n, m) > 65]
    V.check(2, "최솟값 n", ns[0], int(V.ans(2, "2-1")))
    V.check(2, "n=8 경계", S.subs(n, 8), 65)


def p03():
    t = sp.symbols("t", real=True)
    f = lambda u: sp.exp(-u) - sp.exp(-4)
    gf = lambda u: f(u) if f(u) >= 0 else 0
    def h_num(xv):
        xv = sp.nsimplify(xv)
        pts = sorted({xv, xv + 2} | ({sp.Integer(4)} if xv <= 4 <= xv + 2 else set()))
        tot = 0
        for lo, hi in zip(pts, pts[1:]):
            mid = (lo + hi)/2
            tot += sp.integrate(f(t), (t, lo, hi)) if f(mid) >= 0 else 0
        return tot
    H1 = sp.sympify(V.ans(3, "3-1", "H1"), locals={"x": x})
    H2 = sp.sympify(V.ans(3, "3-1", "H2"), locals={"x": x})
    H3 = sp.sympify(V.ans(3, "3-1", "H3"), locals={"x": x})
    for xv in (-3, 0, 1, 2):
        V.check(3, f"x={xv}≤2", sp.N(H1.subs(x, xv), 25), sp.N(h_num(xv), 25))
    for xv in (sp.Rational(5, 2), 3, sp.Rational(7, 2), 4):
        V.check(3, f"x={xv} (2,4]", sp.N(H2.subs(x, xv), 25), sp.N(h_num(xv), 25))
    for xv in (sp.Rational(9, 2), 5, 8):
        V.check(3, f"x={xv}>4", sp.N(H3.subs(x, xv), 25) if H3 != 0 else 0, sp.N(h_num(xv), 25))
    D = h_num(1) - h_num(3) - h_num(5)
    V.check(3, "h(1)-h(3)-h(5)", D, sp.sympify(V.ans(3, "3-1", "D")))
    V.check(3, "조각 연속(x=2)", sp.simplify(H1.subs(x, 2) - H2.subs(x, 2)), 0)
    V.check(3, "조각 연속(x=4)", sp.simplify(H2.subs(x, 4) - H3), 0)

if __name__ == "__main__": V.run(globals())
