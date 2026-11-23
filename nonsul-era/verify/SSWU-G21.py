import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("SSWU-G21", "items/SSWU/PAST")
mp.mp.dps = 30
nn = sp.Symbol("n", positive=True)

def p01():
    # 정n각형을 좌표로 직접 만들어 검산
    def geo(N):
        P = mp.matrix([0, 1]); th = 2*mp.pi/N
        Q = mp.matrix([-mp.sin(th), mp.cos(th)]); R = (P+Q)/2
        area = abs(P[0]*R[1]-P[1]*R[0])/2
        rr = mp.sqrt(R[0]**2+R[1]**2)
        return area, mp.pi*rr**2, 2*mp.pi*rr
    n_ = sp.Symbol("n")
    f1 = sp.sympify(V.ans(1, "1"), locals={"n": n_})
    V.report(1, all(abs(geo(N)[0]-mp.mpf(str(sp.N(f1.subs(n_, N), 30)))) < 1e-20 for N in range(3, 13)), "1-(1) 넓이 n=3..12 좌표 대조")
    a6, b6 = geo(6)[1], geo(6)[2]
    V.check(1, "1-(2)", sp.N(sp.Float(str(b6-2*a6), 30), 20), V.ans(1, "2"), 1e-12)
    x_ = sp.Symbol("x", positive=True)
    lim = sp.limit(sp.pi**2*x_**-2*(2*sp.pi*sp.cos(x_)-2*sp.pi*sp.cos(x_)**2), x_, 0)
    V.check(1, "1-(3) 극한(x=π/n)", lim, V.ans(1, "3"))
    big = mp.mpf(10)**6
    num = big**2*(geo_b := 2*mp.pi*mp.cos(mp.pi/big) - 2*mp.pi*mp.cos(mp.pi/big)**2)
    V.check(1, "1-(3) 수치 n=1e6", sp.Float(str(num), 12), V.ans(1, "3"), 1e-4)

def p02():
    a, b, t_ = sp.symbols("a b t")
    # (가)(다): ∫_1^2 f = F(2)-F(1) = 4 = ∫_1^2 (4-cos πt) → f = 4-cos πx 강제
    up = sp.integrate(4-sp.cos(sp.pi*t_), (t_, 1, 2))
    V.check(2, "∫_1^2(4-cosπt)=F(2)-F(1)", up, 6-2)
    f2 = 4-sp.cos(sp.pi*x)
    V.check(2, "2-(1)", f2.subs(x, sp.Rational(3, 2)), V.ans(2, "1"))
    sb = sp.solve(sp.Eq(b*sp.cos(sp.pi), f2.subs(x, 1)), b)[0]
    sa = sp.solve(sp.Eq(sp.integrate(a*sp.sin(sp.pi*t_)+sb*sp.cos(sp.pi*t_), (t_, 0, 1)), 2), a)[0]
    V.check(2, "2-(2) a", sa, V.ans(2, "2", "a")); V.check(2, "2-(2) b", sb, V.ans(2, "2", "b"))
    f1 = sa*sp.sin(sp.pi*t_)+sb*sp.cos(sp.pi*t_)
    F = lambda u: sp.integrate(f1, (t_, 0, u)) if u <= 1 else 2+sp.integrate(f2.subs(x, t_), (t_, 1, u))
    V.check(2, "2-(3)", F(sp.Rational(1, 2))+F(sp.Rational(3, 2)), V.ans(2, "3"))
    V.check(2, "F(2)=6 일관", F(2), 6)

def p03():
    mp.mp.dps = 30
    s = 2*mp.pi*2*mp.sin(mp.pi/3)            # AB
    r = 3*s/(2*mp.pi)                        # r*(2π/3)=AB
    V.check(3, "3-(1) AB", sp.Float(str(s), 20), S(V.ans(3, "1", "AB")), 1e-12)
    V.check(3, "3-(1) r", sp.Float(str(r), 20), S(V.ans(3, "1", "r")), 1e-12)
    d = r*mp.sqrt(3)                         # O_nO_{n+1}: 호의 중심각 2π/3 ⇒ 현=d
    h = mp.sqrt(r**2-d**2/4)
    O0, O1, P0 = mp.matrix([0, 0]), mp.matrix([d, 0]), mp.matrix([d/2, h])
    u = lambda v: v/mp.norm(v)
    ab = mp.matrix([-(P0-O1)[1], (P0-O1)[0]]); ac = mp.matrix([-(P0-O0)[1], (P0-O0)[0]])
    ab, ac = u(ab), u(ac)
    if ab[1] < 0: ab = -ab
    if ac[1] < 0: ac = -ac
    ang = mp.acos(ab[0]*ac[0]+ab[1]*ac[1])
    V.report(3, abs(ang-mp.pi/3) < 1e-20 and abs((P0+s*ab)[1]-(P0+s*ac)[1]) < 1e-20, "3-(1) 접선 조건에서 ∠A=60°, BC∥x축 (좌표)")
    arc = r*2*mp.asin(d/(2*r))
    V.report(3, abs(arc-s) < 1e-20, "3-(1) 호 P0P1 = AB (좌표)")
    V.check(3, "3-(2) AA'=3·d", sp.Float(str(3*d), 20), V.ans(3, "2"), 1e-12)
    # 색칠 도형: y=-r 접선 위, 두 아랫호 아래 영역 (O1=(d,0), O2=(2d,0))
    f1 = lambda X: r - mp.sqrt(r**2-(X-d)**2)
    f2 = lambda X: r - mp.sqrt(r**2-(X-2*d)**2)
    A = mp.quad(f1, [d, 1.5*d]) + mp.quad(f2, [1.5*d, 2*d])
    V.check(3, "3-(3) 수치적분", sp.Float(str(A), 20), V.ans(3, "3"), 1e-10)

def p04():
    c1 = sum(1 for X in range(-1, 22) for Y in range(-1, 22) if Y >= 0 and X+Y <= 20 and 2*X+Y >= 20)
    V.check(4, "4-(1) 격자점", c1, V.ans(4, "1"))
    N = lambda m: sum(1 for X in range(-m, m+1) for Y in range(-m, m+1) if abs(X)+abs(Y) <= m)
    V.check(4, "4-(2) ΣN(k)", sum(N(m) for m in range(1, 12)), V.ans(4, "2"))
    V.check(4, "N(n)=2n²+2n+1", all(N(m) == 2*m*m+2*m+1 for m in range(1, 30)), True)
    pr = {2, 3, 5, 7, 11, 13, 17, 19}
    c3 = sum(1 for X in range(1, 20) for Y in range(1, 20) for Z in range(1, 20) if X+Y+Z == 20 and X % 2 == 1 and Y % 2 == 0 and Z in pr)
    V.check(4, "4-(3)", c3, V.ans(4, "3"))

if __name__ == "__main__": V.run(globals())
