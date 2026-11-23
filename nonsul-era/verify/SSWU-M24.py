import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("SSWU-M24", "items/SSWU/PAST")
mp.mp.dps = 30

def p01():
    tt = sp.Symbol("tt", real=True)
    f = sp.exp(1 + tt) - 2*tt
    # (1) 최솟값 [증명-수치확인]
    tmin = sp.solve(sp.Eq(sp.diff(f, tt), 0), tt)[0]
    V.check(1, "(1) 최솟값=4-2ln2", sp.simplify(f.subs(tt, tmin)), 4 - 2*sp.log(2))
    V.proof(1, "(1) f>0 (격자)", [v for v in [sp.Rational(i, 10) for i in range(-80, 81)] if sp.N(f.subs(tt, v)) <= 0])
    # (2) g(y)=e^{1+y}-e+1-y^2 : g'=f, g(-1)<0<g(0), 실근 유일
    y_ = sp.Symbol("y_", real=True)
    g = sp.exp(1 + y_) - sp.E + 1 - y_**2
    V.check(1, "(2) g'=f", sp.simplify(sp.diff(g, y_) - f.subs(tt, y_)), 0)
    assert g.subs(y_, -1) < 0 < g.subs(y_, 0)
    roots = {round(float(mp.findroot(lambda z: mp.e**(1+z) - mp.e + 1 - z**2, s)), 10) for s in [-0.9, -0.5, -0.1, 0.5, 2, -3, -10]}
    V.proof(1, "(2) 실근 유일·(-1,0)", [r for r in roots if not -1 < r < 0] + ([roots] if len(roots) != 1 else []))
    # (3) 곡선 ln(e-1+x²+y²)=1+y 위 (a,1) 에서 접선의 기울기: 음함수 미분(F_x/F_y) 으로 독립 계산
    X, Y = sp.symbols("X Y", real=True)
    F_ = sp.log(sp.E - 1 + X**2 + Y**2) - 1 - Y
    a = sp.sqrt(sp.exp(2) - sp.E)
    assert sp.simplify(F_.subs({X: a, Y: 1})) == 0
    slope = sp.simplify((-sp.diff(F_, X)/sp.diff(F_, Y)).subs({X: a, Y: 1}))
    V.check(1, "(3) 기울기", slope, S(V.ans(1, "3")))

def p02():
    n_ = sp.Symbol("n_", positive=True)
    for m in range(1, 8):
        # P_n: 원 x²+y²=m(m+1) 과 y=√x
        xs = [r for r in sp.solve(sp.Eq(x**2 + x - m*(m+1), 0), x) if r > 0]
        Px, Py = xs[0], sp.sqrt(xs[0])
        for nm, val, key in (("x", Px, "x"), ("y", Py, "y")):
            V.check(2, f"(1) P_{m}.{nm}", val, S(V.ans(2, "1", key), locals={"n": n}).subs(n, m))
        # 접선: P·(X,Y)=m(m+1) , 원 C_{m+1}: X²+Y²=(m+1)(m+2)
        X = sp.Symbol("X", real=True)
        Yl = (m*(m+1) - Px*X)/Py
        sol = sorted(sp.solve(sp.Eq(X**2 + Yl**2, (m+1)*(m+2)), X), key=float)
        V.check(2, f"(2) x차 n={m}", sol[1] - sol[0], 2*sp.sqrt(2))
        Q = sp.Matrix([sol[0], Yl.subs(X, sol[0])]); R = sp.Matrix([sol[1], Yl.subs(X, sol[1])])
        cth = sp.simplify(Q.dot(R)/(sp.sqrt(Q.dot(Q))*sp.sqrt(R.dot(R))))
        assert 0 < sp.acos(cth) < sp.pi
        V.check(2, f"(3) cos θ_{m}", cth, S(V.ans(2, "3", "cos"), locals={"n": n}).subs(n, m))
    V.check(2, "(3) lim", sp.limit(S(V.ans(2, "3", "cos"), locals={"n": n}), n, sp.oo), S(V.ans(2, "3", "lim")))

def p03():
    # f, g 를 조건을 만족하는 일차 다항함수로 구성해 모든 값을 독립 확인 (f(1)=-1, f'(1)=c, g(1)=1, g'(1)=d)
    c, d = sp.symbols("c d")
    f = -1 + c*(x - 1)
    # (1) 첫 극한 = 3 → c
    L1 = sp.limit((f**3 + 1)/(x**2 - 1), x, 1)
    cv = sp.solve(sp.Eq(L1, 3), c)[0]
    V.check(3, "(1) f'(1)", cv, S(V.ans(3, "1")))
    fv = f.subs(c, cv)
    # (2) g(1)=1(극한 존재 조건), 둘째 극한 = 2 → d 와 h'(1)
    g = 1 + d*(x - 1)
    L2 = sp.limit((sp.exp(x) - sp.E)*(fv + g)/((x - 1)**2*g), x, 1)
    dv = sp.solve(sp.Eq(L2, 2), d)[0]
    h = fv/g
    h1 = sp.simplify(sp.diff(h, x).subs(x, 1).subs(d, dv))
    V.check(3, "(2) h'(1)", h1, S(V.ans(3, "2")))
    V.check(3, "(2) 극한 재확인(수치)", sp.N(((sp.exp(x) - sp.E)*(fv + g.subs(d, dv))/((x - 1)**2*g.subs(d, dv))).subs(x, 1 + sp.Rational(1, 10**8)), 15), 2, 1e-6)
    gv = g.subs(d, dv)
    k1 = sp.simplify(sp.diff(fv.subs(x, gv), x).subs(x, 1))
    V.check(3, "(3) k'(1)", k1, S(V.ans(3, "3")))

def p04():
    f = lambda v: v - mp.log(v)
    # (1) 최솟값 f(1)=1, f''>0 [증명-수치확인]
    xs = [mp.mpf(i)/50 for i in range(1, 2000)]
    V.check(4, "(1) 임계점·최솟값", sp.solve(sp.diff(x - sp.log(x), x), x)[0], 1)
    V.proof(4, "(1) f>=1", [v for v in xs if f(v) < 1 - mp.mpf(10)**-25])
    V.proof(4, "(1) f''>0", [v for v in xs if sp.diff(x - sp.log(x), x, 2).subs(x, v) <= 0])
    # (2) 격자 반례 탐색
    bad = []
    for i in range(1, 40):
        for j in range(i + 1, 41):
            a_, b_ = mp.mpf(i)/7, mp.mpf(j)/7
            if (a_ + b_)/2 < (b_*mp.log(b_) - a_*mp.log(a_))/(b_ - a_) - mp.mpf(10)**-20:
                bad.append((a_, b_))
    V.proof(4, "(2)", bad)
    # (2) 적분 계산 확인(기호)
    a_, b_ = sp.symbols("a_ b_", positive=True)
    Iab = sp.integrate(x - sp.log(x), (x, a_, b_))
    V.check(4, "(2) 정적분 공식", sp.simplify(Iab - (sp.Rational(1, 2)*(b_**2 - a_**2) + (b_ - a_) - (b_*sp.log(b_) - a_*sp.log(a_)))), 0)
    # (3) n=2..60
    bad = []
    for n_ in range(2, 61):
        lhs = sum((f(k) + f(k + 1))/2 for k in range(1, n_))
        rhs = mp.quad(f, [1, n_])
        if lhs < rhs - mp.mpf(10)**-20: bad.append(n_)
    V.proof(4, "(3) n=2..60", bad)

if __name__ == "__main__": V.run(globals())
