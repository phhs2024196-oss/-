import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SSWU-G23", "items/SSWU/PAST")

def p01():
    a = sp.symbols("a")
    tt = sp.symbols("tt")
    g = x**2 + a*x                                  # 최고차 계수 1, 원점 통과, f 가 사차 → g 는 이차
    f = x*sp.integrate(g.subs(x, tt), (tt, 0, x))
    assert sp.degree(sp.expand(f), x) == 4
    av = sp.solve(sp.Eq(f.subs(x, 3), 0), a)[0]
    g1, f1 = g.subs(a, av), sp.expand(f.subs(a, av))
    V.check(1, "(1) g", g1, S(V.ans(1, "1", "g"), locals={"x": x}))
    V.check(1, "(1) f", f1, S(V.ans(1, "1", "f"), locals={"x": x}))
    avg = (f1.subs(x, 3) - f1.subs(x, 0)) / 3
    bs = [r for r in sp.solve(sp.Eq(sp.diff(f1, x), avg), x) if r.is_real and 0 < r < 3]
    V.check(1, "(2) b", bs[0] if len(bs) == 1 else bs, V.ans(1, "2"))
    f2 = sp.diff(f1, x, 2)
    # (3) |f''|-f'' 를 f'' 의 영근(0, 3/2) 기준 구간마다 부호를 시험점으로 판정해 적분
    pts = sorted([sp.Integer(0), sp.Integer(3)] + [r for r in sp.solve(f2, x) if 0 < r < 3])
    val = 0
    for p_, q_ in zip(pts, pts[1:]):
        sgn = sp.sign(f2.subs(x, (p_ + q_) / 2))
        val += sp.integrate(sp.expand((sgn - 1) * f2), (x, p_, q_))   # |f''|=sgn*f''
    V.check(1, "(3)", val, V.ans(1, "3"))
    num = sp.Integral(sp.Abs(f2) - f2, (x, 0, 3)).evalf(20)
    V.check(1, "(3) 수치", num, V.ans(1, "3"), 1e-9)

def p02():
    A, B = sp.symbols("A B")
    nn = sp.Symbol("nn", positive=True, integer=True)
    # a_n 을 문제 정의대로 수치 적분해 (1) 의 A,B 로 쓴 적분과 n=1..6 에서 비교
    for m in range(1, 7):
        an = sp.sqrt(2) * sp.exp(sp.pi/4) * sp.Integral(sp.exp(-x) * sp.sin(x), (x, sp.pi/4, m*sp.pi + sp.pi/4)).evalf(25)
        Av, Bv = S(V.ans(2, "1", "A")), S(V.ans(2, "1", "B"))
        rhs = sp.Integral(sp.exp(-x) * (Av*sp.sin(x) + Bv*sp.cos(x)), (x, 0, m*sp.pi)).evalf(25)
        V.check(2, f"(1) n={m}", an, rhs, 1e-12)
    # (2) 닫힌형 a_n 이 정의와 일치하는지, 극한
    c = sp.integrate(sp.exp(-x) * (sp.sin(x) + sp.cos(x)), x)
    an = sp.simplify(c.subs(x, nn*sp.pi) - c.subs(x, 0))
    V.check(2, "(2) 닫힌형 n=3", an.subs(nn, 3), sp.sqrt(2) * sp.exp(sp.pi/4) * sp.Integral(sp.exp(-x) * sp.sin(x), (x, sp.pi/4, 3*sp.pi + sp.pi/4)).evalf(25), 1e-12)
    assert abs(sp.exp(-sp.pi)) < 1
    V.check(2, "(2) 극한(n=200 수치)", sp.N(an.subs(nn, 200), 30), V.ans(2, "2"), 1e-12)
    # (3) 부분합 수치
    f = lambda m: sp.N(1 - (-sp.exp(-sp.pi))**m, 40)
    V.check(2, "(3)", sum(f(m) - f(m+1) for m in range(1, 60)), S(V.ans(2, "3")), 1e-12)

def p03():
    # 좌표: O1=(0,0), A=(1,0), O2=(2,0). 반직선 O1P1 위의 점 u(cosθ, sinθ) 가 반원 O2 위: (u cosθ-2)^2+(u sinθ)^2=1
    def geom(th):
        c, s = sp.cos(th), sp.sin(th)
        u = sp.symbols("u", positive=True)
        r = sorted(sp.solve(sp.Eq((u*c - 2)**2 + (u*s)**2, 1), u), key=lambda v: float(v))
        P1 = sp.Matrix([c, s]); P2 = r[0]*sp.Matrix([c, s]); Q = r[1]*sp.Matrix([c, s])
        O1, A, O2 = sp.Matrix([0, 0]), sp.Matrix([1, 0]), sp.Matrix([2, 0])
        def area(*pts):   # 신발끈
            n_ = len(pts); return abs(sum(pts[i][0]*pts[(i+1) % n_][1] - pts[(i+1) % n_][0]*pts[i][1] for i in range(n_))) / 2
        return dict(P1=P1, P2=P2, Q=Q, tri=area(P2, O2, Q), f=area(O1, A, P1), g=area(O1, O2, Q), h=area(A, O2, P2, P1), l=sp.sqrt((P2-P1).dot(P2-P1)))
    G = geom(sp.asin(1/sp.sqrt(5)))
    V.check(3, "(1)", sp.simplify(G["tri"]), S(V.ans(3, "1")))
    lf = lambda key: S(V.ans(3, "2", key), locals={"t": t}) if key != "lim" else S(V.ans(3, "2", key))
    Lexp = S(V.ans(3, "2", "l"), locals={"t": t})
    for th in (sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 10)):
        G = geom(th)
        V.check(3, f"(2) l θ={th}", sp.N(G["l"], 25), sp.N(Lexp.subs(t, th), 25), 1e-12)
        for key in ("f", "g", "h"):
            ex = S(V.ans(3, "3", key), locals={"t": t})
            V.check(3, f"(3) {key} θ={th}", sp.N(G[key], 25), sp.N(ex.subs(t, th), 25), 1e-12)
    # 극한(작은 θ 수치): ℓ/θ², (f+g+h)/θ
    e = sp.Rational(1, 10**5)
    G = geom(e)
    V.check(3, "(2) lim", sp.N(G["l"]/e**2, 20), S(V.ans(3, "2", "lim")), 1e-4)
    V.check(3, "(3) lim", sp.N((G["f"] + G["g"] + G["h"])/e, 20), S(V.ans(3, "3", "lim")), 1e-4)

def p04():
    import itertools
    D = range(10)
    pw5 = list(itertools.product(D, repeat=5))
    n1 = sum(1 for w in pw5 if max(w.count(d) for d in D) >= 3)
    V.check(4, "(1) 전수", n1, V.ans(4, "1"))
    def run3(w):
        return any(w[i] == w[i+1] == w[i+2] for i in range(len(w) - 2))
    n2 = sum(1 for w in pw5 if run3(w))
    V.check(4, "(2) 전수", n2, V.ans(4, "2"))
    # (3) 정책: 연속 3회 이상 없으면 5자리(비밀번호 = 그 5자리), 있으면 뒤 두 자리 추가 → 7자리
    n3 = sum(1 for w in pw5 if not run3(w)) + sum(1 for w in pw5 if run3(w)) * 10**2
    V.check(4, "(3) 정책", n3, V.ans(4, "3"))

if __name__ == "__main__": V.run(globals())
