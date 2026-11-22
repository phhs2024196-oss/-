import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DSW-G26A", "items/DSW/PAST")
a, tt = sp.symbols("a t", real=True)


def radii(N, kmax):
    """원 배치를 기하로 직접 구성: 반직선(각 α=π/a_N) 위 중심 d_k, 반지름 r_k=d_k sinα, 인접 원 외접."""
    al = sp.sin(sp.pi / (3*N + 3))
    r = [al/(1 + al)]                      # (1-r1) sinα = r1
    d = [1 - r[0]]
    for _ in range(kmax - 1):
        dn = sp.symbols("dn", positive=True)
        sol = sp.solve(sp.Eq(d[-1] - dn, d[-1]*al + dn*al), dn)[0]   # d_k - d_{k+1} = r_k + r_{k+1}
        d.append(sp.simplify(sol)); r.append(sp.simplify(sol*al))
    return r, al


def p01():
    r, al = radii(1, 2)
    N6 = 6
    S2 = N6*sp.pi*r[1]**2
    C2 = N6*2*sp.pi*r[1]
    V.check(1, "1-1 S2", S2, V.ans(1, "1-1", "S2"))
    V.check(1, "1-1 C2", C2, V.ans(1, "1-1", "C2"))
    rk = lambda k: sp.Rational(1, 3)**k
    V.check(1, "1-1 r1,r2", [r[0], r[1]], [rk(1), rk(2)])
    k_ = sp.symbols("k_", integer=True, positive=True)
    SS = sp.summation(6*sp.pi*rk(k_)**2, (k_, 1, 10))
    CC = sp.summation(6*2*sp.pi*rk(k_), (k_, 1, 10))
    V.check(1, "1-2 sum S", SS, V.ans(1, "1-2", "SS"))
    V.check(1, "1-2 sum C", CC, V.ans(1, "1-2", "CC"))
    # 1-3: 일반항을 N=2,5 에서 기하 구성값과 대조하고 YAML 의 식을 N 의 함수로 읽어 비교
    Nn = sp.symbols("N", positive=True)
    expr = sp.sympify(V.ans(1, "1-3"), locals={"N": Nn, "k": k})
    for N in (1, 2, 5):
        rr, _ = radii(N, 5)
        for j in range(5):
            V.check(1, f"1-3 N={N} k={j+1}", sp.N(expr.subs({Nn: N, k: j + 1}), 30), sp.N(rr[j], 30))


def p02():
    f = lambda z, aa: aa*z**2 if z < 0 else -aa*z**2
    def S_num(aa, tv):
        c = -(x - tv)**2 + f(tv, aa)
        rts = []
        for br, dom in ((aa*x**2, (-sp.oo, 0)), (-aa*x**2, (0, sp.oo))):
            for z in sp.solve(sp.Eq(c, br), x):
                if dom[0] < z < dom[1] or (z == 0 and dom[1] == sp.oo):
                    rts.append(z)
        rts = sorted(set(rts))
        assert len(rts) == 2, rts
        x1, x2 = rts
        return sp.integrate(sp.Abs(c), (x, x1, x2)) if False else sp.integrate(sp.simplify(-c if c.subs(x, (x1+x2)/2) < 0 else c), (x, x1, x2))
    h = sp.Rational(1, 2)
    V.check(2, "2-1 S(-2)", S_num(h, -2), V.ans(2, "2-1", "S(-2)"))
    V.check(2, "2-1 S(2)", S_num(h, 2), V.ans(2, "2-1", "S(2)"))
    # 2-2: g(a) 는 t<0 의 S(t)/t^3 = -g 로 수치 대조
    pq = V.ans(2, "2-2")
    d = {kk: sp.Integer(vv) for kk, vv in (p.split("=") for p in pq.replace(" ", "").split(";"))}
    ga = lambda aa: (6*aa**4 + d["p"]*aa**3 + d["q"]*aa**2 + d["r"]*aa + d["s"])/(3*(1 + aa)**3)
    for aa in (sp.Rational(1, 3), sp.Rational(3, 4)):
        for tv in (-1, -3):
            V.check(2, f"2-2 a={aa},t={tv}", S_num(aa, tv)/tv**3, -ga(aa))
    # 2-3: h(a)=S(t)/t^3 (t>0) 를 수치로 구해 극한을 대조
    ha_sym = lambda aa: (6*aa**4 - 4*aa**3 + 6*aa**2)/(3*(1 - aa)**3)
    for aa in (sp.Rational(1, 3), sp.Rational(3, 4)):
        for tv in (1, 2):
            V.check(2, f"2-3 h a={aa},t={tv}", S_num(aa, tv)/tv**3, ha_sym(aa))
    gg = ga(a)
    V.check(2, "2-3 lim a->0+ g/h", sp.limit(gg/ha_sym(a), a, 0, "+"), V.ans(2, "2-3", "L1"))
    V.check(2, "2-3 lim a->1- g/((1-a)^3 h)", sp.limit(gg/((1 - a)**3*ha_sym(a)), a, 1, "-"), V.ans(2, "2-3", "L2"))

if __name__ == "__main__": V.run(globals())
