import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp, random
V = Verifier("SSWU-M22", "items/SSWU/PAST")
mp.mp.dps = 30
th_, a_ = sp.Symbol("theta"), sp.Symbol("a")

def N(xx, th): return mp.mpf(1)/2**xx*mp.sin(th*xx)*mp.sin(mp.pi/2-th*xx)

def p01():
    Nx = sp.Rational(1, 2)**x*sp.sin(th_*x)*sp.sin(sp.pi/2-th_*x)
    V.check(1, "1-(1) 극한", sp.limit(Nx/x, x, 0), V.ans(1, "1"))
    s2 = sum(abs(N(i, mp.pi/4)) for i in range(1, 120))
    V.check(1, "1-(2) 급수", sp.Float(str(s2), 20), V.ans(1, "2"), 1e-12)
    th = mp.pi/8
    s3 = sum(mp.log(mp.mpf(1)/2**(i+1)+mp.cos(2*th*i)*N(i, th), 2) for i in range(1, 21))
    V.check(1, "1-(3) 직접합", sp.Float(str(s3), 20), sp.N(S(V.ans(1, "3")), 20), 1e-12)

def p02():
    x = sp.Symbol("x")
    A, B, C = sp.symbols("A B C")
    eq = sp.expand((-x)**3+A*(-x)**2+B*(-x)+C+(x**3+A*x**2+B*x+C))
    sol = sp.solve([eq.coeff(x, 2), eq.coeff(x, 0)], [A, C]); 
    f = x**3+0*x**2+B*x
    bb = sp.solve(f.subs(x, 2), B)[0]
    assert sp.diff(x**3+bb*x, x).subs(x, 2) == 8
    f = (x**3+bb*x)
    V.check(2, "2-(1)", f, V.ans(2, "1"))
    g = x**3-2*x**2-4*x
    crit = [g.subs(x, r) for r in sp.solve(sp.diff(g, x), x)]
    cnt = lambda kk: len(sp.Poly(f-2*x**2-kk, x).real_roots()) if False else len(set(sp.real_roots(sp.Poly(f-2*x**2-kk, x))))
    ks = sorted(S(v) for v in V.ans(2, "2", "k").split(","))
    V.report(2, sorted(crit) == ks and all(cnt(kk) == 2 for kk in ks) and cnt(0) == 3 and cnt(5) == 1, "2-(2) k 조건(극값에서 서로 다른 두 실근, 그 외 1·3개)")
    kk = min(ks); r = sorted(set(sp.real_roots(sp.Poly(f-2*x**2-kk, x))))
    area = sp.integrate(f-(2*x**2+kk), (x, r[0], r[1]))
    V.check(2, "2-(2) 넓이", area, V.ans(2, "2", "area"))
    # 접선 4개와 직사각형
    def tang(m):
        return sorted(sp.simplify(f.subs(x, p)-m*p) for p in sp.solve(sp.diff(f, x)-m, x))
    c1, c2 = tang(1), tang(-1)
    area3 = abs(c1[1]-c1[0])/sp.sqrt(2)*abs(c2[1]-c2[0])/sp.sqrt(2)
    V.check(2, "2-(3) 직사각형", sp.simplify(area3), V.ans(2, "3"))

def p03():
    def r_seq(a, th, K=400):
        out = []
        for _ in range(K):
            # O1=a(cos θ/2, sin θ/2), P1=(p,0), Q1=p(cosθ, sinθ): |P1Q1|=|O1P1|
            c2, s2_ = mp.cos(th/2), mp.sin(th/2)
            A_, B_, C_ = 4*s2_**2-1, 2*a*c2, a*a
            D_ = mp.sqrt(B_**2+4*A_*C_)
            p = min(r for r in ((-B_+D_)/(2*A_), (-B_-D_)/(2*A_)) if r > 0)
            out.append(2*p*mp.sin(th/2)); a = p*mp.cos(th/2)
        return out
    random.seed(1); okall = True; msg = ""
    for _ in range(6):
        th = mp.mpf(random.uniform(0.2, 3.0)); a = mp.mpf(random.uniform(0.5, 3))
        rs = r_seq(a, th); t = mp.tan(th/2); q = 1/(mp.sqrt(3)*t+1)
        r1 = 2*a*t/(mp.sqrt(3)*t+1)
        okall &= abs(rs[0]-r1) < 1e-20 and abs(rs[1]-r1*q) < 1e-20
        okall &= abs(sum(rs)-2*a/mp.sqrt(3)) < 1e-12
        As = sum(mp.sqrt(3)/4*v*v for v in rs)
        form = a*a*t/(mp.sqrt(3)*t+2)
        okall &= abs(As-form) < 1e-15 and As < a*a/mp.sqrt(3)
    V.report(3, okall, "3-(1)(2)(3) 좌표로 만든 r_n 수열: r1,r2·합·넓이합·부등식 (θ,a 무작위 6회) [증명-수치확인]")
    r1 = sp.sympify(V.ans(3, "1", "r1"), locals={"a": a_, "theta": th_}); r2 = sp.sympify(V.ans(3, "1", "r2"), locals={"a": a_, "theta": th_})
    V.check(3, "3-(1) r1,r2 식(θ=1,a=2 수치)", sp.N((r1+r2).subs({a_: 2, th_: 1}), 20), sp.N(sp.Float(str(r_seq(mp.mpf(2), mp.mpf(1))[0]+r_seq(mp.mpf(2), mp.mpf(1))[1]), 20), 20), 1e-12)
    ss = sp.sympify(V.ans(3, "3"), locals={"a": a_, "theta": th_})
    V.check(3, "3-(3) 식(θ=1,a=2 수치)", sp.N(ss.subs({a_: 2, th_: 1}), 20), sp.N(sp.Float(str(sum(mp.sqrt(3)/4*v*v for v in r_seq(mp.mpf(2), mp.mpf(1)))), 20), 20), 1e-12)

def p04():
    import math
    A = {n: {k for k in range(-5, 10) if 4**k >= n and 2**k <= n} if True else None for n in range(1, 11)}
    V.check(4, "4-(1)", len(set().union(*A.values())), V.ans(4, "1"))
    def Nk(k): return sum(1 for nn_ in range(2, 2**k+1) if any(nn_**m == 2**k for m in range(1, k+1)))
    got = "; ".join(f"N{k} = {Nk(k)}" for k in range(1, 11))
    V.report(4, all(Nk(k) == int(V.ans(4, "2", f"N{k}")) for k in range(1, 11)), "4-(2) N_k 전수 열거: " + got)
    def Nbig(kk):  # N at 2^{k+1}: n=2^l, l | 2^{k+1}
        e = k_ = kk; return sum(1 for l in range(1, 2**kk+1) if 2**kk % l == 0)
    V.report(4, all(Nbig(k+1) == k+2 for k in range(1, 12)), "4-(3) N_{2^{k+1}}=k+2 (k=1..11, 약수 열거)")
    k_ = sp.Symbol("k", integer=True, positive=True)
    V.check(4, "4-(3) 급수", sp.summation(1/(k_*(k_+2)), (k_, 1, sp.oo)), V.ans(4, "3"))

if __name__ == "__main__": V.run(globals())
