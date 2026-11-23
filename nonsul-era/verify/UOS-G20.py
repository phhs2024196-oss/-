import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("UOS-G20", "items/UOS/PAST")


def p01():
    expr = S(V.ans(1, "1-1"))
    n_ = sp.symbols("n")
    for N in range(1, 9):
        tot = sum(a*a + b*b + c*c for a in range(1, N + 3) for b in range(1, N + 3) for c in range(1, N + 3) if a + b + c == N + 2)
        V.check(1, f"1-1 n={N} 전수 {tot}", tot, expr.subs(n_, N))
    V.check(1, "인수분해 꼴", sp.factor(sp.summation(sp.Symbol("l")**2*(n - sp.Symbol("l") + 1), (sp.Symbol("l"), 1, n))*3), expr.subs(sp.Symbol("n"), n))


def p02():
    t0 = sp.symbols("t0", positive=True)
    # 독립 계산: θ(t)=∠OFP (F=(0,1), FO=(0,-1), FP=(t, t^2/4-1)), S(t)= 선분 FO, FP 와 포물선으로 둘러싸인 넓이
    tt = sp.symbols("tt", positive=True)
    cth = (1 - tt**2/4)/sp.sqrt(tt**2 + (1 - tt**2/4)**2)
    theta = sp.acos(cth)
    line = 1 + (tt**2/4 - 1)*x/tt
    S_t = sp.integrate(line - x**2/4, (x, 0, tt))
    tstar = sp.nsolve(theta - sp.pi/3, tt, 1.0)
    V.report(2, abs(tstar - 2/math.sqrt(3)) < 1e-12, f"θ=π/3 ↔ t={tstar}")
    val = (sp.diff(S_t, tt)/sp.diff(theta, tt)).subs(tt, tstar)
    V.check(2, "2-1 S'(π/3)", sp.N(val, 30), V.ans(2, "2-1"), 1e-12)


def p03():
    mp.mp.dps = 30
    n_ = sp.symbols("n", positive=True)
    # 닫힌식 S_n 을 수치 적분과 비교(n=5)
    Sn = n_/(n_**2 + 1)*(2*sp.exp(-sp.pi/n_)*(1 - sp.exp(-sp.pi))/(1 - sp.exp(-sp.pi/n_)) + 1 - sp.exp(-sp.pi))
    for N in (1, 5, 8):
        pts = [mp.pi*j/N for j in range(N + 1)]
        num = sum(abs(mp.quad(lambda u: mp.e**(-u)*mp.sin(N*u), [pts[j], pts[j + 1]])) for j in range(N))
        V.check(3, f"S_{N} 닫힌식 vs 수치적분", sp.N(Sn.subs(n_, N), 25), sp.N(num, 25), 1e-18)
    lim = sp.limit(Sn, n_, sp.oo)
    V.check(3, "3-1 극한", lim, V.ans(3, "3-1"))
    V.check(3, "큰 n 수치", sp.N(Sn.subs(n_, 10**6), 20), sp.N(S(V.ans(3, "3-1")), 20), 1e-5)


def p04():
    # (a) 한 변 2인 정오각형 대각선
    d = 2*sp.sin(sp.pi*2/5)/sp.sin(sp.pi/5)
    V.check(4, "4-1 대각선", sp.N(d, 30), V.ans(4, "4-1"))
    # (b) 3D: A(위) B C D E, 변 s. 접으면 B=E=P, |PA|=s,|PC|=s,|PD|=s
    R = 1.0
    ang = {"A": 90, "B": 162, "C": 234, "D": 306, "E": 18}
    pt = {k: sp.Matrix([sp.N(sp.cos(sp.rad(v))), sp.N(sp.sin(sp.rad(v))), 0]) for k, v in ang.items()}
    s = (pt["A"] - pt["B"]).norm()
    px, py, pz = sp.symbols("px py pz", real=True)
    P = sp.Matrix([px, py, pz])
    eqs = [(P - pt[k]).dot(P - pt[k]) - s**2 for k in ("A", "C", "D")]
    sols = sp.solve(eqs, [px, py, pz], dict=True)
    sol = [q for q in sols if q[pz] > 0][0]
    Pv = P.subs(sol)
    ap = Pv - pt["A"]
    def perp(v):
        return v - ap*(v.dot(ap)/ap.dot(ap))
    u, w = perp(pt["C"] - pt["A"]), perp(pt["D"] - pt["A"])
    cth = sp.N(u.dot(w)/(u.norm()*w.norm()), 30)
    V.check(4, "cosθ = √5/5", cth, sp.sqrt(5)/5, 1e-12)
    a_, b_ = sp.Rational(0), sp.Rational(1, 5)
    V.check(4, "cosθ = a+b√5 (a=0,b=1/5)", cth, a_ + b_*sp.sqrt(5), 1e-12)
    V.check(4, "4-2 a+b", a_ + b_, V.ans(4, "4-2"))
    # 해설 중간값: cosα, sin²α
    al = sp.Rational(1, 1)*sp.pi/5
    V.check(4, "cos(π/5)=(1+√5)/4", sp.N(sp.cos(sp.pi/5), 30), (1 + sp.sqrt(5))/4)
    V.check(4, "sin²(π/5)=(5-√5)/8", sp.N(sp.sin(sp.pi/5)**2, 30), (5 - sp.sqrt(5))/8)


if __name__ == "__main__":
    V.run(globals())
