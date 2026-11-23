import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("KW-G20A", "items/KW/PAST")
SY = lambda s, **kw: sp.sympify(s, locals={"t": t, "x": x, "n": n, **kw})

def p01():
    # [1] 넓이 최솟값
    xx = sp.symbols("xx", positive=True)
    S_ = sp.Rational(1, 2)*(2/xx + 1)*(xx + 1)*sp.sin(sp.pi/3)
    crit = [r for r in sp.solve(sp.diff(S_, xx), xx) if r.is_positive]
    V.check(1, "1 x", crit[0], SY(V.ans(1, "1", "x")))
    V.check(1, "1 min", sp.simplify(S_.subs(xx, crit[0])), SY(V.ans(1, "1", "min")))
    # [2](1)(2) 순서쌍 개수 직접 세기
    cnt = lambda m: sum(1 for X in range(1, 3**m) for Y in range(1, 3**m) if X + 2*Y == 3**m)
    a = {m: cnt(m) for m in (1, 2, 3, 4, 5)}
    V.check(1, "2-1 a1", a[1], SY(V.ans(1, "2-1", "a1")))
    V.check(1, "2-1 a2", a[2], SY(V.ans(1, "2-1", "a2")))
    V.check(1, "2-1 a3", a[3], SY(V.ans(1, "2-1", "a3")))
    f22 = SY(V.ans(1, "2-2"))
    for m in (1, 2, 3, 4, 5):
        V.check(1, f"2-2 a_{m}", a[m], f22.subs(n, m))
    # [2](3)
    tot = sum(sp.log(6*f22.subs(n, m) + 3, 3) for m in range(1, 100))
    V.check(1, "2-3 합", sp.nsimplify(sp.N(tot, 30)), SY(V.ans(1, "2-3")))
    # [3] 정사각형: 2^c=2/15, b=c-2, d=c+4, a=b+4
    c_ = sp.log(sp.Rational(2, 15), 2)
    V.check(1, "3-2 c", c_, SY(V.ans(1, "3-2", "c"), log=sp.log))
    b_ = SY(V.ans(1, "3-1", "b"), c=c_); d_ = SY(V.ans(1, "3-2", "d"), c=c_); a_ = b_ + (d_ - c_)
    A = sp.Matrix([a_, 2**(a_+2)+4]); B = sp.Matrix([b_, 2**(b_+2)+4]); C = sp.Matrix([c_, 2**c_]); D = sp.Matrix([d_, 2**d_])
    V.check(1, "3 BA=CD", sp.simplify((A-B) - (D-C)).norm(), 0)
    V.check(1, "3 BC⊥CD", sp.simplify((B-C).dot(D-C)), 0)
    V.check(1, "3 |BC|=|CD|", sp.simplify((B-C).dot(B-C) - (D-C).dot(D-C)), 0)
    V.check(1, "3-1 side", sp.sqrt(sp.simplify((B-C).dot(B-C))), SY(V.ans(1, "3-1", "side")))
    V.report(1, b_ < a_ and c_ < d_, "3 b<a, c<d")

def p02():
    lg = lambda k: len(str(k)) - 1
    # [1](1) 구간별 적분
    I = sum(lg(k)*1 for k in range(1, 2020)) # [log t]=lg(k) on [k,k+1), k=1..2019
    V.check(2, "1-1", I, SY(V.ans(2, "1-1")))
    # [1](2) 계수의 합 P(1)
    V.check(2, "1-2 P(1)", sum(lg(k)*(-1)**k for k in range(1, 2021)), SY(V.ans(2, "1-2")))
    # [1](3) h(x)=P(x)-1/(1+x^2) 가 x<0 에서 감소하고 부호가 한 번 바뀜
    mp.mp.dps = 60
    coef = [lg(k) for k in range(2021)]
    def P(xv):
        return mp.fsum(coef[k]*(-xv)**k for k in range(1, 2021))
    h = lambda xv: P(xv) - 1/(1 + xv**2)
    grid = [mp.mpf(-i)/1000 for i in range(0, 1001)]
    vals = [h(g) for g in grid]
    changes = sum(1 for u, w in zip(vals, vals[1:]) if u*w < 0)
    V.check(2, "1-3 교점 수(구간 [-1,0])", changes, SY(V.ans(2, "1-3")))
    V.report(2, all(u < w for u, w in zip(vals, vals[1:])), "1-3 h 는 [-1,0] 격자에서 x 가 커질수록 감소")
    V.report(2, P(mp.mpf(-1)) == 4953 and h(mp.mpf(-3)) > 0 and h(mp.mpf(-1.5)) > 0, "1-3 P(-1)=4953, x<-1 에서 h>0")
    # [2] 타원 x^2/4+y^2=1
    f = sp.sqrt(1 - x**2/4)
    tt = sp.symbols("tt", positive=True)
    V.check(2, "2-2 ODE f f'=-x/4", sp.simplify(f*sp.diff(f, x) + x/4), 0)
    for xv in (sp.Rational(1, 2), sp.Rational(13, 10), sp.Rational(-17, 10)):
        rhs = -sp.Rational(1, 4)*sp.N(sp.Integral(tt/sp.sqrt(1 - tt**2/4), (tt, 0, abs(xv))), 20) + 1  # f 는 짝함수, x<0 에서도 적분값 동일
        V.check(2, f"2-2 조건식 x={xv}", rhs, sp.N(f.subs(x, xv), 20), tol=1e-12)
    y_ = sp.Symbol("y")
    lhs_, rhs_ = V.ans(2, "2-2").split("=")
    V.check(2, "2-2 YAML 방정식 ↔ f", sp.simplify((SY(lhs_, y=y_)).subs(y_, f) - SY(rhs_)), 0)
    sl = sp.diff(f, x).subs(x, sp.sqrt(3))
    V.check(2, "2-1 f(√3)=1/2", f.subs(x, sp.sqrt(3)), sp.Rational(1, 2))
    V.check(2, "2-1 기울기", sl, SY(V.ans(2, "2-1", "slope")))
    line = sl*(x - sp.sqrt(3)) + sp.Rational(1, 2)
    V.check(2, "2-1 접선", sp.expand(line), sp.expand(SY(V.ans(2, "2-1", "line"))))
    # [3](1) 수치 확인: f 짝, g' 짝, g(0)=1
    bad = []
    for fx in (sp.cos(x), x**2 + 1, sp.exp(x**2)):
        for gx in (x**3 + 1, sp.sinh(x) + 1, x*sp.sin(x)*x + 1 - 0*x):
            if sp.simplify(sp.diff(gx, x) - sp.diff(gx, x).subs(x, -x)) != 0: continue
            tv = sp.Rational(17, 10)
            L_ = sp.N(sp.Integral(fx*gx, (x, -tv, tv)), 20); R_ = sp.N(2*sp.Integral(fx, (x, 0, tv)), 20)
            if abs(L_ - R_) > 1e-12: bad.append((fx, gx))
    V.proof(2, "3-1", bad)
    # [3](2)
    for tv in (sp.Rational(13, 10), 2):
        lhs = sp.N(sp.Integral(2*x*sp.sin(x)/(sp.exp(x) + 1), (x, -tv, tv)), 20)
        V.check(2, f"3-2 t={tv}", lhs, sp.N(SY(V.ans(2, "3-2")).subs(t, tv), 20), tol=1e-12)

if __name__ == "__main__": V.run(globals())
