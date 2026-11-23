import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DDW-G24", "items/DDW/PAST")


def p01():
    f = sp.cos(x) / (2*sp.sqrt(2) - 2*sp.sin(x))
    d = sp.simplify(sp.diff(f, x))
    V.check(1, "1-1 f' 분자", sp.simplify(d*(2*sp.sqrt(2) - 2*sp.sin(x))**2), 2 - 2*sp.sqrt(2)*sp.sin(x))
    V.check(1, "1-1 극값 후보", sp.solveset(sp.sin(x) - 1/sp.sqrt(2), x, sp.Interval(0, 2*sp.pi)),
            sp.FiniteSet(sp.pi/4, 3*sp.pi/4))
    cand = [f.subs(x, v) for v in (0, sp.pi/4, 3*sp.pi/4, 2*sp.pi)]
    V.check(1, "1-1 최솟값", min(cand, key=lambda c: float(c)), V.ans(1, "1"))
    # 1-2: f(x)=sin x on [0,pi], 0 elsewhere
    brk = [0, sp.pi/2, sp.pi, 3*sp.pi/2, 2*sp.pi]
    # 구간별로 조각 식을 직접 지정(각 구간에서 F 의 가지가 하나로 정해짐)
    def piece(h, a_, b_):
        mid = (a_ + b_)/2
        fx = sp.sin(x) if 0 <= mid <= sp.pi else 0
        fs = sp.sin(x - sp.pi/2) if 0 <= mid - sp.pi/2 <= sp.pi else 0
        return h(fx, fs)
    def I(h):
        return sum(sp.integrate(piece(h, a_, b_), (x, a_, b_)) for a_, b_ in zip(brk, brk[1:]))
    V.check(1, "1-2 전체", I(lambda u, w: (u + w)**2), V.ans(1, "2"))
    V.check(1, "1-2 (5)", I(lambda u, w: u**2), sp.pi/2)
    V.check(1, "1-2 (6)", I(lambda u, w: w**2), sp.pi/2)
    V.check(1, "1-2 (7)", I(lambda u, w: u*w), sp.Rational(1, 2))
    # 수치 교차검산
    import mpmath as mp
    Fn = lambda z: mp.sin(z) if 0 <= z <= mp.pi else 0
    num = mp.quad(lambda z: (Fn(z) + Fn(z - mp.pi/2))**2, [0, mp.pi/2, mp.pi, 3*mp.pi/2, 2*mp.pi])
    V.check(1, "1-2 수치", sp.Float(num, 20), sp.sympify(V.ans(1, "2"), locals={"pi": sp.pi}), 1e-9)


def p02():
    tt = sp.symbols("tt", real=True)
    def f_at(xv):
        xv = sp.nsimplify(xv)
        r = sp.Rational(xv)/2
        if r <= 1:
            return sp.integrate(xv - 2*tt, (tt, 0, r)) + sp.integrate(2*tt - xv, (tt, r, 1))
        return sp.integrate(xv - 2*tt, (tt, 0, 1))
    ans = sp.sympify(V.ans(2, "1"), locals={"x": x})
    for xv in (0, sp.Rational(1, 2), 1, sp.Rational(3, 2), 2, 3, 5):
        V.check(2, f"2-1 x={xv}", ans.subs(x, xv), f_at(xv))
    cand = [f_at(sp.Rational(v, 4)) for v in range(0, 41)]
    V.check(2, "2-2 최솟값(격자)", min(cand), V.ans(2, "2", "min"))
    V.check(2, "2-2 x=1", sp.solveset(sp.diff(x**2/2 - x + 1, x), x, sp.Interval(0, 2)), sp.FiniteSet(int(V.ans(2, "2", "x"))))


def p03():
    A, B, AB = sp.Rational(1, 100), sp.Rational(5, 1000), sp.Rational(2, 10000)
    U = A + B - AB
    cond = (A - AB)/U
    V.check(3, "3-1 조건부확률", cond, sp.Rational(49, 74))
    V.check(3, "3-1 a+b", sp.numer(cond) + sp.denom(cond), V.ans(3, "1"))
    # 3-2: 표값(소수 넷째 자리)과 정확한 표준정규분포로 대조
    Phi = lambda z: sp.Rational(1, 2)*(1 + sp.erf(z/sp.sqrt(2)))
    V.check(3, "3-2 P(|Z|<=2)≈0.9544", sp.N(2*(Phi(2) - sp.Rational(1, 2)), 6), sp.Float(0.9544, 6), 5e-4)
    sigma, m = 5, 20
    V.check(3, "3-2 P(X<=25)≈0.8413", sp.N(Phi(sp.Rational(25 - m, sigma)), 6), sp.Float(0.8413, 6), 5e-5)
    V.check(3, "3-2 P(|X-m|<=10)", sp.N(2*(Phi(sp.Rational(10, sigma)) - sp.Rational(1, 2)), 6), sp.Float(0.9544, 6), 5e-4)
    V.check(3, "3-2 표본평균 sd", sp.Rational(sigma, 5), 1)
    p_table = sp.Rational(1, 2) - sp.Rational(3413, 10000)
    V.check(3, "3-2 p(표값)", p_table, V.ans(3, "2"))
    V.check(3, "3-2 p(정확값)", sp.N(1 - Phi(1), 6), sp.sympify(V.ans(3, "2")), 5e-5)

if __name__ == "__main__": V.run(globals())
