#!/usr/bin/env python3
"""TUK-G24A-01~03 (한국공학대 2024 오전) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G24A", "items/TUK/PAST")
mp.mp.dps = 25
th = sp.Symbol("th", real=True)


def kv(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def p01():
    D = (4 * sp.sin(th)) ** 2 - 4 * (2 * sp.cos(th) + 2)
    sols = sorted(sp.solveset(sp.Eq(D, 0), th, sp.Interval.Ropen(0, 2 * sp.pi)), key=lambda r: float(r))
    e = kv(1, "1")
    exp_sols = sorted([S(e["다"].replace("t", "th")), S(e["라"]), S(e["마"])], key=lambda r: float(r))
    V.check(1, "1-1 θ 해집합(다·라·마)", sp.FiniteSet(*sols), sp.FiniteSet(*exp_sols))
    fac = sp.factor(2 * (1 - sp.cos(th) ** 2) - sp.cos(th) - 1)
    ga = S(e["가"].replace("t", "th"), locals={"th": th})
    V.report(1, sp.simplify(-ga * (sp.cos(th) + 1) - (2 * (1 - sp.cos(th) ** 2) - sp.cos(th) - 1)) == 0 or sp.simplify(ga * (sp.cos(th) + 1) - (2 * (1 - sp.cos(th) ** 2) - sp.cos(th) - 1)) == 0, f"1-1 (가) 인수 {ga} : {fac}")
    V.report(1, same(sp.cos(sols[0]), e["나"]), "1-1 (나) cosθ=1/2")
    a = sp.Symbol("a", real=True)
    xa = sp.log(a - 2, 3) + 2  # -3^(x-2)+a=2
    xb = 1                    # 2^x=2
    ok = [r for r in [sp.Integer(5)] if sp.simplify(abs(xa.subs(a, r) - xb) - 2) == 0]
    cands = sp.solve(sp.Eq(sp.log(a - 2, 3) + 1, 2), a) + sp.solve(sp.Eq(sp.log(a - 2, 3) + 1, -2), a)
    cands = [c for c in cands if c > 3]
    V.check(1, "1-2 a (a>3)", cands[0], V.ans(1, "2"))
    r = sp.Symbol("r", positive=True)
    rr = sp.solve(r**2 - 9, r)[0]
    a1 = sp.Symbol("a1", positive=True)
    a1v = sp.solve(sp.Eq((a1 * rr) * (a1 * rr**3) * (a1 * rr**5), 3**15), a1)[0]
    kk = sp.solve(sp.Eq(a1v * rr ** (n - 1), 3**46), n)[0]
    V.check(1, "1-3 k", kk, V.ans(1, "3"))


def p02():
    a, b, c = sp.symbols("a b c")
    f = a * x**2 + b * x + c
    num = f.subs(c, 0)
    a_val = sp.solve(sp.Eq(sp.limit(num**2 / (2 * x**2 * num + 3 * num.subs(x, x**2)), x, sp.oo), 2), a)
    a_val = [v for v in a_val if v != 0][0]
    f1 = num.subs({a: a_val}).subs(x, 1)
    bb = sp.solve(sp.Eq(sp.limit(num.subs(a, a_val) / x, x, 0), 3), b)[0]
    V.check(2, "2-1 f(1)", f1.subs(b, bb), V.ans(2, "1"))
    A = sp.Symbol("A", positive=True)
    fa = -2 * x**3 + 3 * (A + 2) * x**2 - 12 * A * x - sp.Rational(3, 2) * A**2 + 18 * A
    cr = sp.solve(sp.diff(fa, x), x)
    mx = [r for r in cr if sp.simplify(sp.diff(fa, x, 2).subs(x, r)).subs(A, 3) < 0][0]
    g = sp.simplify(fa.subs(x, mx))
    V.check(2, "2-2 lim g'(a)/(a-2)", sp.limit(sp.diff(g, A) / (A - 2), A, 2, "+"), V.ans(2, "2"))
    v = lambda tt: tt**3 - tt if tt <= 1 else tt**2 + tt - 2
    C = sp.Symbol("C")
    Cv = sp.solve(sp.Eq(1 + 1 + C, 0), C)[0]   # v(1)=0 → t^2+t+C
    V.report(2, Cv == -2, "2-3 상수 C=-2 (v 연속)")
    vf = lambda tt: tt**3 - tt if tt <= 1 else tt**2 + tt + float(Cv)
    s_ = mp.quad(lambda tt: abs(vf(tt)), [0, 1, 2])
    V.report(2, abs(s_ - mp.mpf(str(sp.N(S(V.ans(2, "3")), 25)))) < 1e-9, f"2-3 움직인 거리 수치 {s_}")


def p03():
    f = x**3 - x**2 + 2 * x - 1
    fn = sp.lambdify(x, f, "mpmath")
    gn = lambda y_: mp.findroot(lambda u: fn(u) - y_, y_)
    num = mp.quad(gn, [-1, 0, 1])
    V.report(3, abs(num - mp.mpf(str(sp.N(S(V.ans(3, "1")), 25)))) < 1e-9, f"3-1 ∫g 수치 {num}")
    a, b, c = sp.symbols("a b c")
    F = a * x**2 + b * x + c
    rhs = sp.integrate((x**2 + 2 * t) * sp.diff(F, x).subs(x, t), (t, 0, x))
    eqs = sp.Poly(sp.expand(a * x**4 + b * x**2 + c - rhs), x).coeffs()
    sols = sp.solve(eqs + [(a + b) ** 2 - 4], [a, b, c], dict=True)
    good = [s_ for s_ in sols if F.subs(s_).subs(x, 1) > 0]
    V.check(3, "3-2 f'(1)", sp.diff(F.subs(good[0]), x).subs(x, 1) if len(good) == 1 else None, V.ans(3, "2"))
    aa = sp.Symbol("aa", positive=True)
    fx = lambda av: sp.Piecewise((x**2 + av * x, x < 0), (-x**2 + av * x, True))
    fp = lambda av: sp.Piecewise((2 * x + av, x < 0), (-2 * x + av, True))
    I = sp.integrate(sp.Abs(2 * x + aa), (x, -aa, 0)) + sp.integrate(sp.Abs(-2 * x + aa), (x, 0, aa))
    a_val = sp.solve(sp.Eq(sp.simplify(I), 1), aa)[0]
    h = sp.Symbol("h")
    expr = (fx(a_val).subs(x, -a_val + 2 * h) - fx(a_val).subs(x, a_val - h)) / h
    num2 = mp.limit(lambda hh: sp.lambdify(h, expr, "mpmath")(hh), 0)
    V.report(3, abs(num2 - mp.mpf(str(sp.N(S(V.ans(3, "3")), 25)))) < 1e-8, f"3-3 a={a_val}, 극한 수치 {num2}")


if __name__ == "__main__": V.run(globals())
