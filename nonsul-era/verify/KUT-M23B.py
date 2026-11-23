#!/usr/bin/env python3
"""KUT-M23B-01~10 (한국기술교육대 2023학년도 온라인 모의평가 수리논술) — sympy 검산. 기대값은 YAML 에서 V.ans() 로 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-M23B", "items/KUT/PAST")
a, b, c, k, t, p_, q_ = sp.symbols("a b c k t p q", real=True)


def p01():
    f = x**2 + 2 * x + 5
    lo, hi = extrema(f, x, -2, 1)
    V.check(1, "min", 2**lo, V.ans(1, "1", "min"))
    V.check(1, "max", 2**hi, V.ans(1, "1", "max"))


def p02():
    n = sum(1 for xv in range(-9, 400)
            if sp.N(sp.log(xv + 10, 8), 40) < sp.N(2 * sp.log(5, 4), 40))
    V.check(2, "정수 개수", n, V.ans(2, "1"))


def p03():
    r1 = sp.solveset(sp.sin(4 * x) - sp.Rational(1, 3), x, sp.Interval(-sp.pi / 4, sp.pi / 4))
    r2 = sp.solveset(sp.sin(4 * x) + sp.Rational(1, 3), x, sp.Interval(-sp.pi / 4, sp.pi / 4))
    assert len(r1) == 2 and len(r2) == 2
    s = sum(r1) + sum(r2)
    V.check(3, "cos", sp.cos(sp.simplify(s) * sp.pi), V.ans(3, "1"))


def p04():
    A = sp.symbols("A")
    f = -2 * (x - 5) + 3  # f(5)=3, f'(5)=-2 인 일차식 대표(값만 사용)
    g = x**3 * f + 2 * x
    V.check(4, "g'(5)", sp.diff(g, x).subs(x, 5), V.ans(4, "1"))
    assert sp.limit((f - 3) / (x - 5), x, 5) == -2


def p05():
    A = sp.symbols("A")
    f = 4 * x**3 - 2 * A * x + 3
    sol = sp.solve(sp.integrate(sp.diff(f, x), (x, 0, 1)) - A, A)[0]
    V.check(5, "정적분", sp.integrate(f.subs(A, sol), (x, 0, 3)), V.ans(5, "1"))


def p06():
    f = a * x**2 - b * x + c
    sol = sp.solve([f.subs(x, 0), f.subs(x, 1) - 4, sp.simplify(f.subs(x, b / (2 * a)) - sp.Rational(9, 2))], [a, b, c], dict=True)
    sol = [s for s in sol if s[a] < 0 and 0 < s[b] / (2 * s[a]) < 1]
    assert len(sol) == 1
    V.check(6, "f(1/2)", f.subs(sol[0]).subs(x, sp.Rational(1, 2)), V.ans(6, "1"))


def p07():
    f = lambda z: sp.Rational(4, 3) * z**3 - z + 3
    lo, hi = extrema(f(x), x, -1, 1)
    V.check(7, "k max", lo, V.ans(7, "1"))


def p08():
    f = 6 * x**2 - 18 * x
    S1 = sp.integrate(sp.Abs(f), (x, -1, 0))
    S1 = sp.integrate(f, (x, -1, 0))
    S2 = -sp.integrate(f, (x, 0, 3))
    S3 = sp.integrate(f, (x, 3, k))
    sol = [s for s in sp.solve(sp.Eq(2 * (S2 / 3), S1 + S3 - 20), k) if s > 3]
    V.check(8, "k", sol[0], V.ans(8, "1"))


def p09():
    A, B = sp.symbols("a B")
    f = x**3 - A * x**2 + B
    Bsol = sp.solve(f.subs(x, 1) - 3, B)[0]
    V.check(9, "1-1 b", Bsol, V.ans(9, "1-1", "b"))
    fa = f.subs(B, Bsol)
    # x>1 에서 f' >= 0 인 A 의 범위: f'(x)=3x(x-2A/3) ; 임계 A 는 2A/3=1
    d = sp.diff(fa, x)
    ok = all(min(d.subs(A, av).subs(x, 1 + sp.Rational(j, 20)) for j in range(0, 400)) >= 0 for av in [sp.Rational(3, 2), 1, 0, -5]) \
        and min(d.subs(A, sp.Rational(31, 20)).subs(x, 1 + sp.Rational(j, 20)) for j in range(0, 400)) < 0
    V.report(9, ok, "a<=3/2 에서 x>1 증가, a 가 조금 커지면 증가 실패(표본)")
    rng = V.ans(9, "1-1").split(";")[1].replace(" ", "")
    V.report(9, rng == "a<=3/2", f"범위 문자열 {rng}")
    V.check(9, "1-2 f(3) 최솟값", (27 - 9 * A + Bsol).subs(A, sp.Rational(3, 2)), V.ans(9, "1-2"))


def p10():
    m = sp.Symbol("m")
    p = q_ = None
    # (1) |f| 연속: a = -3/(p-6) = 3/(q-6), q-p=m
    P, Q, Aa = sp.symbols("P Q Aa", real=True)
    sol = sp.solve([Aa + 3 / (P - 6), Aa - 3 / (Q - 6), Q - P - m], [Aa, P, Q], dict=True)[0]
    V.check(10, "1-1 a_m", sp.simplify(sol[Aa]), V.ans(10, "1-1"))
    # (2) 교점 개수: |f(x)| = 3/|x-6| (x<p or x>q), 6/m (p<=x<=q); g=|(x-5)(x-7)|
    g = sp.Abs(x**2 - 12 * x + 35)
    total = 0
    for mv in range(1, 11):
        pv, qv, av = 6 - sp.Rational(mv, 2), 6 + sp.Rational(mv, 2), sp.Rational(6, mv)
        pts = set()
        # 구간 [p,q]: g(x)=a
        for lo, hi, expr in [(pv, qv, None)]:
            for s in sp.solve(x**2 - 12 * x + 35 - av, x) + sp.solve(x**2 - 12 * x + 35 + av, x):
                if s.is_real and pv <= s <= qv:
                    pts.add(sp.nsimplify(s))
        # 구간 밖: 3/|x-6| = g(x)
        for hlo, hhi in [(-sp.oo, pv), (qv, sp.oo)]:
            for sgn in (1, -1):
                eq = sp.numer(sp.together(3 / (sgn * (x - 6)) - sgn2 * (x**2 - 12 * x + 35))) if False else None
            for gs in (1, -1):
                for xs in (1, -1):
                    # 3/(xs*(x-6)) = gs*(x^2-12x+35)
                    e = sp.numer(sp.together(3 / (xs * (x - 6)) - gs * (x**2 - 12 * x + 35)))
                    for s in sp.Poly(e, x).nroots(n=30):
                        if abs(sp.im(s)) < 1e-20:
                            sv = sp.re(s)
                            if sv != 6 and ((sv < pv) if hlo == -sp.oo else (sv > qv)):
                                if abs(3 / abs(sv - 6) - abs(sv**2 - 12 * sv + 35)) < 1e-9:
                                    pts.add(sp.Float(sv, 12))
        # 중복 제거(수치)
        vals = sorted({round(float(sp.N(s, 20)), 8) for s in pts})
        total += len(vals)
        V.report(10, True, f"m={mv}: 교점 {len(vals)}")
    V.check(10, "1-2 합", total, V.ans(10, "1-2"))


if __name__ == "__main__":
    V.run(globals())
