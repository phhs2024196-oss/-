#!/usr/bin/env python3
"""KUT-M26 (2026학년도 한국기술교육대 모의논술 01~10) sympy 검산 — 기대값은 YAML 에서 읽는다."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-M26", "items/KUT/PAST")


def p01():
    X, Y, K = sp.symbols("X Y K", positive=True)
    # 8^x = 25^y = k  =>  x = ln k/ln 8, y = ln k/ln 25 ;  1/(3x)+1/(2y)=1
    k_ = sp.symbols("k", positive=True)
    xx, yy = sp.log(k_)/sp.log(8), sp.log(k_)/sp.log(25)
    e = sp.simplify(sp.expand_log(1/(3*xx) + 1/(2*yy), force=True))
    sol = [r for r in sp.solve(sp.Eq(sp.log(8)/(3*sp.log(k_)) + sp.log(25)/(2*sp.log(k_)), 1), k_)]
    V.check(1, "k", sol[0], V.ans(1, "1"))


def p02():
    la, lb = sp.symbols("la lb")   # log_3 a, log_3 b
    s = sp.solve([sp.Eq(la + lb, -2*0 + 4), sp.Eq(lb - la, -2*1 + 4)], [la, lb], dict=True)[0]
    V.check(2, "a+b", 3**s[la] + 3**s[lb], V.ans(2, "1"))


def p03():
    a = sp.symbols("a", real=True)
    # sin+cos=2a/3, sin*cos=(a^2-3)/6 에서 (sin+cos)^2 = 1+2 sin cos
    sols = sp.solve(sp.Eq((2*a/3)**2, 1 + 2*(a**2 - 3)/6), a)
    V.check(3, "a", sols[0], V.ans(3, "1"))
    # 실근 확인: a=0 -> 6x^2-3=0, 근 ±1/sqrt2 = sin,cos 가능
    r = sp.solve(6*x**2 - 3, x)
    V.report(3, len(sols) == 1 and abs(r[0]**2 + r[1]**2 - 1) < 1e-12, f"a=0 근 {r} 제곱합 1")


def p04():
    a, b = sp.symbols("a b")
    f = -2*x**3 + 2*x**2 + a*x + b
    assert sp.limit((f + 2*x**3)/(x**2 + 1), x, sp.oo) == 2
    s = sp.solve([f.subs(x, 0), sp.limit(sp.cancel(f.subs(b, 0)/(x**2 + x)), x, 0) - 4], [a, b], dict=True)[0]
    V.check(4, "f(1)", f.subs(s).subs(x, 1), V.ans(4, "1"))


def p05():
    a, b, c = sp.symbols("a b c")
    f = a*x**2 + b*x + c
    s = sp.solve([sp.diff(f, x).subs(x, 1) - 2, sp.diff(f, x).subs(x, 2), f.subs(x, 2) - 5], [a, b, c], dict=True)[0]
    g = f.subs(s).subs(x, 4*x)
    V.check(5, "g'(-1)", sp.diff(g, x).subs(x, -1), V.ans(5, "1"))


def p06():
    a = sp.symbols("a")
    a0 = sp.solve(sp.Eq(sp.limit(x**2 - a*x, x, 4), 2*4 - 2*a), a)[0]
    f = lambda v: v**2 - a0*v if v < 4 else 2*v - 2*a0
    vals = [f(sp.Rational(i, 100)) for i in range(100, 501)]
    V.check(6, "M+m", max(vals) + min(vals), V.ans(6, "1"))


def p07():
    a, b = sp.symbols("a b")
    h = x**3 - 9*x**2 + a*x + b
    s = sp.solve([h.subs(x, 1), sp.diff(h, x).subs(x, 1)], [a, b], dict=True)[0]
    h = h.subs(s)
    ts = [r for r in sp.solve(sp.diff(h, x), x) if r != 1 and r >= 0]
    V.check(7, "거리", abs(h.subs(x, ts[0])), V.ans(7, "1"))


def p08():
    # f(x)=f(x-1)+2, ∫0^2 f = 10/3 를 만족하는 연속함수 하나를 구성: f(x)=2x+p(x), p 주기 1 (연속), ∫0^1 f = 2/3
    # f(x)=2x + c (0<=x<=1 에서 ∫= 1 + c = 2/3 -> c=-1/3), 주기 확장 f(x)=2(x-floor x)+c+2 floor x
    c = sp.Rational(2, 3) - 1
    f1 = lambda v: 2*v + c          # [0,1]
    I01 = sp.integrate(f1(x), (x, 0, 1))
    I12 = sp.integrate(f1(x - 1) + 2, (x, 1, 2))
    I23 = sp.integrate(f1(x - 2) + 4, (x, 2, 3))
    V.report(8, sp.simplify(I01 + I12 - sp.Rational(10, 3)) == 0, f"구성함수 ∫0^2={I01 + I12}")
    V.check(8, "∫2^3", I23, V.ans(8, "1"))


def p09():
    # 이등분선: |2x-5y|/sqrt29 = |5x+2y|/sqrt29
    X, Y = sp.symbols("X Y", real=True)
    ms = sp.solve(sp.Eq(2*X - 5*Y, 5*X + 2*Y), Y)[0]/X, sp.solve(sp.Eq(2*X - 5*Y, -(5*X + 2*Y)), Y)[0]/X
    slopes = sorted(sp.simplify(m_) for m_ in ms)
    neg = [m_ for m_ in slopes if m_ < 0][0]
    V.check(9, "1-1 기울기", neg, V.ans(9, "1-1"))
    # 제2사분면 점 (-7,3): tanθ=-3/7
    th = sp.atan2(3, -7)
    val = sp.sin(th + sp.pi) + sp.cos(th - sp.pi)
    V.check(9, "1-2", sp.simplify(val), V.ans(9, "1-2"))


def p10():
    tt, A = sp.symbols("t A", real=True)
    v = -tt**2 + 6*tt
    def f(a_):   # 구간별 |v| 적분
        pts = sorted({a_, a_ + 3} | ({6} if a_ < 6 < a_ + 3 else set()))
        return sum(abs(sp.integrate(v, (tt, p, q))) for p, q in zip(pts, pts[1:]))
    f1 = f(sp.Integer(1))
    lim = sp.limit((3*A**2 - 9*A - 18)/A**2, A, sp.oo)   # a>=6 의 f(a)
    chk = all(sp.simplify(f(sp.Integer(a0)) - (3*a0**2 - 9*a0 - 18)) == 0 for a0 in (6, 7, 9))
    V.report(10, chk, "a>=6 에서 f(a)=3a^2-9a-18")
    V.check(10, "1-1", f1 + lim, V.ans(10, "1-1"))
    F2 = sp.Rational(2, 3)*A**3 - 3*A**2 - 9*A + 54   # 3<a<6
    chk2 = all(sp.simplify(f(sp.Rational(a0)) - F2.subs(A, a0)) == 0 for a0 in (4, 5, sp.Rational(7, 2)))
    V.report(10, chk2, "3<a<6 에서 f(a)=2/3 a^3-3a^2-9a+54")
    crit = [r for r in sp.solve(sp.diff(F2, A), A) if 3 < r < 6][0]
    cands = {0: 18, 3: 18, crit: F2.subs(A, crit), 6: 36}
    m_ = min(cands, key=lambda q: float(sp.N(cands[q])))
    # 수치 격자 확인(0<=a<=12, 간격 1/8): 최솟값 위치가 crit 근처
    grid = [sp.Rational(i, 8) for i in range(0, 97)]
    am = min(grid, key=lambda g: float(f(g)))
    V.report(10, abs(float(am - crit)) <= 0.125, f"격자 최소 a={am}, crit={float(crit):.4f}")
    V.check(10, "1-2 m", m_, V.ans(10, "1-2"))


if __name__ == "__main__":
    V.run(globals())
