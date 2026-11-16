#!/usr/bin/env python3
"""J06 정적분을 포함한 관계식으로 함수 결정 — 검산 스크립트 (sympy/mpmath)."""
import sys
import random
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
x, t, y, h = sp.symbols('x t y h', real=True)
F = sp.Function('f')
results = []
random.seed(2027)


def report(pid, ok, msg):
    results.append(bool(ok))
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def zero(expr):
    return sp.simplify(expr) == 0


def rand_pts(k=20, lo=-2, hi=2):
    return [sp.Rational(random.randint(int(lo * 100), int(hi * 100)), 100) for _ in range(k)]


# ---------------------------------------------------------------- J06-01
def p01():
    k = sp.symbols('k')
    ksol = sp.solve(sp.Eq(k, sp.integrate(t * (sp.exp(t) + k), (t, 0, 1))), k)
    ok = ksol == [2]                                                     # (1)
    f = sp.exp(x) + ksol[0]
    ok &= zero(f - (sp.exp(x) + sp.integrate(t * f.subs(x, t), (t, 0, 1))))
    g = f - (sp.E * x + 2)
    # (2) g>=0, 등호 x=1 뿐 [증명-수치확인]
    ok &= sp.solve(sp.diff(g, x), x) == [1] and zero(g.subs(x, 1))
    ok &= all(g.subs(x, p).evalf() >= 0 for p in rand_pts(200, -5, 5))
    area = sp.integrate(g, (x, 0, 1))                                    # (3)
    ok &= zero(area - (sp.E / 2 - 1))
    report("J06-01", ok, f"∫t f=2, f=e^x+2, 접선 아래 넓이={sp.simplify(area)}")


# ---------------------------------------------------------------- J06-02
def p02():
    xp = sp.symbols('x', positive=True)
    tp = sp.symbols('t', positive=True)
    # 미분한 식 f + x f' = 2 f + 3x^2, 초기조건 x=1 대입: f(1)=1
    ode = sp.Eq(F(xp) + xp * F(xp).diff(xp), 2 * F(xp) + 3 * xp ** 2)
    sol = sp.dsolve(ode, F(xp), ics={F(1): 1}).rhs
    ok = zero(sol - (3 * xp ** 2 - 2 * xp))
    ok &= zero(sp.diff(sol / xp, xp) - 3)                                # (1)
    ok &= zero(xp * sol - (2 * sp.integrate(sol.subs(xp, tp), (tp, 1, xp)) + xp ** 3))  # 원식
    roots = sorted(sp.solve(sol, xp))
    area = sp.integrate(-sol, (xp, 0, roots[-1]))                        # (3)
    ok &= area == sp.Rational(4, 27)
    report("J06-02", ok, f"f={sol}, 원식 항등 확인, 넓이={area}")


# ---------------------------------------------------------------- J06-03
def p03():
    sol = sp.dsolve(sp.Eq(F(x).diff(x), 2 * x + F(x)), F(x), ics={F(0): 0}).rhs
    ok = zero(sol - (2 * sp.exp(x) - 2 * x - 2))
    ok &= zero(sol - (x ** 2 + sp.integrate(sol.subs(x, t), (t, 0, x))))  # 원식
    ok &= zero(sp.diff(sp.exp(-x) * sol, x) - 2 * x * sp.exp(-x))       # (1)
    d = sol - x ** 2
    ok &= all(d.subs(x, p).evalf() >= 0 for p in rand_pts(300, 0, 6))   # (2) [증명-수치확인]
    area = sp.integrate(d, (x, 0, 1))
    ok &= zero(area - (2 * sp.E - sp.Rational(16, 3)))
    report("J06-03", ok, f"f={sol}, f>=x^2 (x>=0) 격자, 넓이={sp.simplify(area)}≈{float(area):.5f}")


# ---------------------------------------------------------------- J06-04
def p04():
    # f'(x)=lim [f(x)(e^h-1)+f(h)e^x]/h = f(x)+e^x f'(0)
    sol = sp.dsolve(sp.Eq(F(x).diff(x), F(x) + sp.exp(x)), F(x), ics={F(0): 0}).rhs
    ok = zero(sol - x * sp.exp(x))
    fx = lambda v: sol.subs(x, v)
    ok &= all(zero(fx(a + b) - fx(a) * sp.exp(b) - fx(b) * sp.exp(a)) for a, b in zip(rand_pts(), rand_pts()))
    ok &= sp.diff(sol, x).subs(x, 0) == 1
    crit = sp.solve(sp.diff(sol, x), x)
    m = sol.subs(x, crit[0])
    ok &= crit == [-1] and zero(m + sp.exp(-1))
    area = sp.integrate(-sol, (x, -1, 0))
    ok &= zero(area - (1 - 2 / sp.E))
    report("J06-04", ok, f"f={sol}, 함수방정식 20점 항등, 최솟값={m}, 넓이={sp.simplify(area)}")


# ---------------------------------------------------------------- J06-05
def p05():
    xp = sp.symbols('x', positive=True)
    # f'(x)=f'(1)+f(x)/x , f(1)=0
    sol = sp.dsolve(sp.Eq(F(xp).diff(xp), 1 + F(xp) / xp), F(xp), ics={F(1): 0}).rhs
    ok = zero(sol - xp * sp.log(xp))
    pts = [sp.Rational(random.randint(1, 500), 100) for _ in range(20)]
    ok &= all(zero(sp.expand_log(sol.subs(xp, a * b) - a * sol.subs(xp, b) - b * sol.subs(xp, a), force=True)) for a, b in zip(pts, pts[::-1]))
    ok &= zero(sp.diff(sol, xp).subs(xp, 1) - 1)
    chord = sol.subs(xp, 1) + (sol.subs(xp, sp.E) - sol.subs(xp, 1)) / (sp.E - 1) * (xp - 1)
    ok &= all((chord - sol).subs(xp, 1 + (sp.E - 1) * sp.Rational(i, 100)).evalf() >= -1e-30 for i in range(101))
    area = sp.integrate(chord - sol, (xp, 1, sp.E))
    ok &= zero(area - (sp.E ** 2 - 2 * sp.E - 1) / 4)
    report("J06-05", ok, f"f={sol}, 함수방정식 20점 항등, 현-곡선 넓이={sp.simplify(area)}≈{float(area):.5f}")


# ---------------------------------------------------------------- J06-06
def p06():
    lam, A, u = sp.symbols('lambda A u', real=True)
    eq = sp.Eq(A, sp.integrate(t ** 2 + lam * t * A, (t, 0, 1)))
    ok = sp.solve(eq.subs(lam, 2), A) == []                             # (1) λ=2 해 없음
    Asol = sp.solve(eq, A)[0]
    ok &= zero(Asol - sp.Rational(2, 3) / (2 - lam))
    usol = sp.simplify(lam * Asol)
    I = sp.integrate((x ** 2 + u * x) ** 2, (x, 0, 1))
    ok &= zero(I - (sp.Rational(1, 5) + u / 2 + u ** 2 / 3))            # (2)
    ustar = sp.solve(sp.diff(I, u), u)[0]
    lamstar = sp.solve(sp.Eq(usol, ustar), lam)
    Imin = I.subs(u, ustar)
    ok &= ustar == sp.Rational(-3, 4) and lamstar == [18] and Imin == sp.Rational(1, 80)
    # λ 에 대한 수치 탐색으로 최소 확인
    G = sp.lambdify(lam, I.subs(u, usol))
    grid = [i / 10 for i in range(-500, 1000) if abs(i / 10 - 2) > 1e-9]
    ok &= min(G(v) for v in grid) >= 1 / 80 - 1e-12
    report("J06-06", ok, f"λ=2 무해, u=2λ/(3(2-λ)), 최소 λ={lamstar[0]}, 최솟값={Imin}")


# ---------------------------------------------------------------- J06-07
def p07():
    # 미분: f' = 2 + f + (f - 2x) = 2f + 2 - 2x, f(0)=0
    sol = sp.dsolve(sp.Eq(F(x).diff(x), 2 * F(x) + 2 - 2 * x), F(x), ics={F(0): 0}).rhs
    ok = zero(sol - (sp.exp(2 * x) + 2 * x - 1) / 2)
    ok &= zero(sol - (2 * x + sp.integrate(sp.exp(x - t) * sol.subs(x, t), (t, 0, x))))  # 원식
    g = sp.exp(-x) * sol
    ok &= zero(sp.diff(g, x) - g - 2 * (1 - x) * sp.exp(-x)) and g.subs(x, 0) == 0  # (1)
    slope = sp.diff(sol, x).subs(x, 0)
    ok &= slope == 2
    area = sp.integrate(sol - slope * x, (x, 0, sp.Rational(1, 2)))
    ok &= zero(area - (sp.E / 4 - sp.Rational(5, 8)))
    report("J06-07", ok, f"f={sp.simplify(sol)}, 원식 항등, 넓이={sp.simplify(area)}≈{float(area):.5f}")


# ---------------------------------------------------------------- J06-08
def p08():
    a = sp.symbols('a', real=True)
    eq = sp.expand(sp.integrate((6 * t - a) ** 2, (t, 0, 1)) - a)
    ok = zero(eq - (a ** 2 - 7 * a + 12))                                 # (1)
    sols = sorted(sp.solve(eq, a))
    ok &= sols == [3, 4]
    fs = [6 * x - s for s in sols]
    ok &= all(zero(f - (6 * x - sp.integrate(f.subs(x, t) ** 2, (t, 0, 1)))) for f in fs)  # (2)
    vals = [sp.integrate(sp.Abs(f), (x, 0, 1)) for f in fs]
    ok &= vals == [sp.Rational(3, 2), sp.Rational(5, 3)]
    report("J06-08", ok, f"a∈{sols}, f=6x-3 또는 6x-4, ∫|f|={vals} → 최소 3/2")


# ---------------------------------------------------------------- J06-09
def p09():
    ok = True
    # f(0)(1+f(0))=0 의 해
    c = sp.symbols('c')
    ok &= sorted(sp.solve(sp.Eq(c, 2 * c + c ** 2), c)) == [-1, 0]
    sol = sp.dsolve(sp.Eq(F(x).diff(x), 2 * (1 + F(x))), F(x), ics={F(0): 0}).rhs
    ok &= zero(sol - (sp.exp(2 * x) - 1))
    fx = lambda v: sol.subs(x, v)
    ok &= all(zero(fx(a + b) - fx(a) - fx(b) - fx(a) * fx(b)) for a, b in zip(rand_pts(), rand_pts()))
    ok &= sp.diff(sol, x).subs(x, 0) == 2
    g = sol + sol.subs(x, -x)   # f(x) - (-f(-x))
    ok &= all(g.subs(x, p).evalf() >= 0 for p in rand_pts(200, 0, 3))
    area = sp.integrate(g, (x, 0, sp.log(2) / 2))
    ok &= zero(area - (sp.Rational(3, 4) - sp.log(2)))
    report("J06-09", ok, f"f={sol}, 함수방정식 20점 항등, 넓이={sp.simplify(area)}≈{float(area):.5f}")


# ---------------------------------------------------------------- J06-10
def p10():
    sol = sp.dsolve(sp.Eq(F(x).diff(x), F(x) ** 2), F(x), ics={F(0): 1}).rhs
    ok = zero(sol - 1 / (1 - x))
    ok &= zero(sol - (1 + sp.integrate(sol.subs(x, t) ** 2, (t, 0, x))))   # 원식 (0<=x<1)
    touch = sp.solve(sp.Eq(sol, 4 * x), x)
    ok &= touch == [sp.Rational(1, 2)]
    ok &= all((sol - 4 * x).subs(x, sp.Rational(i, 200)) >= 0 for i in range(0, 200))
    area = sp.integrate(sol - 4 * x, (x, 0, sp.Rational(1, 2)))
    ok &= zero(area - (sp.log(2) - sp.Rational(1, 2)))
    report("J06-10", ok, f"f={sol}, y=4x 와 x=1/2 에서 접함, 넓이={sp.simplify(area)}≈{float(area):.5f}")


for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        fn()
    except Exception as e:  # noqa
        report(fn.__name__, False, f"예외 {e!r}")

passed = sum(results)
print(f"SUMMARY T06: {passed}/10 PASS")
if passed != 10:
    sys.exit(1)
