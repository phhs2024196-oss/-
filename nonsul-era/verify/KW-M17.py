import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-M17", "items/KW/PAST")
R_ = sp.Rational
al = sp.symbols("alpha", real=True)

def p01():
    from math import gcd
    # [1] 17 | 2^n*3^(2n)-1
    V.proof(1, "1 17의 배수 n<=60", [n_ for n_ in range(1, 61) if (2**n_*3**(2*n_) - 1) % 17])
    # [2] 몫·나머지 유일성(전수)
    V.proof(1, "2 (q,r) 유일", [(a, b) for a in range(1, 80) for b in range(1, 40)
            if len([(q, r) for q in range(0, 80) for r in range(0, b) if a == b*q + r]) != 1])
    # [3](1) 반례, (2)
    f = lambda a: gcd(a, 2)
    V.check(1, "3-1 f(4)", f(4), 2); V.check(1, "3-1 f(2)f(2)", f(2)*f(2), 4)
    V.report(1, "false" in V.ans(1, "3-1", "property") and f(2*2) != f(2)*f(2), "3-1 성질 불성립(반례 a=b=2)")
    V.proof(1, "3-2 f(a)=f(b)=1 이면 f(ab)=f(a)f(b)", [(a, b) for a in range(1, 80) for b in range(1, 80)
            if f(a) == 1 and f(b) == 1 and f(a*b) != f(a)*f(b)])
    # [4] x^2+x+1 로 나눈 나머지가 0
    aa, bb = sp.symbols("a b", real=True)
    poly = sp.Poly(aa*x**2016 + bb*x**523 - 6, x)
    rem = sp.rem(poly, sp.Poly(x**2 + x + 1, x))
    sol = sp.solve(rem.all_coeffs(), [aa, bb], dict=True)[0]
    V.check(1, "4 a,b", [sol[aa], sol[bb]], [6, 0])
    V.check(1, "4 ab", sol[aa]*sol[bb], V.ans(1, "4"))
    # [5] f=(x-2)(x-s), g=(x-2)(x-t), Y={2,3,8} 이므로 {s,t}={3,8} -> X 의 나머지 원소
    s_, t_ = sp.symbols("s t")
    ks = set()
    for s_v, t_v in ((3, 8), (8, 3)):
        fx = (x - 2)*(x - s_v); gx = (x - 2)*(x - t_v)
        X = set(sp.solve(fx + gx, x)); Y = set(sp.solve(fx*gx, x))
        V.check(1, "5 Y", sorted(Y), [2, 3, 8])
        ks |= X - {2}
    V.check(1, "5 k", sorted(ks), [sp.sympify(V.ans(1, "5"))])
    # 참고: 이 f=(x-2)(x-3)=x^2-5x+6 은 a=-5 로 자연수 조건과 맞지 않음(공식 문제 조건 모순 의심) — 보고용
    V.report(1, True, "5 [참고] f=(x-2)(x-3) 의 a=-5 ∉ N: 조건 a,b,c,d∈N 과 공식 답이 모순으로 보임(공식 답은 유지)")

def p02():
    T = sp.symbols("T", real=True)
    # [2] tan(alpha+beta), beta=alpha±pi/2, T=tan(alpha)=f'
    ex = sp.sympify(V.ans(2, "2", "tan(alpha+beta)"), locals={"f'(x)": T}) if False else None
    target = (T**2 - 1)/(2*T)
    for sgn in (1, -1):
        e = sp.tan(al + (al + sgn*sp.pi/2))
        V.check(2, f"2 tan(α+β) β=α{'+' if sgn>0 else '-'}π/2", max(abs(sp.N((e.subs(al, sp.atan(T)) - target).subs(T, tv))) for tv in (R_(1, 2), 2, -3, R_(7, 5))), 0, tol=1e-9)
        V.check(2, f"2 tanα·tanβ", sp.simplify((sp.tan(al)*sp.tan(al + sgn*sp.pi/2)).subs(al, sp.atan(T))), -1)
    V.report(2, "(f'(x)**2-1)/(2*f'(x))" in V.ans(2, "2"), "2 YAML 답 형태")
    # [3](1) alpha=atan(f') 의 미분
    F = sp.Function("F")
    d = sp.simplify(sp.diff(sp.atan(sp.diff(F(x), x)), x) - sp.diff(F(x), x, 2)/(1 + sp.diff(F(x), x)**2))
    V.check(2, "3-1 α'=f''/(1+f'^2)", d, 0)
    # [3](2) f=ln sec x
    fl = sp.log(1/sp.cos(x))
    V.check(2, "3-2 f'=tan x", sp.simplify(sp.diff(fl, x) - sp.tan(x)), 0)
    dal = sp.simplify(sp.diff(fl, x, 2)/(1 + sp.diff(fl, x)**2))
    V.check(2, "3-2 α'", dal, V.ans(2, "3-2"))
    # [4] f 일차/상수 ⇒ α'=0, 이차 이상(f''≠0)이면 α'≠0
    c1, c2 = sp.symbols("c1 c2", real=True)
    lin = c1*x + c2
    V.check(2, "4 일차 α'=0", sp.simplify(sp.diff(lin, x, 2)/(1 + sp.diff(lin, x)**2)), 0)
    V.check(2, "4 f''=0 ⇒ f'=C1 ⇒ f=C1x+C2", sp.integrate(sp.integrate(0, x) + c1, x) + c2, lin)
    V.check(2, "4 x^2 이면 α'≠0", bool(sp.simplify((sp.diff(x**2, x, 2)/(1 + sp.diff(x**2, x)**2)).subs(x, 1)) != 0), True)
    # [5]
    g = -sp.sin(x)/(1 + sp.cos(x)**2)
    V.check(2, "5 g", sp.simplify(sp.diff(sp.atan(sp.cos(x)), x) - g), 0)   # α=atan(cos x) 에서 (|α|<π/2 구간 기준) g=α'
    V.check(2, "5 g 직접", sp.simplify(sp.diff(sp.sin(x), x, 2)/(1 + sp.diff(sp.sin(x), x)**2) - g), 0)
    gp = sp.simplify(sp.diff(g, x))
    V.check(2, "5 g'", sp.simplify(gp + sp.cos(x)*(2 + sp.sin(x)**2)/(1 + sp.cos(x)**2)**2), 0)
    crit = sp.solveset(sp.diff(g, x), x, sp.Interval(0, 2*sp.pi))
    V.check(2, "5 극점", sorted(crit, key=float), [sp.pi/2, 3*sp.pi/2])
    vals = {c: sp.simplify(g.subs(x, c)) for c in list(crit) + [0, 2*sp.pi]}
    xM = max(vals, key=lambda c: float(vals[c])); xm = min(vals, key=lambda c: float(vals[c]))
    V.check(2, "5 x_M", xM, sp.sympify(V.ans(2, "5", "xM")))
    V.check(2, "5 x_m", xm, sp.sympify(V.ans(2, "5", "xm")))
    V.check(2, "5 g(x_M)", vals[xM], V.ans(2, "5", "g_xM")); V.check(2, "5 g(x_m)", vals[xm], V.ans(2, "5", "g_xm"))
    # α(x_M)=α(x_m): tan α = cos x = 0 → α=0 (α∈[0,π), α≠π/2)
    V.check(2, "5 α(x_M)", sp.cos(xM), 0); V.check(2, "5 α(x_m)", sp.cos(xm), 0)
    V.check(2, "5 α 값", 0, V.ans(2, "5", "alpha_xM")); V.check(2, "5 α 값", 0, V.ans(2, "5", "alpha_xm"))

if __name__ == "__main__": V.run(globals())
