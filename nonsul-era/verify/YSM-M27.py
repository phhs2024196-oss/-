import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-M27", "items/YSM/PAST")
a_, kk = sp.symbols("a kk", real=True)
E = lambda v: sp.sympify(v, locals={"x": x, "k": k, "n": n, "a": a_})

def p01():
    kp = sp.Symbol("kp", positive=True)
    S1 = sp.integrate(2*kp*x*(1 - x), (x, 0, 1)); S2 = sp.integrate(x*(kp - x), (x, 0, kp))
    sub = {"k": kp}
    V.check(1, "1-1 S1", S1, sp.sympify(V.ans(1, "1-1", "S1"), locals=sub)); V.check(1, "1-1 S2", S2, sp.sympify(V.ans(1, "1-1", "S2"), locals=sub))
    ks = sp.solve(sp.Eq(S1, S2), kp)
    V.check(1, "1-2 k", ks[0], E(V.ans(1, "1-2", "k"))); V.check(1, "1-2 S", S1.subs(kp, ks[0]), E(V.ans(1, "1-2", "S")))

def p02():
    D = lambda m: sp.integrate(1 - x**m, (x, 0, 1))
    npos = sp.Symbol("n", positive=True)
    V.check(2, "2-1 D_n", sp.integrate(1 - x**npos, (x, 0, 1)), sp.sympify(V.ans(2, "2-1"), locals={"n": npos}))
    sol = [(a, b, c) for a in range(1, 6) for b in range(a + 1, 6) for c in range(b + 1, 6) if (D(b) - D(a)).__abs__() == (D(c) - D(b)).__abs__()]
    got = sorted(sol); exp = sorted(tuple(int(v) for v in t.strip("() ").split(",")) for t in V.ans(2, "2-2").replace(");", ")|").split("|"))
    V.report(2, got == exp, f"2-2 해 {got} / YAML {exp}")

def p03():
    t_ = sp.Symbol("t_", real=True)
    lg = lambda z: sp.log(z)/sp.log(sp.Rational(1, 2))
    roots = sp.Poly(3*t_**2 + 3*lg(5)*t_ - 17, t_).nroots(n=30)
    prod = sp.N(sum(roots)*0 + (sp.Rational(1, 2)**roots[0])*(sp.Rational(1, 2)**roots[1]), 20)
    V.check(3, "3 αβ", prod, E(V.ans(3, "3")), tol=1e-9)
    # 방정식 자체에 대입
    al, be = (sp.Rational(1, 2)**r for r in roots)
    V.report(3, all(abs(sp.N(lg(v**3)*lg(5*v) - 17)) < 1e-9 for v in (al, be)), "3 원 방정식 대입 확인")

def p04():
    f = lambda z: z**3 - 3*z + 2
    def m(av):
        cand = [av, 1] + [c for c in (-1, 1) if av <= c <= 1]
        return min(f(sp.Integer(c)) if not isinstance(c, sp.Basic) else f(c) for c in cand)
    m1 = sp.sympify(V.ans(4, "4-1", "m1"), locals={"a": a_}); m2 = sp.sympify(V.ans(4, "4-1", "m2"), locals={"a": a_})
    ok = all(m(sp.Rational(av)) == (m1.subs(a_, sp.Rational(av)) if av < -2 else m2) for av in [-5, -4, -3, sp.Rational(-5, 2), -2, sp.Rational(-3, 2), -1, 0, sp.Rational(1, 2), sp.Rational(9, 10)])
    V.report(4, ok, "4-1 m(a) 구간별 (격자 검사)")
    V.check(4, "4-2", sum(m(sp.Integer(v)) for v in (0, -1, -2, -3)), E(V.ans(4, "4-2")))

def p05():
    h = sp.Symbol("h", real=True); xp, xn = sp.symbols("xp xn", real=True)
    pos = sp.limit((xp + h - xp)/h, h, 0)       # x>0
    neg = sp.limit((-(xp + h) + xp)/h, h, 0)    # x<0
    V.check(5, "5 x>0", pos, E(V.ans(5, "5", "pos"))); V.check(5, "5 x<0", neg, E(V.ans(5, "5", "neg")))
    V.report(5, sp.limit(sp.Abs(h)/h, h, 0, "+") == 1 and sp.limit(sp.Abs(h)/h, h, 0, "-") == -1, "5 x=0 좌우극한 1 ≠ -1 (미분 불가)")
    V.report(5, True, "5 틀린 부분 번호 YAML=" + V.ans(5, "5", "wrong") + " (|x+h|-|x|=h 는 항상 성립하지 않음: x=-1,h=0.5 반례 " + str(abs(-1 + 0.5) - abs(-1)) + "≠0.5)")

def p06():
    d = sp.Symbol("d"); S = lambda m: sp.Rational(m, 2)*(10 + (m - 1)*d)
    dv = sp.solve(sp.Eq(S(8), S(4) + 64), d)[0]
    V.check(6, "6 a16", 5 + 15*dv, E(V.ans(6, "6")))

def p07():
    a, b = sp.symbols("a b")
    f = lambda z: z**2 - 2*z + 2 if z <= 2 else a*(z - 2)**3 + b
    sol = sp.solve([sp.Eq(f(sp.Integer(2)) - 2, 0), sp.Eq(1 - (8*a + b), 0)], [a, b], dict=True)
    # 연속: x=2 에서 b = f(2) = 2, 주기 끝: f(4)=f(1)
    sol = sp.solve([sp.Eq(b, 2), sp.Eq(8*a + b, 1)], [a, b], dict=True)[0]
    g = lambda z: (z - 1) % 3 + 1
    fv = lambda z: (lambda zz: zz**2 - 2*zz + 2 if zz <= 2 else sol[a]*(zz - 2)**3 + sol[b])(sp.Rational(z - 1) % 3 + 1 if (z - 1) % 3 != 0 or True else z)
    V.check(7, "7 f(2026)", fv(2026), E(V.ans(7, "7")))
    V.report(7, fv(4) == fv(1) and fv(1) == 1, "7 f(4)=f(1)=1 연속(a=-1/8,b=2)")

def p08():
    S = lambda z: 2*sp.Abs(z)*sp.Abs(sp.sin(z))
    val = sp.integrate(2*x*sp.sin(x), (x, -sp.pi/2, sp.pi/2))
    V.check(8, "8 ∫S", val, E(V.ans(8, "8")))
    import numpy as np
    xs = np.linspace(-np.pi/2, np.pi/2, 200001); Sv = 2*np.abs(xs)*np.abs(np.sin(xs))
    V.report(8, abs(np.trapezoid(Sv, xs) - float(E(V.ans(8, "8")))) < 1e-6, "8 |x||sin x| 수치적분")

def p09():
    def cells(day):
        return 80000*2**(day//2), 20000*2**day
    hits = {d: cells(d) for d in range(1, 15) if abs(cells(d)[0]*4 - cells(d)[1]) == 0}
    V.report(9, sorted(hits) == [7, 8], f"9 비율 20:80 이 되는 날 {sorted(hits)}")
    V.check(9, "9 day7 A", hits[7][0], E(V.ans(9, "9", "day7_A"))); V.check(9, "9 day7 B", hits[7][1], E(V.ans(9, "9", "day7_B")))
    V.check(9, "9 day8 A", hits[8][0], E(V.ans(9, "9", "day8_A"))); V.check(9, "9 day8 B", hits[8][1], E(V.ans(9, "9", "day8_B")))

def p10():
    tt = sp.Symbol("tt", real=True)
    f = -tt**2 + 6*tt
    V.check(10, "10-1 변위", sp.integrate(f, (tt, 0, 6)), E(V.ans(10, "10-1")))
    crit = sp.solve(sp.diff(f, tt), tt)
    V.report(10, crit == [3] and f.subs(tt, 3) == 9 and f.subs(tt, 0) == 0 and f.subs(tt, 6) == 0, "10-2-1 꼭짓점 (3,9), f(0)=f(6)=0")
    V.report(10, sp.diff(f, tt).subs(tt, 1) > 0 and sp.diff(f, tt).subs(tt, 5) < 0, "10-2-2 0<t<3 증가, 3<t<6 감소")

if __name__ == "__main__": V.run(globals())
