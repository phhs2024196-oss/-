import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy.parsing.sympy_parser import parse_expr
V = Verifier("SHU-G25N", "items/SHU/PAST")
P = lambda s: parse_expr(s.replace("^", "**"), local_dict={"x": x, "pi": sp.pi})
def kv(s):
    return dict(p.split("=", 1) for p in s.replace(" ", "").split(";") if "=" in p)
R = sp.Rational

def p07():
    f = -R(1, 3)**(x - 1) + 1
    mn, mx = extrema(f, x, -3, 5)
    V.check(7, "m", mn, P(V.ans(7, "1", "m")))
    V.check(7, "M", mx, P(V.ans(7, "1", "M")))
    V.report(7, sp.diff(f, x).subs(x, 0) > 0 and sp.diff(f, x).subs(x, 3) > 0, "증가함수")

def p08():
    a, b, c = sp.symbols("a b c", real=True)
    f = a * x**3 + b * x**2 + c * x - R(1, 3)
    d = sp.diff(f, x)
    s = sp.solve([d.subs(x, 0) + 4, d.subs(x, -1), d.subs(x, 2)], [a, b, c], dict=True)
    V.check(8, "해 개수", len(s), 1)
    for k_, v in (("a", a), ("b", b), ("c", c)):
        V.check(8, k_, s[0][v], P(V.ans(8, "1", k_)))
    g = d.subs(s[0])
    V.report(8, g.subs(x, -2) > 0 and g.subs(x, 0) < 0 and g.subs(x, 3) > 0, "x=-1 극대, x=2 극소")

def p09():
    a = sp.symbols("a2:6")
    an = {1: sp.Symbol("a1")}
    A2 = sp.Symbol("A2")
    a2 = sp.solve(sp.Eq(3 * A2 + 5 * (-R(1, 3) * A2), 7), A2)[0]
    V.check(9, "a2", a2, P(V.ans(9, "1", "a2")))
    a3 = -R(1, 3) * a2
    V.check(9, "a3", a3, P(V.ans(9, "1", "a3")))
    a5 = a3 * R(1, 9)
    V.check(9, "a5", a5, P(V.ans(9, "1", "a5")))
    V.check(9, "(가) 확인", 3 * a2 + 5 * a3, 7)

def p10():
    a, b = sp.symbols("a b", real=True)
    av = sp.solve(sp.Eq(1 / (8 * sp.sqrt(2 + a)), R(1, 32)), a)
    V.check(10, "a", av[0], P(V.ans(10, "1", "a")))
    bv = -sp.sqrt(2 + av[0])
    V.check(10, "b", bv, P(V.ans(10, "1", "b")))
    lim = sp.limit((sp.sqrt(x + av[0]) + bv) / (x**2 - 4), x, 2)
    V.check(10, "극한 1/32", lim, R(1, 32))

def p11():
    th = sp.symbols("theta", real=True)
    D4 = sp.cos(th)**2 - sp.sqrt(3) * sp.sin(th) * sp.cos(th)
    cand = sp.solveset(sp.cos(th), th, sp.Interval.open(0, sp.pi)) | sp.solveset(sp.cos(th) - sp.sqrt(3) * sp.sin(th), th, sp.Interval.open(0, sp.pi))
    sols = sorted(cand, key=lambda r: float(r))
    import numpy as np
    grid = np.linspace(0.0005, np.pi - 0.0005, 6283)
    Dv = [float(D4.subs(th, g_)) for g_ in grid[::20]]
    zeros = [g_ for g_, v in zip(grid[::20], Dv) if abs(v) < 1e-3]
    V.report(11, all(any(abs(float(r) - z) < 0.1 for r in sols) for z in zeros), "격자 탐색 D4≈0 인 점은 모두 해 근방")
    exp = sorted([P(s) for s in kv("theta=" + V.ans(11, "1").split("=", 1)[1].replace(",", "|"))["theta"].split("|")], key=lambda r: float(r))
    V.check(11, "θ 집합", sols, exp)
    for r in sols:
        eq = sp.sqrt(3) * x**2 + 2 * sp.cos(r) * x + sp.sin(r) * sp.cos(r)
        V.check(11, f"중근 판별식 θ={r}", sp.simplify(sp.discriminant(eq, x)), 0)

def p12():
    a, b = sp.symbols("a b", real=True)
    f = x**2 + a * x + b
    s = sp.solve([sp.integrate((x - 1) * f, (x, -1, 1)) - R(8, 3), sp.integrate(x**2 * f, (x, -1, 1)) + R(4, 15)], [a, b], dict=True)
    V.check(12, "해 개수", len(s), 1)
    V.check(12, "a", s[0][a], P(V.ans(12, "1", "a")))
    V.check(12, "b", s[0][b], P(V.ans(12, "1", "b")))
    V.check(12, "a-3b=5", (a - 3 * b).subs(s[0]), 5)

def p13():
    a, b, c = sp.symbols("a b c", real=True)
    f = a * x**3 + b * x**2 + c * x
    s = sp.solve([f.subs(x, 1) - 4, sp.diff(f, x).subs(x, 1) - 16, sp.expand(f + f.subs(x, -x)).coeff(x, 2)], [a, b, c], dict=True)
    V.check(13, "해 개수", len(s), 1)
    for k_, v in (("a", a), ("b", b), ("c", c)):
        V.check(13, k_, s[0][v], P(V.ans(13, "1", k_)))
    F = f.subs(s[0])
    V.check(13, "(가) 극한", sp.limit((F - 4) / (x - 1), x, 1), 16)
    V.check(13, "(나) f(x)+f(-x)=0", sp.expand(F + F.subs(x, -x)), 0)
    # 다른 풀이: c=4-a, 극한 2a+4=16
    V.check(13, "ALT a", sp.solve(sp.Eq(2 * a + 4, 16), a)[0], P(V.ans(13, "1", "a")))

def p14():
    sols = []
    for av in range(1, 400):
        L = sp.log(av, 10)
        if av == 10:
            continue
        # 모든 x: (1-L)x^2+2(1-L)x-1<0 ⇔ 1-L<0 이고 D<0 (수치 최댓값으로도 확인)
        k1 = 1 - L
        ok = float(k1) < 0 and float(4 * k1**2 + 4 * k1) < 0
        mx = float(k1 * (-1)**2 + 2 * k1 * (-1) - 1) if float(k1) < 0 else None
        ok2 = mx is not None and mx < 0
        V.report(14, ok == ok2, f"a={av} 판별식·최댓값 방법 일치") if av in (10 + 1, 99, 100, 101, 5) else None
        if ok:
            sols.append(av)
    V.check(14, "범위", (min(sols), max(sols)), (11, 99))
    V.check(14, "합", sum(sols), P(V.ans(14, "1")))
    V.check(14, "등차합 공식", 89 * (11 + 99) // 2, P(V.ans(14, "1")))

def p15():
    a = sp.symbols("a", real=True)
    f0 = x**3 - 6 * x**2 + a * x
    av = sp.solve(sp.diff(f0, x).subs(x, 1), a)[0]
    V.check(15, "a", av, P(V.ans(15, "1", "a")))
    f, g = f0.subs(a, av), f0.subs(a, av) + 1
    def rng(F):
        cr = sorted(sp.solve(sp.diff(F, x), x), key=float)
        return F.subs(x, cr[1]), F.subs(x, cr[0])
    (flo, fhi), (glo, ghi) = rng(f), rng(g)
    V.check(15, "f 극값(극솟,극댓)", (flo, fhi), (0, 4))
    V.check(15, "g 극값(극솟,극댓)", (glo, ghi), (1, 5))
    bs = [bb for bb in range(-20, 21) if 0 < bb < 4 and 1 < bb < 5]
    # 세 점 교점 수 직접 확인
    def n_real(F, bb):
        return len(set(sp.real_roots(sp.Poly(F - bb, x))))
    ok = [bb for bb in range(-20, 21) if n_real(f, bb) == 3 and n_real(g, bb) == 3]
    V.check(15, "정수 b", ok, [int(v) for v in kv(V.ans(15, "1"))["b"].split(",")])
    V.check(15, "합", sum(ok), P(kv(V.ans(15, "1"))["sum"]))

if __name__ == "__main__": V.run(globals())
