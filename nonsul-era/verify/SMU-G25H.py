import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
E = lambda v: sp.sympify(v, locals={"x": x, "n": n})  # 식 답은 vlib 의 실수 기호로 읽기

V = Verifier("SMU-G25H", "items/SMU/PAST")
d = sp.symbols("d", real=True)

def p09():
    ds = [z for z in sp.solve(sp.Abs(30 + 10*d) - sp.Abs(30 + 20*d), d) if z != 0] if False else [z for z in sp.solve((30 + 10*d)**2 - (30 + 20*d)**2, d) if z != 0]
    V.report(9, len(ds) == 1, f"공차 {ds}")
    an = lambda kk: 30 + (kk - 1)*ds[0]
    M = next(kk for kk in range(1, 1000) if an(kk) == -20)
    V.check(9, "1 M", M, V.ans(9, "1"))
    Sn = [sum(an(i) for i in range(1, kk + 1)) for kk in range(1, 100)]
    V.check(9, "2 Smax", max(Sn), V.ans(9, "2"))
    V.check(9, "3 합", sum(sp.Rational(4)/(an(kk)*an(kk + 1)) for kk in range(M, 101)), V.ans(9, "3"))

def p10():
    F = E(V.ans(10, "1"))
    V.check(10, "(가)", sp.limit((F.subs(x, x - 1) + 22)/(x**2 - 9), x, 3), sp.Rational(-3, 2))
    fp = sp.diff(F, x)
    V.report(10, sp.expand(fp.subs(x, sp.Rational(1, 2) - x) - fp.subs(x, sp.Rational(3, 2) + x)) == 0, "(나) 대칭")
    V.report(10, sp.Poly(F, x).LC() == 1 and sp.degree(F, x) == 3, "최고차 1 삼차")
    # 유일성: 일반 삼차로 풀기
    b2, b1, b0 = sp.symbols("b2 b1 b0")
    G = x**3 + b2*x**2 + b1*x + b0; gp = sp.diff(G, x)
    so = sp.solve([G.subs(x, 2) + 22, gp.subs(x, 2)/6 + sp.Rational(3, 2)] + sp.Poly(sp.expand(gp.subs(x, sp.Rational(1, 2) - x) - gp.subs(x, sp.Rational(3, 2) + x)), x).all_coeffs(), [b2, b1, b0], dict=True)
    V.check(10, "1 f 유일", [sp.expand(G.subs(s)) for s in so], [F])
    cr = sp.solve(fp, x)
    V.check(10, "2 극대", max(F.subs(x, c) for c in cr if sp.diff(F, x, 2).subs(x, c) < 0), V.ans(10, "2"))
    V.check(10, "3 극소", min(F.subs(x, c) for c in cr if sp.diff(F, x, 2).subs(x, c) > 0), V.ans(10, "3"))

if __name__ == "__main__": V.run(globals())
