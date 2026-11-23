import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUS-G25A", "items/KUS/PAST")


def p01():
    U = frozenset(range(4))
    subsets = [frozenset(c) for r in range(5) for c in itertools.combinations(range(4), r)]
    bad = []
    for A in subsets:
        for B in subsets:
            env = dict(A=A, B=B, Ac=U - A, Bc=U - B)
            e = {i: eval(V.ans(1, "1", i), {}, env) for i in "124"}
            lines = [(A | B) - (A & B) | A, ((A | B) & (U - (A & B))) | A, e["1"] | A, e["2"],
                     ((B | A) | A) & (((U - B) | (U - A)) | A), e["4"], (B | (A | A)) & ((U - B) | U), B | A, A | B]
            if len(set(lines)) != 1 or (B | A) != (A | B):
                bad.append((set(A), set(B)))
    V.proof(1, "①②④ 각 줄 동치(|U|=4 전수)", bad)
    V.report(1, V.ans(1, "1", "3") == "교환법칙", "③ 교환법칙(B∪A=A∪B 전수 확인)")


def p02():
    nums = sorted((int("".join(map(str, p))) for p in itertools.permutations(range(7), 5)
                   if p[0] != 0 and p[-1] not in (0, 5)), reverse=True)
    V.check(2, "105번째", nums[104], V.ans(2, "1"))
    pre = [sum(1 for v in nums if str(v).startswith(s)) for s in ("65", "64", "635", "634", "632")]
    V.report(2, pre == [48, 36, 9, 6, 6], f"꼴별 개수 {pre}")


def p03():
    from sympy import Rational as R
    xk, S = R(1), None
    for k in range(1, 11):
        yk = xk**2 / 2
        A2 = sp.solve(sp.Eq(0, xk * (x - xk) + yk), x)[0]          # 접선 y=0
        Q = sp.solve(sp.Eq(0, -(x - xk) / xk + yk), x)[0]          # 수직선 y=0
        S = abs(sp.Rational(1, 2) * (Q - A2) * yk)
        xk = A2
    V.check(3, "S10", S, V.ans(3, "1"))


def p04():
    d, r = sp.symbols("d r", positive=True)
    rs = [s for s in sp.solve(r**3 - 4 * r**2 - 12 * r, r) if s > 0]   # b9/b6 등을 r로 전개한 식
    b1, rr = sp.symbols("b1 rr")
    expr = sp.simplify((b1*rr**8)/(b1*rr**5) - 4*(b1*rr**4)/(b1*rr**2) - 12*(b1*rr)/b1)
    rs2 = [s for s in sp.solve(expr, rr) if s > 0]
    dd = sp.log(rs2[0])
    f = (x - 2) * (x - 2 + dd) * (x - 2 - dd)
    g = sp.expand(f.subs(x, x + 2))
    odd = sp.simplify(g + g.subs(x, -x)) == 0
    val = f.subs(x, 1) * sp.diff(sp.log(sp.Abs(f)), x).subs(x, 3)   # 원본은 f(3)<0 이므로 |f| 로 계산
    V.report(4, rs == rs2 == [6] and odd, f"r={rs2}, g 원점대칭={odd}, f(3)={sp.N(f.subs(x,3),5)}")
    V.check(4, "값", sp.simplify(val), V.ans(4, "1"))


def p05():
    r, m = sp.symbols("r m", positive=True)
    D = sp.expand((m - r) ** 2 - (m**2 + 1))
    ms = sp.solve(D, m)[0]
    gx = sp.simplify((r - ms) / (ms**2 + 1))
    V.check(5, "f", ms, sp.sympify(V.ans(5, "1", "f"), locals={"r": r}))
    V.check(5, "g", gx, sp.sympify(V.ans(5, "1", "g"), locals={"r": r}))
    lim = sp.limit(ms * gx.subs(r, 2 * r), r, sp.oo)
    V.check(5, "lim", lim, V.ans(5, "1", "lim"))
    # 접점이 원 위에 있고 접선 위에 있는지 수치 확인
    xr = gx.subs(r, 3); mr = ms.subs(r, 3)
    V.report(5, abs(((xr - 3) ** 2 + (mr * xr + 1) ** 2 - 9).evalf()) < 1e-12, "접점이 원 위")


def p06():
    F = sp.Function("f")
    sol = sp.dsolve(sp.Eq(F(x).diff(x), -2 * x * F(x) ** 2), F(x), ics={F(0): 1}).rhs
    c1 = sp.simplify(sol.diff(x).subs(x, 2) - sp.Rational(-4, 25)) == 0
    val = sp.integrate(x**2 * sol * sol.diff(x), (x, 0, 2))
    V.report(6, c1 and sol.subs(x, 2) == sp.Rational(1, 5), f"f={sol}, f'(2) 일치={c1}")
    V.check(6, "적분", val, V.ans(6, "1"))


if __name__ == "__main__":
    V.run(globals())
