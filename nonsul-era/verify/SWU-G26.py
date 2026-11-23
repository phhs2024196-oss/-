import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SWU-G26", "items/SWU/PAST")
R = sp.Rational
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application as _ima
P = lambda s: parse_expr(s, transformations=standard_transformations + (_ima,), local_dict={'x': x})

def p01():
    # 외분 4:1 (A_{n-2}, A_{n-1}): (4*a_{n-1}-1*a_{n-2})/3
    A = [None, R(1), R(4, 3)]
    for n in (3, 4): A.append((4 * A[n-1] - A[n-2]) / 3)
    B = [None, R(1), R(2, 3)]
    for n in (3, 4): B.append((B[n-2] + 2 * B[n-1]) / 3)  # B_{n-2}B_{n-1}을 2:1 내분 (B_{n-1} 쪽에 가까운 점)
    # 1번째 내분점 OB_1을 2:1: (2*1+1*0)/3 = 2/3
    for n in (2, 3, 4):
        V.check(1, f"1-1 a{n}", A[n], V.ans(1, "1", f"a{n}"))
        V.check(1, f"1-2 b{n}", B[n], V.ans(1, "2", f"b{n}"))
    def a_(n):
        v = [R(1), R(4, 3)]
        for _ in range(3, n + 1): v.append((4 * v[-1] - v[-2]) / 3)
        return v[n-1]
    def b_(n):
        v = [R(1), R(2, 3)]
        for _ in range(3, n + 1): v.append((v[-2] + 2 * v[-1]) / 3)
        return v[n-1]
    V.check(1, "1-3 a2025+2b2025", a_(2025) + 2 * b_(2025), V.ans(1, "3"))

def p02():
    f = x**4 - 6*x**2
    m = sp.diff(f, x).subs(x, -1)
    line = m * (x + 1) + f.subs(x, -1)
    V.check(2, "1 접선", sp.expand(line), P(V.ans(2, "1", "line")))
    roots = sp.solve(sp.Eq(f, line), x)
    other = [r for r in roots if r != -1][0]
    pt = V.ans(2, "1", "point")
    X, Y = [P(s) for s in pt.strip("()").split(",")]
    V.check(2, "1 교점 x", other, X); V.check(2, "1 교점 y", f.subs(x, other), Y)
    S1 = integral(line - f, x, -1, other)
    V.check(2, "2 S1", S1, V.ans(2, "2"))
    l2 = sp.expand(sp.interpolate([(1, f.subs(x, 1)), (other, f.subs(x, other))], x))
    V.check(2, "3 직선", l2, P(V.ans(2, "3", "line")))
    S2 = integral(l2 - f, x, 1, other)
    V.check(2, "3 S2", S2, V.ans(2, "3", "S2"))
    V.check(2, "3 비", S1 / S2, V.ans(2, "3", "ratio"))

def p03():
    a, b, c, p, q, tt = sp.symbols("a b c p q tt", positive=True)
    A, B, C = sp.symbols("A B C", real=True)
    f = A*x**2 + B*x + C
    sol = sp.solve([sp.diff(f, x).subs(x, tt) - p, sp.diff(f, x).subs(x, -tt) + q, f.subs(x, tt) - p*tt], [A, B, C], dict=True)[0]
    # 접점 (-t) 조건도 확인
    V.check(3, "1 접점 -t", (f.subs(sol).subs(x, -tt)), -q * (-tt))
    ans = V.ans(3, "1")
    for nm, sym in (("a", A), ("b", B), ("c", C)):
        V.check(3, f"1 {nm}", sol[sym], S(V.ans(3, "1", nm), locals={"t": tt, "p": p, "q": q}))
    g = sp.expand(f.subs(sol))
    vx = sp.simplify(-sol[B] / (2*sol[A])); vy = sp.simplify(g.subs(x, vx))
    V.check(3, "2 x", vx, S(V.ans(3, "2", "x"), locals={"t": tt, "p": p, "q": q}))
    V.check(3, "2 y", vy, S(V.ans(3, "2", "y"), locals={"t": tt, "p": p, "q": q}))
    # (3) pq=1: y = t/(p+1/p) 의 sup = t/2 (p=1에서 달성; 수치 확인)
    h = vy.subs(q, 1/p)
    crit = sp.solve(sp.diff(h, p), p)
    best = max(sp.simplify(h.subs(p, r)) for r in crit if r.is_positive)
    V.check(3, "3 최대", best, S(V.ans(3, "3"), locals={"t": tt}))

if __name__ == "__main__": V.run(globals())
