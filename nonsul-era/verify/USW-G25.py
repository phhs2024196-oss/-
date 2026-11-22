import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("USW-G25", "items/USW/PAST")
R = sp.Rational
A = lambda no, sub="1", key=None: sp.sympify(V.ans(no, sub, key), locals={"x": x})

def p06():
    a = sp.symbols("a", positive=True)
    f = 9**(-(x - a) - 2); g = R(1, 3)**(2*(x - a**2) + 8)   # y축 대칭 → x→-x, 평행이동 x→x-a
    sol = sp.solve(sp.Eq(sp.expand_log(sp.log(f, 3), force=True) - sp.expand_log(sp.log(g, 3), force=True), 0).simplify(), a)
    V.check(6, "a", sol[0], A(6))

def p07():
    a = sp.symbols("a", positive=True)
    sol = sp.solve(sp.Eq(sp.log(2, 3)*sp.log(a, 8), -1), a)
    V.check(7, "a", sol[0], A(7))

def p08():
    # 좌표: A=(0,0), B=(1/2,√3/2), C=(1,0), D=(5,0); 원: A,B,D 지나는 원, 직선 BC 와 다시 만나는 E
    Ax = sp.Matrix([0, 0]); Bx = sp.Matrix([R(1, 2), sp.sqrt(3)/2]); Cx = sp.Matrix([1, 0]); Dx = sp.Matrix([5, 0])
    assert same((Cx - Ax).norm(), 1) and same((Bx - Ax).norm(), 1) and same((Bx - Cx).norm(), 1)  # ABC 정삼각형
    X, Y, c = sp.symbols("X Y c")
    # 원 x²+y²+Ex... : 중심 (2.5, k) 반지름 R 로 놓고 A,B 통과
    D_, E_ = sp.symbols("D_ E_")
    circ = lambda P: P[0]**2 + P[1]**2 + D_*P[0] + E_*P[1]
    s = sp.solve([circ(Ax), circ(Bx), circ(Dx)], [D_, E_], dict=True)  # circ(Ax)=0 이라 F=0
    s = sp.solve([circ(Bx), circ(Dx)], [D_, E_], dict=True)[0]
    u = sp.symbols("u")
    P = Bx + u*(Cx - Bx)
    us = [r for r in sp.solve(sp.expand(circ(P).subs(s)), u) if not same(r, 0)]
    E = (Bx + us[0]*(Cx - Bx))
    V.check(8, "AE", (E - Ax).norm(), A(8))

def p09():
    n = sp.symbols("n", positive=True)
    tot = 0
    for nn in range(1, 11):
        r = sp.solve(x**2 + x/nn - nn - 1, x)
        tot += 1/r[0] + 1/r[1]
    V.check(9, "합", sp.nsimplify(sp.simplify(tot)), A(9))

def p10():
    t = sp.symbols("t", positive=True)
    OA = t**2 + 1                       # 직선 l: y - t² = -(1/t)(x - t) 의 y절편
    assert sp.simplify(sp.Rational(-1, 1)/t - (-1/t)) == 0
    V.check(10, "극한", limit(OA - sp.sqrt(t**2 + t**4), t, sp.oo), A(10))

def p11():
    F = x**4 + x**3 + 6*x**2 + 4*x + 8
    out = []
    for fa, ga in [(q, sp.cancel(F/q)) for q in [sp.Poly(d, x).as_expr() for d in [x**2 + 4, x**2 + x + 2, x**2 + 1, x + 1]] if sp.rem(F, q, x) == 0]:
        if fa.subs(x, 1) == 5 and sp.diff(fa, x).subs(x, 1) == ga.subs(x, 1)/2:
            out.append(sp.diff(ga, x).subs(x, 1))
    V.check(11, "g'(1)", out, [A(11)])

def p12():
    a, b, k = sp.symbols("a b k", real=True)
    fp = x**3 - 6*x**2 + 2*a*x + b
    sol = sp.solve(sp.Poly(sp.expand(fp - x*(x - k)*(x - 2*k)), x).all_coeffs(), [a, b, k], dict=True)
    sol = [s for s in sol if s[k] != 0][0]
    f = (x**4/4 - 2*x**3 + a*x**2 + b*x).subs(sol)
    vals = [f.subs(x, r) for r in sp.solve(sp.diff(f, x), x)]
    V.check(12, "극댓값", max(vals), A(12, "1", "max")); V.check(12, "극솟값", min(vals), A(12, "1", "min"))

def p13():
    a, b = sp.symbols("a b")
    m = sp.diff(x**3 - 2*x**2, x).subs(x, 1); line = m*(x - 1) + (1 - 2)
    g = x**2 + a*x + b
    sol = sp.solve([sp.diff(g, x).subs(x, -1) - m, g.subs(x, -1) - line.subs(x, -1)], [a, b])
    V.check(13, "a", sol[a], A(13, "1", "a")); V.check(13, "b", sol[b], A(13, "1", "b"))

def p14():
    res = []
    for kk in [R(i, 60) for i in range(1, 60)]:
        xs = sp.solve((x - 1)**2 - (kk*x**2 + 1), x); hi = max(xs)
        S1 = integral(kk*x**2 + 1 - (x - 1)**2, x, 0, hi); S2 = integral((x - 1)**2, x, 0, 1)
        if sp.simplify(S1 - 9*S2) == 0: res.append(kk)
    V.check(14, "k", res, [A(14)])

def p15():
    a, b = sp.symbols("a b"); t_ = sp.symbols("t_")
    cs = sp.symbols("c0:4"); f = sum(c*x**i for i, c in enumerate(cs))
    eq = sp.expand(sp.integrate((x**2 - t**2)*f.subs(x, t), (t, 1, x)) - (x**4 + 2*a*x**3 + b*x**2))
    s = sp.solve(sp.Poly(eq, x).all_coeffs(), list(cs) + [a, b], dict=True)[0]
    V.check(15, "적분", integral(f.subs(s), x, 0, 1), A(15))

if __name__ == "__main__": V.run(globals())
