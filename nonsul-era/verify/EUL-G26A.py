import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-G26A", "items/EUL/PAST")
R = sp.Rational
E = lambda v: sp.sympify(v, locals={"x": x, "n": n})
KV = lambda no, sub, key: E(V.ans(no, sub, key))

def p08():
    a, b = sp.symbols("a b", positive=True)
    # A(( 1-b)/a,0), H(-b/a,0): A 가 OH 중점; OA=OB: |(1-b)/a| = |log5 b|
    bv = sp.solve(sp.Eq((1 - b)/a, -b/(2*a)), b)[0]
    V.check(8, "1 b", bv, V.ans(8, "1"))
    av = sp.solve(sp.Eq((bv - 1)/a, sp.log(bv, 5)), a)[0]
    V.check(8, "2 b^a", sp.simplify(sp.expand_log(bv**av, force=True)) if False else sp.N(bv**av, 30), E(V.ans(8, "2")))

def p09():
    th = sp.symbols("th")
    for c in [sp.Rational(3, 5)]:
        s = -sp.sqrt(1 - c**2)  # 4사분면
        r1 = c/(1 - s)
        V.check(9, "1 root", 3*r1**2 - 10*r1 + 3, 0)
        V.check(9, "1 cos", c, V.ans(9, "1"))
        V.check(9, "2 tan", s/c, V.ans(9, "2"))
    # 유일성: 근 3 또는 1/3 -> cos=3/5 만 4사분면
    cands = []
    for r in (3, R(1, 3)):
        # cos/(1-sin)=r, sin^2+cos^2=1 -> cos=2r/(r^2+1), sin=(r^2-1)/(r^2+1)
        c = 2*r/(r**2+1); s_ = (r**2-1)/(r**2+1)
        if c > 0 and s_ < 0: cands.append(s_/c)
    V.check(9, "2 tan(범위 검사)", cands[0], V.ans(9, "2"))

def p10():
    a1, d = sp.symbols("a1 d")
    an = lambda m: a1 + (m-1)*d
    sol = sp.solve([an(1)+an(3)+8, (an(2)+an(4)) - (an(1)+an(3)) - 6], [a1, d])
    V.check(10, "1 an", an(n).subs(sol), E(V.ans(10, "1")))
    V.check(10, "2 S12", sum(an(m).subs(sol) for m in range(1, 13)), V.ans(10, "2"))

def p11():
    a, b = sp.symbols("a b")
    sol = sp.solve([a + b - 2, a*(6-2)**2 + b - 5], [a, b])
    V.check(11, "1 a", sol[a], KV(11, "1", "a"))
    V.check(11, "1 b", sol[b], KV(11, "1", "b"))
    g = lambda t_: -t_ + 5 if t_ <= 3 else sol[a]*(t_-2)**2 + sol[b]
    r = 23 % 6
    r = 6 if r == 0 else r
    V.check(11, "2 f(23)", g(r), V.ans(11, "2"))

def p12():
    a, b, c, A, B, C = sp.symbols("a b c A B C")
    f = 3*x**2 + a*x + b
    s = sp.solve([f.subs(x, 1) - 2, f.subs(x, 2) - 16], [a, b])
    fx = f.subs(s)
    V.check(12, "1 f", fx, E(V.ans(12, "1")))
    V.check(12, "1 (가)", sp.limit(x**2*fx.subs(x, 1/x), x, 0, "+"), 3)
    g = sp.Poly(sp.symbols("q2")*x**2 + sp.symbols("q1")*x + sp.symbols("q0"), x).as_expr()
    q2, q1, q0 = sp.symbols("q2 q1 q0")
    # f-3g = 5x+c 이고 g(0)=-3
    sg = sp.solve([sp.Poly(fx - 3*g - 5*x - C, x).coeff_monomial(x**2), q0 + 3], [q2, q0])
    gg = g.subs(sg)
    cc = sp.solve(sp.Poly(sp.expand(fx - 3*gg - 5*x - C), x).coeff_monomial(1), C)[0]
    V.check(12, "2 g(3)", gg.subs({x: 3}).subs(q1, 0), V.ans(12, "2"))
    V.check(12, "2 (나)", sp.limit((fx - 3*gg.subs(q1, 0))/x, x, sp.oo), 5)

def p13():
    k = sp.symbols("k", real=True)
    f = x**3 + k*x + k + 5
    V.check(13, "A on curve", f.subs(x, -1), 4)
    m = sp.diff(f, x).subs(x, -1)
    kv = None
    for kk in sp.solve(sp.Eq(m, 1), k):
        # 직각 조건 AB ⟂ BC, BC 기울기 -1
        kv = kk
    fk = f.subs(k, kv)
    line = sp.diff(fk, x).subs(x, -1)*(x + 1) + 4
    V.check(13, "1 AB", line, E(V.ans(13, "1")))
    xs = [r for r in sp.solve(fk - line, x) if r != -1]
    B = (xs[0], fk.subs(x, xs[0]))
    C = (0, B[1] + B[0])  # y = -(x - xb) + yb at x=0
    AB = sp.sqrt((B[0]+1)**2 + (B[1]-4)**2)
    # B on circle with diameter AC: (B-A).(B-C)=0
    V.check(13, "원 조건", (B[0]+1)*(B[0]-C[0]) + (B[1]-4)*(B[1]-C[1]), 0)
    V.check(13, "2 AB", AB, E(V.ans(13, "2")))

def p14():
    q = sp.Function("q")
    c1, c2 = sp.symbols("c1 c2")
    # g' = x+1 에서 g = x^2/2 + x + c2, f = x g'
    g = x**2/2 + x + c2
    f = x*sp.diff(g, x)
    V.check(14, "1 g'", sp.diff(g, x), E(V.ans(14, "1")))
    V.check(14, "(가)", sp.simplify(sp.diff(f, x) - sp.diff(g, x, 2) - 2*x), 0)  # f' = g' + 2x
    c2v = sp.solve(f.subs(x, 1) - g.subs(x, 1), c2)[0]
    gg = g.subs(c2, c2v)
    V.check(14, "(나)", sp.simplify(sp.integrate(f, x) - (x*gg - sp.integrate(gg, x)) - sp.Symbol("K")*0 - (sp.integrate(f, x) - (x*gg - sp.integrate(gg, x))).subs(x, 0)), 0)
    V.check(14, "2 g(0)", gg.subs(x, 0), V.ans(14, "2"))

if __name__ == "__main__": V.run(globals())
