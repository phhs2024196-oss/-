import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
V = Verifier("KW-G21A", "items/KW/PAST")
a_, b_ = sp.symbols("a b", real=True)

def p01():
    # [1](1) 수직·평행
    sol = sp.solve([a_*2*b_ + 2*(-5), a_/2 - (b_+1)/2], [a_, b_], dict=True)
    # a-b=1, ab=5 -> a^2-a... 해는 허수일 수 있어 항등식으로 확인
    V.check(1, "1-1 a^2+b^2=(a-b)^2+2ab", sp.Integer(1)**2 + 2*5, V.ans(1, "1-1"))
    # [1](2) 교점 C 는 A(0,-2),B(0,2) 지름 원 위: C=(2cos q,2 sin q), 넓이=1/2*4*|x|
    q = sp.symbols("q", real=True)
    area = sp.Rational(1, 2) * 4 * sp.Abs(2*sp.cos(q))
    mx = max(sp.N(area.subs(q, sp.pi*i/200)) for i in range(0, 401))
    V.check(1, "1-2 최댓값", sp.nsimplify(mx), V.ans(1, "1-2"), tol=1e-6)
    # [2](1)
    X = sp.cos(t)/t; Y = sp.sin(t)/t
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    V.check(1, "2-1 vx", vx - sp.sympify(V.ans(1, "2-1", "vx"), locals={"t": t}), 0)
    V.check(1, "2-1 vy", vy - sp.sympify(V.ans(1, "2-1", "vy"), locals={"t": t}), 0)
    w2 = sp.simplify(vx**2 + vy**2)
    V.check(1, "2-1 w^2", w2 - sp.sympify(V.ans(1, "2-1", "w"), locals={"t": t})**2, 0)
    # [2](2) 극한: t^n w(t)
    w = sp.sqrt(1 + 1/t**2)/t
    good = [nn for nn in range(1, 6) if sp.limit(t**nn*w, t, sp.oo).is_finite]
    V.check(1, "2-2 n", good[0] if len(good) == 1 else -1, V.ans(1, "2-2"))
    # [2](3) u(pi)>0, u(2pi)<0, u(3pi)>0
    u = vx
    sgn = [sp.sign(sp.simplify(u.subs(t, m*sp.pi))) for m in (1, 2, 3)]
    V.check(1, "2-3 부호", sgn, [1, -1, 1])
    # [3](1)
    om = sp.Rational(-1, 2) + sp.sqrt(3)*sp.I/2
    a1, b1 = sp.Rational(1, 7), sp.Rational(3, 7)
    V.check(1, "3-1 a", a1, V.ans(1, "3-1", "a")); V.check(1, "3-1 b", b1, V.ans(1, "3-1", "b"))
    V.check(1, "3-1 식", sp.simplify(sp.expand(om/(3+om) - (a1+b1*om))), 0)
    # [3](2)
    A = [1, a1, b1]
    sums = [x_+y_ for x_ in A for y_ in A]   # 9 가지 순서쌍
    from collections import Counter
    c = Counter(sums); n9 = len(sums)
    for key, val in (("p2/7", sp.Rational(2, 7)), ("p4/7", sp.Rational(4, 7)), ("p6/7", sp.Rational(6, 7)), ("p8/7", sp.Rational(8, 7)), ("p10/7", sp.Rational(10, 7)), ("p2", 2)):
        V.check(1, f"3-2 {key}", sp.Rational(c[sp.nsimplify(val)], n9), V.ans(1, "3-2", key))
    V.check(1, "3-2 E", sum(s_ for s_ in sums)/n9, V.ans(1, "3-2", "E"))

def p02():
    x_ = x
    a = sp.symbols("a", real=True)
    f = lambda aa: -x**3 + sp.Rational(3, 2)*aa*x**2 - aa
    def g(av):
        return max(f(av).subs(x, p) for p in [0, 1] + [r for r in sp.solve(sp.diff(f(av), x), x) if 0 <= r <= 1])
    for av in [-2, 0, sp.Rational(1, 2), sp.Rational(9, 10), 1, 3]:
        gv = g(sp.nsimplify(av))
        ex = sp.sympify(V.ans(2, "1-1", "g"), locals={"a": a}).subs(a, av)
        V.check(2, f"1-1 g({av})", gv, ex)
    gmin = min(g(sp.Rational(i, 1000)) for i in range(0, 1001))
    V.check(2, "1-2 최솟값(수치)", sp.N(gmin), sp.N(sp.sympify(V.ans(2, "1-2"))), tol=1e-4)
    a_star = sp.sqrt(6)/3
    V.check(2, "1-2 g(sqrt6/3)", a_star**3/2 - a_star, V.ans(2, "1-2"))
    # [2]
    F = sp.log(x)/x
    crit = sp.solve(sp.diff(F, x), x)
    V.check(2, "2-1 a", crit[0], sp.E if V.ans(2, "2-1", "a") == "e" else V.ans(2, "2-1", "a"))
    infl = sp.solve(sp.diff(F, x, 2), x)
    V.check(2, "2-1 b", infl[0], V.ans(2, "2-1", "b"))
    s_ = sp.symbols("s", positive=True)
    I1 = integral(sp.log(t)/t, t, 1, sp.exp(sp.Rational(3, 2)))
    I2 = integral(1/t, t, sp.E, sp.exp(sp.Rational(3, 2)))
    roots = sp.solve(I1*x**2 - x/2 - sp.Rational(5, 4)*I2, x)
    V.check(2, "2-2 roots", roots, [sp.Rational(-5, 9), 1])
    ans = V.ans(2, "2-2"); V.check(2, "2-2 YAML", sorted(sp.sympify(z) for z in ans.split(",")), sorted(roots))
    # 2-3
    V.report(2, sp.N(sp.log(3, 2) - sp.sqrt(sp.Rational(3, 2))) > 0 and V.ans(2, "2-3").startswith("log2(3) >"), "2-3 log2(3)>sqrt(1.5) 수치")
    V.report(2, sp.N(F.subs(x, sp.sqrt(3)) - F.subs(x, sp.sqrt(2))) > 0, "2-3 f(sqrt3)>f(sqrt2)")
    # 2-4
    nn = sp.symbols("nn", integer=True)
    for N in (2, 3, 4):
        an = sp.integrate(-x**(-N)*sp.log(x), (x, sp.exp(sp.Rational(1, 1-N)), 1))
        V.check(2, f"2-4 a_{N}", sp.simplify(an), sp.Rational(1, (N-1)**2))
    tot = sum(sp.sqrt(sp.Rational(1, (m-1)**2)*sp.Rational(1, m**2)) for m in range(2, 2022))
    V.check(2, "2-4 합", tot, V.ans(2, "2-4"))

if __name__ == "__main__": V.run(globals())
