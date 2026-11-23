import sys, re, itertools, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SKU-M27", "items/SKU/PAST")
A_, B_ = sp.symbols("A_ B_", real=True)


def p01():
    # f=2^(x-a)+b ; 점근선 y=b ; 역함수 g: x=b ; 교점 P(b,b)
    a, b = A_, B_
    fx = 2**(x - a) + b
    loc = {"b": b}
    V.check(1, "1 f 점근선 y=", sp.limit(fx, x, -sp.oo), sp.sympify(V.ans(1, "1").split("=")[1], locals=loc))
    # g(x)=a+log2(x-b) 는 f 의 역함수(f(g(x))=x), 정의역 x>b 의 경계 x=b 에서 -oo -> 수직 점근선 x=b
    g = a + sp.log(x - b, 2)
    V.report(1, sp.simplify(sp.expand_log(fx.subs(x, g) , force=True) - x) == 0, "f(g(x))=x")
    V.report(1, sp.limit(g.subs({a: 1, b: 0}), x, 0, "+") == -sp.oo, "2 g: x->b+ 에서 -oo (수직 점근선)")
    V.check(1, "2 g 점근선 x=", b, sp.sympify(V.ans(1, "2").split("=")[1], locals=loc))
    # P=(b,b) on y=2x-3
    bv = sp.solve(sp.Eq(b, 2 * b - 3), b)[0]
    V.check(1, "3 b", bv, V.ans(1, "3"))
    # g(5)=3 (g 는 f 의 역함수) -> 수치 확인
    av = sp.solve(sp.Eq((2**(3 - a) + bv).subs(a, a), 5), a)[0]
    V.check(1, "4 a", av, V.ans(1, "4"))
    fa = (2**(x - av) + bv)
    V.check(1, "g(5)=3 확인(f(3)=5)", fa.subs(x, 3), 5)
    V.check(1, "5 a+b", av + bv, V.ans(1, "5"))


def p02():
    sols = []
    for a1, d in itertools.product(range(-60, 61), range(-30, 31)):
        Sn = lambda n: sp.Rational(n * (2 * a1 + (n - 1) * d), 2)
        if all(Sn(n) >= Sn(5) for n in range(1, 300)) and abs(Sn(4) - Sn(2)) == 11:
            sols.append((a1, d))
    assert len(sols) == 1, sols
    a1, d = sols[0]
    an = lambda n: a1 + (n - 1) * d
    V.check(2, "1 a3+a4", an(3) + an(4), V.ans(2, "1"))
    V.check(2, "2 a1", a1, V.ans(2, "2"))
    V.check(2, "3 d", d, V.ans(2, "3"))
    V.check(2, "4 S10", sum(an(n) for n in range(1, 11)), V.ans(2, "4"))


def p03():
    f = x**4 / 2 - sp.Rational(4, 3) * x**3 - x
    ts = sorted(sp.solve(sp.Eq(sp.diff(f, x), -1), x))
    assert ts == [0, 2], ts
    tp = lambda t0: (t0, f.subs(x, t0))
    V.check(3, "1 접점1", list(S(V.ans(3, "1"))), list(tp(ts[0])))
    V.check(3, "3 접점2", list(S(V.ans(3, "3"))), list(tp(ts[1])))
    line = lambda t0: sp.expand(f.subs(x, t0) + sp.diff(f, x).subs(x, t0) * (x - t0))
    lhs, rhs = V.ans(3, "2").split("=")
    V.check(3, "2 y1", line(ts[0]), sp.sympify(rhs, locals={"x": x}))
    lhs, rhs = V.ans(3, "4").split("=")
    V.check(3, "4 y2", line(ts[1]), sp.sympify(rhs, locals={"x": x}))
    # 두 직선은 y=-x+5 와 평행하고 서로 다름
    V.report(3, line(ts[0]) != line(ts[1]) and sp.diff(line(ts[1]), x) == -1, "기울기 -1, 서로 다른 직선")


def p04():
    f = 2 * x**3 - 12 * x**2 + 18 * x
    crit = sp.solve(sp.diff(f, x), x)
    kmax = [c for c in crit if sp.diff(f, x, 2).subs(x, c) < 0]
    assert len(kmax) == 1
    kv = kmax[0]
    V.check(4, "1 k", kv, V.ans(4, "1"))
    zeros = sorted(sp.solve(f, x))  # 0, 3
    lo, hi = zeros[0], zeros[-1]
    assert all(f.subs(x, lo + (hi - lo) * sp.Rational(i, 30)) >= 0 for i in range(31))
    s1 = integral(f, x, lo, kv)
    s2 = integral(f, x, kv, hi)
    assert s1 <= s2
    V.check(4, "2 S1", s1, V.ans(4, "2"))
    V.check(4, "3 S2", s2, V.ans(4, "3"))
    r = sp.Rational(s1 / s2)
    q, p = sp.fraction(r)
    V.check(4, "4 p-q", p - q, V.ans(4, "4"))


if __name__ == "__main__": V.run(globals())
