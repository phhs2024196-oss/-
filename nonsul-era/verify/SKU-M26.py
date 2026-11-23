import sys, re, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SKU-M26", "items/SKU/PAST")
X = sp.symbols("X", positive=True)


def p01():
    # A: 3 = log_2 x
    xa = sp.solve(sp.Eq(sp.log(X, 2), 3), X)[0]
    V.check(1, "1 A", list(S(V.ans(1, "1"))), [xa, 3])
    # B, C: x = a^3, b^3 with a=2k, b=3k ; AB:BC = 1:4
    kk = sp.symbols("kk", positive=True)
    a, b = 2 * kk, 3 * kk
    sol = sp.solve(sp.Eq(4 * (a**3 - xa), (b**3 - a**3)), kk)
    assert len(sol) == 1
    s = sp.nsimplify(a.subs(kk, sol[0])**3 + b.subs(kk, sol[0])**3)
    q, p = sp.fraction(sp.Rational(s))
    assert sp.gcd(p, q) == 1 and 2 < a.subs(kk, sol[0]) < b.subs(kk, sol[0])
    V.check(1, "2 p+q", p + q, V.ans(1, "2"))


def p02():
    k_ = sp.symbols("k_", positive=True)
    xx = sp.symbols("xx", real=True)
    period = sp.Integer(2) * sp.pi / (2 * k_)
    V.check(2, "1 g(k)", period, sp.sympify(V.ans(2, "1"), locals={"k": k_}))
    f = lambda z: 3 * sp.cos(2 * k_ * z)
    h = sp.simplify(f(period) + sp.cos(period / 3 + sp.Rational(5, 4) * sp.pi))
    assert sp.simplify(h - (3 + sp.cos(sp.pi / (3 * k_) + sp.Rational(5, 4) * sp.pi))) == 0
    # 최대: cos(theta)=1, theta=pi/(3k)+5pi/4 > 5pi/4, theta=2 m pi ; k 최대 = theta 최소
    th = lambda kv: sp.pi / (3 * kv) + sp.Rational(5, 4) * sp.pi
    def best(target_parity):
        # theta = n*pi with n parity target (even -> cos=1, odd -> cos=-1), theta>5pi/4 ; k=pi/(3(theta-5pi/4)) 최대 -> theta 최소
        ns = [m for m in range(1, 40) if m % 2 == target_parity and m * sp.pi > sp.Rational(5, 4) * sp.pi]
        m = min(ns)
        return sp.simplify(sp.pi / (3 * (m * sp.pi - sp.Rational(5, 4) * sp.pi)))
    aa, bb = best(0), best(1)
    V.check(2, "2 a", aa, V.ans(2, "2"))
    V.check(2, "3 b", bb, V.ans(2, "3"))
    # 수치 확인: h(a)=4 (max value), h(b)=2
    V.check(2, "h(a)=4", sp.N(3 + sp.cos(th(aa)), 20), 4)
    V.check(2, "h(b)=2", sp.N(3 + sp.cos(th(bb)), 20), 2)


def p03():
    a_, C = sp.symbols("a_ C")
    fp = a_ * x**2 - 12
    aa = sp.solve(fp.subs(x, -2), a_)[0]
    V.check(3, "1 a", aa, V.ans(3, "1"))
    f = sp.integrate(fp.subs(a_, aa), x) + C
    Cv = sp.solve(f.subs(x, -2), C)[0]
    f = f.subs(C, Cv)
    V.check(3, "2 f", sp.expand(f - sp.sympify(V.ans(3, "2"), locals={"x": x})), 0)
    crit = sp.solve(sp.diff(f, x), x)
    mins = [c for c in crit if sp.diff(f, x, 2).subs(x, c) > 0]
    assert len(mins) == 1
    V.check(3, "3 극솟값", f.subs(x, mins[0]), V.ans(3, "3"))


def p04():
    f = x**2 - 2 * x - 3
    g1 = lambda u: 2 * sp.Abs(u - 1) - 4
    # official claim: g>=f on [-1,3], f>=g on [3,4] (샘플 점검)
    iv = [tuple(int(v) for v in m) for m in re.findall(r"\[(-?\d+),(-?\d+)\]", V.ans(4, "1"))]
    assert iv == [(-1, 3), (3, 4)], iv
    pts = lambda lo, hi: [sp.Rational(lo) + (sp.Rational(hi - lo)) * sp.Rational(i, 20) for i in range(21)]
    ok = all(g1(u) >= f.subs(x, u) for u in pts(*iv[0])) and all(f.subs(x, u) >= g1(u) for u in pts(*iv[1]))
    V.report(4, ok, "1 대소 관계 [-1,3]: g>=f, [3,4]: f>=g")
    gx, fx = (-2 * x - 2, 2 * x - 6), f
    # S = 2( int_1^3 (g-f) + int_3^4 (f-g) ) with limits read from the YAML expression
    lims = [tuple(int(v) for v in m) for m in re.findall(r"int_(\d)\^(\d)", V.ans(4, "2"))]
    assert lims == [(1, 3), (3, 4)], lims
    expr = 2 * (integral(gx[1] - fx, x, 1, 3) + integral(fx - gx[1], x, 3, 4))
    # 독립 계산: 전체 구간 [-2,4] 에서 |f-g| 적분(조각별)
    brk = [-2, -1, 1, 3, 4]
    seg = lambda lo, hi: abs(integral(fx - (gx[0] if hi <= 1 else gx[1]), x, lo, hi))
    tot = sum(seg(lo, hi) for lo, hi in zip(brk, brk[1:]))
    V.check(4, "2 식의 값 = 전체 넓이", expr, tot)
    V.check(4, "3 S", tot, V.ans(4, "3"))


if __name__ == "__main__": V.run(globals())
