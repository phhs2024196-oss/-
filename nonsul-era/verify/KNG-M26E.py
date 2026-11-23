import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import math
V = Verifier("KNG-M26E", "items/KNG/PAST")
a = sp.symbols("a", real=True); h = sp.symbols("h")
from sympy.parsing.sympy_parser import parse_expr
P = lambda q: parse_expr(q, local_dict={"x": x})

def p01():
    # a>1: 증가 -> x=0 에서 최소 log_a 3 - 1 = -3/2 -> a=1/9 (a>1 모순)
    a1 = sp.Integer(3) ** sp.Rational(-2)
    V.check(1, "a>1 후보", a1, sp.Rational(1, 9)); V.report(1, a1 < 1, "a>1 후보가 조건 위반")
    # 0<a<1: 감소 -> x=3 에서 최소 log_a 6 - 1 = -3/2
    a2 = sp.Integer(6) ** sp.Rational(-2)
    V.check(1, "0<a<1 해", a2, V.ans(1, "1", "a"))
    f = lambda av, xv: math.log(xv + 3) / math.log(av) - 1
    xs = [i * 3 / 3000 for i in range(3001)]
    V.check(1, "수치 최솟값(a=1/36)", round(min(f(1 / 36, xv) for xv in xs), 9), sp.Rational(-3, 2))
    bad = [av for av in [1.01 + 0.01 * i for i in range(2000)] if abs(min(f(av, xv) for xv in xs[::100]) + 1.5) < 1e-6]
    V.report(1, not bad, f"a>1 에서 최솟값 -3/2 인 a 없음 {bad[:1]}")

def p02():
    t = sp.symbols("t", positive=True)
    sol = sp.solve(sp.Eq(t + 1 / t, 2), t)
    V.check(1, "tan", sol[0], 1)
    th = sp.Rational(5, 4) * sp.pi
    V.check(1, "theta", th, S(V.ans(2, "1", "theta").replace("pi", "pi")))
    V.report(2, sp.pi < th < sp.Rational(3, 2) * sp.pi and sp.tan(th) == 1, "범위·tan=1")
    V.check(2, "sin+cos", sp.simplify(sp.sin(th) + sp.cos(th)), S(V.ans(2, "1", "sum")))

def p03():
    n_ = sp.symbols("n", integer=True, positive=True)
    Sn = lambda m: 3 ** m - 1
    a1 = Sn(1); r = sp.symbols("r"); rr = sp.solve(sp.Eq(a1 * (1 + r), Sn(2)), r)[0]
    V.check(3, "a1", a1, 2); V.check(3, "r", rr, 3)
    V.report(3, all(Sn(m) - Sn(m - 1) == a1 * rr ** (m - 1) for m in range(2, 12)), "일반항 a_n=2*3^(n-1)")
    val = sp.log(128, a1) + sp.log(sp.Rational(1, 9), rr)
    V.check(3, "값", sp.simplify(val), V.ans(3, "1"))

def p04():
    f1 = lambda xv: xv - 1; f2 = lambda xv: xv - 4; g = 2 * x + a
    left = sp.limit(f1(x) * g, x, 2); right = sp.limit(f2(x) * g, x, 2)
    sol = sp.solve(sp.Eq(left, right), a)
    V.check(4, "a", sol[0], S(V.ans(4, "1", "a")))
    V.check(4, "f(2)g(2)", (f2(x) * g).subs(x, 2), -8 - 2 * a)
    V.check(4, "좌극한", left, 4 + a)

def p05():
    f = lambda u: u ** 2 + 2 * u + 1
    fp = sp.limit((f(x + h) - f(x)) / h, h, 0)
    V.check(5, "f'", fp, P(V.ans(5, "1", "fprime")))
    slope = (f(1) - f(-1)) / sp.Integer(2)
    V.check(5, "평균변화율", slope, 2)
    cs = sp.solve(sp.Eq(fp.subs(x, sp.Symbol("c")), slope), sp.Symbol("c"))
    V.check(5, "c", cs[0], S(V.ans(5, "1", "c"))); V.report(5, -1 < cs[0] < 1, "c in (-1,1)")

def p06():
    f = 2 - x ** 2
    A = sp.integrate(f, (x, 0, 1)); V.check(6, "넓이", A, sp.Rational(5, 3))
    V.report(6, min(f.subs(x, 0), f.subs(x, 1)) >= 1, "[0,1]에서 f>=1 이므로 y=a(a<=1) 아래는 1*a 직사각형")
    av = sp.symbols("av"); sol = sp.solve(sp.Eq(A / 2, 1 * av), av)[0]
    V.check(6, "a", sol, S(V.ans(6, "1", "a")))

if __name__ == "__main__": V.run(globals())
