import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G22A", "items/YSM/PAST")
mm, nn = sp.symbols("m n", positive=True)
E = lambda v: sp.sympify(v, locals={"x": x, "m": mm, "n": nn, "t": t})

def p01():
    from fractions import Fraction as Fr
    # 주머니별 1번 공 확률
    p1 = {"A": Fr(1, 4), "B": Fr(2, 4), "C": Fr(1)}
    pD = lambda n_: sum(Fr(1, 3)*p**n_ for p in p1.values())
    pC = lambda n_: Fr(1, 3)*p1["C"]**n_/pD(n_)
    V.check(1, "1-1 P(D1)", sp.Rational(pD(1).numerator, pD(1).denominator), E(V.ans(1, "1-1", "P")))
    V.check(1, "1-1 P(C|D1)", sp.Rational(pC(1).numerator, pC(1).denominator), E(V.ans(1, "1-1", "Q")))
    # 12개 공 중 하나를 뽑는 동치 풀이: 1번 공 7개 중 C 의 공 4개
    V.report(1, Fr(7, 12) == Fr(4 + 2 + 1, 12) and Fr(4, 7) == pC(1), "1-1 12개 공 동치 풀이 (7/12, 4/7)")
    # 1-2: 일반 n 기호 식
    f = sp.Rational(1, 3)/(sp.Rational(1, 3)*(sp.Rational(1, 4))**nn + sp.Rational(1, 3)*(sp.Rational(1, 2))**nn + sp.Rational(1, 3))
    V.check(1, "1-2 P(C|Dn)", sp.simplify(f), E(V.ans(1, "1-2")))
    # 1-3: 처음으로 0.9 이상이 되는 n, 그 앞은 모두 미만, 단조 증가
    seq = [pC(i) for i in range(1, 12)]
    nmin = next(i for i, v in enumerate(seq, 1) if v >= Fr(9, 10))
    V.check(1, "1-3 최소 n", nmin, E(V.ans(1, "1-3", "n")))
    V.report(1, all(a < b for a, b in zip(seq, seq[1:])), "1-3 P(C|Dn) 단조 증가")

def p02():
    a, b = sp.symbols("a b", positive=True)
    xs = sp.Symbol("xs")
    ta, tb = 2*a*x - a**2, 2*b*x - b**2
    xc = sp.solve(sp.Eq(ta, tb), x)[0]
    area = sp.integrate(x**2 - ta, (x, a, xc)) + sp.integrate(x**2 - tb, (x, xc, b))
    V.check(2, "2-1 넓이", sp.simplify(area), E(V.ans(2, "2-1")).subs({sp.Symbol("a", real=True): a, sp.Symbol("b", real=True): b}) if False else sp.sympify(V.ans(2, "2-1"), locals={"a": a, "b": b}))
    at = (-t + sp.sqrt(t**2 + 16*t))/8; bt = (-3*t + 3*sp.sqrt(t**2 + 4*t))/8
    ft = (mm*bt + nn*at)/(mm + nn)
    L = sp.limit(ft, t, sp.oo)
    V.check(2, "2-2 lim f", sp.simplify(L), E(V.ans(2, "2-2")))
    # 공식 풀이의 중간 극한값
    V.check(2, "2-2 (i)", sp.limit(t - sp.sqrt(t**2 + 4*t), t, sp.oo), -2)
    V.check(2, "2-2 (ii)", sp.limit(t - sp.sqrt(t**2 + 16*t), t, sp.oo), -8)
    V.check(2, "2-2 a->1", sp.limit(at, t, sp.oo), 1); V.check(2, "2-2 b->3/4", sp.limit(bt, t, sp.oo), sp.Rational(3, 4))  # 큰 t 에서는 a>b (제시문 (나)는 a<b 를 요구하지 않음)

def p03():
    r = sp.Rational(65, 100)*sp.Rational(8, 10)
    V.check(3, "3-1 등비급수", 1500*sp.summation(r**n, (n, 1, sp.oo)), E(V.ans(3, "3-1")))
    f = (10*t**2 - 9*t - 9)*sp.exp(-t)
    A = sp.integrate(f, (t, sp.Rational(3, 2), x))
    V.check(3, "3-2 A(x) 닫힌식", sp.simplify(A - ((-10*x**2 - 11*x - 2)*sp.exp(-x) + 41*sp.exp(-sp.Rational(3, 2)))), 0)
    V.check(3, "3-2 lim A", sp.limit(A, x, sp.oo), E(V.ans(3, "3-2")))
    f2 = sp.diff(f, t, 2)
    V.check(3, "3-3 f''", sp.simplify(f2 - (10*t**2 - 49*t + 29)*sp.exp(-t)), 0)
    rts = [s for s in sp.solve(sp.Eq(sp.diff(f, t, 2)*sp.exp(t), 0), t) if s >= sp.Rational(3, 2)]
    V.check(3, "3-3 a", rts[0], E(V.ans(3, "3-3")))
    a0 = float(rts[0]); g = sp.lambdify(t, f2)
    V.report(3, g(a0 - 0.01) < 0 < g(a0 + 0.01), "3-3 f'' 부호 −→+ (a 좌우)")
    V.report(3, abs(float(f2.subs(t, sp.Rational(3, 2)) * sp.exp(sp.Rational(3, 2))) + 22) < 1e-9, "3-3 f''(1.5)=-22e^{-1.5} (공식 표의 e^{1.5} 표기는 오기 의심)")

if __name__ == "__main__": V.run(globals())
