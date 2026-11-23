import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("UOS-M24", "items/UOS/PAST")


def p01():
    # f(x)=x*g(x)/(2π) 는 조건을 만족하는 연속함수 → a_n 수치 적분으로 닫힌 꼴과 합을 검증
    g = lambda u: abs(mp.cos(3 * u)) + abs(mp.sin(u / 2))
    f = lambda u: u * g(u) / (2 * mp.pi)
    mp.mp.dps = 20
    A = mp.quad(f, mp.linspace(0, 2 * mp.pi, 13))
    G = mp.quad(g, mp.linspace(0, 2 * mp.pi, 13))
    V.check(1, "∫g=8", sp.Float(G, 15), 8, 1e-8)
    bad = []
    for m in (1, 2, 3, 4):
        I = mp.quad(f, mp.linspace(0, 2 * m * mp.pi, 12 * m + 1))
        an = (-1) ** m / (2 * m) * I
        if abs(an - (-1) ** m * (A / 2 + 2 * (m - 1))) > 1e-8:
            bad.append(m)
    V.proof(1, "a_n 닫힌 꼴 n=1..4", bad)
    a = sp.Symbol("a")
    nn = sp.Symbol("nn", integer=True, positive=True)
    an = (-1) ** nn * (a / 2 + 2 * (nn - 1))
    total = sum(an.subs({a: sp.Rational(7, 3), nn: i}) for i in range(1, 2025))  # A 값과 무관함도 확인
    V.check(1, "Σa_n (A=7/3)", total, V.ans(1, "1-1"))
    total2 = sum(an.subs({a: A, nn: i}) for i in range(1, 2025))
    V.check(1, "Σa_n (실제 A)", sp.Float(total2, 15), V.ans(1, "1-1"), 1e-6)


def p02():
    import math
    ra = sum(1 for a in range(7) for b in range(7) for c in range(7) if a + b + c == 6)
    pb = sum(1 for a in range(5) for b in range(5) for c in range(5) if a + b + c == 4)
    V.check(2, "(a)", ra * pb, V.ans(2, "2-1"))
    cnt = 0
    for r in ((a, b, 6 - a - b) for a in range(7) for b in range(7 - a)):
        for bl in ((a, b, 4 - a - b) for a in range(5) for b in range(5 - a)):
            if all(r[i] >= bl[i] for i in range(3)):
                cnt += 1
    V.check(2, "(b) 전수", cnt, V.ans(2, "2-2"))
    V.check(2, "(b) 공식 H4*H2", math.comb(6, 4) * math.comb(4, 2), V.ans(2, "2-2"))


def p03():
    t = sp.Symbol("t", real=True)
    arg = sp.exp(3 * t) - 2 * sp.exp(t) + sp.exp(-t)
    V.check(3, "인수분해", arg - sp.exp(-t) * (sp.exp(2 * t) - 1) ** 2, 0)
    dom = sp.Union(sp.Interval.open(-sp.oo, 0), sp.Interval.open(0, sp.oo))
    V.report(3, dom == sp.sympify(V.ans(3, "3-1", "domain"), locals={"Union": sp.Union, "Interval": sp.Interval}), "정의역")
    ft = lambda u: mp.e ** u * mp.log(mp.e ** (3 * u) - 2 * mp.e ** u + mp.e ** (-u))
    mp.mp.dps = 25
    I = mp.quad(ft, [mp.log(2), mp.log(3)])
    V.check(3, "정적분 수치", sp.Float(I, 20), V.ans(3, "3-1", "integral"), 1e-12)
    xx = sp.Symbol("xx", positive=True)
    ex = sp.integrate(sp.log(xx**-1), (xx, 2, 3)) + 2 * sp.integrate(sp.log(xx - 1), (xx, 2, 3)) + 2 * sp.integrate(sp.log(xx + 1), (xx, 2, 3))
    V.check(3, "정적분 기호(분해)", ex, V.ans(3, "3-1", "integral"))


def p04():
    X = sp.Symbol("X", positive=True)
    Y = sp.sqrt(X**2 / 2 + 1)
    P = sp.Matrix([X, Y]); F1 = sp.Matrix([0, sp.sqrt(3)]); F2 = sp.Matrix([0, -sp.sqrt(3)])
    v1, v2 = F1 - P, F2 - P
    cs = sp.simplify(v1.dot(v2) / (v1.norm() * v2.norm()))
    fx = sp.sympify(V.ans(4, "4-1", "f")); gx = sp.sympify(V.ans(4, "4-1", "g"))
    V.check(4, "(a) cos", cs.subs(X, sp.Rational(7, 5)), (fx / gx).subs(sp.Symbol("x"), sp.Rational(7, 5)))
    V.check(4, "(a) cos 2", cs.subs(X, 3), (fx / gx).subs(sp.Symbol("x"), 3))
    x0 = sp.sqrt(26) / sp.sqrt(3)
    V.check(4, "|PF1|+|PF2|=8", sp.simplify((v1.norm() + v2.norm()).subs(X, x0)), 8)
    V.check(4, "|PF1|,|PF2|", (sp.simplify(v1.norm().subs(X, x0)), sp.simplify(v2.norm().subs(X, x0))) == (3, 5), True)
    th = sp.acos(fx / gx).subs(sp.Symbol("x"), X)
    V.check(4, "(b) θ'", sp.simplify(sp.diff(th, X).subs(X, x0)), V.ans(4, "4-2"))


if __name__ == "__main__": V.run(globals())
