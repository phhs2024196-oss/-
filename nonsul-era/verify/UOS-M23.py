import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
from fractions import Fraction
V = Verifier("UOS-M23", "items/UOS/PAST")


def p01():
    X = sp.Symbol("X", positive=True)
    PF1 = sp.Rational(5, 3) * X - 3   # 우측 가지: ex-a
    PF2 = sp.Rational(5, 3) * X + 3
    xs = sp.solve(sp.Eq(PF2 / PF1, sp.Rational(29, 11)), X)
    V.check(1, "x_P", xs[0], 4)
    x0 = xs[0]
    y0 = sp.sqrt(16 * (x0**2 / 9 - 1))
    V.check(1, "P 가 쌍곡선 위", x0**2 / 9 - y0**2 / 16, 1)
    V.check(1, "|PF1|=11/3", sp.sqrt((x0 - 5) ** 2 + y0**2), sp.Rational(11, 3))
    a = sp.Symbol("a")
    a0 = sp.solve(sp.Eq(a * (x0**2 - 25), y0), a)[0]
    par = a0 * (x**2 - 25)
    line = y0 / (x0 + 5) * (x + 5)
    area = sp.integrate(par - line, (x, -5, x0))
    V.check(1, "넓이", area, V.ans(1, "1-1"))


def p02():
    def seqprob(seq):
        w, r, b, p = 2, 1, 1, Fraction(1)
        ok = True
        for c in seq:
            tot = w + r + b
            p *= Fraction({"W": w, "R": r, "B": b}[c], tot)
            if c == "W": w += 1
            elif c == "R": r += 1
            else: b += 1
            if w < r + b: ok = False
        return p, ok
    PA = Fraction(0); PAok = Fraction(0); n = 0
    for seq in set(itertools.permutations("WWWWWRB")):
        p, ok = seqprob(seq)
        PA += p; n += 1
        if ok: PAok += p
    V.check(2, "(a) P(A)", sp.Rational(PA.numerator, PA.denominator), V.ans(2, "2-1"))
    V.check(2, "순열 수", n, 42)
    cond = PAok / PA
    V.check(2, "(b)", sp.Rational(cond.numerator, cond.denominator), V.ans(2, "2-2"))
    nb = sum(1 for s in set(itertools.permutations("WWWWWRB")) if not seqprob(s)[1])
    V.check(2, "|A∩B| 순열 수", nb, 14)


def p03():
    t = sp.Symbol("t", real=True)
    f, g = sp.sin(x), sp.cos(x)
    c1 = sp.integrate(sp.exp(t) * sp.sin(t), (t, 0, x)) - (sp.exp(x) * (f - g) + 1) / 2
    c2 = sp.integrate(sp.exp(t) * sp.cos(t), (t, 0, x)) - (sp.exp(x) * (f + g) - 1) / 2
    V.check(3, "조건(1) 성립(f=sin,g=cos)", sp.simplify(c1), 0)
    V.check(3, "조건(2) 성립", sp.simplify(c2), 0)
    # 조건을 미분해 얻은 선형 관계에서 f'=g, g'=-f 유도
    F, G, Fp, Gp = sp.symbols("F G Fp Gp")
    sol = sp.solve([sp.Eq(F + G, Fp - Gp), sp.Eq(G - F, Fp + Gp)], [Fp, Gp])
    V.check(3, "f'=g", sol[Fp], G)
    V.check(3, "g'=-f", sol[Gp], -F)
    V.check(3, "값", (f**2 + g**2).subs(x, 1), V.ans(3, "3-1"))


def p04():
    import mpmath as mp
    mp.mp.dps = 30
    def r(th):
        A = mp.matrix([0, 0]); B = mp.matrix([2, 0])
        P = 2 * mp.cos(th) * mp.matrix([mp.cos(th), mp.sin(th)])
        Q = B + 2 * mp.cos(3 * th) * mp.matrix([-mp.cos(3 * th), mp.sin(3 * th)])
        # R = AP ∩ BQ
        M = mp.matrix([[P[0], -(Q[0] - B[0])], [P[1], -(Q[1] - B[1])]])
        s = mp.lu_solve(M, B - A)
        R = A + s[0] * P
        d = lambda U, W: mp.sqrt((U[0] - W[0]) ** 2 + (U[1] - W[1]) ** 2)
        a, b, c = d(Q, R), d(R, P), d(P, Q)
        area = abs((Q[0] - R[0]) * (P[1] - R[1]) - (P[0] - R[0]) * (Q[1] - R[1])) / 2
        return 2 * area / (a + b + c)
    th = mp.mpf("1e-8")
    V.check(4, "수치 r/θ (θ=1e-8)", sp.Float(r(th) / th, 15), V.ans(4, "4-1"), 1e-6)
    th0 = mp.mpf("0.3")
    th_ = sp.Symbol("th", positive=True)
    s_ = 2 * sp.sin(3 * th_) * sp.sin(th_) / (sp.sin(4 * th_) + sp.sin(th_) + sp.sin(3 * th_))
    rp = 2 * sp.cos(th_) - 2 * sp.sin(3 * th_) / sp.sin(4 * th_)
    rr = s_ * rp * sp.sin(4 * th_) / (2 * sp.sin(th_))
    V.check(4, "공식 r(0.3) = 기하 r(0.3)", sp.N(rr.subs(th_, sp.Rational(3, 10)), 20), sp.Float(str(r(th0)), 20), 1e-12)
    V.check(4, "공식의 극한", sp.limit(rr / th_, th_, 0, "+"), V.ans(4, "4-1"))


if __name__ == "__main__": V.run(globals())
