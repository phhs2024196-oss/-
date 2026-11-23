import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SWU-M27", "items/SWU/PAST")


def p01():
    tt = sp.symbols("t", positive=True)
    E = lambda v: sp.sympify(v, locals={"t": tt})
    AB = -(1 / tt**2) * (x - tt)
    CD = -tt**2 * (x - 1 / tt)
    xe = sp.solve(sp.Eq(AB, CD), x)[0]
    ye = sp.simplify(AB.subs(x, xe))
    V.check(1, "1-1 x", xe, E(V.ans(1, "1", "x")))
    V.check(1, "1-1 y", ye, E(V.ans(1, "1", "y")))
    # 공통부분 OBED = 삼각형 BOE + 삼각형 EOD
    area = sp.Rational(1, 2) * (1 / tt) * xe + sp.Rational(1, 2) * (1 / tt) * ye
    V.check(1, "1-2 넓이", sp.simplify(area), E(V.ans(1, "2")))
    # 신발끈 공식으로 사각형 O-D-E-B 직접 계산
    P = [(0, 0), (1 / tt, 0), (xe, ye), (0, 1 / tt)]
    sh = sum(P[i][0] * P[(i + 1) % 4][1] - P[(i + 1) % 4][0] * P[i][1] for i in range(4)) / 2
    V.check(1, "1-2 신발끈", sp.simplify(sh), E(V.ans(1, "2")))
    f = E(V.ans(1, "2"))
    V.check(1, "1-3 극한", sp.limit(f.subs(tt, tt) / f.subs(tt, 2 * tt), tt, sp.oo), V.ans(1, "3"))


def p02():
    N = 60
    V.proof(2, "2-1 등식 n<=60", [n_ for n_ in range(1, N + 1)
            if sum((2 * i - 1) ** 2 for i in range(1, n_ + 1)) * 3 != n_ * (2 * n_ - 1) * (2 * n_ + 1)])
    from fractions import Fraction
    V.proof(2, "2-2 부등식 n<=14", [n_ for n_ in range(1, 15)
            if sum(Fraction(1, i) for i in range(1, 2**n_ + 1)) < 1 + Fraction(n_, 2)])
    V.proof(2, "2-3 배수 n<=200", [n_ for n_ in range(1, 201) if (5 ** (2 * n_ - 1) + 4 ** (n_ + 1)) % 21])
    # 귀납 단계 항등식: 5^(2k+1)+4^(k+2) = 4(5^(2k-1)+4^(k+1)) + 21*5^(2k-1)
    V.proof(2, "2-3 귀납단계 항등식", [k_ for k_ in range(1, 50)
            if 5 ** (2 * k_ + 1) + 4 ** (k_ + 2) != 4 * (5 ** (2 * k_ - 1) + 4 ** (k_ + 1)) + 21 * 5 ** (2 * k_ - 1)])
    kk = sp.symbols("k", positive=True)
    V.check(2, "2-1 귀납단계 항등식", sp.expand(sp.Rational(1, 3) * kk * (2 * kk - 1) * (2 * kk + 1) + (2 * kk + 1) ** 2),
            sp.expand(sp.Rational(1, 3) * (kk + 1) * (2 * kk + 1) * (2 * kk + 3)))


def p03():
    nums = range(1, 10)
    perms = list(itertools.permutations(nums, 5))

    def pos(p, v):
        return p.index(v)
    c1 = sum(1 for p in perms if any(p[i:i + 3] == (1, 2, 3) for i in range(3)))
    V.check(3, "3-1", c1, V.ans(3, "1"))
    c2 = sum(1 for p in perms if {1, 2, 3} <= set(p) and pos(p, 1) < pos(p, 2) < pos(p, 3) and pos(p, 3) - pos(p, 1) == 3)
    V.check(3, "3-2", c2, V.ans(3, "2"))
    c3 = sum(1 for p in perms if {1, 2, 3} <= set(p) and abs(pos(p, 1) - pos(p, 2)) == 1
             and (abs(pos(p, 3) - pos(p, 2)) == 1 or (4 in p and abs(pos(p, 3) - pos(p, 4)) == 1)))
    V.check(3, "3-3", c3, V.ans(3, "3"))
    # 대학 풀이 분류: 2와 이웃(180) + 4와만 이웃(100)
    a_ = sum(1 for p in perms if {1, 2, 3} <= set(p) and abs(pos(p, 1) - pos(p, 2)) == 1 and abs(pos(p, 3) - pos(p, 2)) == 1)
    b_ = sum(1 for p in perms if {1, 2, 3, 4} <= set(p) and abs(pos(p, 1) - pos(p, 2)) == 1
             and abs(pos(p, 3) - pos(p, 4)) == 1 and abs(pos(p, 3) - pos(p, 2)) != 1)
    V.check(3, "3-3 분류 180", a_, 180)
    V.check(3, "3-3 분류 100", b_, 100)


if __name__ == "__main__":
    V.run(globals())
