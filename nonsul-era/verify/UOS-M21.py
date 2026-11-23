import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from fractions import Fraction
from math import isqrt
V = Verifier("UOS-M21", "items/UOS/PAST")


def p01():
    a, b = sp.symbols("a b", real=True)
    C = sp.Matrix([(a + b) / 2, a * b])
    # 두 접선의 교점 확인
    sol = sp.solve([sp.Eq(sp.Symbol("Y"), 2 * a * x - a**2).subs(x, sp.Symbol("X")), sp.Eq(sp.Symbol("Y"), 2 * b * x - b**2).subs(x, sp.Symbol("X"))], [sp.Symbol("X"), sp.Symbol("Y")], dict=True)[0]
    V.check(1, "교점 C", sp.simplify(sol[sp.Symbol("X")] - (a + b) / 2) + sp.simplify(sol[sp.Symbol("Y")] - a * b), 0)
    area = sp.integrate((a + b) * x - a * b - x**2, (x, a, b))
    V.check(1, "넓이 (b-a)^3/6", sp.simplify(area - (b - a) ** 3 / 6), 0)
    # 삼각형 ABC 넓이 (좌표 공식)
    A_, B_ = sp.Matrix([a, a**2]), sp.Matrix([b, b**2])
    tri = sp.Abs(sp.Matrix.hstack(B_ - A_, C - A_).det()) / 2
    bb = sp.Symbol("bb", positive=True)  # b-a
    expr = sp.simplify(tri.subs(b, a + bb).subs(sp.Abs(bb**3 / 4), bb**3 / 4))
    V.check(1, "삼각형 = (b-a)^3/4", sp.simplify(expr - bb**3 / 4), 0)
    V.check(1, "값", sp.Rational(360, 4), V.ans(1, "1-1"))
    # 수치 확인: a=0, b=360^(1/3)
    bv = sp.cbrt(360)
    V.check(1, "넓이 60 → 삼각형 90", sp.N(sp.integrate((bv) * x - x**2, (x, 0, bv)), 20), 60)


def p02():
    num = 3*x**4 + 4*x**3*sp.log(x) - 4*x**3 - 8*x**2*sp.log(x) + 3*x**2 + 8*x*sp.log(x) + 4*x + 4
    f = num / (x**2 - 2*x + 2)
    import mpmath as mp
    mp.mp.dps = 30
    fl = sp.lambdify(x, f, "mpmath")
    val = mp.quad(fl, [1, 2])
    V.check(2, "수치 적분", sp.N(sp.sympify(V.ans(2, "2-1")), 25), sp.Float(str(val), 25), 1e-18)
    V.check(2, "분해", sp.simplify(f - ((3*x**4 - 4*x**3 + 3*x**2 + 4*x + 4) / (x**2 - 2*x + 2) + 4*x*sp.log(x))), 0)
    V.check(2, "유리식 분해", sp.simplify((3*x**4 - 4*x**3 + 3*x**2 + 4*x + 4) / (x**2 - 2*x + 2) - (3*x**2 + 2*x + 1 + (2*x - 2) / (x**2 - 2*x + 2) + 4 / (x**2 - 2*x + 2))), 0)
    V.check(2, "부분", integral(3*x**2 + 2*x + 1 + (2*x - 2) / (x**2 - 2*x + 2), x, 1, 2), 11 + sp.log(2))
    V.check(2, "4/(..)", integral(4 / (x**2 - 2*x + 2), x, 1, 2), sp.pi)
    V.check(2, "4x ln x", integral(4*x*sp.log(x), x, 1, 2), 8*sp.log(2) - 3)


# 도로망: y=0 에서 x=1..5, y=1..4 에서 x=0..5, y=5 에서 x=0..2
PTS = {(xx, 0) for xx in range(1, 6)} | {(xx, yy) for yy in range(1, 5) for xx in range(0, 6)} | {(xx, 5) for xx in range(0, 3)}
Cp, Dp = (2, 5), (5, 4)


def reach(start):
    from functools import lru_cache
    @lru_cache(None)
    def go(pt):
        if pt == Cp: return (Fraction(1), Fraction(0))
        if pt == Dp: return (Fraction(0), Fraction(1))
        up = (pt[0], pt[1] + 1) in PTS
        rt = (pt[0] + 1, pt[1]) in PTS
        if up and rt:
            u, r = go((pt[0], pt[1] + 1)), go((pt[0] + 1, pt[1]))
            return (u[0] / 2 + r[0] / 2, u[1] / 2 + r[1] / 2)
        return go((pt[0], pt[1] + 1)) if up else go((pt[0] + 1, pt[1]))
    return go(start)


def p03():
    F = lambda q: sp.Rational(q.numerator, q.denominator)
    pBC, pBD = reach((1, 0))
    pAC, pAD = reach((0, 1))
    V.check(3, "(a) p_BD", F(pBD), V.ans(3, "3-1"))
    V.check(3, "p_BC", F(pBC), sp.Rational(7, 64))
    V.check(3, "p_AC", F(pAC), sp.Rational(22, 64))
    pC = (pAC + pBC) / 2
    V.check(3, "p_C", F(pC), sp.Rational(29, 128))
    V.check(3, "(b)", F((pAC / 2) / pC), V.ans(3, "3-2"))


def a_k(k):
    """격자점 수를 정수 연산으로 직접 센다: curve(x) <= y <= 3*4^k - 2|x|, |x| <= 4^k."""
    K4, K2 = 4**k, 2**k
    cnt = 0
    for xv in range(-K4, K4 + 1):
        top = 3 * K4 - 2 * abs(xv)
        if xv >= 0:
            ylo = -((-xv * xv) // K4)            # ceil(x^2/4^k)
        else:
            m = K4 * (-xv)                        # y >= 2^k sqrt(-x) <=> y^2 >= 4^k * (-x)
            ylo = isqrt(K4 * (-xv))
            if ylo * ylo < m: ylo += 1
            ylo = max(ylo, 0)
        if ylo <= top: cnt += top - ylo + 1
    return cnt


def p04():
    for k in (1, 2, 3, 4):
        V.check(4, f"a_{k}", a_k(k), 3 * 16**k + 4**k + 2**k + 1)
    ex = sp.sympify(V.ans(4, "4-1"), locals={"n": n})
    for nv in (1, 2, 3, 5):
        V.check(4, f"Σ n={nv}", sum(a_k(k) for k in range(1, min(nv, 4) + 1)) if nv <= 4 else sum(3 * 16**k + 4**k + 2**k + 1 for k in range(1, nv + 1)), ex.subs(n, nv))


if __name__ == "__main__": V.run(globals())
