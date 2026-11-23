import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("HSN-G25H", "items/HSN/PAST")

def p10():
    # y=log_{1/4}x -> x축 대칭 -> (x-2, y-1) 이동, 점 (a,2)
    a = sp.symbols("a", positive=True)
    g = -sp.log(a - 2, sp.Rational(1, 4)) + 1
    sol = sp.nsolve(g - 2, a, 5)
    V.check(10, "a", sp.Rational(sp.nsimplify(sol, rational=True)), V.ans(10, "1"))

def p11():
    V.check(11, "정적분", integral(-x * (x - 2), x, 0, 2) + integral(x * (x - 2), x, 2, 3), V.ans(11, "1"))
    V.check(11, "수치", sp.N(sp.Integral(x * sp.Abs(x - 2), (x, 0, 3))), sp.N(S(V.ans(11, "1"))), 1e-3)

def p12():
    a = sp.symbols("a", positive=True)
    lim_r = sp.limit((x**3 - 1) / (a * x**2 - a), x, 1, "+")
    V.check(12, "a", sp.solve(sp.Eq(lim_r, sp.Rational(3, 8)), a)[0], V.ans(12, "1"))

def p13():
    a = sp.symbols("a", real=True)
    # f'(x)=3a x^2+6x+3a >=0 for all x, a>0  <=> 판별식/4 <=0
    disc = sp.discriminant(sp.diff(a * x**3 + 3 * x**2 + 3 * a * x + 1, x), x)
    sol = sp.solve_univariate_inequality(disc <= 0, a, relational=False)
    mn = sp.Intersection(sol, sp.Interval.open(0, sp.oo)).inf
    V.check(13, "a_min", mn, V.ans(13, "1"))

def p14():
    # 구체 수열로 확인: 임의 a_1..a_8 로 시작, a_{n+8}=5-a_n. sum a_{2n}(1..4)=1 이 되게 설정
    import random
    for _ in range(5):
        base = [sp.Rational(random.randint(-9, 9), 3) for _ in range(8)]
        a_ = {i + 1: base[i] for i in range(8)}
        for i in range(9, 80): a_[i] = 5 - a_[i - 8]
        # a_2+a_4+a_6+a_8 = 1 이 되도록 a_8 조정
        a_[8] = 1 - a_[2] - a_[4] - a_[6]
        for i in range(9, 80): a_[i] = 5 - a_[i - 8]
        V.check(14, "합", sum(a_[2 * n] for n in range(1, 37)), V.ans(14, "1"))

def p15():
    # 근 구하기: sin^2 = cos^2 on [-2,4] (x=1/4+k/2)
    roots = [sp.Rational(1, 4) + sp.Rational(k, 2) for k in range(-5, 9)]
    roots = [r for r in roots if -2 <= r <= 4 and sp.simplify(sp.Abs(sp.sin(sp.pi * r))**2 - sp.cos(sp.pi * r)**2) == 0]
    V.check(15, "근의 합", sum(roots), V.ans(15, "1"))
    V.check(15, "근 개수(0..2)", len([r for r in roots if 0 <= r <= 2]), 4)

if __name__ == "__main__": V.run(globals())
