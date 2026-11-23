"""KW-M27 검산: 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
from fractions import Fraction as Fr

V = Verifier("KW-M27", "items/KW/PAST")


def p01():
    # [1] n>=4 에서 n! >= n^2 (수치 증명 확인)
    V.proof(1, "1 n!>=n^2 (4..400)", [n_ for n_ in range(4, 401) if math.factorial(n_) < n_ ** 2])
    V.proof(1, "1 n=3 에서는 성립 안 함(조건 n>=4 의 필요성)", [] if math.factorial(3) < 9 else [3])
    # [2]
    V.proof(1, "2 |sin x|<=|x|", [v for v in [mp.mpf(i) / 50 for i in range(-400, 401)] if abs(mp.sin(v)) > abs(v) + 1e-15])
    f = x ** 2 + 2 * sp.cos(x)
    fl = sp.lambdify(x, f, "mpmath")
    pts = [-mp.pi / 2 + (mp.pi * 1.5) * i / 30000 for i in range(30001)]
    mn, mx = min(fl(v) for v in pts), max(fl(v) for v in pts)
    V.check(1, "2 min", float(mn), float(sp.N(sp.sympify(V.ans(1, "2", "min"), locals={"pi": sp.pi}))), 1e-6)
    V.check(1, "2 max", float(mx), float(sp.N(sp.sympify(V.ans(1, "2", "max"), locals={"pi": sp.pi}))), 1e-6)
    # [3]
    a, b, c, d = sp.symbols("a b c d", real=True)
    F = a * x ** 3 + b * x ** 2 + c * x + d
    conds = [F.subs(x, 0)] + sp.Poly(sp.expand(F.subs(x, -x) - F), x).all_coeffs()
    s0 = sp.solve(conds, [a, c, d], dict=True)[0]
    Fb = F.subs(s0)
    bv = sp.Symbol("bv", positive=True)
    t = sp.Symbol("t")
    integral = sp.integrate(x * sp.exp(bv * x ** 2), (x, 0, 1))
    bsol = sp.nsolve(integral - (sp.exp(2) - 1) / 4, bv, 1.5)
    V.check(1, "3 b", sp.N(bsol, 20), 2, 1e-12)
    V.check(1, "3 f", sp.simplify(Fb.subs(b, sp.Integer(round(float(bsol)))) - sp.sympify(V.ans(1, "3"), locals={"x": x})), 0)
    # [4]
    A, B = sp.symbols("A B", real=True)
    expr = sp.together((A * x + 2) / x + (B * x + B - 2) / (x + 1) - (A ** 2 + B ** 2 - 1) / (x * (x + 1)) - 2)
    num = sp.Poly(sp.expand(sp.numer(expr)), x).all_coeffs()
    sols = sp.solve(num, [A, B], dict=True)
    vals = {sp.simplify(s[A] ** 5 + s[B] ** 5) for s in sols}
    V.check(1, "4 a^5+b^5", vals.pop(), V.ans(1, "4"))
    V.check(1, "4 해의 개수", len(sols), 2)
    # [5]
    pa, pb_a, pb_na = Fr(1, 100), Fr(99, 100), Fr(2, 100)
    post = pa * pb_a / (pa * pb_a + (1 - pa) * pb_na)
    V.check(1, "5 P(A|B)", sp.Rational(post.numerator, post.denominator), V.ans(1, "5"))


def p02():
    n = sp.Symbol("n", positive=True, integer=True)
    # [1]: 닫힌 곱식을 정확히 확인하고 극한
    for nv in (2, 5, 9):
        s = sum(sp.log(1 - sp.Rational(1, (nv + kk) ** 2)) for kk in range(nv + 1))
        V.check(1, f"1 합=ln 곱 n={nv}", sp.N(s - sp.log(sp.Rational((nv - 1) * (2 * nv + 1), 2 * nv ** 2)), 30), 0, 1e-20)
    V.check(1, "1 극한", sp.limit(n * sp.log((n - 1) * (2 * n + 1) / (2 * n ** 2)), n, sp.oo), V.ans(2, "1"))
    # [2](1): g 가 (나)를 만족하고 g 의 역함수 f 가 미분가능
    g = sp.sympify(V.ans(2, "2-1"), locals={"x": x})
    V.check(2, "2-1 (나)", sp.N((g - sp.log(sp.sqrt(2 * x * sp.exp(g) + 1))).subs(x, sp.Rational(7, 3)), 30), 0, 1e-20)
    V.check(2, "2-1 (나) x=-5", sp.N((g - sp.log(sp.sqrt(2 * x * sp.exp(g) + 1))).subs(x, -5), 30), 0, 1e-20)
    # [2](2): g 의 역함수 f 를 직접 구해 극한 계산
    y = sp.Symbol("y", real=True)
    f = sp.sinh(y)  # g(x)=asinh(x) 의 역함수 (g(sinh y)=y 확인)
    V.check(2, "2-2 g(f(y))=y", sp.N(g.subs(x, f).subs(y, sp.Rational(3, 5)) - sp.Rational(3, 5), 30), 0, 1e-20)
    aval = f.subs(y, sp.log(2))
    lim = sp.limit((f.subs(y, 2 * x) - aval) / (x - sp.log(sp.sqrt(2))), x, sp.log(sp.sqrt(2)))
    V.check(2, "2-2 a+b", sp.simplify(aval + lim), V.ans(2, "2-2"))
    # [3]
    fx = sp.exp(sp.sqrt(x) - 1) * sp.sqrt(x)
    V.check(2, "3 (가)", fx.subs(x, 1), 1)
    cond = sp.exp(1 - sp.sqrt(x)) * (fx / (2 * sp.sqrt(x)) - sp.diff(fx, x)) + 1 / (2 * sp.sqrt(x))
    V.check(2, "3 (나)", sp.simplify(cond), 0)
    V.check(2, "3 정적분", sp.simplify(sp.integrate(fx, (x, 1, 4))), sp.sympify(V.ans(2, "3"), locals={"e": sp.E}))
    # [4]: 좌표로 OD, CD, EC 를 구해 평균과 비교
    av, bv = sp.symbols("av bv", positive=True)
    O = sp.Matrix([(av + bv) / 2, 0])
    D = sp.Matrix([av, sp.sqrt(av * bv)])
    C = sp.Matrix([av, 0])
    OD = sp.sqrt(sp.factor(sp.expand(sum((D - O).applyfunc(lambda v: v ** 2)))))
    CD = sp.sqrt(sum((D - C).applyfunc(lambda v: v ** 2)))
    nvec = D - O  # 접선의 법선
    EC = sp.Abs((nvec.T * (C - D))[0]) / sp.sqrt(sp.factor(sp.expand(nvec.dot(nvec))))
    V.check(2, "4 OD=(a+b)/2", sp.simplify(OD - (av + bv) / 2), 0)
    V.check(2, "4 CD=√(ab)", sp.simplify(CD - sp.sqrt(av * bv)), 0)
    V.check(2, "4 EC=2ab/(a+b)", sp.simplify(EC - 2 * av * bv / (av + bv)), 0)
    V.proof(2, "4 부등식", [(p, q) for p in range(1, 30) for q in range(p + 1, 31)
                       if not ((p + q) / 2 > math.sqrt(p * q) > 2 * p * q / (p + q))])


if __name__ == "__main__":
    V.run(globals())
