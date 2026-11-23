"""광운대 2023학년도 자연 1교시(G23A) 문제 1·2 검산 — 기대값은 YAML(V.ans)에서만 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-G23A", "items/KW/PAST")
f_ = sp.Function("f")
LOC = {"x": x, "k": k, "f": f_}


def ex(s):
    return sp.sympify(s, locals=LOC)


def p01():
    # [1](1)
    a = sp.log(4) / sp.log(3)
    V.check(1, "1-1 (1/27)^(a/12)", (sp.Rational(1, 27)) ** (a / 12), V.ans(1, "1-1"))
    # [1](2) 귀류: 3^q=4^p 인 자연수 p,q 가 있는지(없어야 증명과 일치)
    V.proof(1, "1-2 3^q=4^p 자연수해", search(lambda p, q: 3 ** q != 4 ** p, range(1, 120), range(1, 120)))
    # [2] 실근 개수(조밀 격자 부호 변화 + 극값 부호로 단조 구간 확인)
    import numpy as np
    xs = np.linspace(-np.pi, np.pi, 400001)
    fx = np.cos(xs) ** 2 - np.sin(xs) - xs
    cnt = int(np.sum(fx[:-1] * fx[1:] < 0)) + int(np.sum(fx == 0))
    V.check(1, "2 실근 개수(격자)", cnt, V.ans(1, "2"))
    # f 의 증감표 값
    F = sp.cos(x) ** 2 - sp.sin(x)
    V.check(1, "2 f'(x)=-cos x(2 sin x+1)", sp.simplify(sp.diff(F, x) + sp.cos(x) * (2 * sp.sin(x) + 1)), 0)
    # [3]
    PA, PBA, PBAc = sp.Rational(1, 4), sp.Rational(4, 5), sp.Rational(3, 10)
    V.check(1, "3-1 P(A^c∩B)", (1 - PA) * PBAc, V.ans(1, "3-1"))
    PB = PA * PBA + (1 - PA) * PBAc
    V.check(1, "3-2 P(B)", PB, V.ans(1, "3-2"))
    V.check(1, "3-3 P(A|B)", PA * PBA / PB, V.ans(1, "3-3"))
    # [4](1)
    fx_ = sp.exp(x) + sp.sin(x) + x ** 3 + sp.cos(x) * x + x ** 2
    g = (fx_ + fx_.subs(x, -x)) / 2
    h = (fx_ - fx_.subs(x, -x)) / 2
    ok = sp.simplify(g.subs(x, -x) - g) == 0 and sp.simplify(h.subs(x, -x) + h) == 0 and sp.simplify(g + h - fx_) == 0
    V.report(1, ok, "4-1 g 우함수·h 기함수·합 f (표본 f)")
    V.check(1, "4-1 g 공식", ex(V.ans(1, "4-1", "g(x)")).subs(f_(x), fx_).subs(f_(-x), fx_.subs(x, -x)).doit(), g)
    V.check(1, "4-1 h 공식", ex(V.ans(1, "4-1", "h(x)")).subs(f_(x), fx_).subs(f_(-x), fx_.subs(x, -x)).doit(), h)
    # [4](2): I(p)=∫_{-1}^{1} x/((x+p)^2+1) dx, I(p)=0 ⇔ p=0
    p = sp.symbols("p", real=True)
    anti = sp.log((x + p) ** 2 + 1) / 2 - p * sp.atan(x + p)
    V.check(1, "4-2 원시함수 미분", sp.simplify(sp.diff(anti, x) - x / ((x + p) ** 2 + 1)), 0)
    I = sp.lambdify(p, anti.subs(x, 1) - anti.subs(x, -1), "mpmath")
    bad = [pv / 10 for pv in range(-80, 81) if pv != 0 and (I(mp.mpf(pv) / 10) * pv >= 0)]
    V.proof(1, "4-2 p≠0이면 I(p)≠0(부호 -p)", bad)
    V.check(1, "4-2 I(0)", I(mp.mpf(0)), 0)
    gx = -2 * p * x ** 2 / (((x + p) ** 2 + 1) * ((-x + p) ** 2 + 1))
    fx2 = x / ((x + p) ** 2 + 1)
    V.check(1, "4-2 g(x) 식", sp.simplify((fx2 + fx2.subs(x, -x)) / 2 - gx), 0)


def p02():
    # [1](1)
    a = sp.symbols("a", positive=True)
    sol = sorted(set(sp.solve(sp.Eq(sp.Abs(7 * a - 6) / 5, a), a)), key=float)
    V.check(2, "1-1 r1", sol[1], V.ans(2, "1-1", "r1"))
    V.check(2, "1-1 r2", sol[0], V.ans(2, "1-1", "r2"))
    # [1](2): A(2,0), B(0,3/2), P=AB를 1:2 내분, 최소 원 중심 (1/2,1/2) 반지름 1/2
    A, B = sp.Matrix([2, 0]), sp.Matrix([0, sp.Rational(3, 2)])
    P = (1 * B + 2 * A) / 3
    C = sp.Matrix([sp.Rational(1, 2)] * 2)
    r = sp.Rational(1, 2)
    d = sp.sqrt(((P - C).T * (P - C))[0])
    th = 2 * sp.asin(r / d)
    V.check(2, "1-2 점 P x", P[0], sp.Rational(4, 3))
    V.check(2, "1-2 sinθ", sp.sin(th), V.ans(2, "1-2"))
    # [2]
    al, be = sorted(sp.solve(x ** 2 - x - 1, x), key=float)[::-1]
    V.check(2, "2-1 k", sp.solve(sp.Symbol("kk") * (al ** 5 - be ** 5) - 5, sp.Symbol("kk"))[0], V.ans(2, "2-1"))
    n_ = 400
    mp.mp.dps = 60
    aN, bN = mp.mpf(1 + mp.sqrt(5)) / 2, mp.mpf(1 - mp.sqrt(5)) / 2
    lim = (aN ** (n_ + 4) - bN ** (n_ + 4)) / (aN ** n_ - bN ** n_)
    V.check(2, "2-2 극한(n=400 수치)", sp.Float(str(lim), 50), sp.N(ex(V.ans(2, "2-2")), 50), tol=1e-30)
    V.check(2, "2-2 α^4 정확값", sp.simplify(al ** 4), V.ans(2, "2-2"))
    # [3](1)
    kk = sp.symbols("kk", positive=True)
    Fk = sp.exp(x) / (kk * x ** 2 + x + 1)
    d1 = sp.simplify(sp.diff(Fk, x) - sp.exp(x) * x * (kk * x - 2 * kk + 1) / (kk * x ** 2 + x + 1) ** 2)
    V.check(2, "3-1 f' 식", d1, 0)
    x0 = 2 - 1 / kk
    V.check(2, "3-1 극솟값", sp.simplify(Fk.subs(x, x0) - sp.exp(2 - 1 / kk) / (4 * kk - 1)), 0)
    V.check(2, "3-1 극댓값 f(0)", Fk.subs(x, 0), V.ans(2, "3-1", "max"))
    V.check(2, "3-1 극솟값 YAML", sp.simplify(Fk.subs(x, x0) - ex(V.ans(2, "3-1", "min")).subs(k, kk)), 0)
    # k=1/2: f' >= 0 이고 x=0 에서 부호 변화 없음
    Fh = Fk.subs(kk, sp.Rational(1, 2))
    V.check(2, "3-1 k=1/2 f'", sp.simplify(sp.diff(Fh, x) - 2 * x ** 2 * sp.exp(x) / (x ** 2 + 2 * x + 2) ** 2), 0)
    # [3](2) 증명: y' 식, e^x > x^2/2+x+1 > x^2/2 (x>0)
    Y = sp.exp(x) / (x ** 2 / 2 + x + 1)
    V.check(2, "3-2 y' 식", sp.simplify(sp.diff(Y, x) - 2 * x ** 2 * sp.exp(x) / (x ** 2 + 2 * x + 2) ** 2), 0)
    xs = [i / 20 for i in range(1, 2000)]
    V.proof(2, "3-2 e^x > x^2/2+x+1 (x>0)", [v for v in xs if not (mp.e ** v > v ** 2 / 2 + v + 1)])
    V.check(2, "3-2 극한 x/e^x", limit(x / sp.exp(x), x, sp.oo), 0)
    # [3](3)
    gx = sp.exp(-1 / x) / x
    V.check(2, "3-3 우극한", limit(gx, x, 0, "+"), V.ans(2, "3-3"))
    V.check(2, "3-3 좌극한", 0, V.ans(2, "3-3"))


if __name__ == "__main__":
    V.run(globals())
