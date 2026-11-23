#!/usr/bin/env python3
"""TUK-G26A-01~09 (한국공학대 2026 오전) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G26A", "items/TUK/PAST")
mp.mp.dps = 30
th, a, b, c = sp.symbols("theta a b c", real=True)


def kv(no):
    return dict(p.split("=", 1) for p in V.ans(no, "1").replace(" ", "").split(";"))


def p01():
    cos_t = sp.Rational(2, 3)                                   # sin(pi/2+θ)=cosθ
    t0 = 2 * sp.pi - sp.acos(cos_t)                             # 제4사분면의 각
    assert sp.pi * 3 / 2 < t0 < 2 * sp.pi
    V.check(1, "sinθ·tanθ", sp.simplify(sp.sin(t0) * sp.tan(t0)), V.ans(1, "1"))
    V.check(1, "sin(π/2+θ)=2/3 확인", sp.simplify(sp.sin(sp.pi / 2 + t0)), cos_t)


def p02():
    sols = roots_in(2 * sp.cos(x) ** 2 - sp.sin(x) - 1, x, 0, sp.pi)
    V.report(2, len(sols) == 2, f"실근 {sols}")
    V.check(2, "실근의 곱", sp.Mul(*sols), V.ans(2, "1"))


def p03():
    f = lambda aa: -sp.log(x, 3) + aa
    aa = sp.solve(sp.Eq(f(a).subs(x, sp.Rational(1, 27)), 6), a)[0]
    lo, hi = extrema(f(aa), x, sp.Rational(1, 27), 9)
    V.check(3, "a", aa, kv(3)["a"]); V.check(3, "최댓값 6", hi, 6); V.check(3, "b=최솟값", lo, kv(3)["b"])


def p04():
    xa = sp.solve(sp.Eq(2 ** x + 3, 2 ** (x + 1)), x)[0]
    A, B, C = sp.Matrix([xa, 2 ** xa + 3]), sp.Matrix([0, 4]), sp.Matrix([0, 2])
    area = sp.Abs(sp.Matrix.hstack(B - A, C - A).det()) / 2
    V.check(4, "넓이", sp.simplify(area), S(V.ans(4, "1")))
    V.check(4, "B,C 의 y좌표", (2 ** 0 + 3, 2 ** (0 + 1)), (4, 2))


def p05():
    f = 3 * x**2 - 6 * x
    rts = sorted(r for r in sp.solve(f, x) if 0 < r < 2)               # [0,2] 안쪽 부호 변화점
    cuts = [0] + rts + [2]
    absint = sum(abs(sp.integrate(f, (x, cuts[i], cuts[i + 1]))) for i in range(len(cuts) - 1))   # f 는 구간마다 부호 일정
    sols = sp.solve(sp.integrate(f, (x, 0, a)) + absint, a)
    exp = sorted(S(s) for s in V.ans(5, "1").split(","))
    V.check(5, "실수 a 전체", sorted(sols), exp)
    for s in sols:
        V.check(5, f"a={s} 대입", mp.quad(lambda u: 3 * u**2 - 6 * u, [0, float(s)]) + mp.quad(lambda u: abs(3 * u**2 - 6 * u), [0, 2]), 0, 1e-12)


def p06():
    f = -x**2 + 2 * a * x
    xp = [r for r in sp.solve(f, x) if r != 0][0]
    xq = sp.solve(sp.diff(f, x) - 2, x)[0]
    P, Q = sp.Matrix([xp, 0]), sp.Matrix([xq, f.subs(x, xq)])
    det = sp.Matrix.hstack(P, Q).det()
    S_ = a * (a**2 - 1)
    vals = (sp.Rational(13, 10), sp.Rational(27, 10))
    V.check(6, "S(a)=a(a²-1) (a=1.3, 2.7 수치)", [abs(det.subs(a, v)) / 2 for v in vals], [S_.subs(a, v) for v in vals])
    V.check(6, "극한", sp.limit(S_ / (a - 1), a, 1, "+"), V.ans(6, "1"))


def p07():
    p_, q_ = sp.symbols("p q")
    fp = 3 * x**2 + p_ * x + q_
    sol = sp.solve([fp.subs(x, -2) - 4, fp.subs(x, 3) - 4], [p_, q_])
    fp = fp.subs(sol)
    f = sp.integrate(fp, x)                                        # 최고차항의 계수 1 (적분상수는 도함수에 무관)
    roots = sp.solve(sp.Eq(sp.limit((f.subs(x, x + sp.Symbol("h")) - f) / sp.Symbol("h"), sp.Symbol("h"), 0), 4), x)
    V.check(7, "집합의 원소 {-2,3}", sorted(roots), [-2, 3])
    V.check(7, "g(4)=f'(4)", fp.subs(x, 4), V.ans(7, "1"))


def p08():
    f = lambda u: u**3 / 3 - u**2 + sp.Rational(5, 3) * u - 2
    assert sp.diff(f(x), x).subs(x, 1) > 0 and sp.simplify(sp.diff(f(x), x) - ((x - 1) ** 2 + sp.Rational(2, 3))) == 0
    ff = sp.lambdify(x, f(x), "mpmath")
    g = lambda yv: mp.findroot(lambda u: ff(u) - yv, yv)          # 역함수 g(yv): f(u)=yv
    x1 = mp.findroot(lambda u: g(u) + u, -1)                       # g(x)=-x (x<0)
    x2 = mp.findroot(lambda u: g(u) - u, 3)                        # g(x)=x
    h = lambda u: abs(g(u) - abs(u))
    area = mp.quad(h, [x1, 0, x2])
    V.check(8, "교점 x=-1, 3 (g 기준)", (sp.Float(x1, 12), sp.Float(x2, 12)), (-1, 3), 1e-9)
    V.check(8, "넓이(역함수 직접 수치적분)", sp.Float(area, 20), S(V.ans(8, "1")), 1e-9)
    # f 쪽 계산: y=±x 와 y=f(x) 사이
    A = sp.integrate(2 * x, (x, 0, 1)) + sp.integrate(x - f(x), (x, 1, 3))
    V.check(8, "f 쪽 대칭 영역 넓이", A, V.ans(8, "1"))


def p09():
    k_ = sp.Symbol("k", real=True)
    v = sp.diff(x**4 / 4 - sp.Rational(5, 2) * x**3 + 6 * x**2 + k_ * x, x)
    # (나): v=5/2 의 양의 해가 정확히 두 개 → 극댓값 = 5/2 (k 를 찾는다)
    sol_k = sp.solve(sp.Eq(v.subs(x, 1), sp.Rational(5, 2)), k_)
    V.report(9, sol_k == [-3] and sol_k[0] + 2 < 0, f"k={sol_k}, (가) k<-2 만족")
    vk = v.subs(k_, sol_k[0])
    roots = sp.Poly(vk - sp.Rational(5, 2), x).roots() if False else sp.roots(sp.Poly(vk - sp.Rational(5, 2), x))
    pos = sorted(r for r in roots if r > 0)
    V.report(9, pos == [1, sp.Rational(11, 2)], f"v=5/2 의 양의 근 {pos} (p=1, q=11/2)")
    # k 가 다른 값이면 (나) 불성립 (양의 근 개수 != 2)
    for kk in (-2.5, -3.5, -5):
        n_ = len([r for r in sp.Poly(v.subs(k_, kk) - sp.Rational(5, 2), x).nroots() if abs(sp.im(r)) < 1e-9 and sp.re(r) > 0])
        V.report(9, n_ != 2 or kk == -3, f"k={kk} 이면 양의 근 {n_}개 (정확히 2개 아님)")
    V.check(9, "a(q)", sp.diff(vk, x).subs(x, pos[-1]), V.ans(9, "1"))


if __name__ == "__main__":
    V.run(globals())
