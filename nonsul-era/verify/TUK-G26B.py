#!/usr/bin/env python3
"""TUK-G26B-01~09 (한국공학대 2026 오후) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G26B", "items/TUK/PAST")
mp.mp.dps = 30
a, p_, q_, k_ = sp.symbols("a p q k", real=True)


def kv(no):
    return dict(q.split("=", 1) for q in V.ans(no, "1").replace(" ", "").split(";"))


def p01():
    t0 = sp.pi - sp.asin(sp.Rational(4, 5))                        # sinθ=4/5, 제2사분면
    assert sp.pi / 2 < t0 < sp.pi
    V.check(1, "cosθ+tanθ", sp.simplify(sp.cos(t0) + sp.tan(t0)), V.ans(1, "1"))
    V.check(1, "cos(π/2-θ)=4/5 확인", sp.simplify(sp.cos(sp.pi / 2 - t0)), sp.Rational(4, 5))


def p02():
    sols = roots_in(4 * sp.sin(2 * x) - 2, x, 0, sp.pi)
    V.report(2, len(sols) == 2, f"교점 {sols}")
    V.check(2, "β-α", sols[1] - sols[0], V.ans(2, "1"))


def p03():
    sol = sp.solve([3 ** (-2 - p_) + q_, 3 ** (0 - p_) + q_ - 8], [p_, q_], dict=True)
    V.report(3, len(sol) == 1, f"해 {sol}")
    V.check(3, "p", sol[0][p_], kv(3)["p"]); V.check(3, "q", sol[0][q_], kv(3)["q"])


def p04():
    f = sp.log(x - 1, 2) + 2
    xa, xb = sp.solve(sp.Eq(f, 2), x)[0], sp.solve(sp.Eq(f, 5), x)[0]
    A, B, C = sp.Matrix([xa, 2]), sp.Matrix([xb, 5]), sp.Matrix([xb, 0])
    V.check(4, "넓이", sp.Abs(sp.Matrix.hstack(A - C, B - C).det()) / 2, V.ans(4, "1"))
    V.check(4, "A(2,2) B(9,5)", (xa, xb), (2, 9))


def p05():
    c3 = sp.symbols("c0:3")
    f = c3[0] + c3[1] * x + c3[2] * x**2                           # 일반 이차 이하로 놓고 항등식 풀이
    t = sp.Symbol("t")
    eq = sp.expand(sp.integrate(f.subs(x, t), (t, -1, x)) - x**3 - a * x**2 - sp.integrate(f.subs(x, t), (t, -1, 1)))
    sol = sp.solve(sp.Poly(eq, x).coeffs(), [*c3, a], dict=True)
    V.report(5, len(sol) == 1 and sol[0][a] == -1, f"항등식 해 {sol}")
    fx = f.subs(sol[0])
    V.check(5, "g(3)", sp.integrate(fx.subs(x, t), (t, -1, 3)), V.ans(5, "1"))


def p06():
    f = x**2
    Q_y = sp.simplify((-1 / sp.diff(f, x).subs(x, a)) * (0 - a) + a**2)    # 법선의 y절편
    area = a * Q_y / 2                                               # 밑변 OQ(y축 위), 높이 a
    V.check(6, "Q 의 y좌표", Q_y, a**2 + sp.Rational(1, 2))
    V.check(6, "극한", sp.limit(area / a, a, 0, "+"), V.ans(6, "1"))
    # 삼각형 넓이를 행렬식으로 독립 확인
    det = sp.Matrix.hstack(sp.Matrix([a, a**2]), sp.Matrix([0, Q_y])).det()
    V.check(6, "S(a) 행렬식 (a=2)", abs(det.subs(a, 2)) / 2, area.subs(a, 2))


def p07():
    a_ = sp.solve(2 * p_ - 4, p_)[0]                                  # v2(a)=0
    kk = sp.solve(3 * a_**2 + k_ * a_ + k_, k_)[0]
    t = sp.Symbol("t")
    v1, v2 = 3 * t**2 + kk * t + kk, 2 * t - 4
    xp, xq = sp.integrate(v1, (t, 0, a_)), sp.integrate(v2, (t, 0, a_))
    V.report(7, a_ == 2 and kk == -4, f"a={a_}, k={kk}, v1 에서도 t=2 가 부호 변화점: v1={sp.factor(v1)}")
    V.check(7, "거리", sp.Abs(xp - xq), V.ans(7, "1"))


def p08():
    a_ = sp.Symbol("a_", real=True)
    f = lambda u, aa: u**3 / 4 + aa * u - aa**2 + 3
    cand = sp.solve(f(2, a_) - 2, a_)                                  # (나) f(2)=2
    ok_a = [sp.nsimplify(c) for c in cand if c >= 0]                                 # (가) f'>=0 이려면 a>=0
    V.report(8, ok_a == [3], f"a 후보 {cand}, a>=0 → {ok_a}")
    fx = f(x, 3)
    V.report(8, sp.Poly(fx - x, x).count_roots() == 1, "f(x)-x=0 의 서로 다른 실근 1개(x=2)")
    ff = sp.lambdify(x, fx, "mpmath")
    g = lambda yv: mp.findroot(lambda u: ff(u) - yv, yv / 3)           # 역함수
    xe = mp.findroot(lambda u: g(u) + u + 16, -14)                      # g 와 y=-x-16 의 교점 (x≈-14)
    V.check(8, "g 와 y=-x-16 교점 x=-14", sp.Float(xe, 12), -14, 1e-9)
    x2 = mp.findroot(lambda u: g(u) - u, 2)
    # 영역: x∈[-14,-2] 에서 g 위·직선 아래, x∈[-2,2] 에서 g 위·f 아래
    area = mp.quad(lambda u: g(u) - (-u - 16), [xe, -2]) + mp.quad(lambda u: g(u) - ff(u), [-2, x2])
    V.check(8, "넓이(역함수 직접 수치적분)", sp.Float(area, 20), V.ans(8, "1"), 1e-8)
    A = sp.integrate(x - (-x - 16), (x, -8, -2)) + sp.integrate(x - fx, (x, -2, 2))
    V.check(8, "한쪽 영역 넓이 ×2", 2 * A, V.ans(8, "1"))


def p09():
    al = sp.Symbol("alpha", real=True)
    fp = 3 * (x - al) * (x - 3)                                        # (가): f'(3)=0, 극대는 x=α<3
    D = sp.discriminant(sp.expand(fp + 12), x)                         # (나): f'(x)=-12 의 중근
    als = [r for r in sp.solve(D, al) if r < 3]
    V.report(9, als == [-1], f"α 후보 {sp.solve(D, al)}, α<3 → {als}")
    f = sp.integrate(fp.subs(al, als[0]), x) + k_
    V.report(9, sp.expand(f) == x**3 - 3 * x**2 - 9 * x + k_, f"f(x)={sp.expand(f)}")
    # 정수 k 중 서로 다른 세 실근이 되는 최댓값을 직접 열거
    good = [kk for kk in range(-40, 60) if len(sp.Poly(f.subs(k_, kk), x).real_roots()) == 3 and len(set(sp.Poly(f.subs(k_, kk), x).real_roots())) == 3]
    V.check(9, "정수 k 최댓값(열거)", max(good), V.ans(9, "1"))
    V.report(9, min(good) == -4 and good == list(range(-4, 27)), f"가능한 정수 k = {good[0]}..{good[-1]} (-5<k<27)")


if __name__ == "__main__":
    V.run(globals())
