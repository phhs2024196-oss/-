#!/usr/bin/env python3
"""TUK-G25A-01~09 (한국공학대 2025 오전) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G25A", "items/TUK/PAST")
a, b, c, r, L, kk = sp.symbols("a b c r L kk", real=True)


def p01():
    bb = sp.solve(sp.Eq(2 * sp.pi / b, sp.pi / 2), b)[0]
    sol = sp.solve([sp.Eq(a + c, 7), sp.Eq(-a + c, 3)], [a, c])
    V.check(1, "⑴ b", bb, V.ans(1, "1", "⑴"))
    LOC = {"a": a, "c": c}
    V.check(1, "⑵ 최댓값 a+c (a>0)", a + c, sp.sympify(V.ans(1, "1", "⑵"), locals=LOC))
    V.check(1, "⑶ 최솟값 -a+c", -a + c, sp.sympify(V.ans(1, "1", "⑶"), locals=LOC))
    V.check(1, "⑷ a", sol[a], V.ans(1, "1", "⑷"))
    V.check(1, "⑸ c", sol[c], V.ans(1, "1", "⑸"))
    f = sol[a] * sp.sin(bb * x) + sol[c]
    V.check(1, "주기 확인", sp.periodicity(f, x), sp.pi / 2)


def p02():
    sols = [s for s in sp.solve(sp.Eq(2 * L * L, (L + 1) * 1), L) if s != 1]
    assert len(sols) == 1
    V.check(2, "log_a b + log_b a", sols[0] + 1 / sols[0], V.ans(2, "1"))


def p03():
    rr = [s for s in sp.solve(sp.Eq(5 / (r + r**2), 16), r) if s > 0][0]
    V.check(3, "a3/a5", 1 / rr**2, V.ans(3, "1"))
    # 수치 확인: 첫째항 1, 공비 rr
    a_ = lambda n_: rr ** (n_ - 1)  # noqa: E731
    V.check(3, "조건 재확인", 5 * a_(2) / (a_(3) + a_(4)), 16)


def p04():
    f1 = lambda xx: sp.Rational(1, 4) ** (xx - 1) - 1  # noqa: E731
    # 제1사분면: 첫 함수가 양수인 x 는 0<x<1, k = f1(x)-2^x 는 x 에 대해 감소
    F = lambda xx: f1(xx) - 2**xx  # noqa: E731
    alpha, beta = F(1), F(0)
    V.check(4, "α", alpha, -2)
    V.check(4, "β", beta, 2)
    V.check(4, "log_{√2}(β-α)", sp.log(beta - alpha) / sp.log(sp.sqrt(2)), V.ans(4, "1"))
    # 수치 점검: k 가 범위 안/밖에서 제1사분면 교점 유무
    import mpmath as mp

    def has(kv):
        for xs in [i / 1000 for i in range(1, 1000)]:
            if abs(float(F(xs)) - kv) < 2e-3 and float(2**xs + kv) > 0:
                return True
        return False
    V.report(4, has(0) and has(1.9) and has(-1.9) and not has(2.3) and not has(-2.3), "범위 안팎 수치 확인")


def p05():
    P = x**3 - 4 * x**2 + x + 6
    f1 = sp.sqrt(P.subs(x, 1))                 # f(1)=g(1), f(1)>0
    fp1 = sp.diff(P, x).subs(x, 1) / (2 * f1)  # 2 f'(1) f(1) = P'(1)
    V.check(5, "f'(1)", fp1, V.ans(5, "1"))


def p06():
    A = [s for s in sp.solve(sp.Eq((1 + a) ** 2, 4 * a**2), a) if s > 0][0]
    B = [s for s in sp.solve(sp.Eq(4 * A**2, (b - 3) ** 2), b) if s > 0 and s != A][0]
    f = lambda xx: (-xx - A) if xx < 1 else (2 * A if xx == 1 else B * xx - 3)  # noqa: E731
    V.check(6, "f(a-b)+f(a+b)", f(A - B) + f(A + B), V.ans(6, "1"))


def p07():
    sol = [s for s in sp.solve(sp.Eq(sp.integrate(x - (-x**2 + kk * x), (x, kk - 1, kk)) - sp.integrate(-x**2 + kk * x - x, (x, 0, kk - 1)), sp.Rational(2, 3)), kk) if s > 1]
    V.check(7, "k", sol[0], V.ans(7, "1"))
    xs = sp.solve(sp.Eq(-x**2 + kk * x, x), x)
    V.check(7, "P 의 x좌표", [s for s in xs if s != 0][0], kk - 1)


def p08():
    bb = sp.symbols("bb", positive=True)
    aa = 2 + 4 * bb                             # t2-t1 = (a-4b)/6 = 1/3
    t = sp.Symbol("t", real=True)
    v = 6 * t**2 - (aa + 8 * bb) * t + (aa + 2 * bb) * bb
    sols = sp.solve(sp.Eq(sp.integrate(v, (t, 0, bb)), 3), bb)
    bv = [s for s in sols if s > 0][0]
    av = aa.subs(bb, bv)
    assert av > 4 * bv
    vv = v.subs({bb: bv})
    V.check(8, "x(2)", sp.integrate(vv, (t, 0, 2)), V.ans(8, "1"))
    V.check(8, "∫0^b v dt = 3 (v≥0 on [0,b])", sp.integrate(vv, (t, 0, bv)), 3)


def p09():
    ca, cb, cc = sp.symbols("ca cb cc")
    f = -2 * x**2 + 4 * x + 2
    # 항등식 확인
    t_ = sp.Symbol("t_")
    V.check(9, "항등식", sp.integrate(f.subs(x, t_), (t_, 0, x)) - (f - sp.Rational(2, 3) * x**3 + 4 * x**2 - 2 * x - 2), 0)
    # 정답 값 = argmax g(k)=k*sqrt(8-2k)/2 on (0,4)
    g2 = (kk**2 * (8 - 2 * kk)) / 4
    crit = [s for s in sp.solve(sp.diff(g2, kk), kk) if 0 < s < 4]
    V.check(9, "argmax k", crit[0], V.ans(9, "1"))
    # 교점 A,B 와 넓이 직접 계산
    kv = crit[0]
    rts = sp.solve(sp.Eq(f, kv), x)
    area = sp.Abs(rts[0] - rts[1]) * kv / 2
    V.check(9, "g(k)^2", area**2, g2.subs(kk, kv))


if __name__ == "__main__":
    V.run(globals())
