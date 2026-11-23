#!/usr/bin/env python3
"""TUK-G25B-01~09 (한국공학대 2025 오후) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G25B", "items/TUK/PAST")
a, b, kk = sp.symbols("a b kk", real=True)
n_ = sp.Symbol("n_", integer=True, positive=True)


def p01():
    B, A = sp.symbols("B A", positive=True)
    sol = sp.solve([sp.Eq(A, sp.pi / 2 + B), sp.Eq(A + 2 * B, sp.pi)], [A, B])
    V.check(1, "⑴ ∠A", sol[A], V.ans(1, "1", "⑴"))
    V.check(1, "⑵ ∠B", sol[B], V.ans(1, "1", "⑵"))
    # sinA = cosB 확인
    V.check(1, "sin A = cos B", sp.sin(sol[A]), sp.cos(sol[B]))
    R = 2 * sp.sqrt(3) / sp.sin(sol[B]) / 2           # 사인법칙: AC/sin B = 2R (B 의 대변이 AC)
    V.check(1, "⑶ R", R, V.ans(1, "1", "⑶"))
    V.check(1, "⑷ 외접원 넓이", sp.pi * R**2, V.ans(1, "1", "⑷"))
    # 좌표로 독립 확인: AC=2√3, 꼭짓점 A=2π/3 에서 AB=AC, 외접원 반지름 = abc/(4·넓이)
    AB = AC = 2 * sp.sqrt(3)
    BC = sp.sqrt(AB**2 + AC**2 - 2 * AB * AC * sp.cos(sol[A]))
    area = AB * AC * sp.sin(sol[A]) / 2
    V.check(1, "R=abc/4S", AB * AC * BC / (4 * area), R)


def p02():
    s = sp.root(2, 6) / sp.root(4, 3)
    aa = sp.solve(sp.Eq(s * (-a / 2), -1), a)[0]
    V.check(2, "a^2", aa**2, V.ans(2, "1"))
    V.check(2, "기울기 2^(-1/2)", s, 2 ** sp.Rational(-1, 2))


def p03():
    ca, cd = sp.symbols("ca cd")
    S = lambda m: m * ca + m * (m - 1) * cd / 2  # noqa: E731
    expr = sp.expand(S(n_ + 2) - S(n_) - (10 * n_ + 9))
    sol = sp.solve(sp.Poly(expr, n_).all_coeffs(), [ca, cd])
    V.check(3, "a3", sol[ca] + 2 * sol[cd], V.ans(3, "1"))


def p04():
    xi = sp.Symbol("xi", real=True)
    x_int = sp.solve(sp.Eq(sp.log(xi + 1, sp.Rational(1, 2)) + 3, 0), xi)[0]
    V.check(4, "x절편", x_int, 7)
    alpha = sp.solve(sp.Eq(sp.log(7 + kk, 2), 0), kk)[0]
    beta = sp.solve(sp.Eq(sp.log(kk, 2), 3), kk)[0]
    V.check(4, "α", alpha, -6)
    V.check(4, "β", beta, 8)
    V.check(4, "log_4(α+β)", sp.log(alpha + beta) / sp.log(4), V.ans(4, "1"))
    F = lambda xs, kv: float(sp.log(xs + 1) / sp.log(0.5) + 3 - sp.log(xs + kv) / sp.log(2)) if xs + kv > 0 else None  # noqa: E731

    def has(kv):
        pts = [i / 200 for i in range(1, 1400)]
        vals = [(xs, F(xs, kv)) for xs in pts if F(xs, kv) is not None and xs < 7]
        for (x1, f1), (x2, f2) in zip(vals, vals[1:]):
            if f1 * f2 < 0:
                return True
        return False
    V.report(4, has(0) and has(7) and has(-5.9) and not has(8.5) and not has(-6.5), "범위 안팎 수치 확인")


def p05():
    P = x**4 - x**2
    g2 = 4
    f2 = P.subs(x, 2) / g2
    gp2 = 2 * f2
    fp2 = (sp.diff(P, x).subs(x, 2) - f2 * gp2) / g2
    V.check(5, "f'(2)", fp2, V.ans(5, "1"))


def p06():
    bb = [v for v in range(1, 10) if v**2 - 4 < 0][0]          # 판별식 D<0 인 자연수 b
    aa = 2 * bb
    V.check(6, "a", aa, V.ans(6, "1", "a"))
    V.check(6, "b", bb, V.ans(6, "1", "b"))
    V.check(6, "x=0 연속", sp.limit(bb**2 / (x**2 + bb * x + 1), x, 0), sp.Rational(aa**2, 4))
    V.report(6, all(sp.Poly(x**2 + bb * x + 1, x).discriminant() < 0 for _ in [0]), "분모 항상 양수(D<0)")
    # b=2 이상이면 x≠0 에서 분모 0 이 생기는지 확인
    V.report(6, any(sp.solve(x**2 + 2 * x + 1, x)), "b=2 는 분모=0 (x=-1) 이라 불가")


def p07():
    eq = sp.integrate(kk * x - (x**2 - x), (x, 0, kk + 1)) - sp.integrate(x**2 - x - kk * x, (x, kk + 1, 2 * kk))
    sol = [s for s in sp.solve(sp.Eq(eq, sp.Rational(8, 3)), kk) if s > 1]
    V.check(7, "k", sol[0], V.ans(7, "1"))
    xs = sp.solve(sp.Eq(x**2 - x, kk * x), x)
    V.check(7, "P 의 x좌표", [s for s in xs if s != 0][0], kk + 1)


def p08():
    t = sp.Symbol("t", real=True)
    xt = t**3 / 3 + a * t**2 + b * t
    v = sp.diff(xt, t)
    bs = sp.solve(sp.Eq(v.subs(t, 1), 0), b)[0]
    # a<=-2 이면 v 의 큰 영점 -2a-1 >= 3 이므로 [0,3] 에서 v 는 t=1 에서만 부호 변경(+ → -)
    dist = sp.integrate(v.subs(b, bs), (t, 0, 1)) - sp.integrate(v.subs(b, bs), (t, 1, 3))
    sol = [s for s in sp.solve(sp.Eq(dist, sp.Rational(23, 3)), a) if s <= -2]
    av = sol[0]
    V.check(8, "v(6)", v.subs({b: bs}).subs(a, av).subs(t, 6), V.ans(8, "1"))
    V.check(8, "a", av, -3)
    V.check(8, "b", bs.subs(a, av), 5)
    # 수치 확인: ∫|v| (절댓값 직접)
    import mpmath as mp
    vf = sp.lambdify(t, v.subs({b: bs}).subs(a, av), "mpmath")
    V.check(8, "∫0^3|v|", mp.quad(lambda s: abs(vf(s)), [0, 1, 3]), sp.Rational(23, 3), 1e-9)


def p09():
    ca, cb = sp.symbols("ca cb")
    f = sp.Rational(2, 3) * x**3 + 3 * ca * x**2 + 2 * cb * x
    sol = sp.solve([sp.diff(f, x).subs(x, -1), sp.diff(f, x).subs(x, 2)], [ca, cb])
    V.check(9, "a", sol[ca], sp.Rational(-1, 3))
    V.check(9, "b", sol[cb], -2)
    ff = f.subs(sol)
    h = sp.expand(3 * ff - 24 * x - 4)
    V.check(9, "h", h, 2 * x**3 - 3 * x**2 - 36 * x - 4)
    # 서로 다른 네 실근이 되는 자연수 k 의 개수: |h(x)|/4 = k
    cnt = 0
    for kv in range(1, 40):
        roots = set()
        for sgn in (1, -1):
            for r_ in sp.Poly(h - sgn * 4 * kv, x).real_roots():
                roots.add(sp.nsimplify(r_) if r_.is_Rational else round(float(r_), 9))
        if len(roots) == 4:
            cnt += 1
    V.check(9, "자연수 k 의 개수", cnt, V.ans(9, "1"))


if __name__ == "__main__":
    V.run(globals())
