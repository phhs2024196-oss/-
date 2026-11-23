#!/usr/bin/env python3
"""TUK-G24B-01~03 (한국공학대 2024 오후) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G24B", "items/TUK/PAST")
mp.mp.dps = 25
th = sp.Symbol("th", real=True)


def kv(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def close(val, expect):
    return abs(mp.mpf(str(sp.N(val, 25))) - mp.mpf(str(sp.N(S(expect), 25)))) < 1e-9


def p01():
    D = (4 * sp.cos(th)) ** 2 - 8 * (3 - 3 * sp.sin(th))
    sols = sorted(sp.solveset(sp.Eq(D, 0), th, sp.Interval.Ropen(0, 2 * sp.pi)), key=lambda r: float(r))
    e = kv(1, "1")
    exp = sorted([S(e["다"]), S(e["라"]), S(e["마"])], key=lambda r: float(r))
    V.check(1, "1-1 θ 해집합(다·라·마)", sp.FiniteSet(*sols), sp.FiniteSet(*exp))
    ga = S(e["가"].replace("t", "th"), locals={"th": th})
    base = 2 * (1 - sp.sin(th) ** 2) + 3 * sp.sin(th) - 3
    V.report(1, sp.simplify(ga * (sp.sin(th) - 1) - base) == 0 or sp.simplify(ga * (sp.sin(th) - 1) + base) == 0, f"1-1 (가) 인수 {ga}")
    V.report(1, same(sp.sin(sols[0]), e["나"]), "1-1 (나) sinθ=1/2")
    a = sp.Symbol("a", real=True)
    xa = sp.solve(sp.Eq(sp.log(-x, 3), 2), x)[0]
    xb = sp.solve(sp.Eq(sp.log(x, sp.Rational(1, 3)) + a, 2), x)[0]
    V.check(1, "1-2 a", sp.solve(sp.Eq(sp.simplify(xb - xa), 18), a)[0], V.ans(1, "2"))
    r = sp.Symbol("r", positive=True)
    a1 = sp.Symbol("a1", positive=True)
    rr = sp.solve(sp.Eq(r**2, 4), r)[0]  # 2^(a_{n+2}) = 16^(a_n) → a_{n+2}=4a_n
    S_ = lambda m: a1 * (rr**m - 1) / (rr - 1)
    a1v = sp.solve(sp.Eq(S_(8) - S_(4), 160), a1)[0]
    V.check(1, "1-3 a5", a1v * rr**4, V.ans(1, "3"))
    # 수열 직접 전개로 검산
    seq = [a1v, a1v * rr]
    for i in range(6): seq.append(4 * seq[i])
    V.report(1, sum(seq[:8]) - sum(seq[:4]) == 160, "1-3 a_{n+2}=4a_n 전개의 S8-S4=160")


def p02():
    f = x**3 - sp.Rational(3, 2) * x**2 - 3 * x + 4
    fp = sp.diff(f, x)
    roots = sp.solve(sp.Eq(fp, 3), x)
    s_ = sum(roots)
    tang = sp.expand(f.subs(x, s_) + fp.subs(x, s_) * (x - s_))
    V.check(2, "2-1 접선", tang, sp.sympify(V.ans(2, "1").split("=")[1], locals={"x": x}))
    A = sp.Symbol("A", positive=True)
    fa = x**3 - sp.Rational(3, 2) * (A + 1) * x**2 + 3 * A * x - sp.Rational(9, 4) * A**2 + 3 * A
    cr = sp.solve(sp.diff(fa, x), x)
    mn = [rt for rt in cr if sp.simplify(sp.diff(fa, x, 2).subs(x, rt).subs(A, 2)) > 0][0]
    g = sp.simplify(fa.subs(x, mn))
    V.check(2, "2-2 lim g'(a)/(a-1)", sp.limit(sp.diff(g, A) / (A - 1), A, 1, "+"), V.ans(2, "2"))
    x1 = sp.integrate(3 * t**2 - 11 * t + 8, t); x2 = sp.integrate(-3 * t**2 + 5 * t, t)
    meet = [r_ for r_ in sp.solve(sp.Eq(x1, x2), t) if r_ > 0]
    v1 = lambda tt: 3 * tt**2 - 11 * tt + 8
    dist = mp.quad(lambda tt: abs(v1(tt)), [0, 1, float(meet[0])])
    V.report(2, len(meet) == 1 and close(meet[0], 2) and abs(dist - mp.mpf(str(sp.N(S(V.ans(2, "3")), 25)))) < 1e-9, f"2-3 만나는 시각 {meet}, 거리 수치 {dist}")


def p03():
    f = x**2 + 2
    gl = lambda y_: mp.sqrt(y_ - 2)
    # 곡선 y=g(x)=sqrt(x-2) (x>=2) 와 직선 y=(x-2)/3 사이 넓이
    line = lambda u: (u - 2) / 3
    xs = sp.solve(sp.Eq(sp.sqrt(x - 2), (x - 2) / 3), x)
    xs = sorted([r_ for r_ in xs if r_ >= 2])
    area = mp.quad(lambda u: abs(mp.sqrt(u - 2) - line(u)), [float(xs[0]), float(xs[-1])])
    V.report(3, close(area, 0) is False and abs(area - mp.mpf(str(sp.N(S(V.ans(3, "1")), 25)))) < 1e-9, f"3-1 넓이 수치 {area} (교점 {xs})")
    a, b, c, k = sp.symbols("a b c k")
    F = a * x**2 + b * x + c
    lhs = sp.integrate(t * sp.diff(F, x).subs(x, t), (t, 0, x))
    # F(x)=f 의 부정적분 중 F(0)=0 (x=0 대입 조건)
    Fi = sp.integrate(F, x)
    eqs = sp.Poly(sp.expand(lhs - 2 * Fi - k * x), x).coeffs()
    sol = sp.solve(eqs, [b, c], dict=True)[0]
    fx = F.subs(sol)
    kv_ = sp.solve(sp.Eq(fx.subs(x, 0), -3), k)  # 최솟값 -3 (a>0 이면 x=0 에서 최소)
    fk = fx.subs(k, kv_[0])
    av = sp.Symbol("av", positive=True)
    val = sp.integrate(fk, (x, 0, 1))
    V.check(3, "3-2 ∫f 최솟값 (a≥3)", sp.minimum(val, a, sp.Interval(3, sp.oo)), V.ans(3, "2"))
    # 3-3
    A_ = sp.Symbol("A_", positive=True)
    bq, cq = sp.symbols("bq cq")
    ff = x**3 - A_ * x**2 + bq * x + cq
    cq_v = A_                      # f(0)=a
    bq_v = -1                      # f'(0)=-1
    ff = ff.subs({bq: bq_v, cq: cq_v})
    # 수치로 a>1 탐색
    fn = sp.lambdify((x, A_), ff, "mpmath")
    cond_n = lambda av_: mp.quad(lambda u: fn(u, av_), [-1, 1]) - mp.quad(lambda u: abs(fn(u, av_)), [1, av_])
    root = mp.findroot(cond_n, 3.3)
    V.report(3, abs(root - 3) < 1e-9 and close(sp.diff(ff, x).subs(A_, 3).subs(x, 3), V.ans(3, "3")), f"3-3 a={root}, f'(a)={sp.diff(ff,x).subs(A_,3).subs(x,3)}")
    grid = [mp.mpf(1) + mp.mpf(i) / 10 + mp.mpf(1) / 37 for i in range(0, 80)]
    sign_changes = sum(1 for p_, q_ in zip(grid, grid[1:]) if cond_n(p_) * cond_n(q_) < 0)
    V.report(3, sign_changes == 1, "3-3 (1,9) 구간에서 조건 (다)의 해 a 가 하나뿐(부호 변화 1회)")


if __name__ == "__main__": V.run(globals())
