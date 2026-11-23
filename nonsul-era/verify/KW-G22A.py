#!/usr/bin/env python3
"""KW-G22A (광운대 2022학년도 논술고사 자연계열 1교시) — sympy 검산. 기대값은 V.ans() 로만 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: F401,F403
import math
sys.set_int_max_str_digits(0)
import re
import fractions
import numpy as np

V = Verifier("KW-G22A", "items/KW/PAST")
b_, k_, a_, th, X, Y = sp.symbols("b k a th X Y")      # 가정 없는 기호(YAML 문자열 sympify 와 일치)


def num_eq(expr, ans, var, pts, tol=1e-9):
    e = S(ans) if isinstance(ans, str) else ans
    return all(abs(float(expr.subs(var, p)) - float(e.subs(var, p))) < tol for p in pts)


def p01():
    # [1]
    N = (11 * 13 * 15 * 17 - 9) + (11**3 + 5 * 11**2 + 3 * 11 - 9)
    V.check(1, "1 소인수분해", N, V.ans(1, "1"))
    x_ = sp.Symbol("x_")
    V.report(1, sp.expand((x_**2 + 6*x_ - 1) * (x_ + 3)**2 + (x_ - 1) * (x_ + 3)**2 - (x_**2 + 7*x_ - 2) * (x_ + 3)**2) == 0,
             "1 인수분해 항등식 (x^2+6x-1)(x+3)^2+(x-1)(x+3)^2")
    # [2]
    c = b_**2 - 7*b_ + 12
    dist = sp.sqrt((3*b_ + c)**2) / sp.sqrt(10)          # |3b+c|/sqrt(10)
    pts = [-3, -1, 0, 1.5, 2, 5, 9]
    V.report(1, num_eq(dist, V.ans(1, "2-1"), b_, pts), "2-1 거리식")
    area = sp.sqrt(10) / 2 * dist                        # 1/2 * AO * dist, AO = sqrt(10)
    V.report(1, num_eq(area, V.ans(1, "2-2", "area"), b_, pts), "2-2 넓이식")
    bs = np.linspace(-10, 14, 24001)
    ar = [float(area.subs(b_, v)) for v in bs[::200]]
    V.report(1, abs(min(ar) - float(S(V.ans(1, "2-2", "min")))) < 1e-9, f"2-2 최솟값(격자) {min(ar)}")
    # 직접 삼각형 넓이(신발끈) 로 이중 확인
    A, O, B = sp.Matrix([-1, 3]), sp.Matrix([0, 0]), sp.Matrix([b_, c])
    shoelace = sp.Abs((A[0]*(O[1]-B[1]) + O[0]*(B[1]-A[1]) + B[0]*(A[1]-O[1])) / 2)
    V.report(1, all(abs(float(shoelace.subs(b_, p)) - float(area.subs(b_, p))) < 1e-9 for p in pts), "2-2 신발끈 공식 일치")
    # [3](1): 분모 x^2-kx+3 이 0 이 되지 않을 k
    sol = sp.solve_univariate_inequality(k_**2 - 12 < 0, k_, relational=False)
    lo, mid, hi = V.ans(1, "3-1").split("<")
    V.report(1, sol == sp.Interval.open(S(lo), S(hi)) and mid.strip() == "k", f"3-1 k 범위 {sol}")
    # [3](2): h'(x)=3k x^2+2x+4k>0 (x>0)
    xs = np.concatenate([np.geomspace(1e-6, 1e6, 4000)])
    cond = V.ans(1, "3-2").replace("k", "kk")
    bad = []
    for kk in list(np.round(np.arange(-2, 2.001, 0.05), 3)) + [1e-6, -1e-6, 1e-3, -1e-3]:
        inc = bool(np.all(3 * kk * xs**2 + 2 * xs + 4 * kk > 0))
        if inc != bool(eval(cond)):
            bad.append(kk)
    V.report(1, not bad, f"3-2 증가 k 범위 불일치 {bad[:3]}")
    # [4](1): AS=2, SB=18
    a_el = (2 + 18) / sp.Integer(2)
    c_el = a_el - 2
    b2 = a_el**2 - c_el**2
    lhs, rhs = V.ans(1, "4-1").split("=")
    V.report(1, sp.simplify(S(lhs.replace("x", "X").replace("y", "Y")) - (X**2 / a_el**2 + Y**2 / b2)) == 0 and S(rhs) == 1,
             f"4-1 타원 a={a_el}, b^2={b2}")
    # [4](2): a, b, 접선
    a_, b_v = float(a_el), math.sqrt(float(b2))
    V.check(1, "4-2 a", a_, V.ans(1, "4-2", "a"))
    V.check(1, "4-2 b", b_v, V.ans(1, "4-2", "b"))
    xe, ye = 10 * sp.cos(th), 6 * sp.sin(th)
    slope = sp.simplify(sp.diff(ye, th) / sp.diff(xe, th)).subs(th, sp.pi / 3)
    px, py = xe.subs(th, sp.pi / 3), ye.subs(th, sp.pi / 3)
    tang = slope * (X - px) + py
    V.report(1, num_eq(tang, V.ans(1, "4-2", "tangent_y").replace("x", "X"), X, [-3, 0, 1, 5, 10]), "4-2 접선 y")
    # 접선 공식 5x/100 + 3√3 y/36 = 1 과도 일치
    tang2 = (1 - px * X / 100) * 36 / py                   # 접선 공식 x1*x/a^2 + y1*y/b^2 = 1 (a^2=100, b^2=36)
    V.report(1, all(abs(float(tang2.subs(X, q)) - float(tang.subs(X, q))) < 1e-9 for q in (-2, 0, 2, 7)), "4-2 접선 공식 일치")
    # [4](3): l = |SE|, S=(-8,0)
    l = sp.sqrt((10 * sp.cos(th) + 8)**2 + (6 * sp.sin(th))**2)
    ths = np.linspace(0, 2 * math.pi, 200001)
    lv = np.sqrt((10 * np.cos(ths) + 8)**2 + (6 * np.sin(ths))**2)
    V.report(1, abs(lv.min() - float(S(V.ans(1, "4-3", "min")))) < 1e-6
             and abs(ths[lv.argmin()] - float(S(V.ans(1, "4-3", "theta")))) < 1e-4, f"4-3 최소 {lv.min():.6f} at {ths[lv.argmin()]:.6f}")
    # [4](4): T = ∫_{5√2}^{10} 6√(1-x²/100) dx + 1/2 * 5√2 * 3√2   및  1/2 ∫ (x y' - y x') dθ
    T1 = sp.integrate(6 * sp.sqrt(1 - X**2 / 100), (X, 5 * sp.sqrt(2), 10)) + sp.Rational(1, 2) * 5 * sp.sqrt(2) * 3 * sp.sqrt(2)
    T2 = sp.integrate(sp.Rational(1, 2) * (xe * sp.diff(ye, th) - ye * sp.diff(xe, th)), (th, 0, sp.pi / 4))
    V.report(1, same(T1, V.ans(1, "4-4")) and same(T2, V.ans(1, "4-4")), f"4-4 넓이 {sp.simplify(T1)} / {sp.simplify(T2)}")


def p02():
    # [1]
    a, d, c = sp.symbols("a d c")
    poly = sp.expand((X - (a - d)) * (X - a) * (X - (a + d)))
    co = sp.Poly(poly - (X**3 - 9 * X**2 + c * X - 12), X).all_coeffs()
    sol = sp.solve(co, [a, c, d**2 if False else d], dict=True)
    ok_a = all(same(s[a], V.ans(2, "1-1")) for s in sol)
    ok_c = all(same(s[c], V.ans(2, "1-2")) for s in sol)
    V.report(2, bool(sol) and ok_a and ok_c, f"1-1/1-2 a,c 해 {sol}")
    roots = sp.roots(sp.Poly(X**3 - 9 * X**2 + 22 * X - 12, X))
    exp3 = [S(t) for t in V.ans(2, "1-3").split(",")]
    V.report(2, same(list(roots.keys()), exp3) and len(roots) == 3, f"1-3 세 실근 {list(roots.keys())}")
    # [2](1): 항등식 수치 확인
    fr = fractions.Fraction
    fa = math.factorial
    V.proof(2, "2-1", [n for n in range(1, 60)
                       if fr(n + 2, fa(n) + fa(n + 1) + fa(n + 2)) != fr(n + 1, fa(n + 2))])
    # [2](2)
    tot = sum(sp.Rational(n + 2, sp.factorial(n) + sp.factorial(n + 1) + sp.factorial(n + 2)) for n in range(1, 2023))
    V.check(2, "2-2 급수합", tot, V.ans(2, "2-2"))
    # [3](1): h(x)=ln|x|+1/√x 최솟값 > 0
    xx = sp.Symbol("xx", positive=True)
    h = sp.log(xx) + 1 / sp.sqrt(xx)
    cr = sp.solve(sp.diff(h, xx), xx)
    hm = sp.simplify(h.subs(xx, cr[0]))
    grid = np.linspace(1e-9, 1 - 1e-9, 200001)
    V.report(2, cr == [sp.Rational(1, 4)] and float(hm) > 0 and bool(np.all(np.log(grid) > -1 / np.sqrt(grid))),
             f"3-1 h 최솟값 {hm}={float(hm):.4f} at {cr}")
    # [3](2)
    right = sp.limit(xx * sp.log(xx), xx, 0, "+")
    left = -sp.limit(xx * sp.log(xx), xx, 0, "+")      # f(-t)=-f(t) 이므로 lim_{x->0-} f = -lim_{t->0+} f
    V.report(2, same(right, V.ans(2, "3-2", "right")) and same(left, V.ans(2, "3-2", "left")), f"3-2 극한 {right},{left}")
    # [3](3): k=2 로 g(t) 수치 계산
    kv = 2.0
    f = lambda x: kv * x * np.log(np.abs(x))
    xs = np.concatenate([-np.geomspace(1e-12, 60, 400000)[::-1], np.geomspace(1e-12, 60, 400000)])

    def cnt(t):
        v = f(xs) - t
        n = int(np.sum(v[:-1] * v[1:] < 0))
        # 접하는 극점 (x=±1/e): |f-t| 매우 작으면 한 점으로 추가
        n += sum(1 for xc in (-1 / math.e, 1 / math.e) if abs(f(xc) - t) < 1e-12)
        return n
    e_ = math.e
    cand = [-kv / e_, 0.0, kv / e_]
    # t=0 에서는 x=±1 에서 정확히 0 → 격자에서 부호변화로 잡히므로 아래 별도 계산
    def g(t):
        if abs(t) < 1e-15:
            return 2                                   # x=-1, 1 (x=0 은 정의역 밖)
        return cnt(t)
    eps = 1e-6
    jumps = []
    for tc in cand:
        left_v, mid_v, right_v = g(tc - eps), g(tc), g(tc + eps)
        if not (left_v == mid_v == right_v):
            jumps.append((tc, mid_v))
    # 후보 밖에서는 구간 안에서 g 가 상수
    ordered = sorted(cand)
    probes = [ordered[0] - q for q in (0.5, 1, 2, 5, 10)] + [ordered[2] + q for q in (0.5, 1, 2, 5, 10)]
    for lo_, hi_ in zip(ordered[:-1], ordered[1:]):
        probes += [lo_ + (hi_ - lo_) * s_ for s_ in (0.1, 0.3, 0.5, 0.7, 0.9)]
    flat = all(g(t) == g(t - 1e-4) == g(t + 1e-4) for t in probes)        # 후보 밖에서는 g 가 국소 상수
    pairs = re.findall(r"\((.+?),(\d+)\)(?=,|$)", V.ans(2, "3-3"))
    exp_pairs = sorted((float(S(t).subs(k_, kv)), int(v)) for t, v in pairs)
    got_pairs = sorted((float(t), int(v)) for t, v in jumps)
    V.report(2, flat and len(exp_pairs) == len(got_pairs) and all(abs(p[0] - q[0]) < 1e-9 and p[1] == q[1] for p, q in zip(exp_pairs, got_pairs)),
             f"3-3 불연속 (t,g) 계산 {got_pairs} / YAML {exp_pairs}")
    # [3](4): ∫_{-1}^{-1/√e} k x ln(-x) dx = 1
    u = sp.Symbol("u", positive=True)
    I = -sp.integrate(u * sp.log(u), (u, sp.exp(-sp.Rational(1, 2)), 1))   # x=-u 치환
    kk = sp.solve(sp.Eq(k_ * I, 1), k_)[0]
    nonneg = bool(np.all(np.linspace(-1, -1 / math.sqrt(math.e), 1000) * np.log(-np.linspace(-1, -1 / math.sqrt(math.e), 1000)) >= -1e-15))
    V.report(2, same(kk, sp.sympify(V.ans(2, "3-4"), locals={"e": sp.E})) and nonneg, f"3-4 k={sp.simplify(kk)}")


if __name__ == "__main__":
    V.run(globals())
