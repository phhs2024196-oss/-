#!/usr/bin/env python3
"""KAU-G25S-01~02 (한국항공대 2025 이학계열) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,  # noqa: E402
                                        parse_expr, standard_transformations)
from vlib import *  # noqa: E402,F403

V = Verifier("KAU-G25S", "items/KAU/PAST")
TR = standard_transformations + (convert_xor, implicit_multiplication_application)


def P(s, **loc):
    return parse_expr(s, transformations=TR, local_dict={"e": sp.E, "ln": sp.log, "n": n, "x": x, **loc})


def parts(no, sub):
    return dict(p.strip().split("=", 1) for p in V.ans(no, sub).split(";"))


def ints(s):
    return [int(v) for v in re.findall(r"-?\d+", s)]


def p01():
    # 1-1 z=x+iy, z·conj(z)=x²+y²=2, z=k(2+2i)
    a, b, kk = sp.symbols("a b kk", real=True)
    sols = sp.solve([sp.Eq(a**2 + b**2, 2), sp.Eq(a, 2 * kk), sp.Eq(b, 2 * kk)], [a, b, kk], dict=True)
    zs = {sp.simplify(s[a] + sp.I * s[b]) for s in sols}
    exp = V.ans(1, "1").strip("{}").split(",")
    exps = {sp.simplify(sp.sympify(t.replace("i", "I"))) for t in exp}
    V.report(1, zs == exps, f"1-1 A∩B = {zs} / YAML {exps}")
    # 1-2
    A = parts(1, "2")
    h = sp.Symbol("h", real=True)
    g = lambda u: (u + 2) * (u - 3) * (u - 4)
    lm = sp.limit(sp.Abs(g(4 + h)) / h, h, 0, "-")
    lp = sp.limit(sp.Abs(g(4 + h)) / h, h, 0, "+")
    V.report(1, [lm, lp] == [int(v) for v in re.findall(r"미분계수\s*(-?\d+)", A["(1)"])], f"1-2(1) x=4 좌/우 미분계수 {lm}, {lp}")

    def f_branch(u, side):                      # side=+1: u>=0 식, -1: f(x)=f(-x) 로 u<0 식
        return sp.Abs(g(u)) if side > 0 else sp.Abs(g(-u))

    def one_sided(c, direction):
        eps = sp.Rational(1, 10**6)
        pt = c + direction * eps
        side = 1 if pt > 0 else -1
        e = f_branch(x, side)
        inner = g(x) if side > 0 else g(-x)
        return sp.sign(inner.subs(x, pt)) * sp.diff(inner, x).subs(x, c)

    cand = sorted({0} | {r for r in sp.solve(g(x), x) if r > 0} | {-r for r in sp.solve(g(x), x) if r > 0} | {sp.Integer(k_) for k_ in range(-6, 7)})
    bad = []
    for c in cand:
        if c == 0:
            l, r_ = -one_sided(sp.Integer(0), 1), one_sided(sp.Integer(0), 1)       # f 가 짝함수 ⇒ f'(0-) = -f'(0+)
        else:
            l, r_ = one_sided(c, -1), one_sided(c, 1)
        if l != r_:
            bad.append(int(c))
    V.report(1, bad == sorted(ints(A["(2)"])), f"1-2(2) 미분불가능 점 {bad} / YAML {sorted(ints(A['(2)']))}")
    # 1-3
    an = lambda i: 2 * sp.Rational(1, 3) ** (i - 1)
    bn = lambda i: 9 * sp.Rational(1, 27) ** (i - 1)
    R = {i: sp.Tuple(an(i), bn(i)) for i in (1, 2, 3)}
    O = lambda i: sp.Tuple((an(i) + an(i + 1)) / 2, (bn(i) + bn(i + 1)) / 2)
    C = parts(1, "3")
    pts = {m[0]: sp.Tuple(*[sp.Rational(v) for v in m[1].split(",")]) for m in re.findall(r"(\w+)=\(([^()]*)\)", C["(1)"])}
    ok1 = all(pts[nm] == pt for nm, pt in (("R1", R[1]), ("R2", R[2]), ("R3", R[3]), ("O1", O(1)), ("O2", O(2))))
    V.report(1, ok1, f"1-3(1) R1..R3, O1, O2 = {R[1]}, {R[2]}, {R[3]}, {O(1)}, {O(2)}")
    On = sp.Tuple(*P(C["(2)"].split("=", 1)[1]))
    gen = sp.Tuple((2 * sp.Rational(1, 3) ** (n - 1) * (1 + sp.Rational(1, 3))) / 2, (9 * sp.Rational(1, 27) ** (n - 1) * (1 + sp.Rational(1, 27))) / 2)
    V.report(1, all(sp.simplify(u - w) == 0 for u, w in zip(On, gen)), f"1-3(2) O_n = {gen}")
    A_, B_ = sp.symbols("A B")
    sol = sp.solve([A_ * O(1)[0] ** 3 + B_ * O(1)[0] ** 2 - O(1)[1], A_ * O(2)[0] ** 3 + B_ * O(2)[0] ** 2 - O(2)[1]], [A_, B_])
    ab = dict(p.split("=") for p in C["(3)"].replace(" ", "").split(","))
    V.report(1, sol[A_] == sp.Rational(ab["a"]) and sol[B_] == sp.Rational(ab["b"]), f"1-3(3) a,b = {sol}")
    f = sol[A_] * x**3 + sol[B_] * x**2
    m = (O(2)[1] - O(1)[1]) / (O(2)[0] - O(1)[0])
    line = O(1)[1] + m * (x - O(1)[0])
    dist = sp.Abs(f - line) / sp.sqrt(1 + m**2)
    c_ = [sp.Rational(1, 2), sp.Integer(1)] + [r for r in sp.solve(sp.diff(f - line, x), x) if r.is_real and sp.Rational(1, 2) <= r <= 1]
    best = max(c_, key=lambda u: float(dist.subs(x, u)))
    V.check(1, "1-3(4) α²", best**2, C["(4)"])


def p02():
    a, b = sp.symbols("a b", real=True)
    # 2-1
    d1 = (a + a**2 - 4) / sp.sqrt(2)
    d2 = (a - a**2 + 2) / sp.sqrt(2)
    roots = sp.solve(sp.expand(d1**2 - d2**2), a)                       # |d1|=|d2|
    pairs = set()
    for r in roots:
        d = sp.simplify(sp.Abs(d1.subs(a, r)))
        for s in (1, -1):
            pairs.add((sp.simplify(r), sp.simplify(s * d)))
    exp = {sp.Tuple(*P(t.strip())) for t in V.ans(2, "1").split(";")}
    num = lambda S_: sorted((round(float(u), 12), round(float(w), 12)) for u, w in S_)
    V.report(2, num(pairs) == num(exp), f"2-1 순서쌍 {len(pairs)}개 {num(pairs)}")
    # 2-2 n 자연수: n²-2na<0, 0<=a<1 의 실현 가능성
    feas = [i for i in range(1, 40) if sp.solveset(i**2 - 2 * i * a < 0, a, sp.S.Reals).intersect(sp.Interval.Ropen(0, 1)) != sp.S.EmptySet]
    nn = feas[0]
    arange = sp.solveset(nn**2 - 2 * nn * a < 0, a, sp.S.Reals).intersect(sp.Interval.Ropen(0, 1))
    Fx = x**3 + 6 * x**2 + 12 * x + 8
    lo, hi = Fx.subs(x, nn + arange.inf), Fx.subs(x, nn + arange.sup)      # f 증가 ⇒ k=f(n+a) 의 값 범위
    k_exp = [sp.Rational(v) for v in re.findall(r"\d+(?:/\d+)?", V.ans(2, "2"))]
    V.report(2, feas == [1] and sp.expand(Fx - (x + 2) ** 3) == 0 and [lo, hi] == k_exp and sp.diff(Fx, x).is_positive is not False,
             f"2-2 가능한 n={feas}, a∈{arange}, k∈({lo},{hi})")
    # 2-3 교점의 개수를 고정밀로 직접 센다
    mp.mp.dps = 60
    tol = mp.mpf(10) ** -40

    def count(nv):
        c = mp.mpf(nv - 1) / 10
        if c > 1:
            return 0
        L, U = mp.mpf((nv + 1) ** 2), mp.mpf(2 * (nv + 1) ** 2)
        base = mp.asin(c) / mp.pi
        sols = []
        for j in range(-2, int(U) * 10 + 5):
            for off in (base, 1 - base):
                xx = (off + 2 * j) / 20
                if L - tol <= xx <= U + tol:
                    sols.append(xx)
        sols.sort()
        uniq = [s for i, s in enumerate(sols) if i == 0 or abs(s - sols[i - 1]) > tol]
        return len(uniq)

    cnts = [count(i) for i in range(1, 21)]
    V.check(2, "2-3 Σ a_n (a_1=%d, a_2=%d, a_11=%d, a_12=%d)" % (cnts[0], cnts[1], cnts[10], cnts[11]), sum(cnts), V.ans(2, "3"))
    # 2-4 구간별 정적분으로 h(a), 증감 → 극값
    aa = sp.Symbol("aa", positive=True)
    fx = x**2 - 2 * x

    def seg(lo_, hi_, mid):
        s_ = sp.sign(fx.subs(x, mid))
        return sp.integrate(fx + s_ * fx - (-x + 12), (x, lo_, hi_))
    H = sp.expand(seg(-aa, 0, -1) + seg(0, 2, 1) + seg(2, aa, 3))              # a>2 이므로 세 구간
    crit = [r for r in sp.solve(sp.diff(H, aa), aa) if r > 2]
    hv = sp.simplify(H.subs(aa, crit[0]))
    sg = (sp.sign(sp.diff(H, aa).subs(aa, 2.1)), sp.sign(sp.diff(H, aa).subs(aa, 3)))
    K = parts(2, "4")
    V.report(2, sg == (-1, 1) and sp.simplify(hv - P(K["min"])) == 0 and sp.simplify(crit[0] - P(K["at"])) == 0 and sp.expand(H - sp.Rational(4, 3) * (aa**3 - 18 * aa + 2)) == 0,
             f"2-4 h(a)={H}, a={crit}, 극솟값 {hv}, 증감 {sg}")


if __name__ == "__main__":
    V.run(globals())
