#!/usr/bin/env python3
"""TUK-G23B-01~03 (한국공학대 2023 오후) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G23B", "items/TUK/PAST")


def kv(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def p01():
    a, b = sp.log(3, 2), sp.log(5, 2)
    g = lambda u: 2 ** u   # log2 의 역함수
    V.check(1, "1-1 g(3a-2b)", sp.simplify(g(3 * a - 2 * b)), V.ans(1, "1"))
    e = sp.Rational(1, 2) ** (x**2 - 4 * x + 6)
    vals = [e.subs(x, c) for c in [1, 2, 3, 4]] + [e.subs(x, sp.Rational(k_, 4)) for k_ in range(4, 17)]
    V.check(1, "1-2 최댓값", max(vals), kv(1, "2")["max"])
    V.check(1, "1-2 최솟값", min(vals), kv(1, "2")["min"])
    A, M = sp.symbols("A M", positive=True)
    sol = sp.solve([A ** (1 - M) - 1, A ** (3 - M) - 3], [A, M], dict=True)
    ok = [s_ for s_ in sol if s_[A] > 1]
    V.report(1, len(ok) == 1 and same(ok[0][A], kv(1, "3")["a"]) and same(ok[0][M], kv(1, "3")["m"]), f"1-3 a,m = {ok}")
    # 역함수와 교점이 y=x 위: a^(x-m)=x 가 x=1,3 에서 성립
    aa, mm = S(kv(1, "3")["a"]), S(kv(1, "3")["m"])
    V.report(1, same(aa ** (1 - mm), 1) and same(aa ** (3 - mm), 3), "1-3 교점 x=1,3 대입 확인")
    L = sp.Symbol("L", real=True)  # L = log2 a
    f = x**3 - 3 * x**2 + 1 + L
    # 서로 다른 세 실근 ⇔ f(0)f(2)<0
    rng = sp.solveset(f.subs(x, 0) * f.subs(x, 2) < 0, L, sp.S.Reals)
    lo, hi = 2 ** rng.start, 2 ** rng.end
    lo_y, hi_y = V.ans(1, "4").replace(" ", "").split("<a<")
    V.report(1, same(lo, lo_y) and same(hi, hi_y), f"1-4 a 범위 ({lo},{hi})")
    cnt = lambda aval: len(set(sp.Poly(f.subs(L, sp.log(aval, 2)), x).nroots(n=30, maxsteps=200) and [round(float(sp.re(r)), 6) for r in sp.Poly(f.subs(L, sp.log(aval, 2)), x).nroots() if abs(sp.im(r)) < 1e-9]))
    V.report(1, cnt(sp.Rational(3, 4)) == 3 and cnt(1) == 3 and cnt(4) == 3 and cnt(sp.Rational(1, 4)) < 3 and cnt(16) < 3, "1-4 표본 a 에서 서로 다른 실근 개수")


def p02():
    f = x**3 / 3 + x**2 - 3 * x + sp.Rational(2, 3)
    cr = sp.solve(sp.diff(f, x), x)
    V.check(2, "2-1 극솟값", [f.subs(x, r) for r in cr if sp.diff(f, x, 2).subs(x, r) > 0][0], V.ans(2, "1"))
    gp1 = 3
    g1 = sp.Symbol("g1")
    V.check(2, "2-2 h'(1) (g(1) 와 무관)", sp.simplify(sp.diff(f, x).subs(x, 1) * g1 + f.subs(x, 1) * gp1), V.ans(2, "2"))
    h = f - (-3 * x**2 + 17 * x + sp.Rational(1, 3))
    cand = [r for r in sp.solve(sp.diff(h, x), x) if r > 0]
    V.check(2, "2-3 a 최댓값 = min_{x>0} h", min(h.subs(x, r) for r in cand), V.ans(2, "3"))
    ks = []
    for k_ in range(-30, 30):
        pts = [sp.Integer(k_), sp.Integer(k_ + 2)] + [r for r in sp.solve(sp.diff(f, x), x) if k_ <= r <= k_ + 2]
        if min(f.subs(x, q) for q in pts) == -1: ks.append(k_)
    V.check(2, "2-4 k 집합", sp.FiniteSet(*ks), sp.FiniteSet(*[S(v) for v in V.ans(2, "4").replace(" ", "").split(",")]))


def p03():
    s3 = sp.sqrt(3)
    def F(xv):  # 1<=x<3 기본 구간 + (나) 반복
        xv = sp.Rational(xv) if not isinstance(xv, sp.Basic) else xv
        k_ = 0
        while xv < 1: xv += 2; k_ -= 1
        while xv > 3: xv -= 2; k_ += 1
        return xv, k_
    def f(xv):
        u, k_ = F(xv); return (1 - (u - 2) ** 2) * s3 ** k_
    def g(xv):
        u, k_ = F(xv); return (((u - 1) ** 2) if u < 2 else (u - 3) ** 2) * s3 ** k_
    kv3 = kv(3, "1")
    V.report(3, same(f(0), kv3["f(0)"]) and same(g(0), kv3["g(0)"]), f"3-1 f(0)={sp.simplify(f(0))}, g(0)={sp.simplify(g(0))}")
    A, B, C = (5, f(5)), (6, f(6)), (7, f(7))
    ab = sp.Matrix([B[0] - A[0], B[1] - A[1]]); ac = sp.Matrix([C[0] - A[0], C[1] - A[1]])
    cosA = (ab.dot(ac)) / (ab.norm() * ac.norm())
    V.check(3, "3-2 sinA", sp.sqrt(1 - cosA**2), V.ans(3, "2"))
    V.check(3, "3-3 Σf(n)", sp.simplify(sum(f(k_) for k_ in range(1, 14))), V.ans(3, "3"))
    import mpmath as mp
    mp.mp.dps = 25
    s3n = mp.sqrt(3)
    def fn(u):
        k_ = 0
        while u < 1: u += 2; k_ -= 1
        while u > 3: u -= 2; k_ += 1
        return (1 - (u - 2) ** 2) * s3n ** k_
    def gn(u):
        k_ = 0
        while u < 1: u += 2; k_ -= 1
        while u > 3: u -= 2; k_ += 1
        return (((u - 1) ** 2) if u < 2 else (u - 3) ** 2) * s3n ** k_
    num = mp.quad(lambda u: abs(fn(u) - gn(u)), [1, 2, 3, 4, 5])
    V.report(3, abs(num - mp.mpf(str(sp.N(S(V.ans(3, "4")), 25)))) < 1e-9, f"3-4 넓이 수치 {num}")


if __name__ == "__main__": V.run(globals())
