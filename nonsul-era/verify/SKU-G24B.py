#!/usr/bin/env python3
"""서경대 2024학년도 B형 수학 10~15번 검산. 기대값은 V.ans() 로 YAML 에서 읽는다."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F401

V = Verifier("SKU-G24B", "items/SKU/PAST")
X = sp.Symbol("x")


def p10():
    cand = [sp.pi / 3, 2 * sp.pi / 3, 4 * sp.pi / 3, 5 * sp.pi / 3]
    sol = [c for c in cand if sp.simplify(4 * sp.sin(c) ** 2 - 3) == 0 and sp.sin(c) * sp.cos(c) > 0]
    # 구간 전체에서 sin^2 = 3/4 인 해가 위 4개뿐인지 확인
    allr = sp.solveset(4 * sp.sin(X) ** 2 - 3, X, sp.Interval(0, 2 * sp.pi))
    V.report(10, set(allr) == set(cand), f"4sin^2x-3=0 의 해 {sorted(allr, key=float)}")
    quad = sp.expand((x - sol[0]) * (x - sol[1]))
    V.check(10, f"근 {sol}", quad, S(V.ans(10, "1")).subs(X, x))


def p11():
    f = lambda u: sp.Rational(3, 2) ** (u + 1) - 1
    g = lambda u: sp.Rational(3, 2) ** (u + 3) + sp.Rational(5, 2)
    # 일반 해석: 직선 y=-4x+k 와 각 곡선의 교점 P(p), Q(q) 로 두고 PQ^2=17 을 푼다(P·Q 위치 두 경우 모두).
    ks = set()
    for d in (1, -1):                      # p - q = d  (PQ^2=(1+16)(p-q)^2=17)
        q = sp.symbols("q", real=True)
        # f(p)=-4p+k, g(q)=-4q+k, p=q+d  → k 소거
        eq = sp.Eq(f(q + d) + 4 * (q + d), g(q) + 4 * q)
        for r in sp.solve(eq, q):
            ks.add(sp.nsimplify(sp.simplify(g(r) + 4 * r)))
    V.report(11, ks == {-4} or -4 in ks, f"두 경우의 k 집합 {ks}")
    V.check(11, "k", -4 if -4 in ks else None, V.ans(11, "1"))
    if len(ks) > 1:
        V.report(11, True, f"[참고] P,Q 위치 반대 경우에 다른 k 존재: {ks}")


def p12():
    a = lambda m: 1 + (m - 1) * 4
    tot = sum(1 / (sp.sqrt(a(k)) + sp.sqrt(a(k + 1))) for k in range(1, 21))
    V.check(12, "합(50자리 수치)", sp.N(tot, 50), V.ans(12, "1"), tol=1e-30)
    tele = sum((sp.sqrt(a(k + 1)) - sp.sqrt(a(k))) / (a(k + 1) - a(k)) for k in range(1, 21))
    V.check(12, "유리화 후 망원합", sp.N(tele, 50), (sp.sqrt(a(21)) - sp.sqrt(a(1))) / 4, tol=1e-30)


def p13():
    a, b = sp.symbols("a b")
    # f(1)=2: (x^2+ax+b)/(x-3) at x=1 → (1+a+b)/(-2)=2 ; x=3 연속: 3a+b=-9
    s = sp.solve([sp.Eq((1 + a + b) / (1 - 3), 2), sp.Eq(9 + 3 * a + b, 0)], [a, b])
    f = sp.cancel((X**2 + s[a] * X + s[b]) / (X - 3))
    ok = sp.simplify((X - 1) * (X - 3) * f - (X - 1) * (X**2 + s[a] * X + s[b])) == 0
    V.report(13, ok and s == {a: -2, b: -3}, f"a,b={s} 항등식 확인")
    V.check(13, "f(3)", f.subs(X, 3), V.ans(13, "1"))


def p14():
    a, b = sp.symbols("a b")
    f = X**2 * (a * X + b)
    s = sp.solve([sp.Eq(sp.diff(f, X).subs(X, -1), 3), sp.Eq(sp.diff(f, X).subs(X, 1), 9)], [a, b])
    F = f.subs(s)
    V.report(14, s == {a: 2, b: sp.Rational(3, 2)}, f"a,b={s}")
    V.check(14, "∫0^2 f(x+2)dx", sp.integrate(F.subs(X, X + 2), (X, 0, 2)), V.ans(14, "1"))
    V.check(14, "∫2^4 f dx", sp.integrate(F, (X, 2, 4)), V.ans(14, "1"))


def p15():
    N = 2024
    # 전수 확인: 넓이 조건 ∫0^b f = 0 (정수 산술, 5·... 로 통분)
    cnt = 0
    pairs = []
    for b in range(2, N + 1):
        # ∫0^b = b^3(12b^2 - 15(a+b)b + 20ab)/60 = 0 → 12b - 15(a+b) + 20a = 0 → 5a = 3b
        if (3 * b) % 5 == 0:
            a = 3 * b // 5
            if 1 <= a < b:
                pairs.append((a, b))
    # 정적분 직접 확인(표본) 및 위·아래 넓이 비교
    for a, b in pairs[:5] + pairs[-5:]:
        f = X**2 * (X - a) * (X - b)
        A = sp.integrate(f, (X, 0, a))
        B = -sp.integrate(f, (X, a, b))
        assert A == B and A > 0
    # 역방향: 5a≠3b 인 몇 쌍은 넓이가 다름
    for a, b in [(1, 2), (2, 3), (3, 4), (3, 6), (6, 9)]:
        f = X**2 * (X - a) * (X - b)
        assert sp.integrate(f, (X, 0, a)) != -sp.integrate(f, (X, a, b))
    V.check(15, "순서쌍 개수", len(pairs), V.ans(15, "1"))


if __name__ == "__main__":
    V.run(globals())
