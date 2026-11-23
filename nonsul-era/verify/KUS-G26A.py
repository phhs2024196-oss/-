#!/usr/bin/env python3
"""KUS-G26A (고려대 세종 2026학년도 자연계열Ⅰ) 검산 — 설정에서 독립 재계산, 기대값은 YAML 에서 읽는다."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F401,F403

V = Verifier("KUS-G26A", "items/KUS/PAST")


def p01():
    cnt = 0
    for perm in itertools.permutations(range(8), 5):
        val = int("".join(map(str, perm)))      # 맨 앞 0 이면 네 자리 수(예: 0,3,1,5,7 -> 3157)
        if val % 2 == 1 and val < 25000:
            cnt += 1
    V.check(1, "경우의 수(전수 열거)", cnt, V.ans(1, "1"))


def p02():
    a, k, c = sp.symbols("a k c", real=True)
    t = sp.symbols("t", real=True)
    # 일반 이차함수 f=c0+c1 x+c2 x^2 로 두고 조건에서 독립적으로 풀기
    c0, c1, c2 = sp.symbols("c0 c1 c2", real=True)
    f = c0 + c1 * x + c2 * x**2
    I = sp.integrate(f**2 * sp.diff(f, x), (x, 0, 4))
    # I = (f(4)^3 - f(0)^3)/3 확인
    ok_sub = sp.simplify(I - (f.subs(x, 4)**3 - f.subs(x, 0)**3) / 3) == 0
    V.report(2, ok_sub, "치환적분 항등식 I=(f(4)^3-f(0)^3)/3")
    # f(4)=f(0) (실수) -> c1 = -4 c2 (c2!=0). f(1)=0 -> c0 = -c1 - c2 = 3 c2
    sol = sp.solve([f.subs(x, 4) - f.subs(x, 0), f.subs(x, 1)], [c0, c1], dict=True)[0]
    f2 = f.subs(sol)
    V.report(2, sp.simplify(f2 - c2 * (x - 1) * (x - 3)) == 0, f"f=c2(x-1)(x-3): {sp.factor(f2)}")
    # 미분가능: f(3)=e^3+a, f'(3)=e^3
    r = sp.solve([f2.subs(x, 3) - (sp.exp(3) + a), sp.diff(f2, x).subs(x, 3) - sp.exp(3)], [a, c2], dict=True)[0]
    V.check(2, "a=-e^3", r[a], -sp.exp(3))
    fx = f2.subs(c2, r[c2])
    ans = sp.simplify(fx.subs(x, sp.log(r[a]**2)))
    V.check(2, "f(ln a^2)", sp.simplify(sp.expand_log(ans, force=True)), V.ans(2, "1"))


def p03():
    a, b = sp.symbols("a b", positive=True)
    xA = sp.solve(sp.Eq(sp.log(a * x + b), 0), x)[0]      # ax+b=1
    OA = sp.simplify((b - 1) / a)                           # b>1, a>0 -> x_A=(1-b)/a<0
    assert sp.simplify(xA - (1 - b) / a) == 0
    OH = b / a                                              # 점근선 x=-b/a
    OB = sp.log(b)
    bs = sp.solve(sp.Eq(3 * OA, OH), b)[0]
    ia = sp.solve(sp.Eq(OA.subs(b, bs), 2 * OB.subs(b, bs)), a)[0]   # a
    V.check(3, "b", bs, sp.Rational(3, 2))
    V.check(3, "e^(1/a)", sp.simplify(sp.exp(1 / ia)), V.ans(3, "1"))


def p04():
    ks = []
    for k in range(1, 40):
        fk = lambda t: t**3 - sp.Rational(k, 2) * t**2 + 6
        gk = lambda t: sp.Rational(k, 2) * t**2 + sp.Rational(k, 2)
        t = sp.symbols("t", positive=True)
        # <다> : 속도가 같은 t>0 에서 속도 크기가 양의 정수
        sols = sp.solve(sp.Eq(sp.diff(fk(t), t), sp.diff(gk(t), t)), t)
        speeds = [sp.diff(gk(t), t).subs(t, s) for s in sols]
        c_da = all(v.is_integer and v > 0 for v in speeds)
        c_na1 = fk(0) - gk(0) > 0
        # 만나지 않음: h(t)=f-g 가 t>=0 에서 0 이 되지 않는다(h(0)>0 이므로 h>0)
        h = sp.expand(fk(t) - gk(t))
        crit = [s for s in sp.solve(sp.diff(h, t), t) if s >= 0]
        mn = min([h.subs(t, s) for s in crit] + [h.subs(t, 0)])
        if c_da and c_na1 and mn > 0:
            ks.append((k, mn))
    V.report(4, len(ks) == 1, f"조건을 만족하는 k = {ks}")
    kk, mn = ks[0]
    V.check(4, "k", kk, V.ans(4, "1", "k"))
    V.check(4, "최소거리", mn, V.ans(4, "1", "d"))


def p05():
    a, t = sp.symbols("a t", positive=True)
    f = x**2 - 2 * x + a
    t0 = sp.solve(sp.Eq(sp.diff(f, x).subs(x, t), 2), t)[0]
    P = sp.Matrix([t0, f.subs(x, t0)])
    L1 = lambda X: 2 * (X - t0) + f.subs(x, t0)
    L2 = lambda X: -sp.Rational(1, 2) * (X - t0) + f.subs(x, t0)
    A_ = sp.Matrix([0, L1(0)])
    B_ = sp.Matrix([sp.solve(sp.Eq(L2(x), 0), x)[0], 0])
    C_ = sp.Matrix([0, L2(0)])
    # Q on L1 with CQ || AB
    q = sp.symbols("q", real=True)
    Q_ = sp.Matrix([q, L1(q)])
    AB, CQ = B_ - A_, Q_ - C_
    qs = sp.solve(sp.Eq(AB[0] * CQ[1] - AB[1] * CQ[0], 0), q)[0]
    Q_ = Q_.subs(q, qs)
    area = lambda U, W, Z: sp.Abs((W - U)[0] * (Z - U)[1] - (W - U)[1] * (Z - U)[0]) / 2
    S1, S2 = area(P, A_, B_), area(P, C_, Q_)
    asol = [s for s in sp.solve(sp.Eq(sp.simplify(S1 / S2), 4), a) if s.is_positive]
    V.report(5, asol == [2], f"a={asol}")
    a0 = asol[0]
    # 영역 경계: A->B->P (선분) -> 곡선 거꾸로 P->(0,f(0)) -> y축 따라 A (그린 정리)
    fx = f.subs(a, a0)
    pts = {k: m.subs(a, a0) for k, m in dict(A=A_, B=B_, P=P, C=C_).items()}
    def seg(U, W):    # ∮ (x dy - y dx)/2 over segment
        return (U[0] * W[1] - W[0] * U[1]) / 2
    Ap, Bp, Pp = pts["A"], pts["B"], pts["P"]
    Z = sp.Matrix([0, fx.subs(x, 0)])
    curve = sp.integrate(sp.Rational(1, 2) * (x * sp.diff(fx, x) - fx), (x, 2, 0))   # P -> Z (x 감소)
    area_region = sp.Abs(seg(Ap, Bp) + seg(Bp, Pp) + curve + seg(Z, Ap))
    V.check(5, "넓이", sp.simplify(area_region), V.ans(5, "1"))


def p06():
    t = sp.symbols("t", positive=True)
    d = sp.symbols("d", real=True)
    Ax = sp.log(t); Bx = 2 + sp.log(t)
    g = lambda s: sp.exp(s - 2)
    Cy = g(Ax)
    # ∠BDC = 90° : (C-D)·(B-D) = 0, D=(d,0)
    eq = sp.Eq((Ax - d) * (Bx - d) + Cy * t, 0)
    # 방정식을 d 에 대해 풀어 PD*QD=(d-Ax)(Bx-d) 를 직접 계산 → t^2/e^2 와 비교(t 표본)
    prod_t = Cy * t
    for tv in (1, sp.Rational(3, 2), 2):
        roots = sp.solve(eq.subs(t, tv), d)
        ok = all(sp.simplify(((r - Ax) * (Bx - r)).subs(t, tv) - (tv**2 / sp.exp(2))) == 0 for r in roots)
        V.report(6, ok and len(roots) == 2, f"t={tv}: D 의 해 {len(roots)}개, PD*QD=t^2/e^2")
    f_t = sp.simplify(sp.expand_log(prod_t, force=True))
    mn = min(f_t.subs(t, 1), f_t.subs(t, 2))
    V.check(6, "min", sp.simplify(mn), V.ans(6, "1"))
    V.report(6, sp.diff(f_t, t).subs(t, sp.Rational(3, 2)) > 0, "t^2/e^2 증가(구간 내부 샘플)")
    # D 가 실제로 존재(0<PD*QD<=1)
    V.report(6, all(0 < f_t.subs(t, v) <= 1 for v in (1, sp.Rational(3, 2), 2)), "x(2-x)=t^2/e^2 가 해를 가짐")


if __name__ == "__main__":
    V.run(globals())
