#!/usr/bin/env python3
"""UOS-M17 (2017학년도 논술고사 자연계열 4문항) — 열거·sympy·수치 검산. 기대값은 YAML 정답(V.ans)에서 읽는다."""
import sys, pathlib, math, itertools
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
import mpmath as mp

V = Verifier("UOS-M17", "items/UOS/PAST")
N_ = sp.Symbol("n")


def _Sn_members(n_):
    """연립부등식을 정수점에서 직접 판정(삼각부등식은 부동소수 대신 sympy 수치 허용오차)."""
    out = []
    for xv in range(1, 10 * n_):
        if xv * xv - 14 * n_ * xv + 48 * n_ * n_ <= 0:
            continue
        ang = math.pi * xv / (4 * n_)
        if math.sin(ang) - math.cos(ang) < 1 - 1e-12:
            out.append(xv)
    return set(out)


def p01():
    f = sp.sympify(V.ans(1, "1-1"), locals={"n": N_})
    for nn in range(1, 8):
        S = _Sn_members(nn)
        cnt = 0
        for p_ in S:
            for q_ in S:
                if p_ != q_ and (p_ + q_) % 2 == 0:
                    m = (p_ + q_) // 2
                    if m in S and m != p_ and m != q_:
                        cnt += 1
        V.check(1, f"1-1 n={nn} 전수 열거", cnt, f.subs(N_, nn))
    # S_n 의 구조: (0,2n)∪(4n,6n)∪(8n,10n) 의 정수
    ok = all(_Sn_members(nn) == set(range(1, 2 * nn)) | set(range(4 * nn + 1, 6 * nn)) | set(range(8 * nn + 1, 10 * nn)) for nn in range(1, 12))
    V.report(1, ok, "S_n 정수점 = (0,2n)∪(4n,6n)∪(8n,10n) (n=1..11)")
    # 홀짝 개수: 각 구간에 홀수 n개, 짝수 n-1개
    ok2 = all(sum(1 for v in range(lo + 1, hi) if v % 2) == nn and sum(1 for v in range(lo + 1, hi) if v % 2 == 0) == nn - 1
              for nn in range(1, 10) for lo, hi in ((0, 2 * nn), (4 * nn, 6 * nn), (8 * nn, 10 * nn)))
    V.report(1, ok2, "각 구간 홀수 n개·짝수 n-1개")
    # 1-2: 합을 정확한 유리수로
    g = sp.sympify(V.ans(1, "1-2"), locals={"n": N_})
    def a(k_):
        return f.subs(N_, k_)  # YAML 일반항(위에서 n=1..7 전수 열거와 일치 확인)
    for nn in range(2, 15):
        s = sum(sp.Rational(k_**2 - k_) / (a(k_) * a(k_ + 1) - 2 * a(k_) - 2 * a(k_ + 1) + 4) for k_ in range(2, nn + 1))
        V.check(1, f"1-2 n={nn} 합", s, g.subs(N_, nn))
    kk = sp.Symbol("kk")
    V.report(1, sp.simplify((f.subs(N_, kk) - 2) - 2 * (5 * kk - 3) * (kk - 1)) == 0, "a_k-2 = 2(5k-3)(k-1)")


def p02():
    A_, B_ = sp.symbols("a b", positive=True)
    cval = sp.sympify(V.ans(2, "2-1", "c"), locals={"a": A_, "b": B_})
    # 수치: 여러 (a,b) 에서 k(φ) 를 직접 계산(A=타원 위 점, 직선 ⟂OA 와 타원의 교점 B,C)
    def k_of(a, b, phi):
        r = 1 / math.sqrt(math.cos(phi) ** 2 / a**2 + math.sin(phi) ** 2 / b**2)
        A = np.array([r * math.cos(phi), r * math.sin(phi)])
        d = np.array([-math.sin(phi), math.cos(phi)])  # OA 에 수직
        s = 1 / math.sqrt(d[0] ** 2 / a**2 + d[1] ** 2 / b**2)
        B, C = s * d, -s * d
        return float((B - A) @ (C - A))
    phis = np.linspace(0, 2 * math.pi, 100001)
    for (a, b) in ((3, 2), (5, 1), (2, 1.5), (7, 4)):
        ks = np.array([k_of(a, b, ph) for ph in phis[::10]])
        mx, mn = ks.max(), ks.min()
        ex_mx = float(sp.sympify(V.ans(2, "2-2", "max"), locals={"a": A_, "b": B_}).subs({A_: a, B_: b}))
        ex_mn = float(sp.sympify(V.ans(2, "2-2", "min"), locals={"a": A_, "b": B_}).subs({A_: a, B_: b}))
        V.report(2, abs(mx - ex_mx) < 1e-6 and abs(mn - ex_mn) < 1e-6, f"(a,b)=({a},{b}) k 범위 [{mn:.6f},{mx:.6f}] / YAML [{ex_mn}, {ex_mx}]")
        # k=0 인 φ 의 위치: 부호변화 점 → 직선 y=±x 방향(φ=π/4+mπ/2)과 일치, 점 좌표 c
        pg = phis[::10] + 0.000123  # 정확히 0 이 되는 각을 격자가 비껴가도록
        kk = np.array([k_of(a, b, ph) for ph in pg])
        zs = [pg[i] for i in range(len(kk) - 1) if kk[i] * kk[i + 1] < 0]
        want = [math.pi / 4 + m * math.pi / 2 for m in range(4)]
        okz = len(zs) == 4 and all(abs(z - w) < 1e-3 for z, w in zip(sorted(zs), want))
        V.report(2, okz, f"(a,b)=({a},{b}) k=0 인 방향 {len(zs)}개, φ=π/4+mπ/2")
        cv = float(cval.subs({A_: a, B_: b}))
        r2 = (a * b) ** 2 / (a**2 + b**2) * 2
        V.report(2, abs(2 * cv**2 - r2) < 1e-9 and abs(cv**2 / a**2 + cv**2 / b**2 - 1) < 1e-9, f"(a,b)=({a},{b}) A=(c,c) 가 타원 위, c={cv:.6f}")
    # 기호 항등식 k = |OA|^2-|OB|^2
    ph = sp.Symbol("ph", real=True)
    r2 = lambda t_: 1 / (sp.cos(t_) ** 2 / A_**2 + sp.sin(t_) ** 2 / B_**2)
    Av = sp.sqrt(r2(ph)) * sp.Matrix([sp.cos(ph), sp.sin(ph)])
    Bv = sp.sqrt(r2(ph + sp.pi / 2)) * sp.Matrix([-sp.sin(ph), sp.cos(ph)])
    kexp = (Bv - Av).dot(-Bv - Av)
    V.report(2, abs(float((kexp - (r2(ph) - r2(ph + sp.pi / 2))).subs({ph: 0.7, A_: 3, B_: 2}))) < 1e-9, "k = |OA|²-|OB|² (기호 식 수치 대입)")


def p03():
    x_, a_, b_, c_ = sp.symbols("x a b c", real=True)
    # (a) 접점 x=c>0: f(c)=g(c), f'(c)=g'(c) 로 a,b 를 c 로 표현 → |b/(a+c)| 최소
    sol = sp.solve([(x_**3 + a_ * x_**2 + b_ * x_ - 9 * x_).subs(x_, c_), sp.diff(x_**3 + a_ * x_**2 + b_ * x_ - 9 * x_, x_).subs(x_, c_)], [a_, b_], dict=True)
    sol = [s for s in sol]
    # c≠0 일 때의 해
    s0 = sol[0]
    ratio = sp.simplify(sp.Abs(s0[b_] / (s0[a_] + c_)))
    f = sp.lambdify(c_, ratio, "mpmath")
    mp.mp.dps = 30
    best = min(f(mp.mpf(i) / 1000) for i in range(1, 100000))
    V.report(3, abs(best - mp.mpf(V.ans(3, "3-1"))) < mp.mpf("1e-5"), f"3-1 최솟값(수치) {mp.nstr(best, 10)} / YAML {V.ans(3,'3-1')}; a,b = {s0}")
    V.check(3, "3-1 c=3 에서 값", ratio.subs(c_, 3), V.ans(3, "3-1"))
    # (b) 넓이 조건: h(x)=x(x-α)(x-β) 에서 두 영역 넓이가 같으면 β/α
    al, be = sp.symbols("alpha beta", positive=True)
    h = x_ * (x_ - al) * (x_ - be)
    A1 = sp.integrate(h, (x_, 0, al))
    A2 = -sp.integrate(h, (x_, al, be))
    r = sp.solve(sp.Eq(A1, A2).subs(be, sp.Symbol("rr") * al), sp.Symbol("rr"))
    r = [v for v in r if v.is_real and v > 1]
    V.check(3, "3-2 β/α (넓이 같음)", r[0] if r else None, V.ans(3, "3-2"))
    # 수치로 두 영역 넓이 직접 비교 (α=1, β=2)
    a1 = float(sp.integrate((x_ * (x_ - 1) * (x_ - 2)), (x_, 0, 1)))
    a2 = float(-sp.integrate((x_ * (x_ - 1) * (x_ - 2)), (x_, 1, 2)))
    V.report(3, abs(a1 - a2) < 1e-12, f"α=1,β=2: 두 넓이 {a1} = {a2}")


def p04():
    lo = sp.sympify(V.ans(4, "4-1", "lo"))
    hi = sp.sympify(V.ans(4, "4-1", "hi"))
    xs = np.linspace(1e-9, math.pi / 2 - 1e-9, 2_000_001)
    lhs = (1 + np.sin(xs)) ** (1 + np.cos(xs))
    rhs = (1 + np.cos(xs)) ** (1 + np.sin(xs))
    mask = lhs > rhs
    # 해집합 = 연속 구간인지, 경계
    idx = np.flatnonzero(mask)
    V.report(4, len(idx) > 0 and np.all(np.diff(idx) == 1), "해집합이 하나의 구간")
    V.report(4, abs(xs[idx[0]] - float(lo)) < 1e-5 and abs(xs[idx[-1]] - float(hi)) < 1e-5,
             f"해집합 [{xs[idx[0]]:.6f},{xs[idx[-1]]:.6f}] / YAML ({float(lo):.6f},{float(hi):.6f})")
    # 해설 논리: X,Y∈(1,2) ⊂ (1,e) 에서 f=ln x/x 증가, 동치성 점검
    bad = []
    for xv in xs[::997]:
        X, Y = 1 + math.sin(xv), 1 + math.cos(xv)
        if (X**Y > Y**X) != (math.log(X) / X > math.log(Y) / Y):
            bad.append(xv)
    V.proof(4, "X^Y>Y^X ⇔ lnX/X>lnY/Y (표본)", bad)
    t = sp.Symbol("t", positive=True)
    V.report(4, sp.simplify(sp.diff(sp.log(t) / t, t) - (1 - sp.log(t)) / t**2) == 0 and (sp.N(1 - sp.log(2)) > 0), "f'(t)=(1-ln t)/t², 1<t<2<e 에서 f'>0")


if __name__ == "__main__":
    V.run(globals())
