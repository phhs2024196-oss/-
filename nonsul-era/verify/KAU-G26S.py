#!/usr/bin/env python3
"""KAU-G26S-01~02 (한국항공대 2026 이학계열) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from sympy.parsing.sympy_parser import parse_expr  # noqa: E402
from vlib import *  # noqa: E402,F403

V = Verifier("KAU-G26S", "items/KAU/PAST")
mp.mp.dps = 40


def P(s, **loc):
    return parse_expr(s.replace("oo", "oo"), local_dict={"x": x, "t": t, "k": k, "oo": sp.oo, **loc})


def parts(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def close(a, b, tol=1e-12):
    return abs(complex(sp.N(a - b, 30))) < tol


def p01():
    # 1-1(1) [증명-수치확인]: 정의에 중간항을 더하고 뺀 식이 곱의 미분과 같은지 여러 함수쌍으로 극한 계산
    h = sp.Symbol("h", real=True)
    bad = []
    for f, g in [(sp.exp(x), x**3 + 1), (sp.sin(x), sp.log(x**2 + 1)), (x**2 + 3 * x, sp.cos(x)), (sp.atan(x), sp.exp(2 * x))]:
        orig = (f.subs(x, x + h) * g.subs(x, x + h) - f * g) / h
        split = ((f.subs(x, x + h) - f) * g.subs(x, x + h) + f * (g.subs(x, x + h) - g)) / h
        want = sp.diff(f, x) * g + f * sp.diff(g, x)
        if not (sp.simplify(sp.limit(orig, h, 0) - want) == 0 and sp.simplify(sp.limit(split, h, 0) - want) == 0
                and sp.simplify(orig - split) == 0):
            bad.append((f, g))
    V.proof(1, "1-1(1) 곱의 미분법(정의→중간항→분리)", bad)
    # 1-1(2): f(1)=2, f'(1)=1 인 서로 다른 f 로 g'(1)
    for F in (2 + (x - 1) + (x - 1) ** 2, 2 * sp.exp((x - 1) / 2)):
        assert F.subs(x, 1) == 2 and sp.diff(F, x).subs(x, 1) == 1, F
        V.check(1, "1-1(2) g'(1)", sp.diff((x**2 + 1) * F, x).subs(x, 1), V.ans(1, "2"))
    # 1-2: 점화식 a_n = (α+β)a_{n-1} - αβ a_{n-2} (정확한 유리수) 로 100항 합, 복소수 근 직접 계산으로 교차 확인
    z = sp.Symbol("z")
    al, be = sp.solve(4 * z**2 - 2 * z + 1, z)
    an = [sp.Integer(2), sp.Rational(1, 2)]
    for _ in range(2, 101):
        an.append(sp.Rational(1, 2) * an[-1] - sp.Rational(1, 4) * an[-2])
    tot = sum(an[1:101])
    V.check(1, "1-2 Σa_n (점화식 정확값)", tot, P(V.ans(1, "3")))
    mp.mp.dps = 60
    a_, b_ = (mp.mpc(str(sp.re(sp.N(v_, 60))), str(sp.im(sp.N(v_, 60)))) for v_ in (al, be))
    V.report(1, abs(sum(a_**n + b_**n for n in range(1, 101)) - mp.mpf(sp.N(P(V.ans(1, "3")), 60))) < mp.mpf(10) ** -50,
             "1-2 복소수 근으로 직접 합산")
    # 1-3: [1/32, 1] 격자+이분법으로 f(x)=0 의 모든 근
    f3 = lambda xv: mp.sin(mp.pi * mp.log(xv, 2) + mp.pi / 4) + mp.sqrt(2) / 2
    grid = [mp.mpf(1) / 32 + (mp.mpf(31) / 32) * i / 20000 for i in range(20001)]
    roots = []
    if abs(f3(grid[0])) < mp.mpf(10) ** -30:
        roots.append(grid[0])
    for u, v in zip(grid, grid[1:]):
        if f3(u) * f3(v) < 0:
            roots.append(mp.findroot(f3, (u, v), solver="anderson", tol=1e-35, maxsteps=200))
    if abs(f3(grid[-1])) < mp.mpf(10) ** -30:
        roots.append(grid[-1])
    exp4 = P(V.ans(1, "4"))
    V.report(1, len(roots) == 6 and abs(sum(roots) - mp.mpf(sp.N(exp4, 40))) < mp.mpf(10) ** -25,
             f"1-3 근 {len(roots)}개, 합 {mp.nstr(sum(roots), 15)} vs {sp.N(exp4, 15)}")
    # 1-4: a_3=ar²=100 → r=10/√a. (1) 식, (2) 최댓값
    L = sp.Symbol("L", real=True)
    fexp = P(parts(1, "5")["f"], L=L)
    ok = True
    for av in (1, 2, 5, 10, 33, 100):
        r_ = 10 / math.sqrt(av)
        direct = -math.log10(av) ** 3 + math.log10(av * r_) ** 2
        ok &= abs(direct - float(fexp.subs(L, math.log10(av)))) < 1e-9
    V.report(1, ok and parts(1, "5")["L"] == "loga", "1-4(1) a_3=100 조건에서 직접 계산한 값과 대조(a=1~100)")
    g = fexp.subs(L, t)
    mx = extrema(g, t, 0, 2)[1]
    V.check(1, "1-4(2) 최댓값 (t=log a ∈[0,2])", mx, V.ans(1, "6"))
    grid_a = [1 + 99 * i / 100000 for i in range(100001)]
    num_max = max(-math.log10(av) ** 3 + (1 + math.log10(av) / 2) ** 2 for av in grid_a)
    V.report(1, abs(num_max - float(sp.N(P(V.ans(1, "6"))))) < 1e-6, f"1-4(2) a∈[1,100] 격자 최댓값 {num_max:.9f}")


def count_roots(kv):
    """f(x)=k 의 서로 다른 실근 수(구간별 삼차식의 근을 40자리로 구해 정의역 검사, 중근은 반올림으로 병합)."""
    cnt = set()
    for poly, lo, hi in [(x**3 - x - kv, -sp.oo, -1), (-x**3 + x - kv, -1, 1), (x**3 - x - kv, 1, sp.oo)]:
        for r_ in sp.Poly(sp.expand(poly), x).nroots(n=40, maxsteps=500):
            if abs(sp.im(r_)) > 1e-12:
                continue
            rv = float(sp.re(r_))
            if (lo == -sp.oo or rv >= lo - 1e-12) and (hi == sp.oo or rv <= hi + 1e-12):
                # 정의역 경계(-1, 1)에서 이웃 구간과 같은 점이면 한 번만 센다
                cnt.add(round(rv, 9))
    return len(cnt)


def p02():
    fabs = x * sp.Abs((x - 1) * (x + 1))
    # 2-1(1): 구간별 도함수 부호로 증감 구간 재구성
    pieces = [(x**3 - x, -sp.oo, -1), (-x**3 + x, -1, 1), (x**3 - x, 1, sp.oo)]
    crit = {sp.Integer(-1), sp.Integer(1)}
    for e, lo, hi in pieces:
        for c in sp.solve(sp.diff(e, x), x):
            if (lo == -sp.oo or c > lo) and (hi == sp.oo or c < hi):
                crit.add(sp.nsimplify(c))
    crit = sorted(crit, key=lambda c: float(c))
    A = parts(2, "1")

    def parse_ints(sv):
        out = []
        for seg in sv.split("|"):
            lo_s, hi_s = seg[1:-1].split(",")
            out.append((P(lo_s), P(hi_s)))
        return out
    inc, dec = parse_ints(A["inc"]), parse_ints(A["dec"])

    def deriv_at(xv):
        for e, lo, hi in pieces:
            if (lo == -sp.oo or xv >= lo) and (hi == sp.oo or xv < hi):
                return sp.diff(e, x).subs(x, xv)
    ends = sorted({sp.nsimplify(e) for iv in inc + dec for e in iv if e not in (sp.oo, -sp.oo)}, key=lambda c: float(c))
    ok_pts = [close(a_, b_) for a_, b_ in zip(ends, crit)] and len(ends) == len(crit)
    mids = []
    for lo, hi in inc + dec:
        lo_f = float(lo) if lo != -sp.oo else float(crit[0]) - 3
        hi_f = float(hi) if hi != sp.oo else float(crit[-1]) + 3
        mids.append((lo_f + hi_f) / 2)
    ok_sign = all(deriv_at(sp.Float(m_)) > 0 for m_ in mids[:len(inc)]) and all(deriv_at(sp.Float(m_)) < 0 for m_ in mids[len(inc):])
    V.report(2, ok_pts and ok_sign, f"2-1(1) 증감 구간 끝점 {ends} = 임계점·꺾임점 {crit}, 구간별 f' 부호 일치")
    # 2-1(2): 단조 구간의 값 범위로 f(x)=k 의 개수 분류
    cv = sorted({sp.simplify(fabs.subs(x, c)) for c in crit}, key=lambda c: float(c))   # 0, ±2√3/9 (극값·꺾임값)
    cand = []
    for i, c in enumerate(cv):
        cand.append(c)
        if i + 1 < len(cv):
            cand.append((c + cv[i + 1]) / 2)
    cand = [cv[0] - 1] + cand + [cv[-1] + 1]
    K2 = [c for c in cand if count_roots(c) == 2]
    exp = [P(s_) for s_ in parts(2, "2")["k"].split("|")]
    V.report(2, len(K2) == len(exp) and all(close(a_, b_) for a_, b_ in zip(sorted(K2, key=float), sorted(exp, key=float))),
             f"2-1(2) 교점 2개인 k {[sp.nsimplify(c) for c in K2]} (대표값별 근의 개수 {[count_roots(c) for c in cand]})")
    # 2-2(1) [증명-수치확인]
    V.proof(2, "2-2(1) (s+t)/2 ≥ √(st)", [(s_, t_) for s_ in range(1, 60) for t_ in range(1, 60) if (s_ + t_) / 2 < math.sqrt(s_ * t_) - 1e-12])
    # 2-2(2) [증명-수치확인]: a 를 바꿔도 [-3,1] 의 최솟값이 항상 1
    a = sp.Symbol("a", real=True)
    fq = a * x**2 - 2 * (a**2 + 1) * x + 2 * a**2 - a + 3
    bad = []
    for av in [sp.Rational(n_, 7) for n_ in range(-70, 71)] + [100, -100, 1000, -1000]:
        e = fq.subs(a, av)
        cands = [-3, 1] + [c for c in sp.solve(sp.diff(e, x), x) if -3 <= c <= 1]
        if min(e.subs(x, c) for c in cands) != 1:
            bad.append(av)
    V.proof(2, "2-2(2) 최솟값=1 (a=-10..10 격자 143점 외 4점)", bad)
    # 2-3: f 를 삼차 이하 일반 다항식으로 놓고 항등식에서 계수를 모두 푼다
    c3, c2, c1, c0 = sp.symbols("c3 c2 c1 c0")
    f = c3 * x**3 + c2 * x**2 + c1 * x + c0
    u = sp.Symbol("u")
    A1 = sp.integrate((u * sp.diff(f, x).subs(x, u)), (u, 0, 1))
    B = sp.integrate(f.subs(x, u), (u, 0, 2))
    rhs = -4 * x**3 + (sp.Rational(3, 2) * x * A1) ** 2 - 2 * x + B
    eqs = sp.Poly(sp.expand(f - rhs), x).all_coeffs()
    sols = sp.solve(eqs, [c3, c2, c1, c0], dict=True)
    got = sorted([sp.expand(f.subs(s_)) for s_ in sols], key=lambda e: float(e.subs(x, 2)))
    A5 = parts(2, "5")
    exp5 = sorted([P(A5["f1"]), P(A5["f2"])], key=lambda e: float(e.subs(x, 2)))
    V.report(2, len(got) == 2 and all(sp.expand(g_ - e_) == 0 for g_, e_ in zip(got, exp5)), f"2-3 항등식 계수 비교로 구한 f {got}")
    # 2-4(1): 교점·넓이를 직접 구하고 극한
    al = sp.solve(sp.Eq(x**2, -k * x + 2 * k**2), x)           # x<0 인 근
    al = [r_ for r_ in al if sp.simplify(r_.subs(k, 1)) < 0][0]
    be = sp.solve(sp.Eq(x, -k * x + 2 * k**2), x)[0]
    g_ = -k * x + 2 * k**2
    S_k = sp.simplify(sp.integrate(g_ - x**2, (x, al, 0)))
    T_k = sp.simplify(sp.integrate(g_ - x, (x, 0, be)))
    V.check(2, "2-4(1) S(k)=10k³/3", S_k, sp.Rational(10, 3) * k**3)
    V.check(2, "2-4(1) T(k)=2k⁴/(k+1)", T_k, 2 * k**4 / (k + 1))
    V.check(2, "2-4(1) lim S/T", sp.limit(S_k / T_k, k, sp.oo), V.ans(2, "6"))
    # 2-4(2): k=1, B_1 을 (−2,4)를 지나는 직선이 이등분하는 기울기 — 다각형 절단(수치)으로 탐색
    tri = [(0.0, 0.0), (0.0, 2.0), (1.0, 1.0)]

    def clipped_area(m):
        side = lambda p: p[1] - 4 - m * (p[0] + 2)
        poly = []
        for i in range(3):
            p, q = tri[i], tri[(i + 1) % 3]
            sp_, sq_ = side(p), side(q)
            if sp_ >= 0:
                poly.append(p)
            if sp_ * sq_ < 0:
                tt = sp_ / (sp_ - sq_)
                poly.append((p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1])))
        if len(poly) < 3:
            return 0.0
        return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2
    V.check(2, "2-4(2) B_1 넓이", sp.integrate(g_.subs(k, 1) - x, (x, 0, be.subs(k, 1))), 1)
    ms = [-12.000137 + i * 0.0005 for i in range(24001)]
    roots = []
    for m1, m2 in zip(ms, ms[1:]):
        d1, d2 = clipped_area(m1) - 0.5, clipped_area(m2) - 0.5
        if d1 * d2 < 0:
            lo, hi = m1, m2
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if (clipped_area(lo) - 0.5) * (clipped_area(mid) - 0.5) > 0 else (lo, mid)
            roots.append((lo + hi) / 2)
    V.report(2, len(roots) == 1 and abs(roots[0] - float(P(V.ans(2, "7")))) < 1e-9, f"2-4(2) 넓이를 이등분하는 기울기 격자탐색 {roots} vs {V.ans(2, '7')}")
    V.check(2, "2-4(2) 4m²+17m+15=0 의 두 근 중 -2<m<-1", [r_ for r_ in sp.solve(4 * x**2 + 17 * x + 15, x) if -2 < r_ < -1][0], V.ans(2, "7"))


if __name__ == "__main__":
    V.run(globals())
