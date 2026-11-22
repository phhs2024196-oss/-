#!/usr/bin/env python3
"""가천대 2026학년도 수시 논술(실전) 의예 수학(문제 1~8, 보고서 문항카드 91~98, 보고서 표기 G) — sympy 검산.
정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산한 뒤, 그 값을 YAML answer.value 문자열과 대조한다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G26M"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items/GCU/PAST"
x, t, h, n = sp.symbols("x t h n", real=True)


def ans(seq):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return str(d["subparts"][0]["answer"]["value"])


def tex(v):
    v = sp.nsimplify(v)
    if v.is_Rational and v.q != 1:
        s = f"\\frac{{{abs(v.p)}}}{{{v.q}}}"
        return ("-" if v < 0 else "") + s
    return str(v)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def has(seq, *vals):
    a = ans(seq).replace(" ", "")
    return all(tex(v) in a for v in vals)


def cnt(n, v):  # 실수 v 의 n제곱근 중 실수인 것의 개수
    if n % 2:
        return 1
    return 2 if v > 0 else (1 if v == 0 else 0)


def p01():
    vals = set()
    for n in range(2, 14):
        for a in [sp.Rational(-5, 2), -1, sp.Rational(-1, 3), 0, sp.Rational(1, 4), 1, 3]:
            vals.add(cnt(n, (-a) ** n) + (-1) ** n * cnt(n + 1, (-a) ** (n + 1)) + cnt(n + 1, cnt(n + 2, a ** n)))
    M, m = max(vals), min(vals)
    k = sp.symbols("k", positive=True)
    # A: log_M(x+k)+m = log_M(-x+k)+m  -> x=0, 높이 H = log_M k + m ; 정삼각형 한 변 = 2H/sqrt3
    H = sp.log(k, M) + m
    side = 2 * H / sp.sqrt(3)
    def inside(kk):
        v = sp.simplify(side.subs(k, kk))
        return sp.simplify(v - sp.sqrt(3)).is_positive and sp.simplify(2 * sp.sqrt(3) - v).is_positive
    ks = [kk for kk in range(2, 200) if inside(kk)]
    report(1, ans(1).strip() == str(len(ks)), f"M={M}, m={m}, k={ks[0]}..{ks[-1]}, 개수={len(ks)}")


def p02():
    import math
    a1, a2 = math.atan(3 / 4), math.atan(1 / 3)
    r = a2 / a1
    b = math.atan(1 / 4)
    for nn in range(200):  # (다)를 tan b_{n+1} 에 대해 풀어 반복
        an = a1 * r ** nn
        tb = (math.tan(an) + math.tan(b)) / (1 - math.tan(an) * math.tan(b))
        b = math.atan(tb)
        assert 0 < b < math.pi / 2
    exact = sp.Rational(103, 4)
    ok = abs(r - 0.5) < 1e-12 and abs(math.tan(b) - float(exact)) < 1e-6 and has(2, exact)
    # 정확값: b = atan(1/4) + 2*atan(3/4)
    sym = sp.nsimplify(sp.simplify(sp.expand_trig(sp.tan(sp.atan(sp.Rational(1, 4)) + 2 * sp.atan(sp.Rational(3, 4))))))
    report(2, ok and sym == exact, f"공비={r:.12f}, 수치 tan b={math.tan(b):.9f}, 정확값={sym}")


def p03():
    a, b, c, d = sp.symbols("a b c d", real=True)
    f = x ** 3 + b * x ** 2 + c * x + d
    eqs = sp.Poly(sp.expand(f + f.subs(x, 2 * a - x) - 2 * a ** 2), x).coeffs()
    sol = sp.solve(eqs, [b, d], dict=True)
    assert len(sol) == 1
    fs = f.subs(sol[0])
    I = sp.integrate(fs, (x, 0, 2 * a))
    av = [v for v in sp.solve(sp.Eq(I - sp.integrate(x, (x, 0, 2 * a)), 36), a) if v.is_real and v > 1]
    assert len(av) == 1
    fs = fs.subs(a, av[0])
    cv = sp.solve(sp.Eq(sp.integrate((x + 5) ** 2 * fs.subs(x, x + av[0]), (x, -1, 1)), 440), c)
    assert len(cv) == 1
    fs = sp.expand(fs.subs(c, cv[0]))
    fpa = sp.diff(fs, x).subs(x, av[0])
    Iv = sp.integrate(fs, (x, 0, 2 * av[0]))
    a_ = ans(3).replace(" ", "")
    ok = f"f'(a)={fpa}" in a_ and f"dx={Iv}" in a_
    report(3, ok, f"a={av[0]}, f={fs}, f'(a)={fpa}, 적분={Iv}")


def p04():
    import math
    # g(x)=x+(2/pi)sin(pi x/2): g'=0 인 점 집합 {4k+2}, g(4k+2)=4k+2 확인
    g = x + 2 / sp.pi * sp.sin(sp.pi * x / 2)
    crit = sp.solveset(sp.diff(g, x), x, sp.Interval(-20, 20))
    assert set(crit) == {4 * k + 2 for k in range(-5, 5)} and all(g.subs(x, cc) == cc for cc in crit)
    # h 의 치역 (-b, a) 가 {4k+2} 를 피함 + a,b>=0  -> a<=2, b<=2 (a=b=0 포함)
    def ok_ab(a, b):
        if a == 0 and b == 0:
            return True
        lo, hi = -b, a
        return not any(lo < v < hi for v in [-6, -2, 2, 6])
    grid = [sp.Rational(i, 20) for i in range(0, 81)]
    A = [a for a in grid if ok_ab(a, 0)]
    B = [b for b in grid if ok_ab(0, b)]
    assert max(A) == 2 and max(B) == 2
    s = sp.symbols("s")
    first = 4 * s ** 3 - 13 * s ** 2 + 12 * s - 2
    ratio = s ** 3 / 3 - sp.Rational(3, 2) * s ** 2 + 2 * s
    def rng(e):
        pts = [sp.Integer(0), sp.Integer(2)] + [r for r in sp.solve(sp.diff(e, s), s) if 0 <= r <= 2]
        v = [e.subs(s, p) for p in pts]
        return min(v), max(v)
    f_lo, f_hi = rng(first)
    r_lo, r_hi = rng(ratio)
    assert -1 < r_lo and r_hi < 1
    best = max(F / (1 - R) for F in (f_lo, f_hi) for R in (r_lo, r_hi))
    report(4, ans(4).strip() == str(best), f"첫째항 범위=[{f_lo},{f_hi}], 공비 범위=[{r_lo},{r_hi}], 급수 최댓값={best}")


def p05():
    N = 80
    a = [None, 0]
    for nn in range(1, N):
        a.append(sum((-1) ** (nn - k) * k * (nn - k + 1) ** 2 for k in range(1, nn + 1)))
    pat = all(a[2 * j] == j * j for j in range(1, N // 2)) and all(a[2 * j + 1] == -j * (j + 1) for j in range(1, (N - 1) // 2))
    S, ms = 0, []
    for mm in range(1, N):
        S += a[mm]
        if S == -78:
            ms.append(mm)
    report(5, pat and len(ms) == 1 and ans(5).strip() == str(ms[0]), f"a_2n=n^2, a_2n+1=-n(n+1) 확인={pat}, 합=-78 인 m={ms}")


def p06():
    import numpy as np
    def h_count(a, b, t):
        # |t-f| 의 미분불가능점: f(x)=t 이고 부호가 바뀌는 단순근 (f'≠0)
        xs = np.linspace(-15, 60, 300001)
        F = (xs ** 3 + a * xs ** 2 + b * xs) * np.exp(-xs) - t
        sgn = np.sign(F)
        return int(np.sum(sgn[1:] * sgn[:-1] < 0))
    def h0(a, b):  # t=0: 근 x=0 및 x^2+ax+b=0 의 근 중 단순근 개수
        p = sp.Poly(x ** 3 + a * x ** 2 + b * x, x)
        return sum(1 for r, mlt in sp.roots(p, x).items() if r.is_real and mlt == 1)
    eps = 1e-4
    passing = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if h_count(a, b, eps) + h_count(a, b, -eps) + h0(a, b) == 3:
                passing.append((a, b))
    # (가): a=b=0 에서 h 의 불연속점 개수 확인 (후보: 극값 27/e^3, 0)
    fmax = float(27 * sp.exp(-3))
    jumps = [v for v in (0.0, fmax) if h_count(0, 0, v - eps) != h_count(0, 0, v + eps) or v == 0.0]
    t_other = [0.7, 2.0, -1.0, -5.0]
    cont_elsewhere = all(h_count(0, 0, v - eps) == h_count(0, 0, v + eps) for v in t_other)
    f = x ** 3 * sp.exp(-x)
    infl = [r for r in sp.solve(sp.simplify(sp.diff(f, x, 2) * sp.exp(x)), x)]
    fp = sp.diff(f, x)
    prods = [sp.simplify(fp.subs(x, p) * fp.subs(x, q)) for i, p in enumerate(infl) for q in infl[i + 1:]]
    mn = min(prods, key=lambda e: float(e))
    ok = passing == [(0, 0)] and len(jumps) == 2 and cont_elsewhere and sp.simplify(mn + 108 * sp.exp(-6)) == 0 \
        and ans(6).replace(" ", "") == "-\\frac{108}{e^{6}}"
    report(6, ok, f"(나) 만족 정수 (a,b)∈[-3,3]^2={passing}, 불연속 t={jumps}, 변곡점={infl}, 최솟값={mn}")


def p07():
    r2 = sp.sqrt(2)
    A = [None, sp.Integer(1)]
    C = [None, sp.Integer(1)]
    for _ in range(1, 8):
        A.append((A[-1] ** 2 + 2) / (2 * A[-1]))  # 유리수 정확 계산
    for _ in range(1, 200):
        C.append((C[-1] + 2) / (C[-1] + 1))
    bq = lambda v: (v - r2) / (v + r2)  # noqa: E731
    b1, b8 = bq(A[1]), bq(A[8])
    d1, d8 = bq(C[1]), bq(C[8])
    def is_power(base, target):
        out = []
        for k in range(1, 300):
            e = sp.expand(sp.radsimp(base ** k - target))
            if e == 0:
                out.append(k)
        return out
    p = is_power(b1, b8)
    q = is_power(d1, d8)
    m = [k for k in range(1, 200) if C[k] == A[8]]
    a_ = ans(7).replace(" ", "")
    ok = p == [128] and q == [8] and m == [128] and "p=128" in a_ and "q=8" in a_ and "m=128" in a_
    report(7, ok, f"p={p}, q={q}, m={m} (a_8={A[8]})")


def p08():
    a, b, c = sp.symbols("a b c", real=True)
    f = x ** 3 + a * x ** 2 + b * x + c
    fp = sp.diff(f, x)
    av = sp.solve(sp.Eq(sp.diff(fp, x).subs(x, -1), 0), a)[0]  # f' 꼭짓점 x=-1
    fp_a = fp.subs(a, av)
    gx = fp_a.subs(x, -2) * (x + 2) - 3
    bs = [v for v in sp.solve(sp.Eq(gx.subs(x, fp_a.subs(x, 2)), 117), b) if sp.discriminant(fp_a.subs(b, v), x) <= 0]
    assert len(bs) == 1
    cv = sp.solve(sp.Eq(f.subs({a: av, b: bs[0], x: -2}), -3), c)[0]
    fs = f.subs({a: av, b: bs[0], c: cv})
    # (가) 확인: f' >= 0
    mono = sp.solve_univariate_inequality(sp.diff(fs, x) < 0, x, relational=False) == sp.EmptySet
    val = fs.subs(x, 3)
    report(8, mono and ans(8).strip() == str(val), f"a={av}, b={bs[0]}, c={cv}, 증가함수={mono}, f(3)={val}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
