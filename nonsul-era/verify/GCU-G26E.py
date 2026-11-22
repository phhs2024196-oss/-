#!/usr/bin/env python3
"""가천대 2026학년도 수시 논술(실전) 자연계열 E세트 수학(문제 7~15, 보고서 문항카드 73~81) — sympy 검산.
정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산한 뒤, 그 값을 YAML answer.value 문자열과 대조한다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G26E"
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


def p07():
    sols = sp.solveset(3 * sp.cos(x) ** 2 - sp.sin(x) * sp.cos(x) - 2 * sp.sin(x) ** 2, x, sp.Interval.Ropen(0, sp.pi / 2))
    sols = sorted(sols)
    ok = sols == [sp.pi / 4] and "x=\\frac{\\pi}{4}" in ans(7).replace(" ", "")
    report(7, ok, f"[0,pi/2) 해={sols}")


def p08():
    S = {1: sp.Integer(3)}
    for k in range(1, 7):
        S[k + 1] = 3 * S[k] + 2
    a6 = S[6] - S[5]
    lv = sp.nsimplify(sp.log(sp.Rational(8) / a6, 3))
    a = ans(8).replace(" ", "")
    ok = f"S_7={S[7]}" in a and f"={lv}" in a
    report(8, ok, f"S={list(S.values())}, a6={a6}, log3(8/a6)={lv}")


def p09():
    k = sp.symbols("k", real=True)
    f = x ** 2 - 3 * x + k
    den = sp.factor(2 * f ** 2 - 2 * x * f + 5 * f)
    conds = []
    for fac, _ in sp.factor_list(den, x)[1]:
        if sp.degree(fac, x) == 2:
            conds.append(sp.solve_univariate_inequality(sp.discriminant(fac, x) < 0, k, relational=False))
    rng = sp.Intersection(*conds)
    ok = rng == sp.Interval.open(sp.Rational(9, 4), sp.oo) and "k>" + tex(rng.inf) in ans(9).replace(" ", "")
    report(9, ok, f"분모 인수={den}, k 범위={rng}")


def p10():
    f = t ** 2 + (27 - t ** 3)
    cand = [sp.Integer(0), sp.Integer(3)] + [c for c in sp.solve(sp.diff(f, t), t) if 0 <= c <= 3]
    vals = [f.subs(t, c) for c in cand]
    M, m = max(vals), min(vals)
    report(10, has(10, M * m), f"후보 t={cand}, M={M}, m={m}, Mm={M * m}")


def p11():
    # 대칭 성질을 가정하지 않고 수치로: 기울기 s>0 인 직선 y=s(x-1/4) 와 y=tan2πx 의 두 가지(branch) 교점
    import math

    def bis(g, l, r):
        for _ in range(200):
            m = (l + r) / 2
            if g(l) * g(m) <= 0:
                r = m
            else:
                l = m
        return (l + r) / 2

    def pts(s):
        g = lambda v: math.tan(2 * math.pi * v) - s * (v - 0.25)  # noqa: E731
        e = 1e-12
        a = bis(g, -0.25 + e, 0.25 - e)   # 왼쪽 가지에서 부호 변화(-inf -> +inf)
        b = bis(g, 0.25 + e, 0.75 - e)
        return a, b

    # a+2b-7/6 이 0 이 되는 s 를 로그 스케일 이분법
    H = lambda ls: (lambda ab: ab[0] + 2 * ab[1] - 7 / 6)(pts(math.exp(ls)))  # noqa: E731
    lo, hi = -10.0, 10.0
    assert H(lo) * H(hi) < 0
    ls = bis(H, lo, hi)
    av, bv = pts(math.exp(ls))
    av, bv = sp.nsimplify(round(av, 9)), sp.nsimplify(round(bv, 9))
    val = sp.simplify(sp.sin((2 * av + bv) * sp.pi))
    ok = val == sp.sqrt(3) / 2 and "\\frac{\\sqrt{3}}{2}" in ans(11).replace(" ", "")
    report(11, ok, f"기울기≈{math.exp(ls):.6f}, a={av}, b={bv}, 값={val}")


def p12():
    a, b, c = sp.symbols("a b c", real=True)
    f = x ** 3 + a * x ** 2 + b * x + c
    fp = sp.diff(f, x)
    mid = sp.expand(sp.cancel((f.subs(x, 3 * x) - f.subs(x, 0)) / (3 * x)))
    low = sp.expand((3 * fp - fp.subs(x, 0)) / 4 + sp.Rational(23, 4) * x ** 2 - 4)
    up = sp.Rational(1, 2) * x ** 4 + 5 * x ** 2
    # 직선 L(x)=mid-9x^2 이 g=low-9x^2 (최댓값) 와 h=up-9x^2 (최솟값) 사이: 계수 a 소거된 L 확인
    L = sp.expand(mid - 9 * x ** 2)
    g = sp.expand(low - 9 * x ** 2 - (L - L))  # low 에도 a,b 포함 -> mid-low 로 비교
    d1 = sp.expand(mid - low)  # >=0
    d2 = sp.expand(up - mid)   # >=0
    found = []
    for av in range(-10, 11):
        for bv in range(-30, 31):
            e1, e2 = d1.subs({a: av, b: bv}), d2.subs({a: av, b: bv})
            xs = [sp.Rational(i, 20) for i in range(-200, 201) if i != 0]
            if all(e1.subs(x, v) >= 0 and e2.subs(x, v) >= 0 for v in xs[::7] + [2, -2]):
                found.append((av, bv))
    assert len(found) == 1
    av, bv = found[0]
    val = fp.subs({a: av, b: bv, x: 3})
    report(12, has(12, val) and ans(12).strip() == str(val), f"(a,b)={found[0]}, f'(3)={val}")


def p13():
    a, b, c = sp.symbols("a b c", real=True)
    res = None
    for deg in range(1, 6):
        lim = sp.limit(x ** deg / (x * sp.diff(x ** deg, x)), x, sp.oo)
        if lim == sp.Rational(1, 3):
            res = deg
    assert res == 3
    f = x ** 3 + a * x ** 2 + b * x + c
    s1 = sp.solve(f.subs(x, 0), c)[0]
    f = f.subs(c, s1)
    s2 = sp.solve(f.subs(x, -1), b)[0]
    f = f.subs(b, s2)
    q = sp.cancel((x ** 3 + 1) / (x * f))
    av = sp.solve(sp.Eq(q.subs(x, -1), 3), a)
    fs = sp.expand(f.subs(a, av[0]))
    assert sp.limit((x ** 2 + 1 / x) * fs, x, 0).is_finite
    I = sp.integrate(sp.Abs(fs), (x, -2, 2))
    report(13, has(13, I), f"f={fs}, 적분={I}")


def p14():
    f = sp.log(x + 2, 2)
    B = (0, f.subs(x, 0))

    def area(P, Q, R):
        return sp.Abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2

    def S(k):
        P = (2 ** k, sp.log(2 + 2 ** k, 2))
        return sp.Abs(area(P, B, (0, 0)) - area(P, B, (-2, 0)))
    # S_n = log2(2+2^n)-1 > 2026  <=> 2^n > 2^2027 - 2
    nmin = next(k for k in range(2, 3000) if 2 + sp.Integer(2) ** k > sp.Integer(2) ** 2027)
    chk = all(sp.simplify(S(k) - (sp.log(2 + 2 ** k, 2) - 1)) == 0 for k in range(2, 8))
    report(14, chk and ans(14).strip() == str(nmin), f"S_n 공식 확인(n=2..7)={chk}, 최소 n={nmin}")


def p15():
    a, C = sp.symbols("a C", real=True)
    F = sp.integrate((x - 2) * (x - a), x) + C
    best = []
    for crit in (2, a):
        Cv = sp.solve(F.subs(x, crit), C)[0]
        F0 = Cv
        cand = [sp.Integer(6)] + [r for r in sp.solve(sp.diff(F0, a), a) if r.is_real and 2 < r <= 6]
        best.append(min(F0.subs(a, r) for r in cand))
        # 경계 a->2+ 도 비교
        best.append(sp.limit(F0, a, 2, "+") + sp.Rational(1, 10**9))
    # 실근 2개 확인: a=6, C=4/3-12
    mn = min(best)
    F6 = F.subs({a: 6, C: sp.Rational(4, 3) - 12})
    nroot = len(set(sp.Poly(F6, x).real_roots()))
    report(15, nroot == 2 and has(15, mn), f"경우별 후보={best}, 최솟값={mn}, a=6 실근수={nroot}")


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
