#!/usr/bin/env python3
"""가천대 2027학년도 논술 모의고사 B형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-M27B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)




def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p05():
    # f=2^|x|+a, g=log2|x|+b 와 y=3 의 교점 개수를 a 격자에서 수치로 세고, p+q=3 과 f(1)=g(1) 을 만족하는 (a,b) 탐색
    import math

    def n_sol(fun, xs):
        c = 0
        for u, v in zip(xs, xs[1:]):
            fu, fv = fun(u), fun(v)
            if abs(fu) < 1e-12 or fu * fv < 0:
                c += 1
        return c
    xs_f = [-10 + 20 * i / 20000 for i in range(20001)]           # x=0 포함(i=10000)
    xs_g = [s * 2**(e / 100) for s in (-1, 1) for e in range(-3000, 3001)]
    xs_g = sorted(xs_g)
    sols = []
    for k in range(-40, 41):
        av = sp.Rational(k, 4)
        p = n_sol(lambda u: 2**abs(u) + float(av) - 3, xs_f)
        bv = sp.solve(sp.Eq(2 + av, sp.Symbol("b")), sp.Symbol("b"))[0]       # f(1)=2+a, g(1)=b
        gseg = [u for u in xs_g]
        q = sum(n_sol(lambda u: math.log2(abs(u)) + float(bv) - 3, [u for u in gseg if (u > 0) == side]) for side in (True, False))
        if p + q == 3:
            sols.append((av, bv))
    vals = {s0 + s1 for s0, s1 in sols}
    report(5, vals == {6}, f"(a,b) 후보={sols}, a+b={vals} (YAML 6)")


def p06():
    F = sp.symbols("F")
    fac = sp.factor(sp.expand((F**2 + F) * (F - 2 * x) - (F + 1) * (2 * x + 1)))
    branches = [r for r in sp.solve(fac, F)]
    # 연속함수는 두 가지 가지 사이를 교점(x=-1)에서만 바꿀 수 있음 → 4가지 후보
    inter = sp.solve(sp.Eq(branches[0], branches[1]), x)
    cands = []
    for left in branches:
        for right in branches:
            fx = lambda t, L=left, R=right: (L if t <= inter[0] else R).subs(x, t)
            if fx(0) - fx(-3) == 4:
                cands.append(fx(2) - fx(-2))
    report(6, fac is not None and set(cands) == {2}, f"인수분해={fac}, 분기 교점 x={inter}, 값={cands} (YAML 2)")


def p07():
    import mpmath as mpm
    f = lambda t: (mpm.mpf(1) / 2 - mpm.cos(mpm.pi / 2 + t) / 2
                   + (2 * mpm.cos(mpm.pi - t)**2 + 3 * mpm.cos(3 * mpm.pi / 2 - t)) / (4 * mpm.sin(mpm.pi - t) + 6))
    xs = [mpm.pi / 2 + mpm.pi * i / 20000 for i in range(20001)]
    vals = [f(t) for t in xs]
    mx, mn = max(vals), min(vals)
    s = sp.symbols("s", real=True)
    simp = sp.simplify(sp.Rational(1, 2) + s / 2 + (2 * (1 - s**2) - 3 * s) / (4 * s + 6))
    ok = abs(mx - 1.5) < 1e-12 and abs(mn - 0.7) < 1e-12 and sp.simplify(simp - (sp.Rational(1, 2) + 1 / (2 * s + 3))) == 0
    report(7, ok, f"수치 최댓값={float(mx)}, 최솟값={float(mn)}, 정리식={simp} (YAML 3/2, 7/10)")


def p08():
    a, t = sp.symbols("a t", real=True)
    f = x**3 + 5 * x**2 + a * x + 7
    h = sp.expand(f - sp.diff(f, x))
    # x≥0 에서 h≥0 이 되는 a: h(1)=0 이 항상 성립 → 내부 최소 조건 h'(1)=0 으로 후보, 이후 수치로 h≥0 확인
    cand = sp.solve(sp.diff(h, x).subs(x, 1), a)
    good = [av for av in cand if min(float(h.subs({a: av, x: i / 100})) for i in range(0, 2001)) >= -1e-12]
    # 다른 a 는 실패하는지 확인
    others_fail = all(min(float(h.subs({a: av, x: i / 100})) for i in range(0, 501)) < 0 for av in (2, 2.9, 3.1, 4))
    av = good[0]
    fa = f.subs(a, av)
    G = sp.integrate((fa - fa.subs(x, t)) * fa.subs(x, t)**2, (t, -av, x))
    Gp = sp.factor(sp.diff(G, x))
    crit = sp.solve(Gp, x)
    ext = []
    for c in crit:
        if c.is_real:
            l, r = Gp.subs(x, c - sp.Rational(1, 1000)), Gp.subs(x, c + sp.Rational(1, 1000))
            if sp.sign(l) != sp.sign(r):
                ext.append(c)
    report(8, good == [3] and others_fail and ext == [sp.Rational(-1, 3)], f"a={good}, g'(x)={Gp}, 극값 x={ext} (YAML a=3, b=-1/3)")


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
