#!/usr/bin/env python3
"""INHA-G25A (2025학년도 인하대 자연계열 논술 오전·일반) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "INHA-G25A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1-1) Σ sin^2(nπ/16)=9/2 (1-2) ∫e^x sin(x/4)cos(x/4)=2e^π/5+1/5 (1-3) ∫e^x f g<0 [증명-수치확인]."""
    import mpmath as mp
    s1 = sp.nsimplify(sp.N(sum(sp.sin(k * sp.pi / 16) ** 2 for k in range(1, 9)), 50), rational=True)
    ok1 = s1 == sp.Rational(9, 2)
    I2 = sp.simplify(sp.integrate(sp.exp(x) * sp.sin(x / 4) * sp.cos(x / 4), (x, 0, sp.pi)))
    ok2 = sp.simplify(I2 - (sp.Rational(2, 5) * sp.exp(sp.pi) + sp.Rational(1, 5))) == 0
    f = sum(sp.sin(k * x / 8) for k in range(1, 17))
    g = sum(k * sp.cos(k * x / 8) for k in range(1, 17))
    ok3a = sp.simplify(sp.diff(f, x) - g / 8) == 0
    f0, fpi = f.subs(x, 0), sp.nsimplify(sp.N(f.subs(x, sp.pi), 50), tolerance=1e-40)
    mp.mp.dps = 30
    F = sp.lambdify(x, f, "mpmath")
    G = sp.lambdify(x, g, "mpmath")
    lhs = mp.quad(lambda u: mp.e**u * F(u) * G(u), mp.linspace(0, mp.pi, 17))
    rhs = -4 * mp.quad(lambda u: mp.e**u * F(u) ** 2, mp.linspace(0, mp.pi, 17))
    ok3 = ok3a and f0 == 0 and fpi == 0 and lhs < 0 and abs(lhs - rhs) < 1e-15 * abs(rhs)
    report(1, ok1 and ok2 and ok3,
           f"(1) {s1} (2) {I2} (3) f'=g/8 {ok3a}, f(π)={fpi}, ∫e^xfg={mp.nstr(lhs, 12)} = -4∫e^xf^2={mp.nstr(rhs, 12)} [증명-수치확인]")


def _crit_points(expr, var, lo, hi, factors):
    """각 인수(sin/cos(u))의 (lo,hi) 안 영점을 u=kπ(+π/2) 풀이로 구하고, 좌우 부호(수치)로 극대/극소/없음 분류."""
    zs = set()
    for fac in factors:                      # fac = (kind, u): kind 'sin' → u=kπ, 'cos' → u=π/2+kπ
        kind, u = fac
        for k in range(-6, 7):
            tgt = k * sp.pi + (sp.pi / 2 if kind == "cos" else 0)
            for z in sp.solve(sp.Eq(u, tgt), var):
                if z.is_real and lo < z < hi:
                    zs.add(sp.simplify(z))
    zs = sorted(zs, key=lambda z: float(z))
    F = sp.lambdify(var, expr, "mpmath")
    import mpmath as mp
    mp.mp.dps = 30
    mx, mn, none = [], [], []
    for z in zs:
        zf, h = mp.mpf(float(sp.N(z, 30))), mp.mpf("1e-6")
        l, r = F(zf - h), F(zf + h)
        (mx if (l > 0 > r) else mn if (l < 0 < r) else none).append(sp.nsimplify(z))
    return zs, mx, mn, none


def p02():
    """(2-1)(a)(b) f 의 영점·F 극값, (2-2)(a)(b) g 의 영점·G 극값 — 부호 수치 판정."""
    T = sp.Symbol("T", real=True)
    f = sp.cos(T - 1) * sp.sin(T - 2) * sp.sin(T**2)
    zs1, mx1, mn1, _ = _crit_points(f, T, 0, 3, [("cos", T - 1), ("sin", T - 2), ("sin", T**2)])
    exp_z1 = {sp.sqrt(sp.pi), 2, sp.sqrt(2 * sp.pi), 1 + sp.pi / 2}       # YAML 정답 1a
    ok1a = {sp.simplify(z) for z in zs1} == exp_z1
    ok1b = set(mx1) == {2, 1 + sp.pi / 2} and set(mn1) == {sp.sqrt(sp.pi), sp.sqrt(2 * sp.pi)}
    g = sp.cos(T**2 - 1) * sp.sin(T**2 - 2) * sp.cos(T**2 - 3) ** 2 * sp.sin(T**2 - 4) ** 2
    zs2, mx2, mn2, no2 = _crit_points(g, T, -sp.sqrt(sp.pi), sp.sqrt(sp.pi),
                                      [("cos", T**2 - 1), ("sin", T**2 - 2), ("cos", T**2 - 3), ("sin", T**2 - 4)])
    a, b = sp.sqrt(1 + sp.pi / 2), sp.sqrt(2)
    exp_z2 = {s * v for s in (1, -1) for v in (a, b, sp.sqrt(3 - sp.pi / 2), sp.sqrt(4 - sp.pi))}
    ok2a = len(zs2) == 8 and all(any(abs(float(z - e)) < 1e-12 for e in exp_z2) for z in zs2)
    near = lambda S, E: len(S) == len(E) and all(any(abs(float(s - e)) < 1e-12 for e in E) for s in S)
    ok2b = near(mx2, [a, -b]) and near(mn2, [-a, b]) and len(no2) == 4
    report(2, ok1a and ok1b and ok2a and ok2b,
           f"(2-1) 영점 {zs1}, 극대 {mx1}, 극소 {mn1} / (2-2) 영점 {len(zs2)}개, 극대 {mx2}, 극소 {mn2}, 극값 아님 {len(no2)}개")


def p03():
    """(3-1) 접점 직접 계산→넓이 (3-2)(a) h 최댓값 (b) 접점 직접 계산→S(p) (c) S 최솟값. 의예 [문제 1] 공유."""
    a, c, p = sp.symbols("a c p", real=True)
    b = sp.Symbol("b", positive=True)

    def trap(F, x0):
        cs = sp.solve(sp.Eq(F.subs(x, c) + sp.diff(F, x).subs(x, c) * (x0 - c), 0), c)
        c1, c2 = cs
        return sp.simplify(sp.Rational(1, 2) * (F.subs(x, c1) + F.subs(x, c2)) * sp.Abs(c2 - c1))
    A1 = trap(x**2 + b, a)
    ok1 = all(abs(float((A1 - 4 * (a**2 + b) ** sp.Rational(3, 2)).subs({a: av, b: bv}))) < 1e-9
              for av, bv in [(0.3, 1.2), (-2, 0.5), (5, 3)])
    g = x**2 + (p - 2) * x + (p**3 - 4 * p + 4)
    D = sp.expand((p - 2) ** 2 - 4 * (p**3 - 4 * p + 4))
    h, rem = sp.div(D, p + 2, p)
    hc = [r for r in sp.solve(sp.diff(h, p), p) if -2 < r < 2]
    hmax = max([h.subs(p, r) for r in hc])
    ok2a = rem == 0 and hmax == sp.Rational(-15, 16)
    import random
    random.seed(1)
    ok2b = True
    for pv in [random.uniform(-1.99, 1.99) for _ in range(20)]:
        Sv = trap(g.subs(p, pv), 1)
        ok2b &= abs(float(Sv) - 4 * (pv**3 - 3 * pv + 3) ** 1.5) < 1e-8
    r = sp.expand(g.subs(x, 1))
    crit = [q for q in sp.solve(sp.diff(r, p), p) if -2 < q < 2]
    cand = [r.subs(p, q) for q in crit] + [sp.limit(r, p, -2, "+"), sp.limit(r, p, 2, "-")]
    rmin = min(cand)
    attained = any(r.subs(p, q) == rmin for q in crit)
    Smin = 4 * rmin ** sp.Rational(3, 2)
    ok2c = attained and Smin == 4
    report(3, ok1 and ok2a and ok2b and ok2c,
           f"(3-1) 넓이 수치=4(a^2+b)^(3/2) (a) h={h}, max={hmax} (b) S=4g(1)^(3/2), g(1)={r} 20점 일치 (c) min S={Smin}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
