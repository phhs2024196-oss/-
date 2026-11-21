#!/usr/bin/env python3
"""DGU-M20 (2020학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import random
import re
import sys

import sympy as sp

PREFIX = "DGU-M20"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """부피 9√2π^4 고정, 옆넓이 πr√(r²+h²) 최소 → r=3π, h=3√2π."""
    r, h = sp.symbols("r h", positive=True)
    V = sp.Rational(1, 3) * sp.pi * r**2 * h
    hs = sp.solve(sp.Eq(V, 9 * sp.sqrt(2) * sp.pi**4), h)[0]
    Sr = sp.pi * r * sp.sqrt(r**2 + hs**2)
    crit = [c for c in sp.solve(sp.diff(Sr**2, r), r) if c.is_positive]
    r0 = crit[0]
    h0 = sp.simplify(hs.subs(r, r0))
    # 부호 변화(감소→증가)
    d = sp.diff(Sr, r)
    left = d.subs(r, r0 * sp.Rational(9, 10)).evalf()
    right = d.subs(r, r0 * sp.Rational(11, 10)).evalf()
    ok = len(crit) == 1 and sp.simplify(r0 - 3 * sp.pi) == 0 and sp.simplify(h0 - 3 * sp.sqrt(2) * sp.pi) == 0 and left < 0 < right
    report(1, ok, f"r={r0}, h={h0}, S'부호 {float(left):.3g}→{float(right):.3g}")


def p02():
    """P(X<1)≤1/3, P(X<2)≥1/2 → b∈[ln√2, ln(3/2)]; k(b)=∫_1^b f^{-1} → 최대 g(ln√2), 최소 g(ln 3/2)."""
    b, X, y = sp.symbols("b X y", positive=True)
    f = b * sp.exp(-b * X)
    P1 = sp.integrate(f, (X, 0, 1))
    P2 = sp.integrate(f, (X, 0, 2))
    blo = sp.solve(sp.Eq(P2, sp.Rational(1, 2)), b)[0]
    bhi = sp.solve(sp.Eq(P1, sp.Rational(1, 3)), b)[0]
    # P1 증가(b에 대해), P2 증가 → b ≤ bhi, b ≥ blo
    mono = sp.simplify(sp.diff(P1, b)).subs(b, sp.Rational(1, 3)) > 0 and sp.diff(P2, b).subs(b, sp.Rational(1, 3)) > 0
    finv = sp.solve(sp.Eq(f.subs(X, y), x), y)[0]  # x = b e^{-b y} → y
    k = sp.simplify(sp.integrate(finv, (x, 1, b)))
    g_expected_form = 1 - (1 + sp.log(b)) / b
    ok_form = sp.simplify(k - g_expected_form) == 0
    gp = sp.simplify(sp.diff(k, b))
    # b<1 구간에서 감소 확인(수치 격자)
    import numpy as np
    bb = np.linspace(float(blo), float(bhi), 2001)
    kv = sp.lambdify(b, k, "numpy")(bb)
    dec = bool(np.all(np.diff(kv) < 0))
    kmax, kmin = sp.simplify(k.subs(b, blo)), sp.simplify(k.subs(b, bhi))
    ok = (sp.simplify(blo - sp.log(sp.sqrt(2))) == 0 and sp.simplify(bhi - sp.log(sp.Rational(3, 2))) == 0
          and mono and ok_form and dec and abs(float(kmax) - float(kv.max())) < 1e-9 and abs(float(kmin) - float(kv.min())) < 1e-9)
    report(2, ok, f"b∈[{blo},{bhi}], k={k}, k'={gp}, 최대 {float(kmax):.6f}, 최소 {float(kmin):.6f}")


def p03():
    """c: a,b 평면 위·a⊥·단위·x>0, d: a,c 에 수직 단위 x>0, e = (e·a)a+(e·c)c+(e·d)d."""
    a = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    b = sp.Matrix([1, 1, 0])
    p, q = sp.symbols("p q", real=True)
    c = p * a + q * b
    sols = sp.solve([a.dot(c), c.dot(c) - 1], [p, q], dict=True)
    cs = [sp.simplify(c.subs(s)) for s in sols]
    cs = [v for v in cs if v[0] > 0]
    c0 = cs[0]
    d1, d2, d3 = sp.symbols("d1 d2 d3", real=True)
    dv = sp.Matrix([d1, d2, d3])
    sd = sp.solve([a.dot(dv), c0.dot(dv), dv.dot(dv) - 1], [d1, d2, d3], dict=True)
    ds = [sp.simplify(dv.subs(s)) for s in sd if dv.subs(s)[0] > 0]
    d0 = ds[0]
    X, Y, Z = sp.symbols("x y z", real=True)
    e = sp.Matrix([X, Y, Z])
    kk, ll, mm = sp.symbols("k l m")
    co = sp.solve(list(kk * a + ll * c0 + mm * d0 - e), [kk, ll, mm], dict=True)[0]
    ok = (len(cs) == 1 and len(ds) == 1
          and sp.simplify(c0 - sp.Matrix([1, 1, -2]) / sp.sqrt(6)) == sp.zeros(3, 1)
          and sp.simplify(d0 - sp.Matrix([1, -1, 0]) / sp.sqrt(2)) == sp.zeros(3, 1)
          and sp.simplify(co[kk] - (X + Y + Z) / sp.sqrt(3)) == 0
          and sp.simplify(co[ll] - (X + Y - 2 * Z) / sp.sqrt(6)) == 0
          and sp.simplify(co[mm] - (X - Y) / sp.sqrt(2)) == 0)
    report(3, ok, f"c={list(c0)}, d={list(d0)}, 계수 k={co[kk]}, l={co[ll]}, m={co[mm]}")


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
