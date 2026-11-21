#!/usr/bin/env python3
"""DGU-M21 (2021학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-M21"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """사다리 운동 PQ=5, P의 속력 = Q의 속력/2 → P(2√5,0), Q(0,√5)."""
    th, w = sp.symbols("theta omega", positive=True)   # x=5cosθ, y=5sinθ, θ'=ω≠0
    X, Y = 5 * sp.cos(th), 5 * sp.sin(th)
    vx, vy = sp.diff(X, th) * w, sp.diff(Y, th) * w
    # 0<θ<π/2 에서 |vx|=5ω sinθ, |vy|=5ω cosθ
    sol = [r for r in sp.solve(sp.Eq(5 * w * sp.sin(th), sp.Rational(1, 2) * 5 * w * sp.cos(th)), th) if 0 < r < sp.pi / 2]
    P = sp.simplify(X.subs(th, sol[0])); Q = sp.simplify(Y.subs(th, sol[0]))
    ok = len(sol) == 1 and P == 2 * sp.sqrt(5) and Q == sp.sqrt(5) and sp.simplify(-vx.subs(th, sol[0]) / vy.subs(th, sol[0]) - sp.Rational(1, 2)) == 0
    report(1, ok, f"tanθ=1/2 → P=({P},0), Q=(0,{Q})")


def p02():
    """주사위 3번: f(x)=x^3+ax^2+bx+c 가 일대일일 확률 (답 4/9). f' 의 서로 다른 실근 개수로 전수 판정."""
    import itertools
    good = 0
    for a, b, c in itertools.product(range(1, 7), repeat=3):
        fp = sp.Poly(3 * x**2 + 2 * a * x + b, x)
        distinct = len(set(sp.real_roots(fp)))
        good += distinct < 2          # 서로 다른 두 실근이면 극값 존재 → 일대일 아님
    p = sp.Rational(good, 216)
    report(2, p == sp.Rational(4, 9), f"유리한 경우 {good}/216 = {p}")


def p03():
    """f'(2x+sinx)=6x+4sinx, f(0)=0: (1) f(4π)=24π^2, ∫0^π f(2x+sinx)dx=2π^3+9π (2) f''(π+1)=3, m=2, M=10/3."""
    u = sp.symbols("u", real=True)
    g = 2 * x + sp.sin(x)
    # f(g(x)) = ∫0^x f'(g(u)) g'(u) du  (g(0)=0, f(0)=0)
    Fg = sp.integrate((6 * u + 4 * sp.sin(u)) * sp.diff(g, x).subs(x, u), (u, 0, x))
    f4pi = sp.simplify(Fg.subs(x, 2 * sp.pi))
    I = sp.simplify(sp.integrate(Fg, (x, 0, sp.pi)))
    f2g = sp.simplify(sp.diff(6 * x + 4 * sp.sin(x), x) / sp.diff(g, x))   # f''(g(x))
    v = sp.simplify(f2g.subs(x, sp.pi / 2))
    import numpy as np
    xs = np.linspace(0, 2 * np.pi, 200001)
    vals = sp.lambdify(x, f2g, "numpy")(xs)
    m, M = vals.min(), vals.max()
    ok = (sp.simplify(f4pi - 24 * sp.pi**2) == 0 and sp.simplify(I - (2 * sp.pi**3 + 9 * sp.pi)) == 0
          and sp.simplify(g.subs(x, sp.pi / 2) - (sp.pi + 1)) == 0 and v == 3
          and abs(m - 2) < 1e-9 and abs(M - 10 / 3) < 1e-9)
    report(3, ok, f"(1) f(4π)={f4pi}, ∫={I} (2) f''(π+1)={v}, m≈{m:.6f}, M≈{M:.6f} (g 치역 = ℝ, g(x+2π)=g(x)+4π 이므로 한 주기로 충분)")


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
