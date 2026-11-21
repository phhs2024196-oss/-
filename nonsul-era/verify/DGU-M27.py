#!/usr/bin/env python3
"""DGU-M27 (2027학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-M27"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """위치 복원 → 교점 x좌표를 θ로 → 최대점 (답 3π/8)."""
    T, th, a, v = sp.symbols("T theta a v", positive=True)
    xs = sp.integrate(sp.integrate(a, (t, 0, T)) + v * sp.cos(th), (T, 0, t))
    ys = sp.integrate(sp.integrate(-a, (t, 0, T)) + v * sp.sin(th), (T, 0, t))
    roots = [r for r in sp.solve(sp.Eq(ys, 0), t) if r != 0]
    X = sp.simplify(xs.subs(t, roots[0]))
    g = sp.simplify(X * a / (2 * v**2))
    crit = [c for c in sp.solve(sp.diff(g, th), th) if 0 < c <= sp.pi / 2]
    cand = crit + [sp.pi / 2]
    best = max(cand, key=lambda c: float(g.subs(th, c)))
    # 수치로 구간 전체 최대 확인
    import numpy as np
    grid = np.linspace(1e-6, float(sp.pi / 2), 200001)
    gf = sp.lambdify(th, g, "numpy")(grid)
    ok = sp.simplify(best - 3 * sp.pi / 8) == 0 and abs(grid[gf.argmax()] - float(3 * sp.pi / 8)) < 1e-4
    report(1, ok, f"x교점=(2v^2/a)*({g}), argmax θ={best}")


def p02():
    """(1) 총액 E·σ (2) P(총액=59000) (3) P(마지막 C | 59000) — 4^5 전수."""
    import itertools
    price = {"A": 9000, "B": 10000, "C": 10000, "D": 15000}
    outs = list(itertools.product("ABCD", repeat=5))
    tot = [sum(price[c] for c in o) for o in outs]
    N = len(outs)
    E = sp.Rational(sum(tot), N)
    V = sp.Rational(sum(s * s for s in tot), N) - E**2
    sd = sp.sqrt(V)
    ev = [o for o, s in zip(outs, tot) if s == 59000]
    p2 = sp.Rational(len(ev), N)
    p3 = sp.Rational(sum(1 for o in ev if o[-1] == "C"), len(ev))
    ok = (E == 55000 and sp.simplify(sd - 500 * sp.sqrt(110)) == 0
          and p2 == sp.Rational(15, 128) and p3 == sp.Rational(1, 5))
    report(2, ok, f"(1) E={E}, σ={sd} (2) {p2} (3) {p3}")


def p03():
    """사차함수 결정(f' 홀함수, g(4)=1, g'(4)=1/10) → (1) f, f'(2) (2) g(10), g'(10) (3) ∫_4^28 g/√x."""
    b, c, d, e, X = sp.symbols("b c d e X", real=True)
    f = x**4 + b * x**3 + c * x**2 + d * x + e
    fp = sp.diff(f, x)
    # ① f' 이 원점 대칭: f'(-x) = -f'(x) 항등
    odd = sp.Poly(sp.expand(fp.subs(x, -x) + fp), x).coeffs()
    # ② g(4)=1 ⇔ f(1)=4,  g'(4)=1/f'(1)=1/10
    sol = sp.solve(odd + [f.subs(x, 1) - 4, fp.subs(x, 1) - 10], [b, c, d, e], dict=True)
    assert len(sol) == 1
    F = sp.expand(f.subs(sol[0]))
    Fp = sp.diff(F, x)
    # x>0 에서 증가(역함수 존재) 확인
    inc = all(float(Fp.subs(x, s)) > 0 for s in [0.001 * k for k in range(1, 5001)])
    v1 = Fp.subs(x, 2)
    r = [s for s in sp.solve(sp.Eq(F, 10), x) if s.is_real and s > 0]
    g10 = r[0]
    slope = sp.nsimplify(sp.simplify(1 / Fp.subs(x, g10)))
    # (3) 수치 적분(역함수를 수치로) vs 치환 후 sympy 정적분
    lo = [s for s in sp.solve(sp.Eq(F, 4), x) if s.is_real and s > 0][0]
    hi = [s for s in sp.solve(sp.Eq(F, 28), x) if s.is_real and s > 0][0]
    I_sym = sp.simplify(sp.integrate(sp.simplify(x * Fp / sp.sqrt(F)), (x, lo, hi)))
    import mpmath as mp
    Ff = sp.lambdify(x, F, "mpmath")
    ginv = lambda y: mp.findroot(lambda s: Ff(s) - y, 1.5)  # noqa: E731
    I_num = mp.quad(lambda y: ginv(y) / mp.sqrt(y), [4, 10, 28])
    ok = (F == x**4 + 3 * x**2 and inc and v1 == 44 and sp.simplify(g10 - sp.sqrt(2)) == 0
          and sp.simplify(slope - sp.sqrt(2) / 28) == 0
          and sp.simplify(I_sym - (10 * sp.sqrt(7) + 4) / 3) == 0
          and abs(I_num - float(I_sym)) < 1e-9)
    report(3, ok, f"(1) f={F}, f'(2)={v1} (2) g(10)={g10}, 기울기={slope} (3) I={I_sym} (수치 {float(I_num):.10f})")


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
