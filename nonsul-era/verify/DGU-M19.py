#!/usr/bin/env python3
"""DGU-M19 (2019학년도 동국대 모의논술, 문제1 과학논술 제외) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import random
import re
import sys

import sympy as sp

PREFIX = "DGU-M19"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    """(1) R=r=1 곡선이 점 (0, 3^{1/4}(1+√3)/√2) 지날 때 3^{3/4}√2 m = 1-√3 (2) R=3r 곡선 길이 32r."""
    th = sp.symbols("theta", real=True)
    # 위치를 굴림 조건에서 직접 구성: OP 길이 R+r, PM 방향각 = π + (R+r)θ/r
    def M(R, r):
        return ((R + r) * sp.cos(th) + r * sp.cos(sp.pi + (R + r) * th / r),
                (R + r) * sp.sin(th) + r * sp.sin(sp.pi + (R + r) * th / r))
    X1, Y1 = M(1, 1)
    given = (2 * sp.cos(th) - sp.cos(2 * th), 2 * sp.sin(th) - sp.sin(2 * th))
    ok_given = sp.simplify(X1 - given[0]) == 0 and sp.simplify(Y1 - given[1]) == 0
    target = 3 ** sp.Rational(1, 4) * (1 + sp.sqrt(3)) / sp.sqrt(2)
    # x=0 인 θ 들 중 y=target 인 것 (수치 탐색 → 정확값 대입)
    import numpy as np
    xs = sp.lambdify(th, X1, "numpy")
    ys = sp.lambdify(th, Y1, "numpy")
    grid = np.linspace(0, 2 * np.pi, 200001)
    cands = []
    for i in range(len(grid) - 1):
        if xs(grid[i]) * xs(grid[i + 1]) < 0:
            r0 = sp.nsolve(X1, th, (grid[i] + grid[i + 1]) / 2, prec=40)
            if abs(ys(float(r0)) - float(target)) < 1e-9:
                cands.append(r0)
    th0 = cands[0]
    m = (sp.diff(Y1, th) / sp.diff(X1, th)).subs(th, th0)
    val = sp.N(3 ** sp.Rational(3, 4) * sp.sqrt(2) * m, 30)
    ok1 = ok_given and len(cands) == 1 and abs(val - (1 - sp.sqrt(3)).evalf(30)) < 1e-20
    r = sp.symbols("r", positive=True)
    X3, Y3 = M(3 * r, r)
    sp2 = sp.simplify(sp.diff(X3, th) ** 2 + sp.diff(Y3, th) ** 2)
    ok_speed = sp.simplify((sp2 - 64 * r**2 * sp.sin(3 * th / 2) ** 2).rewrite(sp.exp)) == 0
    Lnum = sp.Integral(8 * sp.Abs(sp.sin(3 * th / 2)), (th, 0, 2 * sp.pi)).evalf(30)
    Lex = sum(sp.integrate(8 * sp.sin(3 * th / 2) * s, (th, lo, hi)) for lo, hi, s in
              [(0, 2 * sp.pi / 3, 1), (2 * sp.pi / 3, 4 * sp.pi / 3, -1), (4 * sp.pi / 3, 2 * sp.pi, 1)])
    ok2 = ok_speed and Lex == 32 and abs(Lnum - 32) < 1e-9
    report(2, ok1 and ok2, f"(1) θ0={float(th0):.6f}, 3^(3/4)√2 m={float(val):.12f}(1-√3={float(1-sp.sqrt(3)):.12f}) (2) 속력²={sp2}, L={Lex}·r")


def p03():
    """[증명-수치확인] (1) 1/3≤s'≤1 (2) t/3≤s≤t (3) s(t)=10+t/6 해 존재."""
    import math
    sp_ = lambda u: math.sqrt(2) / (2 * math.sqrt(2) + math.cos(u) + math.sin(u))
    # (1) 격자 + 정확 극값
    u = sp.symbols("u", real=True)
    spe = sp.sqrt(2) / (2 * sp.sqrt(2) + sp.cos(u) + sp.sin(u))
    lo_ex = sp.simplify(spe.subs(u, sp.pi / 4))
    hi_ex = sp.simplify(spe.subs(u, 5 * sp.pi / 4))
    g = [sp_(2 * math.pi * k / 100000) for k in range(100001)]
    ok1 = lo_ex == sp.Rational(1, 3) and hi_ex == 1 and min(g) >= 1 / 3 - 1e-12 and max(g) <= 1 + 1e-12
    # (2)(3) s(t) 수치 적분 (Simpson, 누적)
    T, Nst = 80.0, 160000
    hstep = T / Nst
    s = [0.0]
    for k in range(Nst):
        a0 = k * hstep
        s.append(s[-1] + hstep / 6 * (sp_(a0) + 4 * sp_(a0 + hstep / 2) + sp_(a0 + hstep)))
    bad = sum(1 for k in range(1, Nst + 1) if not (k * hstep / 3 - 1e-9 <= s[k] <= k * hstep + 1e-9))
    hv = [s[k] - 10 - k * hstep / 6 for k in range(Nst + 1)]
    root = next(k * hstep for k in range(Nst) if hv[k] <= 0 < hv[k + 1])
    h12 = s[int(12 / hstep)] - 12
    h60 = s[int(60 / hstep)] - 20
    ok = ok1 and bad == 0 and h12 <= 0 <= h60 and 12 <= root <= 60
    report(3, ok, f"[증명-수치확인] (1) s'∈[{lo_ex},{hi_ex}] (2) t/3≤s≤t 위반 {bad}건 (3) h(12)={h12:.3f}≤0≤h(60)={h60:.3f}, 근 t≈{root:.3f}")


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
