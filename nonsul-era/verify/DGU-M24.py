#!/usr/bin/env python3
"""DGU-M24 (2024학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-M24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """y=x^2, y=4x 사이 밑면, x축 수직 정사각형 단면, y<=4 부분의 부피 (답 47/5)."""
    L = 4 * x - x**2                       # 단면 정사각형의 한 변(밑면 위 선분 길이)
    top = sp.Min(4 * x, 4)                 # y<=4 로 잘린 선분의 위 끝
    width = sp.Max(top - x**2, 0)          # 잘린 뒤 밑변(y방향) 길이
    # 경계점: 4x=4, x^2=4, x^2=4x 의 해 → 각 소구간에서 Min/Max 를 중점으로 확정 후 적분
    bps = sorted({0, 4} | {r for e in (4 * x - 4, x**2 - 4, x**2 - 4 * x) for r in sp.solve(e, x) if 0 <= r <= 4})
    V = 0
    for lo, hi in zip(bps, bps[1:]):
        mid = (lo + hi) / 2
        tp = 4 * x if 4 * mid < 4 else sp.Integer(4)
        wd = tp - x**2 if (tp - x**2).subs(x, mid) > 0 else sp.Integer(0)
        V += sp.integrate(wd * L, (x, lo, hi))
    # 몬테카를로 대신 격자 수치적분으로 교차 확인
    import numpy as np
    xs = np.linspace(0, 4, 400001)
    w = np.clip(np.minimum(4 * xs, 4) - xs**2, 0, None) * (4 * xs - xs**2)
    Vn = np.trapz(w, xs) if hasattr(np, "trapz") else np.trapezoid(w, xs)
    # 대표 오답: 잘린 단면을 정사각형 (4-x^2)^2 로 계산
    wrong = sp.integrate((4 * x - x**2)**2, (x, 0, 1)) + sp.integrate((4 - x**2)**2, (x, 1, 2))
    ok = V == sp.Rational(47, 5) and abs(Vn - 9.4) < 1e-6
    report(1, ok, f"V={V} (수치 {Vn:.6f}), 오답(잘린 단면을 정사각형)={wrong}")


def p02():
    """구 B: x축·y축에 접함, xy평면 단면 넓이 25π. P(10,17,3)에서 원까지 거리 최대·최소와 점."""
    a, b, c, r = sp.symbols("a b c r", positive=True)
    sol = sp.solve([sp.Eq(b**2 + c**2, r**2), sp.Eq(a**2 + c**2, r**2)], [b, r], dict=True)
    s = [d for d in sol if d[b].is_positive][0]           # b=a, r=sqrt(a^2+c^2)
    rho2 = sp.simplify(s[r]**2 - c**2)                    # xy평면 단면 원 반지름^2
    aval = [v for v in sp.solve(sp.Eq(sp.pi * rho2, 25 * sp.pi), a) if v > 0][0]
    cx, cy, rho = aval, s[b].subs(a, aval), sp.sqrt(rho2.subs(a, aval))
    ph = sp.symbols("phi", real=True)
    X, Y = cx + rho * sp.cos(ph), cy + rho * sp.sin(ph)
    D2 = sp.expand((X - 10)**2 + (Y - 17)**2 + 9)
    crit = sp.solve(sp.diff(D2, ph), ph)
    vals = [(sp.nsimplify(sp.simplify(D2.subs(ph, cp))), cp) for cp in crit]
    vals += [(sp.nsimplify(sp.simplify(D2.subs(ph, cp + sp.pi))), cp + sp.pi) for cp in crit]
    mx = max(vals, key=lambda v: float(v[0])); mn = min(vals, key=lambda v: float(v[0]))
    Q = (sp.nsimplify(sp.simplify(X.subs(ph, mx[1]))), sp.nsimplify(sp.simplify(Y.subs(ph, mx[1]))))
    R = (sp.nsimplify(sp.simplify(Y.subs(ph, mn[1]) * 0 + X.subs(ph, mn[1]))), sp.nsimplify(sp.simplify(Y.subs(ph, mn[1]))))
    import numpy as np
    g = np.linspace(0, 2 * np.pi, 200001)
    dn = np.sqrt((5 + 5 * np.cos(g) - 10)**2 + (5 + 5 * np.sin(g) - 17)**2 + 9)
    ok = (sp.sqrt(mx[0]) == 3 * sp.sqrt(37) and sp.sqrt(mn[0]) == sp.sqrt(73)
          and Q == (sp.Rational(40, 13), sp.Rational(5, 13)) and R == (sp.Rational(90, 13), sp.Rational(125, 13))
          and abs(dn.max() - float(3 * sp.sqrt(37))) < 1e-6 and abs(dn.min() - float(sp.sqrt(73))) < 1e-6)
    report(2, ok, f"원 중심({cx},{cy},0) 반지름 {rho}; 최대 {sp.sqrt(mx[0])} at Q{Q+(0,)}, 최소 {sp.sqrt(mn[0])} at R{R+(0,)}")


def p03():
    """윷 1000회 빈도 → 배 통계적 확률 p, 규칙 ②로 d를 θ로 (답 d=7(2π-θ)/13)."""
    freq = {1: 110, 2: 311, 3: 384, 4: 179, 0: 16}       # 배의 개수 : 빈도
    p = sp.Rational(sum(k * f for k, f in freq.items()), 4 * sum(freq.values()))
    th, d = sp.symbols("theta d", positive=True)
    round_area = 10 * 1 * (2 * sp.pi - th)               # 둥근 면 = 높이 10 × 호 길이 r(2π-θ), r=1
    flat_area = 10 * d                                   # 평평한 면 = 높이 10 × 현 d
    # 규칙 ②: P(배):P(등) = 둥근 면 : 평평한 면, 규칙 ③: P(배)=p
    dsol = sp.solve(sp.Eq(p / (1 - p), round_area / flat_area), d)[0]
    wrong = sp.solve(sp.Eq(p / (1 - p), flat_area / round_area), d)[0]
    ok = p == sp.Rational(13, 20) and sp.simplify(dsol - sp.Rational(7, 13) * (2 * sp.pi - th)) == 0
    report(3, ok, f"p={p}, d={sp.factor(dsol)} (대응을 뒤바꾼 오답 d={sp.factor(wrong)})")




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
