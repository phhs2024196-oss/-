#!/usr/bin/env python3
"""CAU-G22A (2022학년도 중앙대 수시 논술 자연계열Ⅰ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G22A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """등번호 1~7 → 의자 0~7 단사 배정 전수(조건 4개) (답 56)."""
    import itertools
    allowed = {k: set(range(0, k + 1)) for k in range(1, 8)}
    allowed[5] &= {4, 5}; allowed[6] &= {4, 5, 6}; allowed[7] &= {6, 7}
    cnt = sum(1 for seats in itertools.permutations(range(8), 7)
              if all(seats[k - 1] in allowed[k] for k in range(1, 8)))
    report(1, cnt == 56, f"전수 {cnt}가지 (오답 재현: 4번 선택지를 늘 1로 보면 {8*6})")


def p02():
    """(2-1) 곡선 길이 수치적분, (2-2) 초점·내적에서 y² 도출 후 최댓값 (답 7/2-2√2, 1/√3)."""
    import mpmath as mp
    mp.mp.dps = 30
    # 정의역: 64(1-√x)^3 ≥ 0 이고 x ≥ 1/2
    yf = lambda X: mp.sqrt(64 * (1 - mp.sqrt(X)) ** 3 / 9)
    L = mp.quad(lambda X: mp.sqrt(1 + mp.diff(yf, X) ** 2), [0.5, 0.75, 1 - mp.mpf(10) ** -12])
    ok1 = abs(L - (mp.mpf(7) / 2 - 2 * mp.sqrt(2))) < 1e-8
    # (2-2)
    X, Y = sp.symbols("X Y", real=True)
    T = sp.symbols("T", nonnegative=True)
    w = sp.symbols("w")
    a2, b2 = 2, 1                     # x²/2 + y² = 1
    c = sp.sqrt(a2 - b2)
    PA = sp.Matrix([-c - X, -Y]); PB = sp.Matrix([c - X, -Y])
    eq = sp.Eq(PA.dot(PB), (2 + T ** 2) / (2 * (1 + T + T ** 2)))
    lhs = sp.expand(PA.dot(PB)).subs(X ** 2, 2 - 2 * Y ** 2).subs(Y ** 2, w)
    y2 = sp.solve(sp.Eq(lhs, eq.rhs), w)[0]
    S2 = (sp.Rational(1, 2) * 2 * c) ** 2 * y2                    # S = (1/2)|AB||y|
    crit = [r for r in sp.solve(sp.numer(sp.together(sp.diff(S2, T))), T) if r.is_positive]
    vals = [sp.nsimplify(sp.simplify(S2.subs(T, r))) for r in crit] + [sp.limit(S2, T, sp.oo), S2.subs(T, 0)]
    mx = max(vals, key=lambda v: float(v))
    ok2 = sp.simplify(mx - 1 / sp.sqrt(3)) == 0
    report(2, ok1 and ok2, f"(1) L={mp.nstr(L, 15)} / (2) 임계점 {crit}, S² 최대 {mx}")


def p03():
    """(3-1) a∈[1,8] 격자로 20X-X² 최대·최소, (3-2) (가)의 두 근 중 (나)를 만족하는 연속 분기로 적분 (답 100,64 / √e)."""
    import mpmath as mp
    N = 200000
    vals = []
    for i in range(N + 1):
        a = 1 + 7 * i / N
        b2 = 8 / a - 1                    # 넓이 4 ⇒ a(b²+1)=8
        if b2 < 0:
            continue
        Xv = 2 * a + b2
        vals.append(20 * Xv - Xv * Xv)
    ok1 = abs(max(vals) - 100) < 1e-6 and abs(min(vals) - 64) < 1e-6
    F = sp.symbols("F")
    eqF = F ** 2 * sp.cos(x) ** 2 - 2 * F * (1 + sp.sin(x)) * sp.cos(x) + (1 + sp.sin(x)) ** 2 * sp.cos(x) ** 2
    roots = sp.solve(eqF, F)
    good = [r for r in roots if sp.simplify(r.subs(x, sp.pi / 6) - 3 * sp.sqrt(3) / 2) == 0]
    f = sp.simplify(good[0])
    integrand = (sp.diff(f, x) * sp.cos(x) - f * sp.sin(x)) * sp.exp(sp.sin(x))
    val = sp.N(sp.Integral(integrand, (x, 0, sp.pi / 6)), 30)
    ok2 = len(good) == 1 and abs(val - sp.sqrt(sp.E).evalf(30)) < 1e-25
    report(3, ok1 and ok2, f"(1) M={max(vals):.6f}, m={min(vals):.6f} / (2) f={f}, 적분={val}")


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
