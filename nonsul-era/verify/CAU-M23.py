#!/usr/bin/env python3
"""CAU-M23 (2023학년도 중앙대 모의논술 자연계열 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-M23"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """a∈{2,3,4}, b∈{4..32} 전수: -48+44L-12L^2+L^3>0, L=log_a(ab) (정확한 유리수 비교) → 55."""
    X = sp.symbols("X")
    poly = sp.expand(-48 + 44 * (1 + X) - 12 * (1 + X) ** 2 + (1 + X) ** 3)
    fac_ok = sp.factor(poly) == sp.factor((X - 1) * (X - 3) * (X - 5))
    cnt = 0
    per = {}
    for a in (2, 3, 4):
        k = 0
        for b in range(4, 33):
            L = sp.log(a * b) / sp.log(a)
            v = sp.nsimplify(-48 + 44 * L - 12 * L ** 2 + L ** 3)
            val = sp.N(v, 50)
            if abs(val) < 1e-40:          # 경계(=0)는 정확히 판정
                val = sp.simplify(v)
            if val > 0:
                k += 1
        per[a] = k
        cnt += k
    # 오답 재현: a=3 에서 b<27 상한을 b<=27 로 셈 → 56
    report(1, fac_ok and cnt == sum(per.values()) and per == {2: 4, 3: 23, 4: 28} and cnt == 55,
           f"인수분해 (X-1)(X-3)(X-5) {fac_ok}, a별 {per}, 합 {cnt}")


def p02():
    """(1) S5=45,S10=140 → a_n, Σ 1/2^n ln(3a_n^2/(a_n+2)) 수치·기호 / (2) ∫_0^{π/2}(2x^2-πx+π^2)sin^2x dx."""
    a, d = sp.symbols("a d")
    sol = sp.solve([sp.Eq(5 * (2 * a + 4 * d) / 2, 45), sp.Eq(10 * (2 * a + 9 * d) / 2, 140)], [a, d])
    an = sol[a] + (n - 1) * sol[d]
    import mpmath as mp
    mp.mp.dps = 40
    f = sp.lambdify(n, sp.log(3 * an ** 2 / (an + 2)) / 2 ** n, "mpmath")
    s1 = mp.nsum(lambda k: f(k), [1, mp.inf])
    ok1 = abs(s1 - mp.log(15)) < mp.mpf(10) ** -30
    I = sp.integrate((2 * x ** 2 - sp.pi * x + sp.pi ** 2) * sp.sin(x) ** 2, (x, 0, sp.pi / 2))
    ok2 = sp.simplify(I - sp.Rational(11, 48) * sp.pi ** 3) == 0
    report(2, ok1 and ok2, f"a_n={sp.expand(an)}, 급수={mp.nstr(s1, 15)} (ln15={mp.nstr(mp.log(15), 15)}), 정적분={sp.simplify(I)}")


def p03():
    """(1) A(0,1)에서 f=(lnx)^2-lnx 에 그은 접선의 접점 → ∫_α^β f / (2) P(t,√6(t-2)^2), S(t), S'(d)=0."""
    b = sp.symbols("b", positive=True)
    f = sp.log(x) ** 2 - sp.log(x)
    fb, dfb = f.subs(x, b), sp.diff(f, x).subs(x, b)
    roots = sp.solve(sp.Eq(1 - fb, dfb * (0 - b)), b)
    roots = sorted(roots, key=lambda r: float(r))
    al, be = roots[0], roots[-1]
    I = sp.simplify(sp.integrate(f, (x, al, be)))
    ok1 = len(roots) == 2 and sp.simplify(I - (3 * sp.exp(3) - 3)) == 0
    # (2) 넓이 = (원 밖 점에서 그은 두 접선과 원) = sqrt(L^2-1) - arctan(sqrt(L^2-1))
    L2 = t ** 2 + 6 * (t - 2) ** 4
    S = sp.sqrt(L2 - 1) - sp.atan(sp.sqrt(L2 - 1))
    dS = sp.diff(S, t)
    crit = [r for r in sp.solve(sp.diff(L2, t), t) if r.is_real]
    # L>1 항상(원 밖) 확인: min L2 > 1
    mins = min(float(L2.subs(t, r)) for r in crit)
    ok2 = crit == [sp.Rational(3, 2)] and abs(float(dS.subs(t, sp.Rational(3, 2)))) < 1e-12 and mins > 1
    # 수치 확인: 넓이 공식 = 연 모양 넓이 - 부채꼴 (점 P 의 실제 접점으로 계산)
    import math
    tt = 0.7
    px, py = tt, math.sqrt(6) * (tt - 2) ** 2
    Lr = math.hypot(px, py)
    th = math.acos(1 / Lr)
    area = 2 * 0.5 * 1 * math.sqrt(Lr ** 2 - 1) - th
    ok3 = abs(area - float(S.subs(t, tt))) < 1e-12
    report(3, ok1 and ok2 and ok3, f"접점 x={roots}, 적분={I}; L^2'=0 의 실근 {crit}, min L^2={mins:.4f}>1")


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
