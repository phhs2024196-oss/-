#!/usr/bin/env python3
"""가천대 2027학년도 논술 모의고사 A형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-M27A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)




def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p05():
    # log_3 a + log_9 b = log_9(a^2 b) 가 자연수 m ⇔ a^2 b = 9^m (m ≥ 1). 정수로 전수 조사
    cnt = 0
    for a in range(1, 101):
        for b in range(1, 101):
            v = a * a * b
            m = 0
            while v % 9 == 0:
                v //= 9
                m += 1
            if v == 1 and m >= 1:
                cnt += 1
    # 부동소수 교차 확인
    import math
    cnt2 = sum(1 for a in range(1, 101) for b in range(1, 101)
               if abs((t := math.log(a, 3) + math.log(b, 9)) - round(t)) < 1e-9 and round(t) >= 1)
    report(5, cnt == 14 and cnt2 == 14, f"순서쌍 개수={cnt} (부동소수 확인 {cnt2}) (YAML 14)")


def p06():
    n = sp.symbols("n", integer=True, positive=True)
    S = lambda m: sp.Rational(7 * m, m**2 + 3 * m + 2)
    a = lambda m: S(m) - S(m - 1) if m > 1 else S(1)
    tot = a(6) + a(7) + a(8)
    v = abs(tot)
    report(6, v == sp.Rational(19, 90) and sp.gcd(v.p, v.q) == 1, f"a6,a7,a8={a(6)},{a(7)},{a(8)}, |합|={v} → q/p={v} (YAML 19/90)")


def p07():
    # BC=a=3b(AC=b). C 각 변수 θ 로 c^2 = a^2+b^2-2ab cosθ, 주어진 식=0 인 cosθ 를 수치 탐색 후 정확값 확인
    import mpmath as mpm
    bb = 1
    aa = 3 * bb

    def F(th):
        c = mpm.sqrt(aa**2 + bb**2 - 2 * aa * bb * mpm.cos(th))
        # 사인법칙: sin A = a sinθ / c, sin B = b sinθ / c
        sA, sB = aa * mpm.sin(th) / c, bb * mpm.sin(th) / c
        return sB**3 - mpm.cos(th) * sA * sB**2 - sB + mpm.cos(th) * sA
    roots = set()
    N = 4000
    for i in range(1, N):
        t0, t1 = mpm.pi * i / N, mpm.pi * (i + 1) / N
        if i + 1 < N and F(t0) * F(t1) < 0:
            roots.add(round(float(mpm.cos(mpm.findroot(F, (t0, t1), solver="bisect"))), 12))
    report(7, roots == {round(1 / 3, 12)}, f"cosC 후보={roots} (YAML 1/3)")


def p08():
    a, b, p, q = sp.symbols("a b p q", real=True)
    f = x**2 + a * x + b
    g = x**2 + p * x + q
    gp1 = sp.diff(g, x).subs(x, 1)
    eqs = [sp.Eq((f + g).subs(x, 1), 0),                                   # (가) 극한 존재 → 분자 0
           sp.Eq(sp.diff(f + g, x).subs(x, 1), 3 * gp1),                    # (가) 극한값 = 미분계수
           sp.Eq(sp.diff(f**2 - g**2, x).subs(x, 1), 6 * g.subs(x, 1)),      # (나)
           sp.Eq(sp.diff(f * g, x).subs(x, 1), 3)]                          # (다)
    sols = sp.solve(eqs, [a, b, p, q], dict=True)
    out = set()
    for s in sols:
        fs, gs = f.subs(s), g.subs(s)
        # 극한 직접 확인
        ok = (sp.limit((fs + gs) / (x - 1), x, 1) == 3 * sp.diff(gs, x).subs(x, 1)
              and sp.limit((fs**2 - gs**2) / (x - 1), x, 1) == 6 * gs.subs(x, 1))
        if ok:
            out.add((fs.subs(x, 1), sp.diff(fs, x).subs(x, 1), (fs + gs).subs(x, 0), sp.diff(fs + gs, x).subs(x, 0)))
    report(8, out == {(3, -2, 5, -7)}, f"(f(1), f'(1), f(0)+g(0), f'(0)+g'(0)) = {out} (YAML 3, -2, 5, -7)")


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
