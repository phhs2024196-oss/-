#!/usr/bin/env python3
"""CAU-G21B (2021학년도 중앙대 수시 논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "CAU-G21B"
WIDTH = 2
RESULTS = []
x, t, n, u = sp.symbols("x t n u", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """O→Q(12,8) 경로 전수(2^12 중 도착 경로) → x축 미접촉 비율 2/3."""
    tot = good = 0
    for steps in itertools.product((1, -1), repeat=12):
        ys = list(itertools.accumulate(steps))
        if ys[-1] != 8:
            continue
        tot += 1
        good += all(y != 0 for y in ys)
    p = sp.Rational(good, tot)
    report(1, (tot, good) == (66, 44) and p == sp.Rational(2, 3), f"도착 경로 {tot}, 미접촉 {good}, p={p}")


def p02():
    """(1) n=s^2 로 두고 기호 합·극한 → 1/31, (2) 함수식 만족 확인 + 정적분 수치."""
    s, k = sp.symbols("s k", positive=True, integer=True)
    nn = s**2
    num = sp.expand(sp.summation(sp.expand((k + s)**4), (k, 1, nn)))
    den = sp.expand(sp.summation(sp.expand((nn + k)**4), (k, 1, nn)))
    L = sp.limit(num / den, s, sp.oo)
    ok1 = L == sp.Rational(1, 31)
    f = lambda X: mp.sqrt(100 - (X - 10)**2) / mp.sqrt(3)
    ok_eq = all(abs(f(20 - X) - mp.sqrt(-X**2 + 20 * X - 2 * f(X)**2)) < 1e-20 for X in [mp.mpf(i) / 7 for i in range(0, 141)])
    I = mp.quad(lambda X: X * f(X), [0, 10])
    target = 1000 / mp.sqrt(3) * (mp.pi / 4 - mp.mpf(1) / 3)
    ok2 = ok_eq and abs(I - target) < 1e-20
    report(2, ok1 and ok2, f"(1) 극한={L}  (2) 함수식 만족={ok_eq}, 적분≈{mp.nstr(I, 15)}")


def p03():
    """(1) 속력 최소 시각·이동거리 5/3, (2) 넓이 수치 최대 → M^2=(3+2√3)/12."""
    xx, yy = t, sp.Rational(2, 3) * (t**2 - 2 * t + 2)**sp.Rational(3, 2)
    v2 = sp.simplify(sp.diff(xx, t)**2 + sp.diff(yy, t)**2)
    t0s = [r for r in sp.solve(sp.diff(v2, t), t) if r.is_real]
    t0 = min(t0s, key=lambda r: v2.subs(t, r))
    dist = sp.integrate(sp.sqrt(sp.factor(v2)), (t, 0, t0))
    ok1 = t0 == 1 and sp.nsimplify(sp.simplify(dist)) == sp.Rational(5, 3)

    def area(th):
        A = (mp.cos(th), mp.sin(th))
        ph = 2 * th
        c, s_ = mp.cos(ph), mp.sin(ph)
        r = 1 / mp.sqrt(c**2 + s_**2 / 4)          # 직선 위 점 (r cos2θ, r sin2θ) 이 타원 위
        B = (r * c, r * s_)
        return abs(A[0] * B[1] - A[1] * B[0]) / 2
    grid = [mp.pi / 2 * i / 4000 for i in range(4001)]
    i0 = max(range(len(grid)), key=lambda i: area(grid[i]))
    thm = mp.findroot(lambda th: mp.diff(area, th), grid[i0])
    M2 = area(thm)**2
    ok2 = abs(M2 - (3 + 2 * mp.sqrt(3)) / 12) < 1e-20 and abs(mp.tan(thm)**2 - (1 + mp.sqrt(3)) / 2) < 1e-20
    report(3, ok1 and ok2, f"(1) t0={t0}, 거리={dist}  (2) M^2≈{mp.nstr(M2, 15)}, tan^2θ≈{mp.nstr(mp.tan(thm)**2, 12)}")

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
