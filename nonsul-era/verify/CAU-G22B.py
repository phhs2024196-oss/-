#!/usr/bin/env python3
"""CAU-G22B (2022학년도 중앙대 수시 논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G22B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def count_rect(m, n):
    """격자 m×n 의 네 점 직사각형(기울어진 것 포함) 중 넓이 1이 아닌 것의 개수 — 대각선 쌍 묶음으로 전수."""
    from collections import defaultdict
    from math import comb
    pts = [(i, j) for i in range(1, m + 1) for j in range(1, n + 1)]
    groups = defaultdict(list)
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            (x1, y1), (x2, y2) = pts[a], pts[b]
            groups[(x1 + x2, y1 + y2, (x1 - x2) ** 2 + (y1 - y2) ** 2)].append((pts[a], pts[b]))
    cnt = 0
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                (p1, p2), (q1, _) = g[i], g[j]
                ux, uy = q1[0] - p1[0], q1[1] - p1[1]
                vx, vy = p2[0] - q1[0], p2[1] - q1[1]
                area = abs(ux * vy - uy * vx)
                if area != 1:
                    cnt += 1
    return cnt


def p01():
    """A(3,100)-A(2,100) 직접 전수 (답 9899)."""
    a3, a2 = count_rect(3, 100), count_rect(2, 100)
    report(1, a3 - a2 == 9899, f"A(3,100)={a3}, A(2,100)={a2}, 차={a3 - a2}")


def p02():
    """(2-1) k 격자·경계에서 서로 다른 실근 개수 직접 계산, (2-2) f = L''+L 로 적분 (답 범위, 3/2 ln5 - 2/5)."""
    K = sp.symbols("K", nonnegative=True)
    def nroots(kv):
        poly = sp.Poly(2 * x ** 3 + 3 * kv * x ** 2 - (2 * kv ** 2 + kv - 2), x)
        return len(set(sp.real_roots(poly)))
    k1 = (-1 + sp.sqrt(17)) / 4
    pred = lambda kv: (0 <= kv < k1) or (1 < kv < 2)
    ok1 = True
    for i in range(0, 301):
        kv = sp.Rational(i, 100)
        if (nroots(kv) == 1) != pred(kv):
            ok1 = False
    for kv in [k1, sp.Integer(1), sp.Integer(2)]:          # 경계: 중근 ⇒ 실근 2개
        poly = 2 * x ** 3 + 3 * kv * x ** 2 - (2 * kv ** 2 + kv - 2)
        if len(set(sp.simplify(r) for r in sp.solve(poly, x) if r.is_real)) != 2:
            ok1 = False
    L = sp.log(1 + x ** 2)
    f = sp.simplify(sp.diff(L, x, 2) + L)          # I''+I=f, I(0)=I'(0)=0
    import mpmath as mp
    fl = sp.lambdify(x, f, "mpmath")
    chk = max(abs(mp.quad(lambda s: fl(s) * mp.sin(X0 - s), [0, X0]) - mp.log(1 + X0 ** 2)) for X0 in [mp.mpf("0.7"), mp.mpf(2), mp.mpf("3.3")])
    val = sp.simplify(sp.integrate(x * f, (x, 0, 2)))
    ok2 = chk < 1e-12 and sp.simplify(val - (sp.Rational(3, 2) * sp.log(5) - sp.Rational(2, 5))) == 0
    report(2, ok1 and ok2, f"(1) k 격자 301점·경계 일치 / (2) f={f}, 등식 오차 {mp.nstr(chk, 3)}, 적분={val}")


def p03():
    """(3-1) 원판∩{y≥tanθ x} 넓이를 수치적분 후 ∫g/cos² , (3-2) t 격자로 단위벡터 내적 최대 (답 π/4, 2e^{2+√3}/(1+e^{4+2√3}))."""
    import mpmath as mp
    mp.mp.dps = 20
    def g2(th):
        k = mp.tan(th)
        lo = lambda X: max(k * X, -mp.sqrt(max(0, 1 - (X - 1) ** 2)))
        up = lambda X: mp.sqrt(max(0, 1 - (X - 1) ** 2))
        x0 = 2 / (1 + k * k)
        return mp.quad(lambda X: max(0, up(X) - lo(X)), [0, x0, 2])
    I = mp.quad(lambda th: g2(th) / mp.cos(th) ** 2, [0, mp.pi / 4])
    ok1 = abs(I - mp.pi / 4) < 1e-8
    c = 1 / (14 + 8 * mp.sqrt(3))
    f = lambda X: c * mp.log(X) * (mp.log(X) - 1) ** 2
    def cosang(tv):
        A = (tv, f(tv)); B = (f(tv), tv)
        return (A[0] * B[0] + A[1] * B[1]) / (A[0] ** 2 + A[1] ** 2)
    best = max((cosang(mp.exp(mp.mpf(i) / 1000)) for i in range(1, 8000)))
    # 국소 정밀화
    tmax = mp.findroot(lambda tv: mp.diff(cosang, tv), mp.exp(2 + mp.sqrt(3)))
    best = max(best, cosang(tmax))
    s = 2 + mp.sqrt(3)
    target = 2 * mp.e ** s / (1 + mp.e ** (2 * s))
    ok2 = abs(best - target) < 1e-12 and abs(f(mp.e ** s) - 1) < 1e-15
    report(3, ok1 and ok2, f"(1) ∫g/cos²={mp.nstr(I, 12)} / (2) 최대 {mp.nstr(best, 15)} (t*≈{mp.nstr(tmax, 8)}), 목표식 {mp.nstr(target, 15)}")


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
