#!/usr/bin/env python3
"""SKKU-G26B (성균관대 2026학년도 수시 논술위주 수리형 2교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import sys

import numpy as np
import sympy as sp

PREFIX = "SKKU-G26B"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_kv(no, sub):
    return {k.strip(): sp.sympify(v) for k, v in (s.split("=", 1) for s in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    X0, Y0 = sp.symbols("X0 Y0")
    seg_y = lambda f, x0: (f.subs(x, t) - t) * x0 + t        # 선분 PQ 위 x=x0 인 점의 y (<제시문1>)
    # (i)
    sols = [s for s in sp.solve(sp.Eq(seg_y(-x**3 + 1, sp.Rational(1, 3)), sp.Rational(44, 81)), t)
            if s.is_real and 0 <= s <= 1]
    ok1 = len(sols) == 1 and sp.simplify(sols[0] - sp.sympify(md_answer(1, "i"))) == 0
    # (ii) y(t) 의 [0,1] 위 치역 = [최솟값, 최댓값] (연속), 임계점 포함
    yt = sp.expand(seg_y(-x**4 + 1, sp.Rational(1, 5)))
    cands = [sp.Integer(0), sp.Integer(1)] + [c for c in sp.solve(sp.diff(yt, t), t) if c.is_real and 0 < c < 1]
    vals = [yt.subs(t, c) for c in cands]
    md2 = sp.sympify(md_answer(1, "ii"))
    ok2 = md2 == sp.Interval(min(vals), max(vals))
    # (iii) 전수: 서로 다른 p,q 에 대해 [0,1] 의 t 를 조밀 탐색 + 각 조각 일차식의 해 확인
    def f3(tv):
        return 1 if tv < sp.Rational(1, 2) else -2 * tv + 2
    cnt = 0
    for p, q in itertools.permutations([3, 4, 5, 6], 2):
        a0, b0 = sp.Rational(1, p), sp.Rational(1, q)
        found = False
        for lo, hi, fe in [(0, sp.Rational(1, 2), sp.Integer(1)), (sp.Rational(1, 2), 1, -2 * t + 2)]:
            for s in sp.solve(sp.Eq((fe - t) * a0 + t, b0), t):
                if lo <= s <= hi and f3(s) == fe.subs(t, s):
                    found = True
        cnt += found
    ok3 = cnt == int(md_answer(1, "iii"))
    report(1, ok1 and ok2 and ok3, f"(i) t={sols} (ii) [{min(vals)},{max(vals)}] (iii) {cnt}")


def make_gh(fexpr):
    fl = sp.lambdify(x, fexpr, "math")

    def g(u):
        r = ((u + 2) % 4) - 2                                   # [-2,2)
        return fl(r) if r <= 0 else fl(-r)

    def h(u):
        return g(u - 1) * abs(g(u) - g(u - 1))
    return g, h


def quad_h(h, a, b, per=1.0, n=64):
    """[a,b] 를 길이 1/2 조각으로 나눠 Gauss-Legendre(조각 안에서 h 는 다항식)."""
    import numpy.polynomial.legendre as L
    xs, ws = L.leggauss(n)
    edges = np.arange(math.floor(a * 2) / 2, b + 0.5, 0.5)
    edges = np.clip(edges, a, b)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        m, r = (lo + hi) / 2, (hi - lo) / 2
        tot += r * sum(w * h(m + r * xx) for xx, w in zip(xs, ws))
    return tot


def p02():
    # (i) H(x)=∫_1^x h, [2,2026] 에서 H=0 의 개수(격자 부호변화 + 정확한 0 점)
    g, h = make_gh(x + 1)
    Hper = [quad_h(h, 1, 1 + k * 0.25) for k in range(17)]   # 한 주기(4) 안의 H 값
    # 한 주기 [1,5] 에서 H 를 0.001 간격으로
    grid = np.linspace(1, 5, 4001)
    Hg = np.array([quad_h(h, 1, v) for v in grid])
    period_int = Hg[-1]
    # [2,2026] = [2,5] ∪ 주기 반복. H(x+4)=H(x)+period_int
    def zeros_in(lo, hi):
        sel = (grid >= lo - 1e-12) & (grid <= hi + 1e-12)
        v = Hg[sel]
        z = int(np.sum(np.abs(v) < 1e-12))
        z += int(np.sum((v[:-1] * v[1:] < 0)))
        return z
    assert abs(period_int) < 1e-12
    # 2026 = 2 + 4*506 ; 주기 (4k-3,4k+1] 단위로 센다: [2,5] 의 영점 + 505 개의 (1,5] 영점 + (1,2] 처리
    z_first = zeros_in(2, 5)
    z_full = zeros_in(1 + 1e-9, 5)
    z_tail = zeros_in(1 + 1e-9, 2)                            # (2025,2026]
    total = z_first + 505 * z_full + z_tail
    ok1 = total == int(md_answer(2, "i"))
    # (ii) 정확 적분(sympy, 조각별 다항식)
    xs = sp.symbols("xs")
    f2 = (x + 1)**2
    def g_sym(u):
        r = sp.Mod(u + 2, 4) - 2
        return f2.subs(x, r) if r <= 0 else f2.subs(x, -r)
    # 주기 4 한 바퀴 [0,4] 를 1/2 조각으로: 조각 안 다항식 = 중점의 분기를 사용
    def piece_poly(lo):
        mid = lo + sp.Rational(1, 4)
        def gexpr(shift):
            r = sp.Mod(mid - shift + 2, 4) - 2
            base = xs - shift - (mid - shift - r)                 # 같은 분기의 국소 좌표
            return f2.subs(x, base) if r <= 0 else f2.subs(x, -base)
        g0, g1 = gexpr(0), gexpr(1)
        sgn = sp.sign((g0 - g1).subs(xs, mid))
        return g1 * sgn * (g0 - g1)
    total2 = 0
    upper = sp.Rational(2025, 2)
    per4 = sum(sp.integrate(piece_poly(sp.Rational(k, 2)), (xs, sp.Rational(k, 2), sp.Rational(k + 1, 2))) for k in range(8))
    nfull = int(upper // 4)                                     # 253 주기
    rem = upper - 4 * nfull
    rem_int = sum(sp.integrate(piece_poly(sp.Rational(k, 2)), (xs, sp.Rational(k, 2), sp.Rational(k + 1, 2)))
                  for k in range(int(rem * 2)))
    total2 = nfull * per4 + rem_int
    _, h2 = make_gh(f2)
    num2 = quad_h(h2, 0, 2025 / 2)
    ok2 = sp.simplify(total2 - sp.sympify(md_answer(2, "ii"))) == 0 and abs(num2 - float(total2)) < 1e-6
    # (iii) 전수 (1<=a<b<=10), 수치 적분
    good = set()
    for a in range(1, 11):
        for b in range(a + 1, 11):
            _, h3 = make_gh(a * x + b)
            if quad_h(h3, 2023, 2025) < -1e-9:
                good.add((a, b))
    md3 = set(tuple(int(v) for v in s.strip(" ()").split(",")) for s in md_answer(2, "iii").strip("{} ").split("),"))
    ok3 = good == md3
    report(2, ok1 and ok2 and ok3, f"(i) {total} (ii) {total2} ≈ {num2:.6f} (iii) {sorted(good)}")


def fB(u):
    if u <= 1:
        return u * u - u + 2
    if u <= 1.5:
        return u + 1
    if u <= 2:
        return -u + 4
    return -u * u + 3 * u


def p03():
    kv = md_kv(3, "i")
    # (i) 격자에서 D1, D2 판정 후 경계 확인
    xs = np.linspace(-5, 5, 200001)
    in1 = np.array([fB(fB(v)) <= fB(v) + 1 + 1e-12 for v in xs])
    in2 = np.array([fB(fB(v)) <= fB(v) + 1e-12 for v in xs])
    a_num = xs[in1].max()
    ok_a = in1[xs <= float(kv["a"]) - 1e-4].all() and not in1[xs >= float(kv["a"]) + 1e-4].any() \
        and abs(a_num - float(kv["a"])) < 1e-4
    b, c, d = (float(kv[k]) for k in "bcd")
    exp2 = (xs <= b) | ((xs >= c) & (xs <= d))
    mism = np.where(in2 != exp2)[0]
    ok_b = all(min(abs(xs[i] - e) for e in (b, c, d)) < 1e-4 for i in mism) and \
        all(fB(fB(e)) <= fB(e) + 1e-12 for e in (b, c, d))
    ok1 = ok_a and ok_b
    # (ii) P 의 x∈D1=(-∞,a], R 의 각도 집합 → 호 길이
    av = float(kv["a"])
    P = np.concatenate([-np.logspace(-6, 6, 400000), np.linspace(-1e-6, av, 400000)])
    angs = []
    for px in P:
        py = fB(px)
        if abs(px) < 1e-15:
            angs.append(-np.pi / 2)
            continue
        m = (py - 1) / px
        rx, ry = -2 * m / (1 + m * m), (1 - m * m) / (1 + m * m)
        angs.append(math.atan2(ry, rx))
    angs = np.sort(np.mod(np.array(angs), 2 * np.pi))
    gaps = np.diff(np.concatenate([angs, [angs[0] + 2 * np.pi]]))
    theta = 2 * np.pi - gaps.max()
    ok2 = abs(math.tan(theta / 2) - float(sp.sympify(md_answer(3, "ii")))) < 1e-3
    # (iii) x∈D2∩(1,∞) 위의 P 에 대해 조건(QR 지름, N,Q,R 시계방향) 만족하는 m 탐색
    def cond(px):
        py = fB(px)
        m = (py - 1) / px
        R = np.array([-2 * m / (1 + m * m), (1 - m * m) / (1 + m * m)])
        d2 = px * px + py * py
        base = np.array([px, py]) / d2
        perp = np.array([-py, px]) * math.sqrt(d2 - 1) / d2
        best = None
        for Q in (base + perp, base - perp):
            N = np.array([0.0, 1.0])
            cr = (Q[0] - N[0]) * (R[1] - N[1]) - (Q[1] - N[1]) * (R[0] - N[0])
            if cr < 0:                                          # 시계 방향
                best = Q
        return m, (best + R) if best is not None else np.array([9.0, 9.0])
    def brentq(fn, lo, hi, xtol=1e-14):
        flo = fn(lo)
        for _ in range(200):
            mid = (lo + hi) / 2
            fm = fn(mid)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return (lo + hi) / 2
    xsP = np.linspace(1 + 1e-6, 2, 20001)
    F = np.array([cond(v)[1][0] for v in xsP])
    roots = []
    for i in range(len(xsP) - 1):
        if F[i] * F[i + 1] < 0:
            r = brentq(lambda v: cond(v)[1][0], xsP[i], xsP[i + 1], xtol=1e-14)
            m, res = cond(r)
            if np.linalg.norm(res) < 1e-8:
                roots.append(m)
    kv3 = md_kv(3, "iii")
    cub = lambda m: m**3 + float(kv3["p"]) * m**2 + float(kv3["q"]) * m + float(kv3["r"])
    rat = all(v.is_rational for v in kv3.values())
    ok3 = rat and len(roots) == 1 and abs(cub(roots[0])) < 1e-9
    report(3, ok1 and ok2 and ok3, f"(i) a≈{a_num:.5f} mism={len(mism)} (ii) theta={theta:.6f} tan={math.tan(theta / 2):.6f} "
           f"(iii) m={roots}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"예외 {type(e).__name__}: {e}")
    print(f"SUMMARY {PREFIX}: {sum(RESULTS)}/{len(RESULTS)} PASS")
    sys.exit(0 if all(RESULTS) else 1)
