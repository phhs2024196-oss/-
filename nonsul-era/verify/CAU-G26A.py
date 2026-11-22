#!/usr/bin/env python3
"""CAU-G26A (2026학년도 중앙대 수시 논술 자연계열Ⅰ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G26A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """원탁 8자리 전수(학생 모두 구별, 1학년 고정) → P(1학년 옆에 2학년) (답 4/5)."""
    import itertools
    studs = ["2a", "2b", "3a", "3b", "4a", "4b", "4c"]
    tot = good = 0
    for perm in itertools.permutations(studs):
        seat = ["1"] + list(perm)
        g = lambda i: seat[i % 8][0]
        if any(g(i) == "2" and g(i + 1) == "2" for i in range(8)):
            continue
        if g(1) == "3" or g(7) == "3":
            continue
        a, b = [i for i in range(8) if g(i) == "2"]
        if (b - a) % 8 in (2, 6):          # 두 2학년이 같은 학생과 이웃
            continue
        tot += 1
        good += g(1) == "2" or g(7) == "2"
    p = sp.Rational(good, tot)
    report(1, p == sp.Rational(4, 5), f"전체 {tot}, 이웃 {good}, P={p}")


def p02():
    """(2-1) 자연수 m,n,k 전수 → (4,5,1); (2-2) 미분가능 → a=-3, g(k) 불연속점 {-1,0,1,2π/√3}."""
    from fractions import Fraction
    best = None
    for m_ in range(1, 10):
        for n_ in range(1, 10):
            k_ = 10 - m_ - n_                     # f(1)=1+m+n+k=11
            if k_ < 1 or m_ * m_ - 3 * n_ <= 0:   # f'(x)=3x^2+2mx+n 이 서로 다른 두 실근
                continue
            r = sp.Poly(3 * x**2 + 2 * m_ * x + n_, x).all_roots()
            v = sp.nsimplify(sp.simplify(1 / r[0] + 1 / r[1]))
            if best is None or v > best[0]:
                best = (v, (m_, n_, k_))
    ok1 = best[1] == (4, 5, 1) and best[0] == sp.Rational(-8, 5)
    a = sp.symbols("a", real=True)
    fR = 4 * sp.sin(x) ** 3 - 3 * sp.sin(x)
    fL = 3 / sp.pi**2 * x**3 + a * x
    av = sp.solve(sp.Eq(sp.diff(fL, x).subs(x, 0), sp.diff(fR, x).subs(x, 0)), a)[0]
    fLv = fL.subs(a, av)
    import mpmath as mp

    def g(k):
        k = mp.mpf(k)
        roots = set()
        # x<0: (3/π²)x³ - 3x = k  (x∈[-π,0))
        for r in mp.polyroots([3 / mp.pi**2, 0, -3, -k], maxsteps=200, extraprec=60):
            if abs(mp.im(r)) < 1e-12 and -mp.pi - 1e-12 <= mp.re(r) < -1e-12:
                roots.add(round(float(mp.re(r)), 9))
        # x≥0: s=sin x, 4s³-3s=k, s∈[0,1]
        for r in mp.polyroots([4, 0, -3, -k], maxsteps=200, extraprec=60):
            if abs(mp.im(r)) < 1e-12 and -1e-12 <= mp.re(r) <= 1 + 1e-12:
                s = min(max(mp.re(r), 0), 1)
                x1 = mp.asin(s)
                for xx in (x1, mp.pi - x1):
                    roots.add(round(float(xx), 9))
        return len(roots)

    crit_vals = sorted({-1.0, 0.0, 1.0, float(2 * mp.pi / mp.sqrt(3))})
    # 극값·끝값 후보를 독립적으로: f 의 임계점 값
    cands = set()
    for c in sp.solve(sp.diff(fLv, x), x):
        if -sp.pi <= c < 0:
            cands.add(float(fLv.subs(x, c)))
    for c in [sp.pi / 6, sp.pi / 2, 5 * sp.pi / 6, 0, sp.pi, -sp.pi]:
        cands.add(float((fR if c >= 0 else fLv).subs(x, c)))
    eps = 1e-7
    disc = sorted(c for c in cands if not (g(c - eps) == g(c) == g(c + eps)))
    grid_ok = all(g(-3 + 0.0137 * i) == g(-3 + 0.0137 * i + 1e-6) for i in range(600)
                  if all(abs(-3 + 0.0137 * i - c) > 1e-4 for c in cands))
    ok2 = av == -3 and len(disc) == 4 and all(abs(d - c) < 1e-9 for d, c in zip(disc, crit_vals)) and grid_ok
    report(2, ok1 and ok2, f"(2-1) {best[1]} 에서 1/α+1/β={best[0]}; (2-2) a={av}, 불연속 k={[round(d, 6) for d in disc]}")


def p03():
    """(3-1) 리만합 n=200000 직접 계산; (3-2) m(t) 수치 최소화 + 수치적분 → a=53/2, b=-5, c=-7/4, d=-1/2."""
    import mpmath as mp
    N = 200000
    s = mp.mpf(0)
    for k in range(1, N + 1):
        u = mp.mpf(k) / N
        num = (3 + u) * mp.log(3 * N + k) - (2 + u) * mp.log(2 * N + k) - mp.log(N)
        den = (2 + u) * (3 + u) * mp.log(2 + u) * mp.log(3 + u)
        s += num / den
    s /= N
    exact = sp.integrate(1 / ((2 + x) * sp.log(2 + x)) - 1 / ((3 + x) * sp.log(3 + x)), (x, 0, 1))
    tgt1 = 2 * sp.log(sp.log(3)) - sp.log(sp.log(2)) - sp.log(sp.log(4))
    ok1 = abs(s - mp.mpf(sp.N(tgt1, 30))) < 1e-4 and sp.simplify(sp.expand_log(exact - tgt1, force=True)) == 0

    def m(tv):
        tv = mp.mpf(tv)
        f = lambda xx: abs(xx) + 2 * mp.sqrt((xx - tv) ** 2 + 3 * tv)   # PO+PA+PB, A(t,√(3t)), B(t,-√(3t))
        lo, hi = mp.mpf(-1), tv + 1                                      # f 는 볼록 → 삼분 탐색
        for _ in range(200):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if f(m1) < f(m2):
                hi = m2
            else:
                lo = m1
        return f((lo + hi) / 2)

    mp.mp.dps = 20
    I = mp.quad(lambda tv: mp.log(m(tv)), [mp.mpf(1) / 2, 1, 4])
    tgt2 = sp.Rational(53, 2) * mp.log(2) - 5 * mp.log(5) - sp.Rational(7, 4) * mp.log(7) - sp.Rational(1, 2)
    ok2 = abs(I - tgt2) < 1e-10
    report(3, ok1 and ok2, f"(3-1) 리만합 {mp.nstr(s, 10)} vs {sp.N(tgt1, 10)}; (3-2) ∫ln m = {mp.nstr(I, 15)} vs {mp.nstr(mp.mpf(tgt2), 15)}")


def p04():
    """(4-1) 차집합 육각형 + 반지름 2 원판의 민코프스키 합 넓이(Steiner) → 72+36√6+4π;
    (4-2) 접점 원을 매개화해 z≤0 인 부분의 호 길이 수치 계산 → √5π."""
    import mpmath as mp
    # 삼각형 ABC: AB=5, BC=6, CA=7 (B 원점, C (6,0))
    B = sp.Matrix([0, 0]); C = sp.Matrix([6, 0])
    ax = sp.Rational(5**2 - 7**2 + 6**2, 12)
    A = sp.Matrix([ax, sp.sqrt(25 - ax**2)])
    V = [A, B, C]
    diffs = [V[i] - V[j] for i in range(3) for j in range(3) if i != j]
    # 볼록 껍질(각도 정렬) — 6개 차벡터가 모두 꼭짓점
    import math
    diffs.sort(key=lambda v: math.atan2(float(v[1]), float(v[0])))
    hexA = sp.Rational(1, 2) * sum(diffs[i][0] * diffs[(i + 1) % 6][1] - diffs[(i + 1) % 6][0] * diffs[i][1] for i in range(6))
    per = sum(sp.sqrt((diffs[(i + 1) % 6] - diffs[i]).dot(diffs[(i + 1) % 6] - diffs[i])) for i in range(6))
    area = sp.simplify(hexA + 2 * per + sp.pi * 2**2)     # Steiner: A + r·L + πr², r=2
    ok1 = sp.simplify(area - (72 + 36 * sp.sqrt(6) + 4 * sp.pi)) == 0
    # (4-2)
    T = sp.Matrix([2 * sp.sqrt(5) / 3, 14 * sp.sqrt(5) / 3, 5 * sp.sqrt(5) / 3])
    R = 10
    OT = sp.sqrt(T.dot(T))
    d = R**2 / OT                      # 접점 원 중심까지 거리 (OP·OT 사영)
    rho = sp.sqrt(R**2 - d**2)
    u = T / OT
    e1 = sp.Matrix([-u[1], u[0], 0]); e1 = e1 / sp.sqrt(e1.dot(e1))
    e2 = u.cross(e1)
    Dc = d * u
    mp.mp.dps = 30
    Kn = 200000
    zc, z1, z2 = (mp.mpf(sp.N(v, 30)) for v in (Dc[2], e1[2], e2[2]))
    neg = sum(1 for i in range(Kn) if zc + mp.mpf(sp.N(rho, 30)) * (z1 * mp.cos(2 * mp.pi * (i + 0.5) / Kn) + z2 * mp.sin(2 * mp.pi * (i + 0.5) / Kn)) <= 0)
    L = 2 * mp.pi * mp.mpf(sp.N(rho, 30)) * neg / Kn
    ok2 = abs(L - mp.sqrt(5) * mp.pi) < 1e-3 and sp.simplify(rho - 2 * sp.sqrt(5)) == 0
    report(4, ok1 and ok2, f"(4-1) 넓이 {area}; (4-2) 반지름 {sp.simplify(rho)}, z≤0 호 길이 {mp.nstr(L, 8)} vs √5π={mp.nstr(mp.sqrt(5) * mp.pi, 8)}")


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
