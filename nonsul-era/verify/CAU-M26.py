#!/usr/bin/env python3
"""CAU-M26 (2026학년도 중앙대 모의논술 자연계열) — sympy·전수·수치 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "CAU-M26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """남3·여3 일렬(같은 성별 비구별) × 손 들기 부분집합 전수 → P(남학생 2명 이상 손) (답 67/257)."""
    tot = good = 0
    cnt_by_k = {}
    for pos in itertools.combinations(range(6), 3):          # 남학생 자리
        sex = ["M" if i in pos else "F" for i in range(6)]
        for mask in range(64):
            raised = [i for i in range(6) if mask >> i & 1]
            # 손 든 학생을 순서대로 보면, 연달아 나오는 두 학생은 '이웃하여 손을 든' 것 → 성별이 달라야 함
            if any(sex[a] == sex[b] for a, b in zip(raised, raised[1:])):
                continue
            tot += 1
            cnt_by_k[len(raised)] = cnt_by_k.get(len(raised), 0) + 1
            good += sum(sex[i] == "M" for i in raised) >= 2
    p = sp.Rational(good, tot)
    by_k = [cnt_by_k[k] for k in range(7)]
    ok = p == sp.Rational(67, 257) and by_k == [20, 120, 180, 120, 60, 12, 2]
    report(1, ok, f"전체 {tot} (손 든 수별 {by_k}), 남 2명 이상 {good}, P={p}")


def p02():
    """(2-1) 정수 a,b 전수 → f(1) 집합 {22/3, 16/3}; (2-2) 넓이를 수치적분 → p=-1/6, q=-2, r=2."""
    vals = set()
    for a in range(-3, 4):
        for b in range(0, 4):
            f = x**3 / 3 + a * x**2 + 4 * b * x
            crit = [r for r in sp.solve(sp.diff(f, x), x) if r.is_real]
            if len(crit) != 2:          # 극대·극소가 모두 존재해야 함
                continue
            if sp.simplify(sum(f.subs(x, r) for r in crit) - 12) == 0:
                vals.add(f.subs(x, 1))
    ok1 = vals == {sp.Rational(22, 3), sp.Rational(16, 3)}
    # (2-2) f(x)=2f(x-2) → [2026,2027] 에서 f(x)=2^{1013} f(x-2026). S/2^{1012} = ∫_0^1 |u - 2u sin(pi u)| du
    g = lambda u: abs(u - 2 * u * mp.sin(mp.pi * u))
    S = mp.quad(g, [0, mp.mpf(1) / 6, mp.mpf(5) / 6, 1])
    # 주기 확장 관계 자체도 수치 확인: f(x)=2^{k} f(x-2k)
    def F(xx):
        k = int(mp.floor(xx / 2))
        return 2**k * (xx - 2 * k) * mp.sin(mp.pi * (xx - 2 * k))
    xs = mp.mpf(2026) + mp.mpf("0.37")
    rel = abs(F(xs) / 2**1012 - 2 * (xs - 2026) * mp.sin(mp.pi * (xs - 2026))) < mp.mpf(10) ** -20
    p_, q_, r_ = sp.Rational(-1, 6), -2, 2
    target = mp.mpf(-1) / 6 + q_ / mp.pi + r_ * mp.sqrt(3) / mp.pi
    ok2 = abs(S - target) < mp.mpf(10) ** -20 and rel
    report(2, ok1 and ok2, f"(2-1) f(1)∈{sorted(vals)}; (2-2) S/2^1012={mp.nstr(S, 15)} vs p+q/π+r√3/π={mp.nstr(target, 15)}")


def p03():
    """(3-1) 음함수 미분 + 정의역 → a∈{1/2, 1+√2}; (3-2) 최댓값 (1+√5)/2."""
    X, Y = sp.symbols("X Y", positive=True)
    F = X * Y * (5 - X - Y) - 1          # ln x + ln y + ln(5-x-y) = 0 ⇔ xy(5-x-y)=1 (x,y,5-x-y>0)
    dydx = -sp.diff(F, X) / sp.diff(F, Y)
    sols = sp.solve([F, sp.numer(sp.together(dydx + 1))], [X, Y], dict=True)
    aset = set()
    for s in sols:
        xv, yv = s[X], s[Y]
        if all(v.is_real for v in (xv, yv)) and xv > 0 and yv > 0 and 5 - xv - yv > 0:
            if sp.simplify(dydx.subs(s) + 1) == 0:
                aset.add(sp.nsimplify(sp.simplify(xv)))
    ok1 = aset == {sp.Rational(1, 2), 1 + sp.sqrt(2)}
    f = sp.sin(x) * (sp.sin(x) + 2 * sp.cos(x))
    xs = [-mp.pi / 2 + mp.pi * i / 20000 for i in range(20001)]
    fn = sp.lambdify(x, f, "mpmath")
    mx = max(fn(v) for v in xs)
    crit = sp.solve(sp.Eq(sp.tan(x) ** 2 - sp.tan(x) - 1, 0), x)
    exact = max([sp.nsimplify(sp.simplify(f.subs(x, c))) for c in crit if -sp.pi / 2 < c < sp.pi / 2] + [f.subs(x, sp.pi / 2)])
    ok2 = sp.simplify(exact - (1 + sp.sqrt(5)) / 2) == 0 and abs(mx - (1 + mp.sqrt(5)) / 2) < 1e-8
    report(3, ok1 and ok2, f"(3-1) a∈{sorted(aset, key=float)}; (3-2) 격자최대 {mp.nstr(mx, 12)}, 정확값 {exact}")


def p04():
    """(4-1) 정육각형 좌표 격자 탐색 → 최솟값 √3, 최댓값 4; (4-2) 원 결정 + 수치적분 넓이 → 21√3/32 - π/4."""
    s3 = mp.sqrt(3)
    # 한 변 2, 중심 원점: A(-2,0), B(-1,-√3), C(1,-√3), D(2,0), E(1,√3), F(-1,√3)
    A, B, C, E, Fp = (-2, 0), (-1, -s3), (1, -s3), (1, s3), (-1, s3)
    M = ((E[0] + Fp[0]) / 2, (E[1] + Fp[1]) / 2)
    N = 400
    vals = []
    for i in range(N + 1):
        P = (A[0] + (M[0] - A[0]) * i / N, A[1] + (M[1] - A[1]) * i / N)
        for j in range(N + 1):
            Q = (B[0] + (C[0] - B[0]) * j / N, B[1] + (C[1] - B[1]) * j / N)
            vx = (P[0] - A[0]) + (Q[0] - E[0])
            vy = (P[1] - A[1]) + (Q[1] - E[1])
            vals.append(mp.sqrt(vx * vx + vy * vy))
    mn, mxv = min(vals), max(vals)
    ok1 = abs(mn - s3) < 1e-12 and abs(mxv - 4) < 1e-12
    # (4-2) C1: 중심 (0,k), 반지름 R, P(t,t^2) 에서 포물선에 접함(법선 위 중심); C2: 반지름 R/2, x축에 접함, P 에서 접함(반대쪽 법선)
    tt = sp.symbols("tt", positive=True)
    nx, ny = -2 * tt, 1                       # P 에서 포물선 법선 방향(위쪽)
    nl = sp.sqrt(nx**2 + ny**2)
    # C1 중심 = P + R*(n/|n|) 의 x좌표 0 → R = t|n|/(2t)
    R1 = tt * nl / (2 * tt)
    O2y = tt**2 - (R1 / 2) * ny / nl          # C2 중심 = P - (R1/2) n/|n|
    sol = [s for s in sp.solve(sp.Eq(O2y, R1 / 2), tt) if s.is_positive]
    tv = sol[0]
    R1v = sp.simplify(R1.subs(tt, tv))
    O1y = sp.simplify(tv**2 + R1v * ny / nl.subs(tt, tv))
    O2x = sp.simplify(tv - (R1v / 2) * (-2 * tv) / nl.subs(tt, tv))
    O2y_v = sp.simplify(O2y.subs(tt, tv))
    r2 = R1v / 2
    # 경계를 따라 그린 정리(신발끈)로 넓이: O → Q(x축) → (C2 호) → P → (C1 호) → T → O(y축)
    o1y, R1n = mp.mpf(sp.N(O1y, 30)), mp.mpf(sp.N(R1v, 30))
    o2x, o2y, rr = (mp.mpf(sp.N(v, 30)) for v in (O2x, O2y_v, r2))
    Pm = (mp.mpf(sp.N(tv, 30)), mp.mpf(sp.N(tv**2, 30)))
    a2P = mp.atan2(Pm[1] - o2y, Pm[0] - o2x)          # C2 위 P 의 각
    a1P = mp.atan2(Pm[1] - o1y, Pm[0])                # C1 위 P 의 각
    K = 4000
    pts = [(mp.mpf(0), mp.mpf(0)), (o2x, mp.mpf(0))]
    a_start, a_end = 3 * mp.pi / 2, a2P              # Q(270°) → P
    pts += [(o2x + rr * mp.cos(a_start + (a_end - a_start) * i / K), o2y + rr * mp.sin(a_start + (a_end - a_start) * i / K)) for i in range(1, K + 1)]
    b_start, b_end = a1P + 2 * mp.pi if a1P < 0 else a1P, 3 * mp.pi / 2   # P → T(270°)
    pts += [(R1n * mp.cos(b_start + (b_end - b_start) * i / K), o1y + R1n * mp.sin(b_start + (b_end - b_start) * i / K)) for i in range(1, K + 1)]
    area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2
    target = 21 * s3 / 32 - mp.pi / 4
    ok2 = sp.simplify(tv - sp.sqrt(3) / 2) == 0 and abs(area - target) < 1e-6
    report(4, ok1 and ok2, f"(4-1) 최소 {mp.nstr(mn, 10)}, 최대 {mp.nstr(mxv, 10)}, 합 {mp.nstr(mn + mxv, 10)}; (4-2) t={tv}, 넓이 {mp.nstr(area, 15)}")


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
