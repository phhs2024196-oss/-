#!/usr/bin/env python3
"""경북대 2024학년도 수시 논술(AAT) 자연계열Ⅰ (KNU-G24A-01~02) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp
import mpmath as mp

PREFIX = "KNU-G24A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return None if v is None else sp.sympify(v)


def p01():
    mp.mp.dps = 60
    r3 = mp.sqrt(3)
    A1, B1, C1 = (mp.mpf(0), 1 / r3), (mp.mpf(-0.5), -1 / (2 * r3)), (mp.mpf(0.5), -1 / (2 * r3))

    def div(P, Q, m):  # P→Q 를 1:m 으로 내분
        return ((m * P[0] + Q[0]) / (m + 1), (m * P[1] + Q[1]) / (m + 1))

    def dist(P, Q):
        return mp.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)

    ok1 = ok2 = True
    for m in range(1, 6):
        A, B, C = A1, B1, C1
        for n in range(1, 2025):
            g = ((A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3)
            sc = dist(A, B)
            ok1 &= abs(g[0]) < mp.mpf(10) ** -40 * max(1, 1) and abs(g[1]) < mp.mpf(10) ** -40
            if n <= 60:  # 정삼각형(상대오차)
                ok2 &= abs(dist(B, C) - sc) < sc * mp.mpf(10) ** -40 and abs(dist(C, A) - sc) < sc * mp.mpf(10) ** -40
            A, B, C = div(A, B, m), div(B, C, m), div(C, A, m)
    a1 = load_answer(1, 0)
    ok1 &= tuple(a1) == (0, 0)
    # 1-3(1): m=2 넓이 합(정확 계산: 내분 점화식을 sympy 유리수+√3 로 6단계 → 공비 확인 후 등비합)
    s3 = sp.sqrt(3)
    P = [(sp.Integer(0), 1 / s3), (-sp.Rational(1, 2), -1 / (2 * s3)), (sp.Rational(1, 2), -1 / (2 * s3))]

    def area(T):
        (a, b), (c, d), (e, f) = T
        return sp.Abs(sp.nsimplify(((c - a) * (f - b) - (e - a) * (d - b)) / 2))

    areas = []
    for _ in range(6):
        areas.append(sp.simplify(area(P)))
        A, B, C = P
        P = [tuple((2 * A[i] + B[i]) / 3 for i in range(2)), tuple((2 * B[i] + C[i]) / 3 for i in range(2)),
             tuple((2 * C[i] + A[i]) / 3 for i in range(2))]
    ratios = {sp.simplify(areas[i + 1] / areas[i]) for i in range(5)}
    r = ratios.pop() if len(ratios) == 1 else None
    S = sp.simplify(areas[0] / (1 - r)) if r is not None and abs(r) < 1 else None
    ok3 = S is not None and sp.simplify(S - load_answer(1, 2)) == 0
    # 1-3(2): m=2, A_n 의 동경 각으로 x축 위 개수·A_2024 사분면
    A, B, C = A1, B1, C1
    k = 0
    quad = None
    for n in range(1, 2025):
        ang = mp.atan2(A[1], A[0])
        if abs(mp.sin(ang)) < mp.mpf(10) ** -30:
            k += 1
        if n == 2024:
            quad = (1 if A[1] > 0 else 4) if A[0] > 0 else (2 if A[1] > 0 else 3)
        A, B, C = div(A, B, 2), div(B, C, 2), div(C, A, 2)
    ok4 = (k, quad) == tuple(load_answer(1, 3))
    report(1, ok1 and ok2 and ok3 and ok4,
           f"무게중심=원점(m=1..5, n≤2024)={ok1}, 정삼각형(n≤60)={ok2} [증명-수치확인], 공비={r}, Σs_n={S}, k={k}, ℓ={quad}")


def p02():
    # 2-1: 원(O,A,B 지남) 위 제1사분면 점 C, C에서 AB 수선의 발 H 가 AB 를 1:3 내분
    A = sp.Point(18, 0); B = sp.Point(0, 24); O = sp.Point(0, 0)
    circ = sp.Circle(O, A, B)
    H = sp.Point((3 * A.x + B.x) / 4, (3 * A.y + B.y) / 4)
    perp = sp.Line(A, B).perpendicular_line(H)
    Cs = [P for P in circ.intersection(perp) if P.x > 0 and P.y > 0]
    C = Cs[0]
    AC, BC = sp.simplify(A.distance(C)), sp.simplify(B.distance(C))
    a1 = load_answer(2, 0)
    ok1 = len(Cs) == 1 and sp.simplify(AC - a1[0]) == 0 and sp.simplify(BC - a1[1]) == 0
    # 2-2: 차수 m: 좌변 m(m+2), 우변 m(2m+1) — 121t^3 때문에 m>=1, 같아야 하므로 m=1
    ms = [m for m in range(0, 10) if (m >= 1) and m * (m + 2) == m * (2 * m + 1)]
    b, c0 = sp.symbols("b c0")
    p = lambda u: b * u + c0
    eq = sp.Poly(sp.expand(p(t ** 2 * p(2 * t)) - p(t * p(t ** 2)) - 121 * t ** 3), t).coeffs()
    sols = [s for s in sp.solve(eq, [b, c0], dict=True) if s[b] > 0]
    pp = sols[0][b] * x + sols[0][c0]
    ok2 = ms == [1] and len(sols) == 1 and sp.simplify(pp - load_answer(2, 1).subs(sp.Symbol("x"), x)) == 0
    # 2-3
    f = pp / (x ** 2 + 1) ** 2
    df = sp.diff(f, x)
    crit = sp.solve(sp.numer(sp.together(df)), x)
    mins = [r for r in crit if df.subs(x, r - sp.Rational(1, 100)) < 0 < df.subs(x, r + sp.Rational(1, 100))]
    maxs = [r for r in crit if df.subs(x, r - sp.Rational(1, 100)) > 0 > df.subs(x, r + sp.Rational(1, 100))]
    a3 = load_answer(2, 2)
    ok3 = len(mins) == 1 and len(maxs) == 1 and sp.simplify(mins[0] - a3[0]) == 0 and sp.simplify(maxs[0] - a3[1]) == 0
    # 2-4: n=1..9, 수학이 도착 시각 T (거리 적분 = AC) vs 폭탄 BC/n
    T = sp.Symbol("T", positive=True)
    good = []
    for n in range(1, 10):
        dist = sp.integrate(n ** 3 * pp.subs(x, t) / (t ** 2 + n ** 2) ** 2, (t, 0, T))
        roots = [r for r in sp.solve(sp.Eq(dist, AC), T) if r.is_positive]
        if roots and min(roots) < BC / n:
            good.append(n)
    ok4 = good == [int(load_answer(2, 3))]
    report(2, ok1 and ok2 and ok3 and ok4,
           f"C={C}, AC={AC}, BC={BC}, deg후보={ms}, p={pp}, 극소={mins}, 극대={maxs}, 먼저 도착하는 n={good}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda q: int(q[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
