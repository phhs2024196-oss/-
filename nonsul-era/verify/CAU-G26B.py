#!/usr/bin/env python3
"""CAU-G26B (2026학년도 중앙대 수시 논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "CAU-G26B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """팀 구성(2가지)·통과 여부 전수(정확한 분수) → P(4명 모두 합격) (답 53/128)."""
    from fractions import Fraction as Fr
    pairings = [(("g", "a"), ("n", "b")), (("g", "b"), ("n", "a"))]   # 남 g,n / 여 a,b
    total = Fr(0)
    for t1 in pairings:
        for t2 in pairings:
            pr_team = Fr(1, 4)
            for o1 in itertools.product([1, 0], repeat=2):
                p1 = Fr(1)
                for r in o1:
                    p1 *= Fr(1, 2) if r else Fr(1, 2)
                for o2 in itertools.product([1, 0], repeat=2):
                    p2 = Fr(1)
                    for j, team in enumerate(t2):
                        if team in t1:                       # 두 번 연속 같은 팀
                            prev = o1[t1.index(team)]
                            q = Fr(3, 4) if prev else Fr(1, 4)
                        else:                                 # 처음 팀이 됨
                            q = Fr(1, 2)
                        p2 *= q if o2[j] else 1 - q
                    passed = set()
                    for team, r in list(zip(t1, o1)) + list(zip(t2, o2)):
                        if r:
                            passed |= set(team)
                    if passed == {"g", "n", "a", "b"}:
                        total += pr_team * p1 * p2
    report(1, total == Fr(53, 128), f"P(전원 합격)={total}")


def p02():
    """(2-1) a_1 ≤ 200000 전수 → {6,7,8,81}; (2-2) 항등식에서 a, f(0) 독립 결정 → a=1, 극솟값 3, 극댓값 5e^{-3}+4."""
    def nxt(v):
        return 3 * (v + 2) if v % 3 == 1 else 3 * (v + 1) if v % 3 == 2 else v // 3
    good = []
    for a1 in range(1, 200001):
        seen, v = set(), a1
        while v not in seen:
            seen.add(v)
            if len(seen) > 5:
                break
            v = nxt(v)
        if len(seen) == 5 and v in seen:
            good.append(a1)
    ok1 = good == [6, 7, 8, 81]
    a, c0, tt, u = sp.symbols("a c0 tt u", real=True)
    rhs = -x * sp.exp(-x) + x**2 + a * x
    # 좌변 L(x)=∫_0^x u f'(x-u)du 는 L(0)=0, L'(x)=f(x)-f(0) → L'(0)=0
    av = sp.solve(sp.Eq(sp.diff(rhs, x).subs(x, 0), 0), a)[0]
    f = sp.diff(rhs.subs(a, av), x) + c0                 # f(x) = L'(x) + f(0)
    assert sp.simplify(f.subs(x, 0) - c0) == 0
    # 항등식 자체 확인
    L = sp.integrate((u * sp.diff(f, x).subs(x, x - u)), (u, 0, x))
    ident = sp.simplify(L - rhs.subs(a, av)) == 0
    g = sp.simplify(f.subs(x, tt) - tt * sp.diff(f, x).subs(x, tt))      # 접선의 y절편
    I = sp.integrate(g, (tt, sp.log(2), sp.log(4)))
    c0v = sp.solve(sp.Eq(sp.simplify(I), 4 * sp.log(2) - sp.log(2) ** 2 / 2), c0)[0]
    gv = g.subs(c0, c0v)
    cr = sp.solve(sp.diff(gv, tt), tt)
    vals = {}
    for c in cr:
        sec = sp.diff(gv, tt, 2).subs(tt, c)
        vals["min" if sec > 0 else "max"] = sp.simplify(gv.subs(tt, c))
    ok2 = av == 1 and ident and c0v == 3 and vals.get("min") == 3 and sp.simplify(vals.get("max") - (5 * sp.exp(-3) + 4)) == 0
    report(2, ok1 and ok2, f"(2-1) a1={good}; (2-2) a={av}, f(0)={c0v}, 극값 {vals}")


def p03():
    """(3-1) 교점·수치적분 → 3/2 ln2 - 1; (3-2) 이등변 내접삼각형 꼭지각 격자 탐색 → 최댓값 525/16."""
    import mpmath as mp
    y1 = (3 * sp.exp(2 * x) - 6) / (sp.exp(2 * x) + 3 * sp.exp(x) + 2)
    y2 = sp.exp(x) - sp.Rational(3, 2)
    w = sp.symbols("w", positive=True)
    roots = sorted(sp.log(r) for r in sp.solve(sp.Eq((3 * w**2 - 6) * 2, (2 * w - 3) * (w**2 + 3 * w + 2)), w))
    fn = sp.lambdify(x, y1 - y2, "mpmath")
    area = mp.quad(lambda v: abs(fn(v)), [float(roots[0]), float(roots[-1])])
    ok1 = roots == [0, sp.log(2)] and abs(area - (mp.mpf(3) / 2 * mp.log(2) - 1)) < 1e-12
    best = mp.mpf(0)
    Kn = 200000
    for i in range(1, Kn + 1):
        A = (mp.pi / 2) * i / Kn                     # 꼭지각 A ∈ (0, π/2], 외접원 반지름 1
        BC = 2 * mp.sin(A)
        AB = 2 * mp.cos(A / 2)                       # 이등변: B = C = (π-A)/2, AB = 2 sin C
        val = BC**4 + mp.mpf(15) / 4 * (2 * AB**2)
        best = max(best, val)
    ok2 = abs(best - mp.mpf(525) / 16) < 1e-6
    report(3, ok1 and ok2, f"(3-1) 교점 x={roots}, 넓이 {mp.nstr(area, 15)}; (3-2) 격자 최대 {mp.nstr(best, 12)} (525/16={525/16})")


def p04():
    """(4-1) 접점 매개변수 연립(수치) → 사각형 ABCD 넓이 32; (4-2) 좌표로 S, V, 이면각 → SVsinα=√30."""
    import mpmath as mp
    mp.mp.dps = 30
    s6 = mp.sqrt(6)

    def geo(uu, vv):
        x1, y1 = s6 * mp.cosh(uu), s6 * mp.sinh(uu)      # 제1사분면 접점
        x2, y2 = -s6 * mp.cosh(vv), s6 * mp.sinh(vv)     # 제2사분면 접점
        # 접선 x1 X - y1 Y = 6 과 Y=-X, Y=X 의 교점
        A = (6 / (x1 + y1), -6 / (x1 + y1)); B = (6 / (x1 - y1), 6 / (x1 - y1))
        C = (6 / (x2 + y2), -6 / (x2 + y2)); D = (6 / (x2 - y2), 6 / (x2 - y2))
        det = x1 * (-y2) - (-y1) * x2
        E = ((6 * (-y2) - (-y1) * 6) / det, (x1 * 6 - 6 * x2) / det)
        return A, B, C, D, E

    def tri(P, Q, R):
        return abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2

    def eqs(uu, vv):
        A, B, C, D, E = geo(uu, vv)
        OA, OD = mp.sqrt(A[0] ** 2 + A[1] ** 2), mp.sqrt(D[0] ** 2 + D[1] ** 2)
        return [3 * OA - 2 * OD, tri(A, D, E) - 4]

    sols = set()
    for u0 in [0.3, 0.6, 1.0, 1.5]:
        for v0 in [0.3, 0.6, 1.0, 1.5]:
            try:
                r = mp.findroot(eqs, (u0, v0))
            except Exception:  # noqa: BLE001
                continue
            if r[0] > 0 and r[1] > 0:
                A, B, C, D, E = geo(r[0], r[1])
                quad = abs(sum(P[0] * Q[1] - Q[0] * P[1] for P, Q in [(A, B), (B, C), (C, D), (D, A)])) / 2
                sols.add(round(float(quad), 9))
    ok1 = sols == {32.0}
    P = {"A": (0, 0, 2), "B": (2, 0, 2), "C": (2, 2, 2), "D": (0, 2, 2),
         "E": (0, 0, 0), "F": (2, 0, 0), "G": (2, 2, 0), "H": (0, 2, 0)}
    P = {k: sp.Matrix(v) for k, v in P.items()}
    M = (P["A"] + P["E"]) / 2
    Bp, Dp, Gp = P["B"], P["D"], P["G"]
    S = sp.sqrt((Bp - M).cross(Gp - M).dot((Bp - M).cross(Gp - M))) / 2
    V = abs((Bp - M).dot((Dp - M).cross(Gp - M))) / 6
    n1 = (Dp - M).cross(Bp - M); n2 = (Dp - M).cross(Gp - M)
    cosv = abs(n1.dot(n2)) / sp.sqrt(n1.dot(n1) * n2.dot(n2))
    sina = sp.sqrt(1 - cosv**2)
    val = sp.simplify(S * V * sina)
    ok2 = sp.simplify(val - sp.sqrt(30)) == 0
    report(4, ok1 and ok2, f"(4-1) ABCD 넓이 해집합 {sols}; (4-2) S={sp.simplify(S)}, V={V}, sinα={sp.simplify(sina)}, SVsinα={val}")


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
