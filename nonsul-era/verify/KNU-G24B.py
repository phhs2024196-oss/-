#!/usr/bin/env python3
"""경북대 2024학년도 수시 논술(AAT) 자연계열Ⅱ (KNU-G24B-01~02) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp
import mpmath as mp

PREFIX = "KNU-G24B"
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
    U = lambda k: 2 * abs(k) - abs(k - 1) + abs(k - 2) + 1
    # 1-4 후보: 일반 사차 p(x)=a x^4+b x^3+c x^2+d x+e 에서 p'(x)=4a(x-al)(x-be)(x-ga) 의 근을 조건 (ㄱ)으로 탐색
    #   f=p∘g 일대일 ⇔ (k-1,U(k)) 안에 p' 의 부호변화(=극값점)가 없음. 극값점 집합 R 에서 A 를 재현하는 (al,be,ga) 를 격자에서 찾기
    def Aset(roots, ks):
        return [k for k in ks if not any(k - 1 < r < U(k) for r in roots)]
    ks = [sp.Rational(i, 4) for i in range(-40, 81)]
    target = [k for k in ks if (1 <= k <= 2) or k >= 9]
    grid = [sp.Rational(i, 2) for i in range(-6, 25)]
    found = [(a_, b_, c_) for a_ in grid for b_ in grid for c_ in grid
             if a_ < b_ < c_ and Aset([a_, b_, c_], ks) == target]
    ok0 = found == [(0, 4, 8)]
    # 1-2: k=1, x→-∞ 에서 안쪽 → 0 ⇒ p(0) = 4
    k1in = sp.limit(((1 - 1) * sp.exp(-x) + U(1)) / (sp.exp(-x) + 1), x, -sp.oo)
    ok2 = k1in == 0 and load_answer(1, 1) == 4
    # 1-3: k=0,3,7 의 치역 구간 = (-1,2),(2,6),(6,14), k∉A
    ivs = [(k - 1, U(k)) for k in (0, 3, 7)]
    ok3 = ivs == [(-1, 2), (2, 6), (6, 14)] and all(not ((1 <= k <= 2) or k >= 9) for k in (0, 3, 7))
    # g' > 0 모든 k(격자)
    ok3 &= all(U(k) - (k - 1) > 0 for k in ks)
    # 1-4: p 결정, h(m) 불연속점이 64 가 되는 a
    a = sp.Symbol("a", positive=True)
    p = sp.integrate(4 * a * x * (x - 4) * (x - 8), x) + 4
    ok4 = p.subs(x, 0) == 4
    aval = load_answer(1, 3)
    P = sp.Poly(p.subs(a, aval), x)

    def h(m):
        return sp.Poly(P.as_expr() - m * x - 4, x).sqf_part().count_roots()
    jumps = [m for m in range(-20, 150) if h(m - sp.Rational(1, 1000)) != h(m) or h(m + sp.Rational(1, 1000)) != h(m)]
    ok4 &= jumps == [0, 64]
    report(1, ok0 and ok2 and ok3 and ok4,
           f"A 재현 극값점(반정수 격자)={found}, k=1 안쪽극한={k1in}, 치역구간={ivs}, a={aval}: h 불연속 m={jumps} [증명-수치확인]")


def p02():
    a = sp.Symbol("a", nonnegative=True)
    # 2-1(1): P(0,a), Q(√(1-a²),0), R = 정삼각형의 셋째 꼭짓점 중 정사각형 안쪽(±60° 회전 중 a=0 에서 안에 있는 쪽)
    Px, Py, Qx, Qy = 0, a, sp.sqrt(1 - a ** 2), 0
    cands = []
    for sgn in (1, -1):
        c, s_ = sp.Rational(1, 2), sgn * sp.sqrt(3) / 2
        vx, vy = Px - Qx, Py - Qy
        cands.append((sp.simplify(Qx + c * vx - s_ * vy), sp.simplify(Qy + s_ * vx + c * vy)))
    inside = [R for R in cands if all(0 <= float(R[i].subs(a, v)) <= 1 + 1e-12
                                      for v in (0, 0.3, 0.6, 0.86) for i in (0, 1))]
    R = inside[0]
    ans1 = load_answer(2, 0)
    A_ = sp.Symbol("a")
    ok1 = len(inside) == 1 and all(
        abs(float((R[i] - ans1[i].subs(A_, a)).subs(a, v))) < 1e-12 for v in (0, 0.2, 0.5, 0.8, 0.99) for i in (0, 1))
    # 2-1(2): 0<=a<=√3/2 에서 y = c1 x + c2 √(c3 - x²)
    mp.mp.dps = 40
    c1, c2, c3 = [mp.mpf(sp.N(v, 50)) for v in load_answer(2, 1)]
    vals = [mp.mpf(k) / 200 * mp.sqrt(3) / 2 for k in range(201)]
    Rf = [sp.lambdify(a, R[i], "mpmath") for i in (0, 1)]
    ok2 = all(abs(Rf[1](v) - (c1 * Rf[0](v) + c2 * mp.sqrt(max(c3 - Rf[0](v) ** 2, 0)))) < 1e-25 for v in vals)
    # 전 경로: P 가 OA 위 (θ∈[0,π/2]) 를 정사각형 중심 기준 -90° 회전으로 4번 반복
    def rot(pt, k):
        x0, y0 = pt[0] - 0.5, pt[1] - 0.5
        for _ in range(k):
            x0, y0 = y0, -x0          # 시계 방향 90°: O→A→B→C 대응
        return (x0 + 0.5, y0 + 0.5)
    th = sp.Symbol("th")
    Rth = (sp.sin(th + sp.pi / 6), sp.sin(th + sp.pi / 3))
    ok_param = all(abs(float((R[i] - Rth[i].subs(th, sp.asin(v))).subs(a, v))) < 1e-12 for v in (0.1, 0.5, 0.9) for i in (0, 1))
    # 2-2: R 이 변 위 — θ∈[0,π/2] 에서 x∈{0,1} 또는 y∈{0,1} 의 해, 4회전
    sols = set()
    for e in (Rth[0] - 1, Rth[1] - 1, Rth[0], Rth[1]):
        for r in sp.solveset(e, th, sp.Interval(0, sp.pi / 2)):
            pt = (sp.nsimplify(Rth[0].subs(th, r)), sp.nsimplify(Rth[1].subs(th, r)))
            for k in range(4):
                q = rot((float(pt[0]), float(pt[1])), k)
                sols.add((round(q[0], 9), round(q[1], 9)))
    ans2 = {(round(float(u), 9), round(float(v), 9)) for (u, v) in load_answer(2, 2)}
    ok3 = sols == ans2
    # 2-3: 닫힌 자취의 넓이(신발끈) vs 답
    N = 20000
    pts = []
    for k in range(4):
        for j in range(N):
            tv = mp.pi / 2 * j / N
            pts.append(rot((float(mp.sin(tv + mp.pi / 6)), float(mp.sin(tv + mp.pi / 3))), k))
    area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                   for i in range(len(pts)))) / 2
    ok4 = abs(area - float(load_answer(2, 3))) < 1e-6
    report(2, ok1 and ok2 and ok_param and ok3 and ok4,
           f"R(a)={R}, (2) 항등={ok2}, R 이 변 위 점 {len(sols)}개 일치={ok3}, 자취 넓이≈{area:.9f} vs {float(load_answer(2,3)):.9f}")


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
