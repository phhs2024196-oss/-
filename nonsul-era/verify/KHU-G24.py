#!/usr/bin/env python3
"""KHU-G24 (경희대 2024학년도) — sympy 검산
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "KHU-G24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def perp_bisector_axes(P, Q):
    """선분 PQ(P는 x축 위, Q는 y축 위)의 수직이등분선과 x축, y축의 교점"""
    X, Y = sp.symbols("X Y")
    Mx, My = (P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2
    # 수직이등분선: (Q-P)·((X,Y)-M)=0
    line = (Q[0] - P[0]) * (X - Mx) + (Q[1] - P[1]) * (Y - My)
    xi = sp.solve(line.subs(Y, 0), X)[0]
    yi = sp.solve(line.subs(X, 0), Y)[0]
    return (sp.nsimplify(xi), sp.Integer(0)), (sp.Integer(0), sp.nsimplify(yi))


def p01():
    # (1) 수직이등분선과 쌍곡선 x^2-y^2=b 의 접선 조건
    P1, Q1 = (sp.Integer(8), sp.Integer(0)), (sp.Integer(0), sp.Integer(4))
    X, Y, b = sp.symbols("X Y b")
    line = (Q1[0] - P1[0]) * (X - 4) + (Q1[1] - P1[1]) * (Y - 2)
    yl = sp.solve(line, Y)[0]
    quad = sp.expand(X**2 - yl**2 - b)
    bsol = sp.solve(sp.discriminant(quad, X), b)
    ok1 = bsol == [12]
    # (2) 반복 작도 → 넓이 수열, 빗변 중점의 사분면 분류 → A1 과 A2+A3+A4 비교
    P, Q = P1, Q1
    areas, quads = [], []
    for _ in range(12):
        S = sp.Abs(P[0] * Q[1]) / 2
        mx, my = P[0] / 2, Q[1] / 2
        qd = 1 if (mx > 0 and my > 0) else 2 if (mx < 0 and my > 0) else 3 if (mx < 0 and my < 0) else 4
        areas.append(sp.nsimplify(S)); quads.append(qd)
        P, Q = perp_bisector_axes(P, Q)
    ratio = sp.simplify(areas[1] / areas[0])
    geo = all(sp.simplify(areas[i + 1] / areas[i] - ratio) == 0 for i in range(11))
    per = all(quads[i] == quads[i + 4] for i in range(8))
    r4 = ratio**4
    A = {k: sum(areas[i] for i in range(4) if quads[i] == k) / (1 - r4) for k in (1, 2, 3, 4)}
    claim = sp.simplify(A[1] - (A[2] + A[3] + A[4])) > 0
    ok2 = geo and per and ratio == sp.Rational(9, 16) and not bool(claim)
    report(1, ok1 and ok2, f"(1) b={bsol}  (2) 넓이비={ratio}, 사분면 순서={quads[:4]}, A1={A[1]}, A2+A3+A4={sp.simplify(A[2]+A[3]+A[4])} → 명제 {'참' if claim else '거짓'}")


def p02():
    # 앞면 16, 뒷면 3 인 길이 19 배열 전수 조사(뒷면 위치 선택 19C3)
    import itertools
    c1 = c2 = 0
    for pos in itertools.combinations(range(19), 3):
        s = ["H"] * 19
        for i in pos:
            s[i] = "T"
        w = "".join(s)
        if "TT" in w:
            continue
        c1 += 1
        runs = [r for r in w.split("T") if r]
        if all(len(r) >= 2 for r in runs):
            c2 += 1
    ok = c1 == 680 and c2 == 310
    report(2, ok, f"(1) 뒷면 비연속 {c1}  (2) 앞면 연속 2회 이상+뒷면 비연속 {c2} (전수 19C3)")


def p03():
    import random
    a, b = sp.symbols("a b", positive=True)
    x1, y1 = sp.symbols("x1 y1", real=True)
    A = sp.Matrix([0, 0, 2]); Q = sp.Matrix([x1, y1, 0])
    P = (1 * Q + 4 * A) / 5
    sol = sp.solve([P[0] - a, P[1] - a, P[2] - b, P[0]**2 + P[1]**2 + (P[2] - 1)**2 - 1], [a, b, x1, y1], dict=True)
    sol = [s for s in sol if s[a].is_positive and s[b].is_positive]
    s = sol[0]
    Pv = sp.Matrix([s[a], s[a], s[b]]); Qv = sp.Matrix([s[x1], s[y1], 0])
    ok1 = len(sol) == 1
    # (2) 구 위의 점 R 에 대해 정사영 넓이 최대화: 구면 매개화 격자 탐색
    import math
    Qn = [float(v) for v in Qv]
    best = (-1, None)
    N = 400
    for i in range(1, N):
        th = math.pi * i / N
        for j in range(N):
            ph = 2 * math.pi * j / N
            R = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), 1 + math.cos(th))
            ar = abs(R[0] * Qn[1] - R[1] * Qn[0]) / 2   # 정사영 삼각형 O R' Q 넓이
            if ar > best[0]:
                best = (ar, R)
    # 기호 계산: 최대일 때 R=(±1/√2·(1,-1)… ) → R' 가 OQ 에 수직인 단위벡터, z=1
    u = sp.Matrix([1, -1, 0]) / sp.sqrt(2)
    R = sp.Matrix([u[0], u[1], 1])
    proj = sp.Abs((R[0] * Qv[1] - R[1] * Qv[0])) / 2
    AR = (R - sp.Matrix([0, 0, 2])).norm(); RQ = (Qv - R).norm(); AQ = (Qv - sp.Matrix([0, 0, 2])).norm()
    tri = (sp.Matrix([R - sp.Matrix([0, 0, 2])]).reshape(3, 1)).cross(Qv - sp.Matrix([0, 0, 2])).norm() / 2
    cos = sp.nsimplify(sp.simplify(proj / tri))
    ok2 = abs(best[0] - float(proj)) < 1e-3 and sp.simplify(proj - 2) == 0 and sorted(map(sp.simplify, [AR, RQ, AQ]), key=float) == [sp.sqrt(2), 3 * sp.sqrt(2), 2 * sp.sqrt(5)] and cos == sp.Rational(2, 3)
    report(3, ok1 and ok2, f"(1) P={list(Pv)}, Q={list(Qv)}  (2) 격자최대 {best[0]:.5f}, 변 {AR},{RQ},{AQ}, cosθ={cos}")


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
