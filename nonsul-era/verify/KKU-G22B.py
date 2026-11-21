#!/usr/bin/env python3
"""KKU-G22B (건국대 2022학년도 수시 논술 자연계 B 수학 문제 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KKU-G22B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
LOC = {"e": sp.E, "E": sp.E}


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")




def p01():
    """1-1: B(-1,0), D(1,0). 사인법칙 대신 외접원 중심을 (0,k) 로 두고 원 위 점 A 에서 sin∠BAD=1/2 가 되는 k 를 직접 구함
    (k 의 부호 두 가지), 같은 방법으로 BCD 원(sin∠BCD=1/3). 두 원 위 점 쌍 거리 최대를 격자+국소최적화로 수치 탐색.
    1-2: 꼭짓점 좌표로 넓이(신발끈) 직접 계산해 합."""
    import math
    import mpmath as mp
    kk = sp.symbols("k", real=True)
    B = sp.Matrix([-1, 0]); D = sp.Matrix([1, 0])

    def centers(sinv):
        out = []
        for sgn in (1, -1):
            # 원 위 점 A=(0, k + sgn*R) (BD 의 수직이등분선 위, 중심 기준 sgn 쪽)에서 sin∠BAD 를 계산해 sinv 와 같게
            Rr = sp.sqrt(1 + kk**2)
            A = sp.Matrix([0, kk + Rr]) if sgn == 1 else sp.Matrix([0, kk - Rr])
            u, v = B - A, D - A
            cross = sp.Abs(u[0]*v[1] - u[1]*v[0])
            sol = sp.nsolve(cross / (sp.sqrt(u.dot(u)) * sp.sqrt(v.dot(v))) - sinv, kk, 0.5 if sgn == 1 else -0.5)
            out.append(float(sol))
        allk = set()
        for k0 in out:
            allk.add(round(k0, 12)); allk.add(round(-k0, 12))
        return sorted(allk)

    best = 0
    for k1 in centers(sp.Rational(1, 2)):
        for k2 in centers(sp.Rational(1, 3)):
            R1, R2 = math.sqrt(1 + k1**2), math.sqrt(1 + k2**2)
            f = lambda a, b: math.hypot(R1*math.cos(a) - R2*math.cos(b), k1 + R1*math.sin(a) - k2 - R2*math.sin(b))
            M = 360
            g = max(((f(2*math.pi*i/M, 2*math.pi*j/M), i, j) for i in range(M) for j in range(M)))
            a0, b0 = 2*math.pi*g[1]/M, 2*math.pi*g[2]/M
            h = 2*math.pi/M
            for _ in range(40):
                cand = [(f(a0+da, b0+db), a0+da, b0+db) for da in (-h, 0, h) for db in (-h, 0, h)]
                _, a0, b0 = max(cand); h /= 1.6
            best = max(best, f(a0, b0))
    ans1 = float(sp.sympify(md_answer(1, "1")))
    ok1 = abs(best - ans1) < 1e-9
    S = 0
    for m in range(1, 101):
        P = [(0, -1), (m - 1, sp.cos((m - 1) * sp.pi / 2)), (m, sp.cos(m * sp.pi / 2))]
        area = sp.Abs((P[1][0]-P[0][0])*(P[2][1]-P[0][1]) - (P[2][0]-P[0][0])*(P[1][1]-P[0][1])) / 2
        S += area**2
    ok2 = S == sp.sympify(md_answer(1, "2"))
    report(1, ok1 and ok2, f"max PQ(수치)={best:.12f}; Σa_n^2={S}")


def p02():
    """2-1: B(-s,s^2), C(s,s^2), A(0,3). S(s)=∫(s^2-x^2)dx, θ(s)=acos(AB·AC/|AB||AC|), dS/dθ=S'(s)/θ'(s), BC=2 → s=1.
    2-2: D 를 원 위에서 돌리고 C=D±π/3(CD=2), P=AC∩BD 로 정해 'P 원 밖, C∈선분AP, D∈선분BP' 인 경우의 AP 최대를 수치 탐색."""
    import math
    s = sp.symbols("s", positive=True)
    S = sp.integrate(s**2 - x**2, (x, -s, s))
    AB = sp.Matrix([-s, s**2 - 3]); AC = sp.Matrix([s, s**2 - 3])
    th = sp.acos(AB.dot(AC) / (sp.sqrt(AB.dot(AB)) * sp.sqrt(AC.dot(AC))))
    sv = sp.solve(sp.Eq(2 * s, 2), s)[0]
    dSdth = sp.nsimplify(sp.simplify((sp.diff(S, s) / sp.diff(th, s)).subs(s, sv)))
    ok1 = sp.simplify(dSdth - sp.sympify(md_answer(2, "1"))) == 0

    R = 2.0
    A = (-math.sqrt(2), math.sqrt(2)); Bp = (math.sqrt(2), math.sqrt(2))

    def AP(dl, sgn):
        D = (R*math.cos(dl), R*math.sin(dl)); cl = dl + sgn*math.pi/3
        C = (R*math.cos(cl), R*math.sin(cl))
        d1 = (C[0]-A[0], C[1]-A[1]); d2 = (D[0]-Bp[0], D[1]-Bp[1])
        den = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(den) < 1e-12 or math.hypot(*d1) < 1e-9 or math.hypot(*d2) < 1e-9:
            return None
        u = ((Bp[0]-A[0])*d2[1] - (Bp[1]-A[1])*d2[0]) / den
        w = ((Bp[0]-A[0])*d1[1] - (Bp[1]-A[1])*d1[0]) / den
        P = (A[0]+u*d1[0], A[1]+u*d1[1])
        if not (u > 1 + 1e-12 and w > 1 + 1e-12 and math.hypot(*P) > R + 1e-12):
            return None
        return math.dist(A, P)

    best, arg = 0, None
    M = 200000
    for sgn in (1, -1):
        for i in range(M):
            v = AP(2*math.pi*i/M, sgn)
            if v is not None and v > best:
                best, arg = v, (2*math.pi*i/M, sgn)
    ans2 = float(sp.sympify(md_answer(2, "2")))
    ok2 = abs(best - ans2) < 1e-6 and best <= ans2 + 1e-9
    report(2, ok1 and ok2, f"dS/dθ={dSdth}; max AP(수치)={best:.9f} at D각={math.degrees(arg[0]):.3f}°")


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
