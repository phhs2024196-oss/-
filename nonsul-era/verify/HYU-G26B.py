#!/usr/bin/env python3
"""HYU-G26B (한양대 2026학년도 논술 자연계열 오후1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "HYU-G26B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def parts(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 1. A1=(s1,2^s1), A2=(s1, 2^(s1-3)+35), A3 = A2 에서 y축 방향 수선과 y=2^x 의 교점. A2A3=1.
    s1 = sp.symbols("s1", real=True)
    yA2 = 2**(s1 - 3) + 35
    x3 = sp.log(yA2, 2)
    w = sp.symbols("w", positive=True)                       # w = 2^{s1}
    # A2A3=1 ⇔ x3 = s1-1 ⇔ 2^(s1-1) = 2^(s1-3)+35
    wv = sp.solve(sp.Eq(w / 2, w / 8 + 35), w)[0]
    s1v = sp.log(wv, 2)
    ok_s = abs(float(s1v - 1 - x3.subs(s1, s1v))) < 1e-12 and float(s1v) >= 6
    s2v = s1v - 1
    area = sp.integrate(2**x - yA2.subs(s1, s1v), (x, s2v, s1v))    # 곡선과 수평선 y=y(A2) 사이
    area = sp.simplify(area)
    ok1 = ok_s and abs(float(area - sp.sympify(md_answer(1, "1")))) < 1e-12
    # 2. 점의 작도를 그대로 반복: A_{2k-1}=(x,2^x) → A_{2k}=(x, 2^(x-3)+35) → A_{2k+1}=(log2 y, y)
    y1 = sp.symbols("y1", positive=True)
    y = y1
    for _ in range(4):
        xk = sp.log(y, 2)
        y = sp.simplify(sp.expand_log(2**(xk - 3), force=True) + 35)
    sols = sp.solve(sp.Eq(y, sp.Rational(10725, 256)), y1)
    ok2 = len(sols) == 1 and sp.log(sols[0], 2) >= 6 and sols[0] == sp.sympify(md_answer(1, "2"))
    # 3. a=cosφ, b=sinφ 격자 탐색 + 조건 (1)(2)(구간별 부호 일정) 확인 + 수치적분
    ts1 = np.linspace(0, np.pi, 4001)[1:-1]
    ts2 = ts1 + np.pi
    tt = np.linspace(0, 2 * np.pi, 20001)
    best = []
    for ph in np.linspace(0, 2 * np.pi, 20000, endpoint=False):
        a, b = np.cos(ph), np.sin(ph)
        g = lambda u: a * np.sin(u) + b * np.abs(np.sin(2 * u))
        g1, g2 = g(ts1), g(ts2)
        if not ((g1 >= -1e-12).all() or (g1 <= 1e-12).all()):
            continue
        if not ((g2 >= -1e-12).all() or (g2 <= 1e-12).all()):
            continue
        val = 1013 * np.trapezoid(np.abs(g(tt)), tt)            # |g| 의 주기 2π
        best.append((val, a, b))
    vmin = min(v for v, _, _ in best)
    near = [(a, b) for v, a, b in best if v < vmin + 1e-3]
    md = parts(md_answer(1, "3"))
    mdmin = sp.sympify(md["min"])
    mdpairs = [tuple(map(sp.sympify, p)) for p in sp.sympify(md["pairs"])]
    ok3 = abs(vmin - float(mdmin)) < 1.0                    # 격자 간격(φ)에 따른 오차; 정확값은 아래에서 확인
    # 격자 최소점이 md 의 네 순서쌍 근처에만 있는지
    ok3 = ok3 and all(min(abs(a - float(p[0])) + abs(b - float(p[1])) for p in mdpairs) < 5e-3 for a, b in near)
    ok3 = ok3 and all(any(abs(a - float(p[0])) + abs(b - float(p[1])) < 5e-3 for a, b in near) for p in mdpairs)
    # md 순서쌍에서의 정확한 값(고정밀 적분)과 조건 확인
    mp.mp.dps = 30
    for pa, pb in mdpairs:
        A, B = mp.mpf(sp.N(pa, 40)), mp.mpf(sp.N(pb, 40))
        gg = lambda u: A * mp.sin(u) + B * abs(mp.sin(2 * u))
        I = 1013 * mp.quad(lambda u: abs(gg(u)), [k * mp.pi / 2 for k in range(5)])
        s1_ = [gg(mp.pi * k / 400) for k in range(1, 400)]
        s2_ = [gg(mp.pi + mp.pi * k / 400) for k in range(1, 400)]
        cond = (all(v >= -1e-25 for v in s1_) or all(v <= 1e-25 for v in s1_)) and \
            (all(v >= -1e-25 for v in s2_) or all(v <= 1e-25 for v in s2_))
        ok3 = ok3 and cond and abs(I - mp.mpf(sp.N(mdmin, 40))) < 1e-20 and sp.simplify(pa**2 + pb**2 - 1) == 0
    report(1, ok1 and ok2 and ok3,
           f"s1=log2(280/3) area={area} y1={sols} min~{vmin:.6f} argmin-count={len(near)}")


def p02():
    a = sp.symbols("a", positive=True)
    s3 = sp.sqrt(3)
    B, C, D = sp.Matrix([-3, -s3, 0]), sp.Matrix([3, -s3, 0]), sp.Matrix([0, 2 * s3, 0])
    O = sp.Matrix([0, 0, 0])
    h = sp.sqrt(a**2 - 12)
    A = sp.Matrix([0, 0, h])
    ok0 = all(sp.simplify(sp.sqrt((A - V).dot(A - V)) - a) == 0 for V in (B, C, D)) and \
        all(sp.simplify(sp.sqrt((U - V).dot(U - V)) - 6) == 0 for U, V in ((B, C), (C, D), (D, B)))
    u = sp.symbols("u")
    P = C + u * (D - C)
    P = P.subs(u, sp.solve(P[1], u)[0])                    # O 를 지나고 BC(x축)에 평행 → y=0
    tri = lambda U, V, W: sp.sqrt(((V - U).cross(W - U)).dot((V - U).cross(W - U))) / 2
    prod = sp.simplify(tri(D, O, P) * tri(A, O, D) * tri(A, O, P) * tri(A, P, D))
    md1 = sp.sympify(md_answer(2, "1"), locals={"a": a})
    ok1 = ok0 and all(abs(float((prod - md1).subs(a, v))) < 1e-9 for v in (3.6, 4, 5.5, 9))
    # 2. 평면 EFGH: 법선 z, 평면 FJKG: 법선 x(IJ∥BC 에 수직인 옆면), 평면 EIJF: 법선 y
    nACD = (C - A).cross(D - A)
    m = sp.symbols("m", positive=True)                       # m = AO = sqrt(a^2-12) > 0
    nm = nACD.subs(a, sp.sqrt(m**2 + 12))
    cosv = [sp.simplify(sp.sqrt(nm[i]**2) / sp.sqrt(nm.dot(nm))) for i in (2, 0, 1)]   # θ1, θ2, θ3
    target = sp.sqrt(6) / 3
    sols = set()
    for i in range(3):
        for r in sp.solve(sp.Eq(cosv[i], target), m):
            if all(float(cosv[i].subs(m, r)) >= float(c.subs(m, r)) - 1e-12 for c in cosv):
                sols.add(sp.nsimplify(sp.sqrt(r**2 + 12)))
    md2 = set(sp.sympify(md_answer(2, "2").replace("{", "[").replace("}", "]")))
    ok2 = sols == {sp.nsimplify(v) for v in md2}
    # 3. a=4: 정육면체 한 변 s, 앞면 y=y0. E,F(위 앞) ∈ 평면 ABC, G,H(위 뒤) ∈ ACD, ABD
    s, y0 = sp.symbols("s y0", real=True)
    A4 = A.subs(a, 4)
    plane = lambda U, V, W, X: ((V - U).cross(W - U)).dot(X - U)
    E = sp.Matrix([-s / 2, y0, s]); F = sp.Matrix([s / 2, y0, s])
    G = sp.Matrix([s / 2, y0 + s, s]); H = sp.Matrix([-s / 2, y0 + s, s])
    I_ = sp.Matrix([-s / 2, y0, 0]); J = sp.Matrix([s / 2, y0, 0])
    sol = [d for d in sp.solve([plane(A4, B, C, F), plane(A4, C, D, G)], [s, y0], dict=True) if d[s] > 0]
    sv = sp.nsimplify(sp.simplify(sol[0][s]))
    sub = sol[0]
    ok_h = sp.simplify(plane(A4, B, D, H).subs(sub)) == 0 and sp.simplify(plane(A4, B, C, E).subs(sub)) == 0
    # 정사영: EIJF 네 꼭짓점을 평면 ACD 에 사영해 넓이 계산
    nn = (C - A4).cross(D - A4)
    nn = nn / nn.norm()
    proj = lambda X: X - nn * nn.dot(X)
    Ep, Ip, Jp, Fp = (proj(X.subs(sub)) for X in (E, I_, J, F))
    v1, v2 = Ip - Ep, Fp - Ep                                 # EIJF 는 평행사변형(정사각형의 사영)
    area = sp.nsimplify(sp.simplify(v1.cross(v2).norm()))
    ok3 = ok_h and abs(float(area - sp.sympify(md_answer(2, "3")))) < 1e-12
    report(2, ok1 and ok2 and ok3, f"P={list(P)} prod={prod} a={sols} side={sv} proj={area}")


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
