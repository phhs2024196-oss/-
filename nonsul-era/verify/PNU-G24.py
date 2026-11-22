#!/usr/bin/env python3
"""PNU-G24 (부산대 2024학년도 수시 논술전형 자연계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "PNU-G24"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/PNU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def n_distinct_real_roots(poly, x):
    import numpy as np
    coeffs = [float(c) for c in sp.Poly(sp.expand(poly), x).all_coeffs()]
    rs = np.roots(coeffs)
    reals = sorted(float(r.real) for r in rs if abs(r.imag) < 1e-6)
    out = []
    for r in reals:
        if not out or abs(r - out[-1]) > 1e-4:
            out.append(r)
    return out


def p01():
    x = sp.symbols("x", real=True)
    a, t = sp.symbols("a t", real=True)
    # f(1)=f'(1)=0 인 일대일대응 삼차함수 → f = a(x-1)^3 (중근 x=1, 극값 없음)
    # (1-1) 증가: f=f^{-1} 의 근 = y=x 와의 교점, 2개 → 접함 (기울기 1)
    s1 = [d for d in sp.solve([a * (t - 1)**3 - t, 3 * a * (t - 1)**2 - 1], [a, t], dict=True) if d[a] > 0]
    f1 = s1[0][a] * (x - 1)**3
    ok1 = len(s1) == 1 and sp.simplify(f1 - sp.sympify(md_answer(1, "1"), locals={"x": x})) == 0
    roots1 = n_distinct_real_roots(f1 - x, x)
    ok1 = ok1 and len(roots1) == 2
    # (1-2) 감소: 교점에서 f'(t) = 1/f'(t) → f'(t) = -1
    s2 = [d for d in sp.solve([a * (t - 1)**3 - t, 3 * a * (t - 1)**2 + 1], [a, t], dict=True) if d[a] < 0]
    f2 = s2[0][a] * (x - 1)**3
    ok2 = len(s2) == 1 and sp.simplify(f2 - sp.sympify(md_answer(1, "2"), locals={"x": x})) == 0
    # f(x)=f^{-1}(x) ⇔ f(f(x)) = x (f 전단사)
    roots2 = n_distinct_real_roots(f2.subs(x, f2) - x, x)
    finv = lambda y: 1 - sp.real_root(sp.Rational(27, 16) * y, 3) if y >= 0 else 1 + sp.real_root(-sp.Rational(27, 16) * y, 3)
    g = lambda v: f2.subs(x, v) - finv(v)
    ivt = g(-1) > 0 and g(0) < 0
    ok2 = ok2 and len(roots2) >= 3 and ivt and s2[0][t] == sp.Rational(1, 4)
    report(1, ok1 and ok2, f"(1-1) f={f1}, 교점 {len(roots1)}개 | (1-2) f={f2}, f=f^-1 실근 {[round(r,4) for r in roots2]}, g(-1)>0,g(0)<0={ivt} [증명-수치확인]")


def p02():
    t = sp.symbols("t", positive=True)
    X = sp.symbols("X", real=True)
    P = sp.Matrix([t, t**2 / 2])
    nrm = sp.Matrix([-t, 1]) / sp.sqrt(t**2 + 1)          # 접선 (1,t)에 수직, y성분 양(조건 ii)
    Q = P + nrm
    loc = {"t": t}
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "1").replace(" ", "").split(";"))
    ok1 = sp.simplify(Q[0] - sp.sympify(parts["f(t)"], locals=loc)) == 0 and \
        sp.simplify(Q[1] - sp.sympify(parts["g(t)"], locals=loc)) == 0
    slope = sp.simplify(sp.diff(Q[1], t) / sp.diff(Q[0], t))
    tang = slope * (X - Q[0]) + Q[1]
    rts = sp.solve(sp.Eq(X**2 / 2, tang), X)
    A, B = rts
    l = sp.sqrt((A - B)**2 + (A**2 / 2 - B**2 / 2)**2)
    lim = sp.limit(l / (t * sp.sqrt(t)), t, sp.oo)
    ok2 = sp.simplify(slope - t) == 0 and sp.simplify(lim - sp.sympify(md_answer(2, "2"))) == 0 and \
        abs(float((l - 2 * sp.sqrt(2) * (t**2 + 1)**sp.Rational(3, 4)).subs(t, 1.7))) < 1e-12
    I = sp.integrate(sp.simplify(Q[0] * sp.diff(Q[1], t) / t), (t, sp.sqrt(3), 2 * sp.sqrt(2)))
    Inum = sp.Integral(Q[0] * sp.diff(Q[1], t) / t, (t, sp.sqrt(3), 2 * sp.sqrt(2))).evalf(30)
    ans3 = sp.sympify(md_answer(2, "3"))
    ok3 = sp.simplify(I - ans3) == 0 and abs(Inum - ans3) < 1e-20
    report(2, ok1 and ok2 and ok3, f"Q=({sp.simplify(Q[0])}, {sp.simplify(Q[1])}) slope={slope} lim={lim} I={sp.nsimplify(I)}")


def p03():
    import numpy as np
    r3 = sp.sqrt(3)
    B = sp.Matrix([0, 0, 0]); C = sp.Matrix([4, 0, 0]); A = sp.Matrix([2, 2 * r3, 0])
    up = lambda p, h: p + sp.Matrix([0, 0, h])
    D, E, F = up(A, 8), up(B, 8), up(C, 8)
    G = F + (D - F) / 4                                  # FG = 1 (DF = 4)
    Gp = sp.Matrix([G[0], G[1], 0])                      # G 의 수선의 발
    H = (B + C) / 2
    u, v = sp.symbols("u v", real=True)
    sol = sp.solve(list(A + u * (H - A) - (B + v * (Gp - B)))[:2], [u, v], dict=True)[0]
    Pp = A + sol[u] * (H - A)
    tval = sp.simplify(sol[u] / (1 - sol[u]))           # AP':P'H
    on_AC = sp.simplify((Gp - A).cross(C - A).norm()) == 0
    ok1 = on_AC and sp.simplify(tval - sp.sympify(md_answer(3, "1"))) == 0
    # (3-2) P: BG 위, Q: HI 위, 둘 다 P' 위(수직) / I = A + (0,0,s), PQ = 26/7
    s = sp.symbols("s", positive=True)
    lamP = sol[v]                                        # BP'/BG'
    hP = lamP * 8
    lamQ = sp.simplify((Pp - H).norm() / (A - H).norm())  # HP'/HA
    hQ = lamQ * s
    sv = sp.solve(sp.Eq(hP - hQ, sp.Rational(26, 7)), s)[0]
    I = up(A, sv)
    # 넓이 13/7 = (1/2)(26/7) r → r = 1 : 직선 PQ 로부터 거리 1 (원기둥)
    r = sp.Rational(13, 7) * 2 / sp.Rational(26, 7)
    dist_line = lambda p, a1, b1: ((p - a1).cross(b1 - a1)).norm() / (b1 - a1).norm()
    inside = all(dist_line(Pp, a1, b1) > r for a1, b1 in [(A, B), (A, C)]) and (Pp - B).norm() > r and (Pp - Gp).norm() > r
    areaEGI = ((G - E).cross(I - E)).norm() / 2
    areaProj = ((Gp - B).cross(A - B)).norm() / 2
    cos = sp.nsimplify(sp.simplify(areaProj / areaEGI))
    S = sp.simplify((sp.pi * r**2 / 2) / cos)
    ok2 = inside and sp.simplify(S - sp.sympify(md_answer(3, "2"))) == 0 and sv == 6
    # 몬테카를로: 삼각형 EGI 안에서 직선 PQ 까지 거리 < 1 인 부분의 넓이 (= 곡선과 EG 로 둘러싸인 부분)
    rng = np.random.default_rng(7)
    N = 2_000_000
    a1, b1 = rng.random(N), rng.random(N)
    m = a1 + b1 > 1
    a1[m], b1[m] = 1 - a1[m], 1 - b1[m]
    e, g, i = (np.array([float(c) for c in M]) for M in (E, G, I))
    pts = e + np.outer(a1, g - e) + np.outer(b1, i - e)
    dxy = np.hypot(pts[:, 0] - float(Pp[0]), pts[:, 1] - float(Pp[1]))
    mc = float(areaEGI) * np.mean(dxy < 1)
    ok2 = ok2 and abs(mc - float(S)) / float(S) < 0.01
    report(3, ok1 and ok2, f"t={tval} AI={sv} cos={cos} S={S}≈{float(S):.5f} MC≈{mc:.5f}")


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
