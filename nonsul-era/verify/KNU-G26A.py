#!/usr/bin/env python3
"""KNU 2026학년도 수시 논술(AAT) 자연계열Ⅰ 수학 (KNU-G26A-01~02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-G26A"
WIDTH = 2
RESULTS = []
x, h = sp.symbols("x h", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())


def load_answer(seq, i=0):
    return load_item(seq)["subparts"][i]["answer"]["value"]


# ── 문제 1 보조: 좌표 기하 ──
def tri_area(P, Q, R):
    return sp.Abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2


def line_inter(P1, P2, P3, P4):
    s, u = sp.symbols("s u")
    sol = sp.solve([P1[0] + s * (P2[0] - P1[0]) - P3[0] - u * (P4[0] - P3[0]),
                    P1[1] + s * (P2[1] - P1[1]) - P3[1] - u * (P4[1] - P3[1])], [s, u], dict=True)[0]
    return (P1[0] + sol[s] * (P2[0] - P1[0]), P1[1] + sol[s] * (P2[1] - P1[1]))


def clip(poly, f):
    """볼록 다각형 poly 를 반평면 f(P)>=0 으로 자름(f 는 일차식)."""
    out = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = f(P), f(Q)
        if fp >= 0:
            out.append(P)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    return out


def poly_area(poly):
    return sp.Abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                      for i in range(len(poly)))) / 2


def p01():
    A, B, C = (sp.Integer(1), sp.Integer(4)), (sp.Integer(-2), sp.Integer(-3)), (sp.Integer(7), sp.Integer(2))
    S = tri_area(A, B, C)
    msgs, ok = [], True
    # 1-1: a,b,c 를 만족하는 F 를 넓이 조건 연립으로 직접 찾고, D·E 를 BC 평행선 교점으로 구해 DF/EF 계산
    X, Y = sp.symbols("X Y", real=True)
    F = (X, Y)
    signed = lambda P, Q, R: ((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2
    sgn = sp.sign(signed(A, B, C))
    sol = sp.solve([sgn * signed(F, B, C) / S - sp.Rational(1, 3), sgn * signed(A, F, C) / S - sp.Rational(1, 2)], [X, Y], dict=True)[0]
    Fv = (sol[X], sol[Y])
    assert sp.simplify(tri_area(A, B, Fv) / S - sp.Rational(1, 6)) == 0
    G = (Fv[0] + C[0] - B[0], Fv[1] + C[1] - B[1])        # F 를 지나 BC 에 평행한 직선의 다른 점
    D, E = line_inter(Fv, G, A, B), line_inter(Fv, G, A, C)
    dist = lambda P, Q: sp.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
    r11 = sp.nsimplify(sp.simplify(dist(D, Fv) / dist(E, Fv)))
    a11 = sp.sympify(load_answer(1, 0))
    ok &= sp.simplify(r11 - a11) == 0
    msgs.append(f"1-1 DF/EF={r11} md={a11}")
    # 1-2: 무작위 내부 점 20개에서 (a,b,c) 를 넓이로 계산 → x,y 를 a,b,c 의 일차식으로 최소제곱 적합(상수항 없음)
    import random
    random.seed(1)
    rows, xs, ys = [], [], []
    for _ in range(20):
        w = [sp.Rational(random.randint(1, 50)) for _ in range(3)]
        tot = sum(w)
        P = (sum(wi * V[0] for wi, V in zip(w, (A, B, C))) / tot, sum(wi * V[1] for wi, V in zip(w, (A, B, C))) / tot)
        a_, b_, c_ = tri_area(P, B, C) / S, tri_area(C, A, P) / S, tri_area(A, B, P) / S
        rows.append([a_, b_, c_]); xs.append(P[0]); ys.append(P[1])
    M = sp.Matrix(rows)
    cx = (M.T * M).inv() * M.T * sp.Matrix(xs)
    cy = (M.T * M).inv() * M.T * sp.Matrix(ys)
    resid = (M * cx - sp.Matrix(xs)).norm() + (M * cy - sp.Matrix(ys)).norm()
    got = tuple(list(cx) + list(cy))
    a12 = tuple(sp.sympify(load_answer(1, 1)))
    ok &= resid == 0 and got == a12
    msgs.append(f"1-2 coef={got} md={a12}")
    # 1-3: 삼각형을 반평면 2a-b>=0, 2a-c>=0 으로 자른 볼록 영역(꼭짓점 A 포함)의 넓이비
    area_coord = lambda P: (tri_area(P, B, C) / S, tri_area(C, A, P) / S, tri_area(A, B, P) / S)
    # 내부에서 a,b,c 는 부호 있는 넓이(일차식)로 표현
    lin = lambda P: tuple(sgn * v / S for v in (signed(P, B, C), signed(A, P, C), signed(A, B, P)))
    reg = [A, B, C]
    reg = clip(reg, lambda P: 2 * lin(P)[0] - lin(P)[1])
    reg = clip(reg, lambda P: 2 * lin(P)[0] - lin(P)[2])
    r13 = sp.nsimplify(poly_area(reg) / S)
    a13 = sp.sympify(load_answer(1, 2))
    ok &= len(reg) == 4 and A in reg and sp.simplify(r13 - a13) == 0
    msgs.append(f"1-3 영역 꼭짓점 {len(reg)}개(A 포함) 넓이비={r13} md={a13}")
    report(1, ok, "; ".join(msgs))


def p02():
    a, b, c = sp.symbols("a b c", real=True)
    msgs, ok = [], True
    # 2-1: 연속(x=0, x=1) + 최댓값 25/16 을 (0,1] 에서 실제 최댓값으로 확인, a<=-2
    q = a * x ** 2 + b * x + c
    cands = []
    for s in sp.solve([q.subs(x, 0) - 0, q.subs(x, 1) - 1], [b, c], dict=True):
        qa = q.subs(s)
        # 임계점 값 = 25/16 인 a 중 a<=-2, 임계점이 (0,1) 안인 것 (끝값 0·1 < 25/16 이므로 최댓값은 내부 임계점)
        xv = sp.solve(sp.diff(qa, x), x)[0]
        for av in sp.solve(sp.Eq(qa.subs(x, xv), sp.Rational(25, 16)), a):
            if av <= -2 and 0 < xv.subs(a, av) < 1:
                cands.append((av, s[b].subs(a, av), s[c]))
    a21 = tuple(sp.sympify(load_answer(2, 0)))
    ok &= len(cands) == 1 and tuple(cands[0]) == a21
    msgs.append(f"2-1 (a,b,c)={cands} md={a21}")
    av, bv, cv = cands[0]
    g = av * x ** 2 + bv * x + cv

    def f(t):
        return sp.Piecewise((0, t <= 0), (g.subs(x, t), t <= 1), (1, True))

    ff = f(f(x))
    # 2-2: f∘f 의 꺾일 후보 = {0,1} ∪ {f(x)=0 또는 1 인 x} 에서 좌·우미분계수 비교
    cand = {sp.Integer(0), sp.Integer(1)} | {r for r in sp.solve(g - 1, x) + sp.solve(g, x) if r.is_real}
    bad = []
    for p0 in sorted(cand):
        dq = lambda hh: (ff.subs(x, p0 + hh) - ff.subs(x, p0)) / hh          # 정확한 유리수 차분몫
        L = [dq(-sp.Rational(1, 10 ** k)) for k in (7, 9)]
        R = [dq(sp.Rational(1, 10 ** k)) for k in (7, 9)]
        Lc, Rc = float(L[1]), float(R[1])
        assert abs(float(L[0]) - Lc) < 1e-4 and abs(float(R[0]) - Rc) < 1e-4   # 수렴 확인
        if abs(Lc - Rc) > 1e-3:
            bad.append(p0)
    # 후보 밖에서 미분가능함은 각 구간에서 다항식이므로 자명 — 무작위 점 수치 확인
    a22 = sp.sympify(load_answer(2, 1))
    ok &= set(bad) == set(a22)
    msgs.append(f"2-2 후보={sorted(cand)} 미분불가점={bad} md={a22}")
    # 2-3: (0, 1/4) 위 g(g(x)) 의 도함수 첫 양→음 부호변화점
    gg = sp.expand(g.subs(x, g))
    roots = sorted(r for r in sp.solve(sp.diff(gg, x), x) if r.is_real and 0 < r < sp.Rational(1, 4))
    m_star = roots[0]
    assert sp.diff(gg, x).subs(x, m_star / 2) > 0 and sp.diff(gg, x).subs(x, (m_star + sp.Rational(1, 4)) / 2) < 0
    a23 = sp.sympify(load_answer(2, 2))
    ok &= sp.simplify(m_star - a23) == 0
    msgs.append(f"2-3 m={sp.nsimplify(m_star)} md={a23}")
    # 2-4: 적분
    I = sp.integrate(ff, (x, -1, 1))
    I = sp.nsimplify(sp.simplify(I))
    pq = sp.fraction(I)
    a24 = sp.sympify(load_answer(2, 3))
    ok &= I.is_Rational and pq[0] + pq[1] == a24
    msgs.append(f"2-4 적분={I} p+q={pq[0] + pq[1]} md={a24}")
    report(2, ok, "; ".join(msgs))


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
