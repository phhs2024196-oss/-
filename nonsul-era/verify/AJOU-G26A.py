#!/usr/bin/env python3
"""AJOU-G26A (아주대 2026학년도 논술고사 자연계열(오전)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "AJOU-G26A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def md_pairs(no: int, sub: str) -> dict:
    """'k=3; limit=1/2' 또는 'radius: ..; circle: ..' 꼴을 사전으로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = re.split(r"[:=]", seg, maxsplit=1)
        out[k.strip()] = v.strip()
    return out


def md_set(no: int, sub: str) -> list:
    return [sp.sympify(s) for s in md_answer(no, sub).strip().strip("{}").split(",")]


def same(a, b, tol=1e-10) -> bool:
    return abs(complex(sp.N(a, 30)) - complex(sp.N(b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _tangent(f, a):
    return sp.diff(f, x).subs(x, a) * (x - a) + f.subs(x, a)


def _k_and_limit(expr, var, point, dirn="+"):
    """lim expr/var^k 이 0 이 아닌 유한값이 되는 k(정수 탐색)와 그 값."""
    for k in range(-3, 8):
        L = sp.limit(expr / var**k, var, point, dirn)
        if L.is_finite and L != 0:
            return k, L
    return None, None


def p01():
    tp = sp.symbols("t", positive=True)
    fR, fL = x**2 * (x + 1), -x**2 * (x + 1)
    # (1) 큰 t: -1-t<-1 → fL, -1+t>-1 → fR
    l1, l2 = _tangent(fL, -1 - tp), _tangent(fR, -1 + tp)
    a_t = sp.solve(sp.Eq(l1, l2), x)[0]
    ok1 = same(sp.limit(a_t, tp, sp.oo), sp.sympify(md_answer(1, "1-1")))
    # (2) 작은 t: ±t > -1 → fR
    l3, l4 = _tangent(fR, -tp), _tangent(fR, tp)
    qx = sp.solve(sp.Eq(l3, l4), x)[0]
    qy = l3.subs(x, qx)
    R, V = sp.solve(l3, x)[0], sp.solve(l4, x)[0]
    S = -(V - R) * qy / 2                      # 작은 t 에서 qy<0, V>R → 부호 보정한 넓이
    assert float(S.subs(tp, sp.Rational(1, 10))) > 0
    k, L = _k_and_limit(sp.simplify(S), tp, 0)
    d = md_pairs(1, "1-2")
    ok2 = k == int(d["k"]) and same(L, sp.sympify(d["limit"]))
    # (2-1) p=3,q=4, y=20
    g34 = sp.Piecewise((16 - (x - 3)**2, (x > -1) & (x < 7)), ((x - 3)**2 - 16, True))
    xs = [r for r in sp.solve((x - 3)**2 - 16 - 20, x)]
    B = sp.integrate(20 - g34, (x, min(xs), max(xs)))
    ok3 = same(B, sp.sympify(md_answer(1, "2-1")))
    # (2-2) 전수: P(6,9)와 정수점 (X,g(X)) 를 잇는 직선(기울기별 1개)에서 교점 수 세기, X∈[-80,120]
    def g(X):
        return 25 - (X - 10)**2 if 5 < X < 15 else (X - 10)**2 - 25
    def inter(m):
        pts = set()
        for s, lo_ok in ((-1, lambda v: 5 < v < 15), (1, lambda v: v <= 5 or v >= 15)):
            # s((x-10)^2-25) = 9 + m(x-6)
            for r in sp.solve(s * ((x - 10)**2 - 25) - 9 - m * (x - 6), x):
                if r.is_real and lo_ok(float(r)):
                    pts.add(sp.nsimplify(r))
        return pts
    slopes = {sp.Rational(g(X) - 9, X - 6) for X in range(-80, 121) if X != 6}
    cnt = 0
    for m in slopes:
        pts = inter(m)
        if len(pts) >= 3 and sum(1 for p_ in pts if p_.is_integer) >= 2:
            cnt += 1
    # 범위 밖 X 의 기울기는 -1 미만 또는 9 초과임을 확인(교점 2개뿐)
    edge = inter(sp.Rational(g(-80) - 9, -86)), inter(sp.Rational(g(120) - 9, 114))
    ok4 = cnt == int(sp.sympify(md_answer(1, "2-2"))) and all(len(e) == 2 for e in edge)
    # (3) T(m): x>=0 부분 넓이, 최소
    m = sp.symbols("m", positive=True)
    Px, Qx = sp.solve(sp.Eq(m * x, 4 - 2 * x), x)[0], sp.solve(sp.Eq(m * x, 5 * x - 10), x)[0]
    T = sp.integrate(4 - 2 * x - m * x, (x, 0, Px)) + sp.integrate(m * x - (4 - 2 * x), (x, Px, 2)) \
        + sp.integrate(m * x - (5 * x - 10), (x, 2, Qx))
    crit = [r for r in sp.solve(sp.diff(T, m), m) if r.is_real and 0 < float(r) < 5]
    import numpy as np
    grid = np.linspace(0.01, 4.99, 4999)
    Tf = sp.lambdify(m, T)
    mgrid = grid[int(np.argmin(Tf(grid)))]
    ok5 = len(crit) == 1 and same(crit[0], sp.sympify(md_answer(1, "3"))) and abs(mgrid - float(crit[0])) < 2e-3
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"lim a={sp.limit(a_t, tp, sp.oo)} k={k},L={L} B={B} lines={cnt} m*={sp.nsimplify(crit[0])} "
           f"[{ok1},{ok2},{ok3},{ok4},{ok5}]")


def p02():
    import numpy as np
    def f(tv):
        return -sp.Rational(1, 4) * tv - 2 if tv <= 2 else -sp.Rational(5, 2) + sp.sin(2 * sp.pi * tv)
    def g(tv):
        return sp.Integer(4) if tv <= 3 else 4 + sp.sin(2 * sp.pi * tv)
    # (1) AB^2 = c^2(1+f^2) = 25, c^2 = 10/4
    c2 = sp.Rational(10, 4)
    tt = sp.symbols("tt", real=True)
    sol1 = sp.solveset(sp.Eq(c2 * (1 + (-tt / 4 - 2)**2), 25), tt, sp.Interval(0, 2))
    sol2 = sp.solveset(sp.Eq(c2 * (1 + (-sp.Rational(5, 2) + sp.sin(2 * sp.pi * tt))**2), 25), tt,
                       sp.Interval.Lopen(2, 4))
    got = sorted(list(sol1) + list(sol2), key=float)
    ok1 = sorted(md_set(2, "1-1"), key=float) == got
    # (2) 외접원 반지름 R=abc/(4K), s→0+
    s = sp.symbols("s", positive=True)
    c = sp.sqrt(5)
    A, B = sp.Matrix([0, -f(0) * c]), sp.Matrix([c, 0])
    P = B + s / sp.sqrt(1 + 16) * sp.Matrix([1, g(0)])
    a_, b_, cc = (A - B).norm(), (P - B).norm(), (A - P).norm()
    K = ((A - B)[0] * (P - B)[1] - (A - B)[1] * (P - B)[0]) / 2
    R = a_ * b_ * cc / (4 * sp.Abs(K))
    ok2 = same(sp.limit(sp.simplify(R), s, 0, "+"), sp.sympify(md_answer(2, "1-2")))
    # (3) 예각 tan = |(f-g)/(1+fg)| 최소. 구간별로 정확히 최소화
    u = sp.symbols("u", real=True)                       # u = sin 2πt
    e1 = sp.simplify(((-tt / 4 - 2) - 4) / (1 + (-tt / 4 - 2) * 4))
    e2 = sp.simplify(((-sp.Rational(5, 2) + u) - 4) / (1 + (-sp.Rational(5, 2) + u) * 4))
    e3 = sp.simplify(((-sp.Rational(5, 2) + u) - (4 + u)) / (1 + (-sp.Rational(5, 2) + u) * (4 + u)))
    # 세 식 모두 해당 구간에서 양수(예각) — 수치 확인
    assert all(float(e1.subs(tt, v)) > 0 for v in np.linspace(0, 2, 41))
    assert all(float(e.subs(u, v)) > 0 for e in (e2, e3) for v in np.linspace(-1, 1, 41))
    m1 = sp.minimum(e1, tt, sp.Interval(0, 2))
    m2 = sp.minimum(e2, u, sp.Interval(-1, 1))           # t∈(2,3] 에서 sin2πt 는 [-1,1] 전체
    m3 = sp.minimum(e3, u, sp.Interval(-1, 1))           # t∈(3,4] 도 한 주기
    exact = min([m1, m2, m3], key=float)
    ts = np.linspace(0, 4, 400001)
    num = min(abs(float((f(v) - g(v)) / (1 + f(v) * g(v)))) for v in ts[::400])
    ok3 = same(exact, sp.sympify(md_answer(2, "1-3"))) and abs(num - float(exact)) < 1e-3
    # (2-1) 호 Q1Q2
    def Q(mv):
        mv = sp.nsimplify(mv)
        return sp.Matrix([10 * mv / (1 + mv**2), 5 * (mv**2 - 1) / (1 + mv**2)])
    q1, q2 = Q(2), Q(5 * sp.sqrt(3) - 8)
    ang = sp.acos(sp.simplify(q1.dot(q2) / 25))
    ok4 = same(5 * ang, sp.sympify(md_answer(2, "2-1")))
    # (2-2) S(m) = 접선 x절편·y절편 /2, m→∞
    mm = sp.symbols("m", positive=True)
    Qm = sp.Matrix([10 * mm / (1 + mm**2), 5 * (mm**2 - 1) / (1 + mm**2)])
    assert sp.simplify(Qm.dot(Qm) - 25) == 0 and sp.simplify(Qm[1] - (mm * Qm[0] - 5)) == 0
    Sm = sp.simplify((25 / Qm[0]) * (25 / Qm[1]) / 2)
    k, L = _k_and_limit(Sm, mm, sp.oo, "-")
    d = md_pairs(2, "2-2")
    ok5 = k == int(d["alpha"]) and same(L, sp.sympify(d["limit"]))
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"t={got} limR={sp.limit(sp.simplify(R), s, 0, '+')} tanθ={exact} arc={sp.nsimplify(5*ang)} α={k},L={L} "
           f"[{ok1},{ok2},{ok3},{ok4},{ok5}]")


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
