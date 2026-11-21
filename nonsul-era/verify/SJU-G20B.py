#!/usr/bin/env python3
"""SJU-G20B (세종대 2020학년도 수시 논술고사[B] 자연계열) — sympy 검산. 정답 하드코딩 금지, 설정으로부터 독립 재계산."""
import re
import sys
from fractions import Fraction as Fr
from itertools import product, combinations

import sympy as sp

PREFIX = "SJU-G20B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(no: int, sub: str) -> dict:
    """'k1 = v1; k2 = v2' 형식의 정답을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def S(s: str):
    return sp.sympify(s, locals={"E": sp.E, "e": sp.E, "x": x, "t": t, "M": sp.Symbol("M", positive=True)})


def same(a, b, tol=1e-12) -> bool:
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 or abs(complex(sp.N(a - b, 30))) < tol
    except Exception:  # noqa: BLE001
        return abs(complex(sp.N(a - b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    xx = sp.Symbol("x", positive=True)
    r = (3 * xx**2 - 3 * xx + 7) / (xx**2 + 1)
    # (나)(다)로부터 f, g 를 직접 결정
    g = 1 / sp.sqrt(1 + r**2)
    f = r * g
    ok0 = sp.simplify(f**2 + g**2 - 1) == 0
    lim1 = sp.limit(sp.diff(f, xx) / sp.diff(g, xx), xx, 3)
    ok1 = same(lim1, S(md_answer(1, "1")))
    lim2 = sp.limit(f, xx, sp.oo)
    ok2 = same(lim2, S(md_answer(1, "2")))
    crit = [c for c in sp.solve(sp.diff(g, xx), xx) if c.is_positive]
    gmax = max(g.subs(xx, c) for c in crit)
    grid = max(float(g.subs(xx, i / 100)) for i in range(1, 3000))
    ok3 = same(gmax, S(md_answer(1, "3"))) and grid <= float(gmax) + 1e-12
    report(1, ok0 and ok1 and ok2 and ok3, f"lim f'/g'={lim1} lim f={lim2} max g={sp.nsimplify(gmax)} at {crit}")


def p02():
    import mpmath as mp
    X, Y = sp.symbols("X Y", real=True)
    a = sp.Symbol("a", positive=True)
    # (1) 원 C 까지 최단거리 = |중심거리 - 1/2| ; 격자에서 y≥0 해 찾기 → 수치로 f(x) 와 비교
    fexpr = sp.sqrt(x**2 / 3 + 1) - 1
    okf = True
    for xv in [-5, -1.3, 0, 0.7, 4, 20]:
        fv = mp.findroot(lambda yv: abs(mp.sqrt(xv**2 + (yv - 1)**2) - 0.5) - 2 * (yv + 0.25), 1 + abs(xv))
        okf = okf and abs(fv - float(fexpr.subs(x, xv))) < 1e-9
    # 점근선: 기울기 = lim f/x, 절편 = lim f - m x
    m1 = sp.limit(fexpr / x, x, sp.oo); c1 = sp.limit(fexpr - m1 * x, x, sp.oo)
    m2 = sp.limit(fexpr / x, x, -sp.oo); c2 = sp.limit(fexpr - m2 * x, x, -sp.oo)
    mine = {(sp.nsimplify(m1), c1), (sp.nsimplify(m2), c2)}
    md = set()
    for seg in md_answer(2, "1").split(";"):
        rhs = S(seg.split("=", 1)[1])
        md.add((sp.nsimplify(sp.diff(rhs, x)), rhs.subs(x, 0)))
    ok1 = okf and mine == md
    # (2) g 를 조건에서 수치 확인 후 적분
    gexpr = sp.sqrt((x - a)**2 / 3 + 1) + sp.Rational(1, 4)
    okg = True
    for av, xv in [(1.0, 0.3), (2.5, -1.0), (0.4, 3.0)]:
        gv = mp.findroot(lambda yv: mp.sqrt((xv - av)**2 + (yv + 1.75)**2) - 0.5 - 2 * yv, 2)
        okg = okg and abs(gv - float(gexpr.subs({a: av, x: xv}))) < 1e-9
    I = sp.simplify(sp.integrate((x - a) * gexpr, (x, a, a + 1)))
    ok2 = okg and same(I, S(md_answer(2, "2")))
    # (3) a 의 최댓값: min_x (g-f) >= 0 를 이분법으로 (수치)
    def ok_a(av):
        h = sp.lambdify(x, gexpr.subs(a, av) - fexpr)
        return min(h(v) for v in [i * 0.5 for i in range(-200, 20001)]) >= -1e-12 and \
            float(sp.limit(gexpr.subs(a, av) - fexpr, x, sp.oo)) >= -1e-12
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok_a(mid) else (lo, mid)
    amax_exact = sp.solve(sp.limit(gexpr - fexpr, x, sp.oo), a)[0]
    ok3 = abs(lo - float(amax_exact)) < 1e-6 and same(amax_exact, S(md_answer(2, "3")))
    report(2, ok1 and ok2 and ok3, f"asym={mine} int={I} a_max={amax_exact} (bisect {lo:.8f})")


def p03():
    import mpmath as mp
    O = sp.Matrix([-2, 1, 0]); nrm = sp.Matrix([2, -1, 2]); R2 = 16
    tt = sp.Symbol("tt")
    Cp = O + tt * nrm
    t0 = sp.solve(nrm.dot(Cp) - 4, tt)[0]
    C = Cp.subs(tt, t0)
    ok1 = list(C) == list(S(md_answer(3, "1")))
    d2 = (nrm.dot(O) - 4)**2 / nrm.dot(nrm)
    rad = sp.sqrt(R2 - d2)
    u = sp.Matrix([1, 2, 0]) / sp.sqrt(5)                 # 평면 안 수평 단위벡터
    v = nrm.cross(u) / sp.sqrt(nrm.dot(nrm))
    th = sp.Symbol("th", real=True)
    P = C + rad * (sp.cos(th) * u + sp.sin(th) * v)
    D2 = sp.simplify(P[0]**2 + P[1]**2 - C[0]**2 - C[1]**2)   # C'=(C0,C1)
    D2 = sp.simplify((P[0] - C[0])**2 + (P[1] - C[1])**2)
    f = sp.lambdify(th, D2, "mpmath")
    grid = [2 * mp.pi * i / 3600 for i in range(3600)]
    vals = [f(q) for q in grid]
    mx, mn = max(vals), min(vals)
    def pts(target):
        out = set()
        for q, val in zip(grid, vals):
            if abs(val - target) < 1e-12:
                qq = mp.findroot(lambda z: mp.diff(f, z), q)
                Pq = [float(sp.N(P[i].subs(th, float(qq)))) for i in range(2)]
                out.add((round(Pq[0], 9), round(Pq[1], 9)))
        return out
    pmax, pmin = pts(mx), pts(mn)
    mdmax = {(round(float(p[0]), 9), round(float(p[1]), 9)) for p in S(md_answer(3, "2"))}
    mdmin = {(round(float(p[0]), 9), round(float(p[1]), 9)) for p in S(md_answer(3, "3"))}
    ok2 = pmax == mdmax
    ok3 = pmin == mdmin
    report(3, ok1 and ok2 and ok3, f"C={list(C)} r={rad} max={pmax} min={pmin}")


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
