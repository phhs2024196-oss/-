#!/usr/bin/env python3
"""SJU-G20A (세종대 2020학년도 수시 논술고사[A] 자연계열) — sympy 검산. 정답 하드코딩 금지, 설정으로부터 독립 재계산."""
import re
import sys
from fractions import Fraction as Fr
from itertools import product, combinations

import sympy as sp

PREFIX = "SJU-G20A"
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
    tt = sp.Symbol("t", positive=True)
    F = sp.Function("F")
    gfun = 2 / tt - sp.log(tt)
    # (1) t=1 접선: (0,g(1)), (h(1),0)
    slope = (0 - gfun.subs(tt, 1)) / (sp.Rational(2, 3) - 0)
    f1 = gfun.subs(tt, 1) + slope * 1
    ok1 = same(f1, S(md_answer(1, "1")))
    # (2)(3) ODE f - t f' = g 를 직접 풀기 (dsolve) + 초기조건 f(1), f'(1)
    sol = sp.dsolve(sp.Eq(F(tt) - tt * F(tt).diff(tt), gfun), F(tt)).rhs
    C1 = [s for s in sol.free_symbols if s.name.startswith("C")][0]
    sol = sol.subs(C1, sp.solve(sol.subs(tt, 1) - f1, C1)[0])
    ok_slope = same(sp.diff(sol, tt).subs(tt, 1), slope)
    f2 = sp.simplify(sp.diff(sol, tt, 2))
    ok2 = sp.simplify(f2 - S(md_answer(1, "2")).subs(t, tt)) == 0
    ok3 = sp.simplify(sol - S(md_answer(1, "3")).subs(t, tt)) == 0
    hh = sp.simplify(tt - sol / sp.diff(sol, tt)).subs(tt, 1)            # x절편 검산
    report(1, ok1 and ok2 and ok3 and ok_slope and hh == sp.Rational(2, 3), f"f(1)={f1} f''={f2} f={sol} h(1)={hh}")


def p02():
    A = [-1, -1, -1, 0, 1, 1]
    B = [-1, 0, 0, 1]
    dist = {-1: Fr(0), 0: Fr(0), 1: Fr(0)}
    same_and_1 = Fr(0)
    for i, j in combinations(range(6), 2):
        pa = Fr(1, 15)
        box = B + [A[i], A[j]] if A[i] == A[j] else B
        for bb in box:
            dist[bb] += pa * Fr(1, len(box))
            if bb == 1 and A[i] == A[j]:
                same_and_1 += pa * Fr(1, len(box))
    p1 = dist[1]
    ok1 = same(sp.Rational(p1.numerator, p1.denominator), S(md_answer(2, "1")))
    c = same_and_1 / p1
    ok2 = same(sp.Rational(c.numerator, c.denominator), S(md_answer(2, "2")))
    pr = sum(dist[a] * dist[b] * dist[d] for a, b, d in product((-1, 0, 1), repeat=3) if Fr(a + b + d, 3) <= Fr(7, 10))
    ok3 = same(sp.Rational(pr.numerator, pr.denominator), S(md_answer(2, "3")))
    ok_alt = dist[-1] == Fr(53, 180) and dist[0] == Fr(41, 90)
    report(2, ok1 and ok2 and ok3 and ok_alt, f"P(X)={dict((k, str(v)) for k, v in dist.items())} P(same|X=1)={c} P(Xbar<=0.7)={pr}")


def p03():
    a, b, c, s = sp.symbols("a b c s", real=True)
    P = sp.Matrix([1, 0, 4]); Q = sp.Matrix([a, b, c])
    L = P + s * (Q - P)
    s0 = sp.solve(L[2], s)[0]
    yq = sp.simplify(L[1].subs(s, s0))
    ok1 = sp.simplify(yq - sp.sympify(md_answer(3, "1"), locals={"a": a, "b": b, "c": c})) == 0
    # (2) 접점 집합: 구 위 + (Q-C)·(Q-P)=0 의 차 → 평면; 수치로 접점 무작위 생성해 평면 확인
    X, Y, Z = sp.symbols("x y z", real=True)
    sph = X**2 + Y**2 + (Z - 2)**2 - 1
    perp = X * (X - 1) + Y * Y + (Z - 2) * (Z - 4)
    plane = sp.expand(perp - sph)
    mdpl = sp.sympify(md_answer(3, "2").replace("=", "-(") + ")", locals={"x": X, "y": Y, "z": Z})
    ok2 = sp.simplify(plane / sp.Poly(plane, X).coeffs()[0] - sp.expand(mdpl) / sp.Poly(sp.expand(mdpl), X).coeffs()[0]) == 0
    # (3) 곡선 C 위를 매개화(평면∩구)해 수치 최대
    import mpmath as mp
    n = mp.matrix([1, 0, 2]) / mp.sqrt(5)
    cen = mp.matrix([0, 0, 2]) + n * ((5 - 4) / mp.sqrt(5)) / 1          # 중심 (0,0,2) 에서 평면까지
    dist = abs(0 + 2 * 2 - 5) / mp.sqrt(5)
    rad = mp.sqrt(1 - dist**2)
    cen = mp.matrix([0, 0, 2]) + n * dist
    u = mp.matrix([0, 1, 0]); v = mp.matrix([2, 0, -1]) / mp.sqrt(5)
    pt = lambda th: cen + rad * (mp.cos(th) * u + mp.sin(th) * v)
    fy = lambda th: 4 * pt(th)[1] / (4 - pt(th)[2])
    th0 = max((2 * mp.pi * i / 20000 for i in range(20000)), key=fy)
    thm = mp.findroot(lambda q: mp.diff(fy, q), th0)
    Qm = pt(thm)
    md = S(md_answer(3, "3"))
    ok3 = all(abs(Qm[i] - float(md[i])) < 1e-9 for i in range(3)) and abs(sph.subs({X: md[0], Y: md[1], Z: md[2]})) == 0
    report(3, ok1 and ok2 and ok3, f"yQ'={yq} plane={plane}=0 Qmax=({mp.nstr(Qm[0],10)},{mp.nstr(Qm[1],10)},{mp.nstr(Qm[2],10)}) max y={mp.nstr(fy(thm),10)}")


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
