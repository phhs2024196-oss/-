#!/usr/bin/env python3
"""EWHA-G24A (이화여대 2024학년도 수시 논술 자연계열Ⅱ) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import pathlib
import re
import sys

import mpmath as mp
import sympy as sp
import yaml

PREFIX = "EWHA-G24B"
RESULTS = []
mp.mp.dps = 30
x = sp.symbols("x", real=True)


def md_answer(no, sub):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:02d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:02d} {'PASS' if ok else 'FAIL'}  {msg}")


def I(c, sgn=1):
    """∫_{-π/2}^{π/2} ln(c^2+1+2·sgn·c·sinθ) dθ 고정밀 수치적분"""
    return mp.quad(lambda t: mp.log(c**2 + 1 + 2 * sgn * c * mp.sin(t)), [-mp.pi / 2, mp.pi / 2])


def p01():
    avals = [mp.mpf(v) for v in ("-0.93", "-0.6", "-0.2", "0", "0.35", "0.77", "0.95")]
    ok1 = all(abs(I(a, -1) - I(a, 1)) < 1e-15 for a in avals)                       # (1) [증명-수치확인]
    ok2 = all(abs(I(a) - I(a ** (2**n)) / 2**n) < 1e-15 for a in avals for n in range(1, 7))   # (2)
    ok3 = all(2 * mp.pi * mp.log(1 - a ** (2**n)) - 1e-20 <= I(a ** (2**n)) <= 2 * mp.pi * mp.log(1 + a ** (2**n)) + 1e-20
              for a in avals for n in range(1, 8))                                    # (3)
    val = max(abs(I(a)) for a in avals)                                               # (4)
    ok4 = val < 1e-15 and sp.nsimplify(md_answer(1, "4")["value"]) == 0
    kinds = all(md_answer(1, s)["kind"] == "proof" for s in "123")
    report(1, ok1 and ok2 and ok3 and ok4 and kinds,
           f"(1)~(3) [증명-수치확인] a 7점·n=1..7, (4) max|I(a)|={mp.nstr(val, 3)}")


def p02():
    f = x**2 * sp.exp(x)
    fp = sp.diff(f, x)
    crit = sp.solve(sp.Eq(fp, 0), x)
    f2 = sp.diff(f, x, 2)
    mx = [f.subs(x, c) for c in crit if f2.subs(x, c) < 0]
    mn = [f.subs(x, c) for c in crit if f2.subs(x, c) > 0]
    ans1 = dict(seg.split("=") for seg in md_answer(2, "1")["value"].replace(" ", "").split(";"))
    ok1 = sp.simplify(sp.sympify(ans1["max"]) - mx[0]) == 0 and sp.simplify(sp.sympify(ans1["min"]) - mn[0]) == 0
    # (2) 정수 전수(-300..300): f(n) < c < f(n+1)
    def sols(c):
        return [n for n in range(-300, 301) if sp.N(f.subs(x, n) - c, 50) < 0 < sp.N(f.subs(x, n + 1) - c, 50)]
    sa, sb = sols(4 * sp.exp(-2)), sols(sp.exp(-1))
    md = dict(seg.split("=") for seg in md_answer(2, "2")["value"].replace(" ", "").split(";"))
    ok2 = sa == sorted(sp.sympify(md["a"])) and sb == sorted(sp.sympify(md["b"]))
    # (3) k>0 에서 f(x)=k 의 실근이 모두 정수인 최소 k
    cands = sorted({f.subs(x, n) for n in range(-40, 6) if f.subs(x, n) > 0}, key=float)
    best = None
    g0 = sp.lambdify(x, f, "mpmath")
    for k in cands:
        rts = []
        for lo, hi in [(-60, -2), (-2, 0), (0, 10)]:
            g = lambda u, k=k: g0(u) - mp.mpf(sp.N(k, 40))
            a_, b_ = mp.mpf(lo), mp.mpf(hi)
            if abs(g(a_)) < 1e-25:
                rts.append(a_)
            elif abs(g(b_)) < 1e-25:
                rts.append(b_)
            elif g(a_) * g(b_) < 0:
                rts.append(mp.findroot(g, (a_, b_), solver="bisect"))
        if rts and all(abs(r - mp.nint(r)) < 1e-12 for r in rts):
            best = k
            break
    ok3 = best is not None and sp.simplify(best - sp.sympify(md_answer(2, "3")["value"])) == 0
    report(2, ok1 and ok2 and ok3, f"max={mx[0]} min={mn[0]} a={sa} b={sb} kmin={best}")


def segment_area(A, B, R=4, N=4000):
    """호 AB(원점 중심 반지름 R, 짧은 호)와 현으로 둘러싸인 넓이 — 다각형 근사 + Richardson"""
    def poly(Nn):
        a0, a1 = mp.atan2(A[1], A[0]), mp.atan2(B[1], B[0])
        pts = [(R * mp.cos(a0 + (a1 - a0) * i / Nn), R * mp.sin(a0 + (a1 - a0) * i / Nn)) for i in range(Nn + 1)]
        s = sum(pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1] for i in range(Nn))
        s += pts[-1][0] * pts[0][1] - pts[0][0] * pts[-1][1]
        return abs(s) / 2
    return (4 * poly(2 * N) - poly(N)) / 3


def S_theta(A, B, th):
    P = (mp.cos(th) - 1, mp.sin(th))
    return abs((A[0] - P[0]) * (B[1] - P[1]) - (B[0] - P[0]) * (A[1] - P[1])) / 2


def p03():
    s6, s2 = sp.sqrt(6), sp.sqrt(2)
    A = (2 * s2, 2 * s2); B = (s6 - s2, s6 + s2)
    X, Y = sp.symbols("X Y")
    lhs, rhs = md_answer(3, "1")["value"].split("=")
    md_line = sp.sympify(lhs.replace("y", "Y").replace("x", "X")) - sp.sympify(rhs.replace("y", "Y").replace("x", "X"))
    ok1 = all(sp.simplify(md_line.subs({X: P[0], Y: P[1]})) == 0 for P in (A, B)) and md_line.free_symbols == {X, Y}
    ang = sp.acos(sp.nsimplify(sp.simplify((A[0] * B[0] + A[1] * B[1]) / 16)))
    arc = sp.simplify(4 * ang)
    ok2 = abs(sp.N(arc - sp.sympify(md_answer(3, "2")["value"]), 40)) < 1e-30
    An = tuple(mp.mpf(sp.N(v, 40)) for v in A); Bn = tuple(mp.mpf(sp.N(v, 40)) for v in B)
    seg = segment_area(An, Bn)
    Sf = lambda t: seg + S_theta(An, Bn, t)
    grid = [2 * mp.pi * i / 3600 for i in range(3600)]
    dS = lambda t: mp.diff(Sf, t)
    tmax = mp.findroot(dS, max(grid, key=Sf)); tmin = mp.findroot(dS, min(grid, key=Sf))
    val = (Sf(tmax) - Sf(tmin)) ** 2
    ok3 = abs(val - mp.mpf(sp.N(sp.sympify(md_answer(3, "3")["value"]), 40))) < 1e-15
    report(3, ok1 and ok2 and ok3, f"arc={arc} M={mp.nstr(Sf(tmax), 12)} m={mp.nstr(Sf(tmin), 12)} (M-m)^2={mp.nstr(val, 15)}")


if __name__ == "__main__":
    fns = sorted((n for n in dir() if re.fullmatch(r"p\d+", n)), key=lambda s: int(s[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
