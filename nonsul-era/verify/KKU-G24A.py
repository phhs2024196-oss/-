#!/usr/bin/env python3
"""KKU-G24A (건국대 2024학년도 KU논술우수자 자연계 A) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "KKU-G24A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


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


import mpmath as mp
from sympy.utilities.iterables import multiset_permutations


def p01():
    m = sp.symbols("m", positive=True)
    c = sp.sqrt(13) / 3
    ms = sp.solve(sp.Eq(c / sp.sqrt(m**2 + 1), 1), m)          # 접선 기울기(양수)
    f = x**3 / 6 + 1 / (2 * x)
    xs = sorted(r for r in sp.solve(sp.Eq(ms[0] * x, f), x) if r.is_positive)
    # 수치: 원점→곡선 위 점 선분이 원 내부를 지나면 가려짐
    cy = float(c)
    def visible(xv):
        yv = xv**3 / 6 + 1 / (2 * xv)
        L2 = xv * xv + yv * yv
        s = max(0.0, min(1.0, (cy * yv) / L2))
        d2 = (s * xv) ** 2 + (s * yv - cy) ** 2
        return d2 >= 1 - 1e-12
    grid = [0.01 + k * 0.001 for k in range(5000)]
    vis = [g for g in grid if visible(g)]
    ok_rng = abs(min(vis) - float(xs[0])) < 2e-3 and abs(max(vis) - float(xs[-1])) < 2e-3 and \
        all(visible(g) for g in grid if float(xs[0]) + 1e-3 < g < float(xs[-1]) - 1e-3)
    L = sp.integrate(sp.sqrt(sp.simplify(1 + sp.diff(f, x)**2)).factor(), (x, xs[0], xs[-1]))
    Ln = mp.quad(lambda u: mp.sqrt(1 + (u**2 / 2 - 1 / (2 * u**2))**2), [float(xs[0]), float(xs[-1])])
    md = sp.sympify(md_answer(1, "1"))
    ok = ok_rng and sp.simplify(L - md) == 0 and abs(Ln - float(md)) < 1e-12
    report(1, ok, f"m={ms} x={xs} visible[{min(vis):.3f},{max(vis):.3f}] L={sp.nsimplify(L)}")


def p02():
    letters = "AAABBBBCCCDD"
    c1 = c2 = 0
    for p in multiset_permutations(list(letters)):
        s = "".join(p)
        if "DD" not in s:
            c1 += 1
        if s.index("A") < s.index("B"):
            c2 += 1
    ok = c1 == int(md_answer(2, "1")) and c2 == int(md_answer(2, "2"))
    report(2, ok, f"전수 (1)={c1} (2)={c2}")


def p03():
    mp.mp.dps = 30
    def S(tv):
        a = mp.findroot(lambda u: mp.e**(2 * u) - mp.e**(-(u - tv)) - 1, 0.5)
        return mp.quad(lambda u: mp.e**(-(u - tv)) + 1 - mp.e**(2 * u), [0, a])
    d = mp.diff(S, mp.log(6))
    md = sp.sympify(md_answer(3, "1"))
    report(3, abs(d - mp.mpf(float(md))) < 1e-15, f"S'(ln6) 수치미분={mp.nstr(d, 18)}")


def p04():
    mp.mp.dps = 30
    V = [(0, 0), (1, 0), (1, 1), (0, 1)]
    def cosa(ph):
        px, py = 3 * mp.cos(ph), 3 * mp.sin(ph)
        vs = [(vx - px, vy - py) for vx, vy in V]
        best = 1
        for i in range(4):
            for j in range(i + 1, 4):
                u, w = vs[i], vs[j]
                cv = (u[0] * w[0] + u[1] * w[1]) / (mp.sqrt(u[0]**2 + u[1]**2) * mp.sqrt(w[0]**2 + w[1]**2))
                best = min(best, cv)
        return best
    def cosf(ph):
        px, py = 3 * math.cos(ph), 3 * math.sin(ph)
        vs = [(vx - px, vy - py) for vx, vy in V]
        return min((u[0] * w[0] + u[1] * w[1]) / (math.hypot(*u) * math.hypot(*w))
                   for i, u in enumerate(vs) for w in vs[i + 1:])
    grid = [math.pi / 2 + math.pi / 2 * k / 20000 for k in range(1, 20000)]
    vals = [cosf(g) for g in grid]
    k = min(range(len(vals)), key=lambda i: vals[i])
    lo, hi = mp.mpf(grid[max(k - 1, 0)]), mp.mpf(grid[min(k + 1, len(grid) - 1)])
    for _ in range(200):  # 삼분 탐색
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if cosa(m1) < cosa(m2):
            hi = m2
        else:
            lo = m1
    ph = (lo + hi) / 2
    pr = parts(4, "1")
    ok = abs(cosa(ph) - mp.mpf(str(sp.N(pr["cos_alpha"], 30)))) < 1e-20 and \
        abs(3 * mp.cos(ph) - mp.mpf(str(sp.N(pr["Px"], 30)))) < 1e-8 and \
        abs(3 * mp.sin(ph) - mp.mpf(str(sp.N(pr["Py"], 30)))) < 1e-8
    report(4, ok, f"min cos={mp.nstr(cosa(ph), 15)} P=({mp.nstr(3*mp.cos(ph), 12)},{mp.nstr(3*mp.sin(ph), 12)})")


def parts(no, sub):
    return {k.strip(): sp.sympify(v) for k, v in (s.split("=", 1) for s in md_answer(no, sub).split(";"))}


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
