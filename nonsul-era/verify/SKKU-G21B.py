#!/usr/bin/env python3
"""SKKU-G21B (성균관대 2021학년도 수시 논술우수전형 자연계 2교시 수학 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "SKKU-G21B"

WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str, loc=None) -> dict:
    """'key = expr; key2 = expr2' 형식의 정답 문자열을 dict 로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = seg.split("=", 1)
        out[k.strip()] = sp.sympify(v.strip(), locals=loc or {})
    return out


def eq(a, b) -> bool:
    return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 if not isinstance(a, tuple) else all(eq(u, w) for u, w in zip(a, b))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


x, y = sp.symbols("x y", real=True)


def p01():
    cx, cy, R = 12, sp.Rational(15, 2), 6
    f = sp.expand((x - cx) ** 2 + (x**2 - cy) ** 2)
    crit = [c for c in sp.solve(sp.diff(f, x), x) if c.is_real]
    xb = min(crit, key=lambda c: f.subs(x, c)); fmin = f.subs(x, xb)
    P1 = md_parts(1, "1")
    ok1 = eq(sp.sqrt(fmin), P1["min_dist"]) and P1["radius"] == R and sp.sqrt(fmin) > R
    B = (xb, xb**2)
    d = sp.sqrt((B[0] - cx) ** 2 + (B[1] - cy) ** 2)
    A = (cx + R * (B[0] - cx) / d, cy + R * (B[1] - cy) / d)
    P2 = md_parts(1, "2")
    ok2 = eq(tuple(P2["A"]), A) and eq(tuple(P2["B"]), B)
    m = sp.symbols("m", real=True)
    slopes = sp.solve(sp.Eq((m * (cx - B[0]) - (cy - B[1])) ** 2, R**2 * (m**2 + 1)), m)   # 중심-직선 거리 = R
    vert_ok = abs(cx - B[0]) > R                                                           # x=3 은 접선 아님
    md_t = sp.sympify(md_answer(1, "3").split("=", 1)[1], locals={"x": x})
    md_slopes = sorted([sp.diff(e, x) for e in md_t], key=float)
    ok3 = vert_ok and len(slopes) == 2 and all(eq(u, w) for u, w in zip(sorted(slopes, key=float), md_slopes)) \
        and all(eq(e.subs(x, B[0]), B[1]) for e in md_t)
    m1, m2 = slopes
    tanth = sp.Abs((m1 - m2) / (1 + m1 * m2))
    th = sp.atan(tanth)                                                                     # 0<θ≤π/2
    sinth = sp.nsimplify(sp.simplify(sp.sin(th)))
    ok4 = eq(sinth, sp.sympify(md_answer(1, "4")))
    report(1, ok1 and ok2 and ok3 and ok4, f"min d^2={fmin} B={B} slopes={slopes} sin={sinth}")


def p02():
    a = sp.symbols("a", positive=True)
    t = sp.symbols("t", positive=True)
    f = sp.integrate(1 / t - 1 / a, (t, 1, a))
    g = (a - 1) * (1 / a)
    h = sp.simplify(f - g)
    ok1 = sp.simplify(h - sp.sympify(md_answer(2, "1"), locals={"a": a})) == 0
    crit = [c for c in sp.solve(sp.diff(h, a), a) if c > 1]
    mn = h.subs(a, crit[0])
    import numpy as np
    hf = sp.lambdify(a, h)
    ok2 = len(crit) == 1 and sp.diff(h, a, 2).subs(a, crit[0]) > 0 and \
        hf(np.linspace(1.0001, 1e4, 200001)).min() >= float(mn) - 1e-12 and eq(mn, sp.sympify(md_answer(2, "2")))
    L3 = sp.limit(h / (a - 1), a, 1)
    ok3 = eq(L3, sp.sympify(md_answer(2, "3")))
    k = 2 * sp.sqrt(a) - h
    kf = sp.lambdify(a, k)
    grid = np.concatenate([np.linspace(1 + 1e-9, 10, 100001), np.logspace(1, 12, 100001)])
    k1 = sp.limit(k, a, 1, "+")
    md_k1 = sp.sympify(md_answer(2, "4").split(";")[1].split("=", 1)[1])
    ok4 = kf(grid).min() > 0 and eq(k1, md_k1)                                             # [증명-수치확인]
    L5 = sp.limit(h / a, a, sp.oo)
    ok5 = eq(L5, sp.sympify(md_answer(2, "5")))
    report(2, ok1 and ok2 and ok3 and ok4 and ok5, f"h={h} min={mn} lim1={L3} min k={kf(grid).min():.4f} k(1)={k1} liminf={L5}")


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
