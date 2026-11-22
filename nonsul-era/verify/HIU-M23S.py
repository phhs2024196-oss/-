#!/usr/bin/env python3
"""HIU-M23S (홍익대 2023학년도 세종캠퍼스 자연계열 논술 예시문제) — sympy 검산.
규칙: 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "HIU-M23S"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def parts(s: str) -> dict:
    """'a = 1; b = 2' 형식 → {'a': '1', 'b': '2'}"""
    return {kk.strip(): v.strip() for kk, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    Y = sp.symbols("Y", real=True)
    base = 2**x
    refl = base.subs(x, -x)                                   # y축 대칭
    f = refl.subs(x, x - 2) - 8                                # x축 +2, y축 -8
    g = sp.solve(sp.Eq(f.subs(x, Y), x), Y)[0]                 # 역함수
    a1 = parts(md_answer(1, "1"))
    md_g = sp.sympify(a1["g"], locals={"x": x})
    okg = all(abs(float((g - md_g).subs(x, v))) < 1e-12 for v in (-7, -4, 0, 5))
    av = sp.solve(sp.Eq(g, 0), x)[0]
    bv = g.subs(x, 0)
    val = sp.nsimplify(sp.simplify(av**2 + bv))
    ok = okg and val == sp.sympify(a1["a**2 + b"]) and sp.simplify(f.subs(x, bv)) == 0
    report(1, ok, f"f={f} g={sp.simplify(g)} a={av} b={sp.simplify(bv)} a^2+b={val}")


def p02():
    a = sp.symbols("a", real=True)
    f = -2 * x**3 + 3 * (a - 1) * x**2 + 6 * a * x
    found = []
    for av in [sp.Rational(i, 4) for i in range(0, 41)] + list(sp.solve(sp.Eq(f.subs(x, a), 0), a)):
        if av < 0:
            continue
        fa = f.subs(a, av)
        crit = sp.solve(sp.diff(fa, x), x)
        maxima = [c for c in crit if sp.diff(fa, x, 2).subs(x, c) < 0]
        if maxima and any(fa.subs(x, c) == 0 for c in maxima):
            found.append(av)
    found = sorted(set(found))
    ok = found == [0] and f.subs({a: 0, x: 1}) == sp.sympify(md_answer(2, "1"))
    report(2, ok, f"a candidates={found} f(1)={f.subs({a: 0, x: 1})}")


def p03():
    c = sp.symbols("c0:4", real=True)
    f = sum(c[i] * x**i for i in range(4))                     # 3차 이하 다항식으로 일반화
    tt = sp.symbols("tt", real=True)
    eq = sp.expand(f - (2 * x**2 - 4 * x - sp.integrate(sp.diff(f, x).subs(x, tt), (tt, 0, x))))
    sol = sp.solve(sp.Poly(eq, x).all_coeffs(), c, dict=True)[0]
    F = f.subs(sol)
    roots = sorted(sp.solve(F, x))
    area = sum(sp.Abs(sp.integrate(F, (x, lo, hi))) for lo, hi in zip(roots, roots[1:]))   # 근 사이 구간별 |적분|
    ok = sp.simplify(area - sp.sympify(md_answer(3, "1"))) == 0
    report(3, ok, f"f={F} roots={roots} area={area}")


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
