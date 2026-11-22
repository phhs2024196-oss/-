#!/usr/bin/env python3
"""HIU-M27S (홍익대 2027학년도 모의논술 세종 자연) — sympy 검산.
규칙: 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "HIU-M27S"
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
    y = 2 * sp.sin(x) ** 2 - 4 * sp.cos(x)
    crit = [c for c in sp.solve(sp.diff(y, x), x) if 0 <= c <= 2 * sp.pi] + [0, 2 * sp.pi, sp.pi]
    mx = max(sp.nsimplify(y.subs(x, c)) for c in crit)
    import numpy as np
    grid = max(2 * np.sin(v) ** 2 - 4 * np.cos(v) for v in np.linspace(0, 2 * np.pi, 20001))
    ok = sp.simplify(mx - sp.sympify(md_answer(1, "1"))) == 0 and abs(grid - float(mx)) < 1e-6
    report(1, ok, f"max={mx} grid={grid:.6f}")


def p02():
    X = sp.symbols("X", real=True)
    # 진수 조건 X>-2, X≠0 아래 log2(X+2)=log4(X^2) ⇔ (X+2)^2 = X^2 (양변 모두 log4 로)
    cands = [s for s in sp.solve(sp.Eq((X + 2) ** 2, X**2), X) if s > -2 and s != 0]
    chk = all(abs(float(sp.log(c + 2, 2) - sp.log(c**2, 4))) < 1e-12 for c in cands)
    ok = chk and cands == [sp.sympify(md_answer(2, "1"))]
    report(2, ok, f"solutions={cands}")


def p03():
    N = sp.symbols("N", integer=True, positive=True)
    S = lambda m: m**2 + 3 * m
    a = sp.expand(S(N) - S(N - 1))
    ok1 = a.subs(N, 1) == S(1) and sp.simplify(a - sp.sympify(md_answer(3, "1"), locals={"n": N})) == 0
    tot = sum(S(2 * j) - S(2 * j - 1) for j in range(1, 6))       # 정의로 직접 합
    ok2 = tot == sp.sympify(md_answer(3, "2"))
    report(3, ok1 and ok2, f"a_n={a} sum={tot}")


def p04():
    a, b, c = sp.symbols("a b c", real=True)
    f = a * t**3 + b * t**2 + c * t
    e1 = sp.Eq(sp.limit(b * t**3 / f, t, sp.oo), 2)
    e2 = sp.Eq(f.subs(t, 1), 0)                                   # 분모→0 이므로 분자→0
    sols = sp.solve([e1, e2], [b, c], dict=True)[0]
    fs = f.subs(sols)
    e3 = sp.Eq(sp.limit(fs.subs(t, t - 1) / (t - 2), t, 2), 4)
    av = sp.solve(e3, a)
    F = sp.expand(fs.subs(a, av[0]))
    ok_lim = sp.limit(b * t**3 / f, t, sp.oo).subs(sols).subs(a, av[0]) == 2 and \
        sp.limit(F.subs(t, t - 1) / (t - 2), t, 2) == 4
    ok1 = len(av) == 1 and sp.simplify(F - sp.sympify(md_answer(4, "1"), locals={"t": t})) == 0 and ok_lim
    acc = sp.diff(F, t, 2).subs(t, 3)
    ok2 = acc == sp.sympify(md_answer(4, "2"))
    report(4, ok1 and ok2, f"f={F} a(3)={acc}")


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
