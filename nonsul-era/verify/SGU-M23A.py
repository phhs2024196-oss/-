#!/usr/bin/env python3
"""SGU-M23A (서강대 2023학년도 모의논술 1차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M23A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def same(e1, e2, var):
    return all(abs(complex(sp.N((e1 - e2).subs(var, v), 30))) < 1e-20 for v in (sp.Rational(1, 7), 1, sp.Rational(5, 2), 13))


def p01():
    a, r = sp.symbols("a r", positive=True)
    X = sp.symbols("x", positive=True)
    loc = {"a": a, "x": X}
    s = sp.sqrt(1 + 4 * a**2)
    P = sp.Matrix([a, a**2])
    down = sp.Matrix([2 * a, -1]) / s          # l 아래쪽 법선 단위벡터
    up = -down
    # (1-1) 중심 = P + r*down, x축에 접함: |C_y| = r
    sols1 = [rv for eq in ((P + r * down)[1] - r, (P + r * down)[1] + r) for rv in sp.solve(eq, r) if all(rv.subs(a, v) > 0 for v in (sp.Rational(1, 5), 1, 7))]
    g = sp.simplify(sols1[0])
    ok1 = len(sols1) == 1 and same(g, sp.sympify(md_answer(1, "1"), locals=loc), a)
    # (1-2) 중심 = P + r*up, y축에 접함: |C_x| = r
    sols2 = [rv for eq in ((P + r * up)[0] - r, (P + r * up)[0] + r) for rv in sp.solve(eq, r) if all(rv.subs(a, v) > 0 for v in (sp.Rational(1, 5), 1, 7))]
    h = sp.simplify(sols2[0])
    ok2 = len(sols2) == 1 and same(h, sp.sympify(md_answer(1, "2"), locals=loc), a)
    # (1-3)
    f = sp.simplify((a * h / g).subs(a, X))
    l0, linf = sp.limit(f, X, 0, "+"), sp.limit(f, X, sp.oo)
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "3").replace(" ", "").split(";"))
    ok3 = sp.simplify(l0 - sp.sympify(parts["right"])) == 0 and sp.simplify(linf - sp.sympify(parts["infinity"])) == 0
    # (1-4) 감소 확인(격자) + 치역 = (linf, l0)
    fl = sp.lambdify(X, f)
    xs = [10 ** (k / 100) for k in range(-400, 401)]
    ok4 = all(fl(u) > fl(v) for u, v in zip(xs, xs[1:]))
    rng = sp.Interval.open(sp.Min(l0, linf), sp.Max(l0, linf))
    md4 = sp.sympify(md_answer(1, "4"), locals={"Interval": sp.Interval})
    ok4 = ok4 and md4 == rng
    report(1, ok1 and ok2 and ok3 and ok4, f"g={g} h={h} lim0+={l0} lim∞={linf} range={rng}")


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
