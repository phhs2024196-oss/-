#!/usr/bin/env python3
"""가천대 2024학년도 논술 모의고사 A형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = "GCU-M24A"
WIDTH = 2
RESULTS = []
x, t, a, b, k, n, r = sp.symbols("x t a b k n r", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int) -> str:
    """YAML 의 answer.value (문자열)."""
    it = yaml.safe_load((ROOT / "items/GCU/PAST" / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def num(s: str):
    return sp.nsimplify(sp.sympify(s))


def nums(s: str):
    """답 문자열에서 동그라미 번호를 뺀 수들을 차례로."""
    s = re.sub(r"[①②③④]", " ", s)
    return [sp.Rational(v) for v in re.findall(r"-?\d+(?:/\d+)?", s)]


def p05():
    fp = sp.diff(x**3 + 3 * a * x**2 + x, x)
    disc = sp.discriminant(fp, x)
    amax = max(sp.solve(sp.Eq(disc, 0), a))
    ok_edge = sp.discriminant(fp.subs(a, amax + sp.Rational(1, 1000)), x) > 0
    report(5, ok_edge and sp.simplify(amax - num(ans(5))) == 0, f"판별식={sp.factor(disc)}, a 최댓값={amax}")


def p06():
    sols = []
    for s in sp.solve([1 - a + b], [b], dict=True):            # 분자 → 0
        expr = (x**2 + a * x + b).subs(s) / (x**2 - 1)
        L = sp.limit(sp.cancel(expr), x, -1)
        for av in sp.solve(sp.Eq(L, sp.Rational(1, 2)), a):
            sols.append((av, s[b].subs(a, av)))
    got = nums(ans(6))
    report(6, len(sols) == 1 and list(sols[0]) == got, f"(a,b)={sols}")


def p07():
    th = sp.pi + sp.asin(sp.Rational(3, 4))                    # sinθ=-3/4, cosθ<0 (제3사분면)
    ok = sp.simplify(sp.sin(sp.pi + th)) == sp.Rational(3, 4) and sp.cos(th).is_negative
    val = sp.simplify(sp.tan(th))
    report(7, ok and sp.simplify(val - num(ans(7))) == 0, f"sinθ={sp.simplify(sp.sin(th))}, cosθ={sp.simplify(sp.cos(th))}, tanθ={sp.radsimp(val)}")


def p08():
    a1, d = sp.symbols("a1 d")
    an = lambda m: a1 + (m - 1) * d
    S = lambda m: sum(an(j) + an(7) for j in range(1, m + 1))
    eqs = [S(m) - S(13 - m) for m in range(1, 13)] + [S(15) - 60]
    sol = [s for s in sp.solve(eqs, [a1, d], dict=True) if s[d] != 0]
    val = an(5).subs(sol[0])
    report(8, len(sol) == 1 and val == num(ans(8)), f"{sol}, a5={val}")


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
