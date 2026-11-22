#!/usr/bin/env python3
"""SMWU-G26P (숙명여대 2026학년도 논술우수자전형 자연계열(약학부)) — sympy 검산. 문항 1은 일반(SMWU-G26-01)과 동일하므로 verify/SMWU-G26.py::p01 에서 검산."""
import re
import sys

import sympy as sp

PREFIX = "SMWU-G26P"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
n = sp.symbols("n", positive=True, integer=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    """'b_n = ...; c_n = ...' 형식을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    # (2-1) 해집합을 solveset 으로 구하고 e^{-x} 를 적분
    dom = sp.Interval(0, 2 * sp.pi)
    sol = sp.solveset(sp.sin(x) <= sp.Rational(1, 2), x, dom)
    ivs = sol.args if isinstance(sol, sp.Union) else (sol,)
    I1 = sum(sp.integrate(sp.exp(-x), (x, iv.start, iv.end)) for iv in ivs)
    ok1 = sp.simplify(I1 - sp.sympify(md_answer(2, "1"))) == 0
    # (2-2) 한 주기 [2(n-1)π, 2nπ] 마다 solveset 으로 해 구간을 구해 50주기 합(기호), k 와 비교
    base = sp.solveset(sp.sin(x) <= -sp.Rational(1, 2), x, sp.Interval(0, 2 * sp.pi))
    ivs2 = base.args if isinstance(base, sp.Union) else (base,)
    one = sum(sp.integrate(sp.exp(-x), (x, iv.start, iv.end)) for iv in ivs2)   # n=0 주기 값
    total = sum(one * sp.exp(-2 * sp.pi * m) for m in range(50))                  # 주기마다 e^{-2π} 배
    k = sp.sympify(md_answer(2, "2"))
    import mpmath as mp
    mp.mp.dps = 80
    lhs = sp.N(total, 70); rhs = sp.N((sp.exp(5 * sp.pi / 6) - sp.exp(sp.pi / 6)) * k, 70)
    # 별도 수치 확인: 첫 3주기를 직접 수치적분
    direct = mp.quad(lambda u: mp.exp(-u) * (1 if mp.sin(u) <= -0.5 else 0), mp.linspace(0, 6 * mp.pi, 601))
    first3 = sp.N(sum(one * sp.exp(-2 * sp.pi * m) for m in range(3)), 30)
    ok2 = abs(lhs - rhs) < sp.Float(10)**-60 and abs(direct - mp.mpf(str(first3))) < 1e-4
    report(2, ok1 and ok2, f"I1={sp.simplify(I1)}  sum50 match; base={ivs2}")


def p03():
    f = sp.log(sp.sqrt((1 + x) / (1 - x)))
    # (3-1) [증명-수치확인] h=f-x, h' = x²/(1-x²), h(0)=0, 격자에서 h>0
    h = f - x
    hp = sp.simplify(sp.diff(h, x))
    ok1 = sp.simplify(hp - x**2 / (1 - x**2)) == 0 and h.subs(x, 0) == 0
    import numpy as np
    hl = sp.lambdify(x, h)
    ok1 &= all(hl(v) > 0 for v in np.linspace(1e-3, 0.999, 1000))
    # (3-2) YAML 의 g 가 f 의 역함수인지, 점근선 y=±1, 변곡점 (0,0)
    gstr = md_answer(3, "2").split(";")[0].split("=", 1)[1]
    g = sp.sympify(gstr, locals={"x": x})
    ok2 = all(abs(float(f.subs(x, g.subs(x, v))) - v) < 1e-12 for v in (-3, -0.7, 0.2, 1.5, 4))
    ok2 &= sp.limit(g, x, sp.oo) == 1 and sp.limit(g, x, -sp.oo) == -1
    g2 = sp.diff(g, x, 2)
    ok2 &= sp.simplify(g2.subs(x, 0)) == 0 and float(g2.subs(x, -0.5)) > 0 > float(g2.subs(x, 0.5))
    ok2 &= sp.simplify(sp.diff(g, x).subs(x, 0)) == 1
    fpp = sp.simplify(sp.diff(f, x, 2))
    ok2 &= sp.simplify(fpp - 2 * x / (1 - x**2)**2) == 0
    report(3, ok1 and ok2, f"[증명-수치확인] h'={hp}; g=f^-1 ok, asymptotes ±1, inflection 0")


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
