#!/usr/bin/env python3
"""EWHA-M25A (이화여대 2025학년도 모의논술 자연계열Ⅰ) — sympy 검산.
정답 하드코딩 비교 금지: 문항 YAML 의 answer 를 읽어 독립 재계산값과 비교."""
import itertools
import math
import pathlib
import random
import re
import sys

import sympy as sp
import yaml

PREFIX = "EWHA-M25A"
RESULTS = []
x, t, m = sp.symbols("x t m", real=True)


def md_answer(no, sub):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:02d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:02d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    lo, hi = x**2 / 2, -x**2 + sp.Rational(9, 2) * x
    xs = sp.solve(sp.Eq(lo, hi), x)
    A = sp.integrate(hi - lo, (x, min(xs), max(xs)))
    ok1 = sp.simplify(A - sp.sympify(md_answer(1, "1"))) == 0
    # (2) 수평선: 격자로 S 의 단면 길이 최대를 직접 측정 후, 해석식 후보와 비교
    import numpy as np
    X = np.linspace(0, 3, 300001)
    LO, HI = X**2 / 2, -X**2 + 4.5 * X

    def hlen(tv):
        inside = X[(LO <= tv) & (tv <= HI)]
        return inside.max() - inside.min() if inside.size else 0.0
    T = np.linspace(0, 81 / 16, 2001)
    num2 = max(hlen(v) for v in T)
    # 해석: 0<=t<=9/2 구간의 임계점
    ft = sp.sqrt(2 * t) - (9 - sp.sqrt(81 - 16 * t)) / 4
    tc = sp.nsolve(sp.diff(ft, t), t, 3)
    exact2 = sp.nsimplify(ft.subs(t, sp.Rational(27, 8)))
    ok2 = abs(tc - sp.Rational(27, 8)) < 1e-12 and abs(num2 - float(exact2)) < 1e-3 and \
        sp.simplify(exact2 - sp.sympify(md_answer(1, "2"))) == 0
    # (3) 원점 직선: 교점 사이 거리 최대를 수치로
    def rlen(mv):
        pts = X[(LO <= mv * X + 1e-12) & (mv * X <= HI + 1e-12)]
        if pts.size == 0:
            return 0.0
        return (pts.max() - pts.min()) * math.sqrt(1 + mv * mv)
    M = np.linspace(0.001, 4.499, 4499)
    num3 = max(rlen(v) for v in M)
    h = (sp.Rational(9, 2) - m)**2 * (1 + m**2)
    crit = [c for c in sp.solve(sp.diff(h, m), m) if sp.Rational(3, 2) <= c < sp.Rational(9, 2)]
    exact3 = max([sp.sqrt(h.subs(m, c)) for c in crit] + [2 * sp.Rational(3, 2) * sp.sqrt(1 + sp.Rational(9, 4))],
                 key=lambda v: float(v))
    ok3 = abs(num3 - float(exact3)) < 1e-3 and sp.simplify(exact3 - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"A={A} maxf~{num2:.5f}({exact2}) maxg~{num3:.5f}({sp.simplify(exact3)})")


def p02():
    sols = [(a, b) for a in range(1, 45) for b in range(1, 45) if a * a + b * b == 2025]
    # (1) [증명-수치확인] 모든 해의 기울기 0<n/(m+45)<1
    from fractions import Fraction
    ok1 = all(0 < Fraction(b, a + 45) < 1 for a, b in sols)
    # (2) YAML 식이 원·직선 위에 있는지 (기호) + 각 해에서 재현
    p, q = sp.symbols("p q", positive=True)
    expr = sp.sympify(md_answer(2, "2"), locals={"p": p, "q": q})
    on_circle = sp.simplify(expr[0]**2 + expr[1]**2 - 2025) == 0
    on_line = sp.simplify(expr[1] - p / q * (expr[0] + 45)) == 0
    rep = all(tuple(expr[i].subs({p: Fraction(b, a + 45).numerator, q: Fraction(b, a + 45).denominator})
                    for i in range(2)) == (a, b) for a, b in sols)
    ok2 = on_circle and on_line and rep
    # (3) 전수
    ans = set(sp.sympify(md_answer(2, "3").replace("{", "[").replace("}", "]")))
    ok3 = {tuple(map(int, s)) for s in ans} == set(sols)
    report(2, ok1 and ok2 and ok3, f"sols={sols}")


def p03():
    # [증명-수치확인] f(x)=c ln x 꼴 무작위 c 로 (1)~(5) 성질 확인 + f(1) 재계산
    c = sp.symbols("c")
    f1 = sp.solve(sp.Eq(sp.Symbol("u"), 2 * sp.Symbol("u")), sp.Symbol("u"))[0]   # f(1)=f(1)+f(1)
    ok1 = sp.sympify(md_answer(3, "1")) == f1
    random.seed(1)
    ok = True
    for _ in range(200):
        cv = random.uniform(-3, 3)
        F = lambda z: cv * math.log(z)
        a, b, xv = (random.uniform(0.05, 20) for _ in range(3))
        n, mm = random.randint(1, 9), random.randint(1, 9)
        qv = random.choice([-1, 0, 1]) * n / mm
        ok &= abs(F(a / b) - (F(a) - F(b))) < 1e-9
        ok &= abs(F(xv**n) - n * F(xv)) < 1e-8 and abs(F(xv**(1 / n)) - F(xv) / n) < 1e-9
        ok &= abs(F(xv**qv) - qv * F(xv)) < 1e-8
        hh = 1e-7
        ok &= abs((F(xv + hh) - F(xv - hh)) / (2 * hh) - cv / xv) < 1e-5
    report(3, ok1 and ok, f"f(1)={f1} 성질(1)~(5) 무작위 200점 확인 [증명-수치확인]")


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
