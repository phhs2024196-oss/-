#!/usr/bin/env python3
"""EWHA-M25B (이화여대 2025학년도 모의논술 자연계열Ⅱ) — sympy 검산.
정답 하드코딩 비교 금지: 문항 YAML 의 answer 를 읽어 독립 재계산값과 비교."""
import math
import pathlib
import random
import re
import sys

import sympy as sp
import yaml

PREFIX = "EWHA-M25B"
RESULTS = []
x, th = sp.symbols("x theta", real=True)


def md_answer(no, sub):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:02d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:02d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    g = x**2 - 3
    I_fg = -6                                           # (라): ∫_0^2 (f+g) dx
    I_fmg = I_fg - 2 * sp.integrate(g, (x, 0, 2))      # ∫_0^2 (f-g)
    area = 2 * I_fmg
    # 조건을 모두 만족하는 예시 f 로 일관성 확인: f = g + k(4-x^2)
    k = sp.symbols("k")
    fex = g + k * (4 - x**2)
    kv = sp.solve(sp.integrate(fex + g, (x, 0, 2)) + 6, k)[0]
    fex = fex.subs(k, kv)
    n = sp.symbols("n", positive=True, integer=True)
    j = sp.symbols("j", integer=True)
    rs = sp.limit(sp.summation((fex + g).subs(x, j / n), (j, 1, 2 * n)) / n, n, sp.oo)
    meets = sorted((r, fex.subs(x, r)) for r in sp.solve(sp.Eq(fex, g), x))
    ok_ex = fex.subs(x, 2) == 1 and sp.simplify(fex - fex.subs(x, -x)) == 0 and rs == -6 and kv > 0
    ans1 = {tuple(p) for p in sp.sympify(md_answer(1, "1").replace("{", "[").replace("}", "]"))}
    ok1 = ans1 == {(sp.Integer(a), sp.Integer(b)) for a, b in meets}
    # (2) [증명-수치확인] 예시 f 에서 [-2,2] 격자 f>=g
    ok2 = all(float((fex - g).subs(x, -2 + 4 * i / 1000)) >= -1e-12 for i in range(1001))
    ok3 = sp.simplify(area - sp.sympify(md_answer(1, "3"))) == 0 and \
        sp.integrate(fex - g, (x, -2, 2)) == area
    report(1, ok_ex and ok1 and ok2 and ok3, f"meets={meets} ∫0^2(f-g)={I_fmg} area={area} 예시 f=g+{kv}(4-x^2)")


def p02():
    # [증명-수치확인] f(x)=A^x (A>1) 이 조건 (가)~(다)를 만족함을 확인하고 (1)~(4) 성질을 무작위 점에서 확인
    random.seed(2)
    ok = True
    for _ in range(200):
        A = random.uniform(1.01, 5)
        F = lambda z: A**z
        b, c, xv = (random.uniform(-3, 3) for _ in range(3))
        grid = [b + c + 0.01 * i for i in range(500)]
        ok &= abs(min(F(z) for z in grid) - F(b) * F(c)) < 1e-9          # (다)
        nn, mm = random.randint(1, 6), random.randint(1, 6)
        ok &= abs(F(nn * xv) - F(xv)**nn) < 1e-6 * F(nn * xv) and abs(F(xv / nn) - F(xv)**(1 / nn)) < 1e-9
        qv = random.choice([-1, 0, 1]) * nn / mm
        ok &= abs(F(qv * xv) - F(xv)**qv) < 1e-6 * max(1, F(qv * xv))
        h = 1e-6
        d = (F(xv + h) - F(xv - h)) / (2 * h); d0 = (F(h) - F(-h)) / (2 * h)
        ok &= abs(d - d0 * F(xv)) < 1e-4 * max(1, d)
    report(2, ok, "f=A^x 무작위 200개로 (다)와 성질 (1)~(4) 확인 [증명-수치확인]")


def p03():
    f = x**4 * (1 - x)**4 / (1 + x**2)
    J = sp.integrate(f, (x, 0, 1))
    I = lambda n: sp.integrate(sp.tan(th)**n, (th, 0, sp.pi / 4))
    comb = sp.simplify(I(4) - 4 * I(5) + 6 * I(6) - 4 * I(7) + I(8))
    ok1 = sp.simplify(J - comb) == 0
    ok2 = sp.simplify(J - sp.sympify(md_answer(3, "2"))) == 0
    K = sp.integrate(x**4 * (1 - x)**4, (x, 0, 1))
    ok3 = sp.simplify(K - sp.sympify(md_answer(3, "3"))) == 0
    lo, hi = sp.Rational(22, 7) - K, sp.Rational(22, 7) - K / 2
    ok3 &= lo == sp.Rational(1979, 630) and hi == sp.Rational(3959, 1260) and lo <= sp.pi <= hi
    ok3 &= sp.simplify(sp.integrate(f - x**4 * (1 - x)**4 / 2, (x, 0, 1)) - (J - K / 2)) == 0
    report(3, ok1 and ok2 and ok3, f"∫f={sp.simplify(J)} combo={comb} ∫x^4(1-x)^4={K} [{lo},{hi}]")


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
