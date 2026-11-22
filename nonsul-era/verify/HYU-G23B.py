#!/usr/bin/env python3
"""HYU-G23B (한양대 2023학년도 논술 자연계열 오후1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
증명 소문항은 핵심 등식·부등식을 기호/수치로 확인 [증명-수치확인]."""
import re
import sys
from itertools import product
from math import comb

import mpmath as mp
import sympy as sp

PREFIX = "HYU-G23B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # <가> f(x) = ∫_0^x (x t - t^2) e^{x-t} dt  (sympy 직접 적분)
    f = sp.simplify(sp.integrate((x * t - t**2) * sp.exp(x - t), (t, 0, x)))
    f2 = sp.simplify(sp.diff(f, x, 2))
    # (1) f'' 부호 변화 지점 = 변곡점
    zs = sp.solve(sp.Eq(f2, 0), x)
    sign_change = all(float(f2.subs(x, z - d)) * float(f2.subs(x, z + d)) < 0 for z in zs for d in (0.1, 1, 3))
    infl = [sp.Tuple(z, sp.simplify(f.subs(x, z))) for z in zs]
    md1 = sp.sympify(md_answer(1, "1"))
    ok1 = sign_change and len(infl) == 1 and sp.Tuple(*md1) == infl[0] \
        and float(f2.subs(x, -1)) < 0 and float(f2.subs(x, 1)) > 0
    # (2) 접선 ℓ1(0), ℓ2(2); 영역 = ∫_0^{x0} f + ∫_{x0}^2 (f - ℓ2)
    df = sp.diff(f, x)
    l1 = f.subs(x, 0) + df.subs(x, 0) * x
    l2 = f.subs(x, 2) + df.subs(x, 2) * (x - 2)
    x0 = sp.solve(sp.Eq(l2, l1), x)[0]
    area = sp.simplify((sp.integrate(f - l1, (x, 0, x0)) + sp.integrate(f - l2, (x, x0, 2))).rewrite(sp.exp))
    # 곡선이 ℓ2 위·ℓ1 위(x≥0)에 있음을 격자로 확인
    above = all(float((f - l2).subs(x, v)) >= -1e-12 and float((f - l1).subs(x, v)) >= -1e-12
                for v in [i / 50 for i in range(0, 101)])
    md2 = sp.sympify(md_answer(1, "2"))
    ok2 = above and sp.simplify(area - md2) == 0 and abs(sp.N(area - md2, 30)) < 1e-25
    # (3) [증명-수치확인] g = e^x(sin2πx + 2πx cos2πx) 가 <나> 를 만족하는지 + 부등식
    g = sp.exp(x) * (sp.sin(2 * sp.pi * x) + 2 * sp.pi * x * sp.cos(2 * sp.pi * x))
    gp = sp.diff(g, x)
    cond_ok = True
    mp.mp.dps = 30
    gpl = sp.lambdify(t, gp.subs(x, t), "mpmath")
    for xv in (0.3, 0.77, 1.9):
        lhs = mp.e ** (-xv) * mp.quad(gpl, [0, xv])
        rhs = mp.quad(lambda s: mp.e ** (-s) * gpl(s), [0, xv]) - xv * mp.sin(2 * mp.pi * xv)
        cond_ok &= abs(lhs - rhs) < mp.mpf("1e-20")
    cond_ok &= g.subs(x, 0) == 0
    G = sp.integrate(sp.expand(g), (x, 0, 2023))
    mp.mp.dps = 50
    Gv = sp.N(G / (4046 * sp.pi * sp.exp(2023)), 40)
    bound = sp.integrate(sp.exp(x) * (1 + 2 * sp.pi * x), (x, 0, 2023))
    ok3 = cond_ok and Gv < 1 and sp.simplify(bound - (4046 * sp.pi * sp.exp(2023) + (1 - 2 * sp.pi) * (sp.exp(2023) - 1))) == 0
    report(1, ok1 and ok2 and ok3, f"[{ok1},{ok2},{ok3}] " + f"f={f} f''={f2} infl={infl} area={area} [증명-수치확인] ∫g/(4046πe^2023)={sp.N(Gv, 12)}")


def p02():
    # (1) 정육면체 좌표: F(0,0,0) B(0,0,1) A(0,1,1) D(1,1,1) H(1,1,0), P=(s,1,1), 0≤s≤1/2
    s = sp.symbols("s", real=True)
    P, B, H = sp.Matrix([s, 1, 1]), sp.Matrix([0, 0, 1]), sp.Matrix([1, 1, 0])
    PB, PH = B - P, H - P
    area2 = sp.expand(PB.dot(PB) * PH.dot(PH) - PB.dot(PH) ** 2) / 4   # (넓이)^2 (Lagrange 항등식)
    sols = [r for r in sp.solve(sp.Eq(area2, sp.Rational(7, 16)), s) if 0 <= r <= sp.Rational(1, 2)]
    # 정사영(z=0): P'(s,1) B'(0,0) H'(1,1) 삼각형 넓이 (신발끈)
    proj = sp.Abs(sp.Rational(1, 2) * (s * (0 - 1) + 0 * (1 - 1) + 1 * (1 - 0)))
    pv = sp.simplify(proj.subs(s, sols[0]))
    ok1 = len(sols) == 1 and sp.simplify(pv - sp.sympify(md_answer(2, "1"))) == 0
    # (2) A_n 전수 열거
    def A(m):
        return sum(1 for a, b, c in product(range(1, m + 1), repeat=3) if a < b + c and b < a + c and c < a + b)
    nn = sp.symbols("n")
    expr = sp.sympify(md_answer(2, "2"), locals={"n": nn})
    vals = [A(m) for m in range(1, 27)]
    ok2 = all(vals[m] - vals[m - 1] == expr.subs(nn, m) for m in range(1, 26))
    # (3) [증명-수치확인] 3의 배수 n=3..600, 정확한 정수 비교 + min f(x)(x>1) 확인
    ok3 = all(10**n_ * sum(comb(n_, k) for k in range(n_ // 3 + 1)) < 19**n_ for n_ in range(3, 601, 3))
    fx = x ** sp.Rational(-2, 3) * (x + 1)
    cr = list(sp.solveset(sp.diff(fx, x), x, sp.Interval.open(1, sp.oo)))
    fmin = fx.subs(x, cr[0])
    ok3 &= len(cr) == 1 and float(fmin) < 1.9
    report(2, ok1 and ok2 and ok3, f"(1)AP={sols} proj={pv} (2)A_n(1..6)={vals[:6]} (3)n≤600 OK, min f=f({cr[0]})={sp.N(fmin, 8)} [증명-수치확인]")


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
