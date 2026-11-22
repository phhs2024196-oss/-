#!/usr/bin/env python3
"""HYU-G23A (한양대 2023학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from fractions import Fraction
from math import comb

import sympy as sp

PREFIX = "HYU-G23A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def foot_P(c, s):
    """A(-1,0), R(c,s), H(c,0) (c=cos t, s=sin t); H 에서 직선 AR 에 내린 수선의 발 P (사영으로 독립 계산)."""
    A = sp.Matrix([-1, 0])
    R = sp.Matrix([c, s])
    H = sp.Matrix([c, 0])
    d = R - A
    P = A + d * ((H - A).dot(d) / d.dot(d))
    return P


def p01():
    c = sp.symbols("c", real=True)
    s = sp.symbols("s", positive=True)
    O, B = sp.Matrix([0, 0]), sp.Matrix([1, 0])
    # (1) ∠ROB = π/3
    P1 = foot_P(sp.Rational(1, 2), sp.sqrt(3) / 2)
    v1 = sp.nsimplify(sp.simplify((P1 - O).dot(P1 - O) + (P1 - B).dot(P1 - B)))
    ok1 = sp.simplify(v1 - sp.sympify(md_answer(1, "1"))) == 0
    # (2) t∈[0,π] ↔ c=cos t ∈[-1,1] (단조), s^2=1-c^2
    P = foot_P(c, s)
    op2 = sp.factor(sp.simplify(P.dot(P)).subs(s, sp.sqrt(1 - c**2)))
    crit = [r for r in sp.solve(sp.diff(op2, c), c) if r.is_real and -1 < r < 1]
    cands = [sp.simplify(op2.subs(c, r)) for r in crit] + [sp.limit(op2, c, -1, "+"), op2.subs(c, 1)]
    m2 = min(cands, key=lambda e: float(e))
    import numpy as np
    f = sp.lambdify(c, op2, "numpy")
    grid = min(f(np.linspace(-1 + 1e-9, 1, 200001)))
    md2 = sp.sympify(md_answer(1, "2"))
    ok2 = sp.simplify(sp.sqrt(m2) - md2) == 0 and abs(grid - float(md2) ** 2) < 1e-9
    # (3) V = ∫ y^2 dx,  x(c) 는 c 에 대해 증가(c:-1→1 ↔ x:-1→1)
    X = sp.simplify(P[0].subs(s, sp.sqrt(1 - c**2)))
    Y2 = sp.simplify((P[1] ** 2).subs(s, sp.sqrt(1 - c**2)))
    V = sp.simplify(sp.integrate(sp.cancel(Y2 * sp.diff(X, c)), (c, -1, 1)))
    md3 = sp.sympify(md_answer(1, "3"))
    ok3 = sp.simplify(V - md3) == 0 and sp.simplify(X.subs(c, -1) + 1) == 0 and sp.simplify(X.subs(c, 1) - 1) == 0
    report(1, ok1 and ok2 and ok3, f"(1)={v1} (2)OP^2={op2} min={m2} grid={grid:.12f} (3)x(c)={X} V={V}")


def p02():
    # (1) a_n = (n+1)∫_0^β sin^n x cos x dx, 합 1/6 → tan β
    b, s = sp.symbols("b s", positive=True)
    k = sp.symbols("k", integer=True, positive=True)
    u = sp.symbols("u", positive=True)
    # 치환 u = sin x (du = cos x dx): ∫_0^β sin^k x cos x dx = ∫_0^{sin β} u^k du
    chk = sp.integrate((sp.sin(x) ** 3 * sp.cos(x)), (x, 0, sp.pi / 5)) - sp.integrate(u**3, (u, 0, sp.sin(sp.pi / 5)))
    an = sp.simplify((k + 1) * sp.integrate(u**k, (u, 0, s)))
    ssum = sp.summation(an, (k, 1, sp.oo))
    ssum = sp.piecewise_fold(ssum)
    ssum = ssum.args[0][0] if isinstance(ssum, sp.Piecewise) else ssum
    sols = [r for r in sp.solve(sp.Eq(ssum, sp.Rational(1, 6)), s) if 0 < r < 1]
    tanb = sp.simplify(sols[0] / sp.sqrt(1 - sols[0] ** 2))
    ok1 = sp.simplify(chk) == 0 and len(sols) == 1 and sp.simplify(tanb - sp.sympify(md_answer(2, "1"))) == 0
    # (2) 정규분포 조건 — Φ 로 직접 방정식 풀기(대칭성 가정 없이 수치 풀이)
    from scipy.optimize import brentq
    from scipy.stats import norm
    # (나),(다) 는 m 이 정해지면 σ, n 을 결정. (가): Φ((m-8)/σ)+Φ((m-8)√n/σ)=1 은 m 에 대해 단조 → 근 m=8 확인
    # (다): 2σ/√n = 1 → σ = √n/2. (나): P(X≥12)+P(X̄≥7.5)=1
    def cond_b(nn, mm):
        sg = (nn ** 0.5) / 2
        return norm.sf((12 - mm) / sg) + norm.sf((7.5 - mm) / (sg / nn ** 0.5)) - 1
    def cond_a(mm, nn):
        sg = (nn ** 0.5) / 2
        return norm.sf((8 - mm) / sg) + norm.sf((8 - mm) / (sg / nn ** 0.5)) - 1
    # m 과 n 동시 풀이: 고정점 반복
    mm, nn = 7.0, 10.0
    for _ in range(200):
        mm = brentq(lambda v: cond_a(v, nn), -100, 100)
        nn = brentq(lambda v: cond_b(v, mm), 1.0001, 10000)
    sg = nn ** 0.5 / 2
    tot = mm + sg + nn
    ok2 = abs(round(nn) - nn) < 1e-6 and abs(tot - float(sp.sympify(md_answer(2, "2")))) < 1e-6
    # (3) 짝수 확률 — 정확한 유리수 합으로 여러 p 에서 비교
    p = sp.symbols("p")
    expr = sp.sympify(md_answer(2, "3"), locals={"p": p})
    ok3 = True
    for pv in (Fraction(1, 3), Fraction(2, 7), Fraction(1, 2), Fraction(9, 10)):
        ev = sum(comb(2023, j) * pv ** j * (1 - pv) ** (2023 - j) for j in range(0, 2024, 2))
        ok3 &= sp.Rational(ev.numerator, ev.denominator) == expr.subs(p, sp.Rational(pv.numerator, pv.denominator))
    report(2, ok1 and ok2 and ok3, f"(1)sinβ={sols} tanβ={tanb} (2)m={mm:.6f} σ={sg:.6f} n={nn:.6f} sum={tot:.6f} (3)exact p∈{{1/3,2/7,1/2,9/10}} {ok3}")


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
