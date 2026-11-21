#!/usr/bin/env python3
"""SJU-G21B (세종대 2021학년도 수시 논술고사 [B]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G21B"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import mpmath as mp
    p = sp.symbols("p", real=True)
    # (1-1) 독립 유도: sqrt(1+p^2) = p + 2e^{-x} 를 p 에 대해 풀기
    sols = [s for s in sp.solve(sp.Eq(1 + p**2, (p + 2 * sp.exp(-x))**2), p)]
    fp = sols[0]
    F = sp.integrate(fp, (x, 0, x)) + 1
    ok1 = sp.simplify(F - sp.sympify(md_answer(1, "1"), locals={"x": x})) == 0
    # (나) 직접 확인: 길이 = f(t) - 2e^{-t} + 1
    Fl = sp.lambdify(x, F, "mpmath"); Fd = sp.lambdify(x, sp.diff(F, x), "mpmath")
    okN = all(abs(mp.quad(lambda s: mp.sqrt(1 + Fd(s)**2), [0, v]) - (Fl(v) - 2 * mp.e**(-v) + 1)) < 1e-12 for v in [0.5, 1, 3])
    # (1-2) 최소점, 접선, 넓이: 영역 = {0<=x<=a, max(ℓ(x), b) <= y <= f(x)} 를 수치 적분
    a = sp.nsolve(sp.diff(F, x), x, 0.7); b = F.subs(x, a)
    ell = sp.diff(F, x).subs(x, 0) * x + F.subs(x, 0)
    lam = sp.lambdify(x, ell, "mpmath")
    area = mp.quad(lambda s: Fl(s) - max(lam(s), b), [0, float(sp.nsolve(ell - b, x, 0.3)), a])
    ok2 = abs(area - float(sp.sympify(md_answer(1, "2")))) < 1e-12 and float(sp.diff(F, x, 2).subs(x, a)) > 0
    # (1-3) A, 그리고 A 를 지나는 직선들 중 (0,1) 까지 거리 최대인 기울기 — 각도 격자 탐색
    T = sp.symbols("T")
    tm = sp.solve(sp.diff(-T**2 + 4 * T + 1, T), T)[0]
    Ax = (-T**2 + 4 * T + 1).subs(T, tm)
    import math
    best = max((abs(math.sin(th) * (0 - float(Ax)) - math.cos(th) * (1 - 0)), th) for th in [k * math.pi / 200000 for k in range(200000)])
    slope = math.tan(best[1])
    md = sp.sympify(md_answer(1, "3").split("=")[1], locals={"x": x})
    ok3 = abs(float(sp.diff(md, x)) - slope) < 1e-2 and abs(float(md.subs(x, Ax))) < 1e-12 and float(Ax) == 5
    report(1, ok1 and okN and ok2 and ok3, f"f={F} a={float(a):.8f} b={float(b):.6f} area={float(area):.12f} A=({Ax},0) slope~{slope:.5f}")


def p02():
    import itertools
    from fractions import Fraction as Fr
    from math import comb
    # (2-1) X 직접 열거
    xs = [X for X in range(11) if Fr(2, 10) <= 1 - Fr(2 * X - 10, 10)**2 <= Fr(4, 10)]
    md = sorted(int(v) for v in md_answer(2, "1").strip("{} ").split(","))
    ok1 = xs == md
    # (2-2) 초기하 확률 정확 합(모든 (a,b) 열거)
    pr = Fr(0)
    for a in range(6):
        for b in range(6):
            c = 10 - a - b
            if 0 <= c <= 10 and a > b > c and 5 - a >= 0 and 5 - b >= 0:
                pr += Fr(comb(10, a) * comb(10, 5 - a), comb(20, 5)) * Fr(comb(10 - a, b) * comb(10 - (5 - a), 5 - b), comb(15, 5))
    ok2 = pr.numerator == int(md_answer(2, "2"))
    # (2-3) 묶음 수 5인 배열 직접 세기: +1 위치 조합 C(20,10) 전수
    cnt = 0
    for pos in itertools.combinations(range(20), 10):
        s = [0] * 20
        for q in pos:
            s[q] = 1
        runs = 1 + sum(1 for i in range(19) if s[i] != s[i + 1])
        cnt += runs == 5
    ok3 = cnt == int(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"X={xs} P={pr} count={cnt}")


def p03():
    import random
    import numpy as np
    xs = np.linspace(-10, 10, 2001)
    # (3-1) [증명-수치확인] m 격자 각각에 대해 x^3 < m x 인 x 존재
    ok1 = all(np.any(xs**3 < m * xs) for m in np.linspace(-50, 50, 1001))
    # (3-2)(3-3) [증명-수치확인] f''>=0 인 함수 몇 개에 대해 A_f = {f'(0)}
    random.seed(1)
    fams = [sp.exp(x) + x**2, sp.log(1 + sp.exp(x)), x**4 - 3 * x, sp.sqrt(1 + x**2) + 2 * x, sp.exp(-x) + x**4]
    ok23 = True
    for F in fams:
        assert all(sp.diff(F, x, 2).subs(x, v) >= 0 for v in [sp.Rational(k, 4) for k in range(-40, 41)])
        fl = sp.lambdify(x, F, "numpy"); d0 = float(sp.diff(F, x).subs(x, 0)); f0 = float(F.subs(x, 0))
        inA = lambda m: np.all(fl(xs) >= f0 + m * xs - 1e-12)
        ok23 &= inA(d0) and not inA(d0 + 0.05) and not inA(d0 - 0.05)
    report(3, ok1 and ok23, "[증명-수치확인] x^3: m∈[-50,50] 모두 반례 존재 / f''>=0 5종에서 f'(0)∈A_f, f'(0)±0.05 ∉ A_f")


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
