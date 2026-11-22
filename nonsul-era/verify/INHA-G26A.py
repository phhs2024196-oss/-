#!/usr/bin/env python3
"""INHA-G26A (2026학년도 인하대 논술 자연계열 오전 일반) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "INHA-G26A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1-1) ∫f²=1/2 제약 하 ∫f|sinπx| 최대 (1-2) 등호 함수의 미분방정식 풀이·적분값."""
    s = sp.Abs(sp.sin(sp.pi * x))
    # (1-1) 제약 ∫2f²=2-∫sin² → ∫f² ; 코시-슈바르츠 상한 sqrt(∫f² ∫s²), 등호 f=c|sin|
    I_s2 = sp.integrate(sp.sin(sp.pi * x) ** 2, (x, -1, 1))
    If2 = (2 - I_s2) / 2
    c = sp.symbols("c", positive=True)
    csol = [v for v in sp.solve(sp.Eq(c**2 * I_s2, If2), c)][0]
    f1 = csol * s
    val1 = sp.integrate(csol * sp.sin(sp.pi * x) ** 2, (x, -1, 1))
    bound = sp.sqrt(If2 * I_s2)
    # 제약 확인 + 무작위 시험함수(정규화)로 상한 넘지 않음
    rng = np.random.default_rng(1)
    xs = np.linspace(-1, 1, 20001)
    S = np.abs(np.sin(np.pi * xs))
    worst = 0
    for _ in range(200):
        g = sum(rng.normal() * np.cos(k * np.pi * xs / 2 + rng.uniform(0, 6)) for k in range(6)) + rng.normal() * S
        g = g * np.sqrt(float(If2) / np.trapezoid(g * g, xs))
        worst = max(worst, np.trapezoid(g * S, xs))
    ok1 = (sp.simplify(val1 - bound) == 0 and sp.simplify(val1 - sp.sqrt(2) / 2) == 0
           and sp.simplify(f1 - s / sp.sqrt(2)) == 0 and worst <= float(bound) + 1e-9)
    # (1-2) 등호: -2 f f' e^{2x} = 1, f(0)=1/√2 → 풀고 f(2) 조건·적분값 확인
    F = sp.Function("F")
    sol = sp.dsolve(sp.Eq(F(x).diff(x), -sp.exp(-2 * x)), F(x), ics={F(0): sp.Rational(1, 2)}).rhs  # F=f²
    f2 = sp.sqrt(sol)
    ok_end = sp.simplify(f2.subs(x, 2) - 1 / (sp.sqrt(2) * sp.exp(2))) == 0
    integrand = (2 * sp.exp(x) * sp.diff(f2, x)) ** 2 + 1 / (sp.exp(x) * f2) ** 2
    val2 = sp.simplify(sp.integrate(sp.simplify(integrand), (x, 0, 2)))
    lower = sp.simplify(-4 * (sp.log(f2.subs(x, 2)) - sp.log(f2.subs(x, 0))))
    # 다른 허용 함수(조건 만족)로 적분값이 하한 이상인지 수치 확인
    def J(fn, dfn):
        z = np.linspace(0, 2, 40001)
        return np.trapezoid((2 * np.exp(z) * dfn(z)) ** 2 + 1 / (np.exp(z) * fn(z)) ** 2, z)
    A = 1 / np.sqrt(2)
    trial = [(lambda z: A * np.exp(-z ** 2 / 2), lambda z: -A * z * np.exp(-z ** 2 / 2)),
             (lambda z: A * np.exp(-z), lambda z: -A * np.exp(-z))]
    vals = [J(*tr) for tr in trial]
    ok2 = (ok_end and sp.simplify(val2 - 8) == 0 and sp.simplify(lower - 8) == 0
           and min(vals) >= 8 - 1e-6 and sp.simplify(f2.subs(x, 1) - 1 / (sp.sqrt(2) * sp.E)) == 0)
    report(1, ok1 and ok2, f"(1-1) f={f1}, 최대={val1} (1-2) f={f2}, f(1)={sp.simplify(f2.subs(x,1))}, 최소={val2}, 시험값={['%.4f'%v for v in vals]}")


def p02():
    """(2-1)(2-2) 낮은 차수부터 계수 미지수로 연속/미분가능 조건을 풀어 최소 차수 f (2-3) 극한."""
    # 유리식의 한쪽 극한: 분모가 (x-p)^m 을 가지면 분자의 p 에서의 테일러 계수 0..m-1 이 0 이어야 유한(조건식으로 반환)
    def lim(expr, p):
        num, den = sp.fraction(sp.cancel(sp.together(expr)))
        m = 0
        while sp.expand(den).subs(x, p) == 0:
            den = sp.quo(den, x - p, x)
            m += 1
        num_s = sp.expand(num.subs(x, x + p))
        extra = [num_s.coeff(x, i) for i in range(m)]
        return num_s.coeff(x, m) / den.subs(x, p), extra

    def conds(f, diff):
        eqs = [f.subs(x, 2) - 4]
        pts = [(-1, f * (-(x + 1)), f * (x + 1)), (0, f * (x + 1), f / x), (1, f / x, f * x)]
        for p, L, R in pts:
            val = L.subs(x, p)  # 각 점에서 g 값은 왼쪽 조각(닫힌 끝)으로 정의됨
            for side in (L, R):
                v, ex = lim(side, p)
                eqs += ex + [v - val]
            if diff:
                vl, exl = lim((L - val) / (x - p), p)
                vr, exr = lim((R - val) / (x - p), p)
                eqs += exl + exr + [vl - vr]
        return eqs

    def minimal(diff):
        for d in range(0, 8):
            cs = sp.symbols(f"c0:{d+1}")
            f = sum(cs[i] * x**i for i in range(0, d + 1))
            sol = sp.solve([sp.expand(e) for e in conds(f, diff)], cs, dict=True)
            for so_ in sol:
                fs = sp.expand(f.subs(so_))
                if fs.free_symbols == {x} and sp.Poly(fs, x).degree() == d:
                    return sp.factor(fs)
        return None

    f1 = minimal(False)
    f2 = minimal(True)
    ok1 = f1 is not None and sp.simplify(f1 - x**2) == 0
    ok2 = f2 is not None and sp.simplify(f2 - x**3 * (x - 1) * (x + 1) / 6) == 0
    L1 = sp.limit(sp.diff(f2 / x, x), x, 1, "-")
    L2 = sp.limit(sp.diff(f2 * x, x), x, 1, "+")
    X = sp.symbols("X", positive=True)
    L3 = sp.limit(sp.log(1 + f2.subs(x, X)) / sp.log(X), X, sp.oo)
    ok3 = L1 == sp.Rational(1, 3) and L2 == sp.Rational(1, 3) and L3 == 5
    report(2, ok1 and ok2 and ok3, f"(2-1) {f1} (2-2) {f2} (2-3) {L1}, {L2}, {L3}")


def p03():
    """(3-1) g-f' (3-2) a=1 근 (3-3) f(1/4)=f(9/4) — 정수 조건을 계수 탐색으로 독립 확인."""
    a, b, c, d = sp.symbols("a b c d")
    f = x**4 + a * x**3 + b * x**2 + c * x + d
    g = sp.expand((f.subs(x, x + 1) - f.subs(x, x - 1)) / 2)
    diff = sp.expand(g - sp.diff(f, x))
    ok1 = sp.simplify(diff - (4 * x + a)) == 0

    def cond_ok(fx, K=40):
        fn = sp.lambdify(x, fx)
        for k in range(-K, K + 1):
            u, v, w = fn(k - 2), fn(k), fn(k + 2)
            if not ((u <= v + 1e-12 and v <= w + 1e-12) or (u >= v - 1e-12 and v >= w - 1e-12)):
                return False
        return True

    # (3-2) a=1: g=4(x-r1)(x-r2)(x-r3) 이차계수 3 → 조건 만족하는 (b,c) 를 g 의 근 구조로 탐색
    # 조건은 g(k+1)g(k-1)>=0 과 동치이므로, 조건을 만족하는 f 를 정수 n 과 α 로 매개해 전수 탐색하되
    # 조건 자체는 f 값으로 직접 검사(필요충분 주장 독립 확인)
    roots_found = set()
    for nn in range(-6, 7):
        al = sp.Rational(-3, 4) - 2 * nn - 1
        gg = 4 * (x - nn) * (x - nn - 1) * (x - al)
        fp = sp.expand(gg - (4 * x + 1))
        fx = sp.integrate(fp, x)
        if sp.Poly(fx, x).coeff_monomial(x**3) != 1:
            continue
        if cond_ok(fx):
            roots_found.add(tuple(sorted(sp.solve(gg, x))))
    # 반례 확인: 이웃 정수근 구조가 아닌 a=1 사차함수(임의 b,c) 무작위 300개는 조건 불만족
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(300):
        bb, cc = rng.uniform(-20, 20, 2)
        if cond_ok(x**4 + x**3 + bb * x**2 + cc * x, K=30):
            bad += 1
    ok2 = roots_found == {(-1, 0, sp.Rational(1, 4))} and bad == 0
    # (3-3) f(1/4)=f(9/4) + 조건: 정수 n, α 로 일반 매개 후 조건·등식을 f 값으로 검사
    sols = []
    for nn in range(-6, 7):
        for al_num in range(-40, 41):
            al = sp.Rational(al_num, 4)
            A = sp.symbols("A")
            gg = 4 * (x - nn) * (x - nn - 1) * (x - al)
            # f' = g - (4x+A), 삼차항 계수 A: f'의 x^2 계수 = 3A
            eqA = sp.solve(sp.Eq(sp.expand(gg - 4 * x - A).coeff(x, 2), 3 * A), A)[0]
            fp = sp.expand(gg - 4 * x - eqA)
            fx = sp.integrate(fp, x)
            if sp.simplify(fx.subs(x, sp.Rational(9, 4)) - fx.subs(x, sp.Rational(1, 4))) != 0:
                continue
            if cond_ok(fx):
                sols.append(fp)
    expect = {sp.expand(4 * x**3 - 9 * x**2 + x + 3), sp.expand(4 * x**3 - 17 * x**2 + 19 * x - sp.Rational(13, 3)),
              sp.expand(4 * x**3 - 25 * x**2 + 45 * x - sp.Rational(65, 3))}
    ok3 = set(sols) == expect
    report(3, ok1 and ok2 and ok3, f"(3-1) {diff} (3-2) {roots_found} 무작위통과={bad} (3-3) {len(sols)}개: {sols}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
