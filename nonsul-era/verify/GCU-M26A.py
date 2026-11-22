#!/usr/bin/env python3
"""가천대 2026학년도 논술 모의고사 A형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-M26A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p05():
    sols = [k for k in range(-50, 51)
            if sp.Rational(1, 3) ** (k + 2) < 27 < sp.Rational(1, 9) ** (3 * k)]
    M, N = sum(sols), sp.prod(sols)
    val = sp.simplify(sp.log(sp.Abs(M) ** N, 10))
    report(5, val == 24, f"정수해={sols}, M={M}, N={N}, log|M|^N={val}")


def p06():
    # g = 1/(f^2+3x f) 연속 <=> f(x)(f(x)+3x) != 0 for all real x. 판별식 조건을 sympy 로 풀어 범위 재계산
    k = sp.symbols("k", real=True)
    f = sp.Rational(1, 2) * x**2 - x + k
    conds = []
    for q in (f, sp.expand(f + 3 * x)):
        disc = sp.discriminant(q, x)
        conds.append(sp.solve_univariate_inequality(disc < 0, k, relational=False))
    rng = sp.Intersection(*conds)
    # 경계 근방 수치 확인: k=2 에서는 f+3x=0 인 실근이 생김, k=2.01 에서는 없음
    ok_num = (sp.real_roots(sp.Poly(sp.expand((f * (f + 3 * x)).subs(k, sp.Rational(201, 100))), x)) == []
              and len(sp.real_roots(sp.Poly(sp.expand((f * (f + 3 * x)).subs(k, 2)), x))) > 0)
    report(6, rng == sp.Interval.open(2, sp.oo) and ok_num, f"k 범위={rng} (YAML: k>2)")


def p07():
    # a=BC=4, 3sinA=2sinC, sinA+2sinC=2sinB → 사인법칙으로 c=3a/2, b=(a+2c)/2
    a = sp.Integer(4)
    c = sp.Rational(3, 2) * a
    b = (a + 2 * c) / 2
    cosB = (a**2 + c**2 - b**2) / (2 * a * c)
    S = sp.nsimplify(sp.Rational(1, 2) * a * c * sp.sqrt(1 - cosB**2))
    # 삼각형 성립 확인
    tri = a + b > c and a + c > b and b + c > a
    report(7, tri and sp.simplify(S - 3 * sp.sqrt(15)) == 0, f"b={b}, c={c}, cosB={cosB}, S={S} (YAML 3√15)")


def p08():
    a, b, h = sp.symbols("a b h", real=True)
    f = lambda t: t**2 + a * t + b
    tot = sum(sp.limit((f(kk - 3 + 4 * h) - f(kk - 3 - 2 * h)) / (2 * h), h, 0) for kk in range(1, 10))
    asol = sp.solve(sp.Eq(tot, 0), a)[0]
    fp = sp.diff(f(x), x)
    bsol = sp.solve(sp.Eq(f(3).subs(a, asol), fp.subs({x: 3, a: asol})), b)[0]
    val = f(asol).subs({a: asol, b: bsol})
    report(8, val == 37, f"a={asol}, b={bsol}, f(a)={val} (YAML 37)")


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
