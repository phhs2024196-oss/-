#!/usr/bin/env python3
"""SYU-G25H (삼육대 2025학년도 논술 인문계열 수학 문제 10~15) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 items/SYU/PAST/<ID>.yaml 의 answer.value 와 비교.
p<문항번호>() 함수를 추가하기만 하면 자동으로 실행된다."""
import itertools
import pathlib
import re
import sys

import sympy as sp
import yaml
from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                        standard_transformations)

PREFIX = "SYU-G25H"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def latex_to_sympy(s):
    """'\\dfrac{13\\sqrt{3}}{12}', '1-\\dfrac{\\sqrt{2}}{2}', -557 같은 간단한 값만 처리."""
    if isinstance(s, (int, float)):
        return sp.nsimplify(s)
    s = str(s).replace("\\dfrac", "\\frac")
    for _ in range(5):
        s = re.sub(r"\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    tr = standard_transformations + (implicit_multiplication_application,)
    return sp.simplify(parse_expr(s, transformations=tr))


def load_answer(seq: int):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return latex_to_sympy(d["subparts"][0]["answer"]["value"])


def real_kth_root(v, k):
    """정수 v 의 실수 k제곱근(교과서 정의: k 홀수면 유일한 실근, k 짝수면 v>=0 에서 양의 근). 없으면 None."""
    if k % 2 == 0:
        return None if v < 0 else sp.root(v, k)
    return sp.sign(v) * sp.root(abs(v), k)


def p10():
    # 조건: 정수 a, b, 자연수 k (3<k<50) 에 대해 실수 k제곱근의 곱 = -2^(1/5).
    # 양변을 k제곱하면 (k 홀수일 때) ab = (-1)^k 2^(k/5). 짝수 k 는 두 근 모두 음이 아닌 값 → 곱이 음수 불가.
    ks, abs_ = [], set()
    target = -sp.root(2, 5)
    for k in range(4, 50):
        found = set()
        if k % 2 == 0:
            continue  # 짝수 k: 실수 k제곱근은 음이 아니므로 곱이 음수일 수 없음(전수 근거는 아래 보조 검사)
        val = target**k  # = ab (실수 k제곱근은 k 홀수에서 곱셈적)
        val = sp.nsimplify(sp.simplify(val))
        if val.is_integer:
            # 실제로 정수쌍 (a,b) 가 존재하는지: a=-1, b=-val 또는 a=1, b=val 등
            a, b = 1, int(val)
            lhs = real_kth_root(a, k) * real_kth_root(b, k)
            if sp.simplify(lhs - target) == 0:
                found.add(a * b)
        if found:
            ks.append(k)
            abs_ |= found
    # 짝수 k 보조 검사: 작은 정수 범위에서 해 없음
    even_ok = all(real_kth_root(a, k) is None or real_kth_root(b, k) is None
                  or sp.N(real_kth_root(a, k) * real_kth_root(b, k)) >= 0
                  for k in (4, 6, 10) for a in range(-40, 41) for b in range(-40, 41))
    A, B = sum(ks), sum(abs_)
    got = A + B
    report(10, even_ok and sp.simplify(got - load_answer(10)) == 0, f"k={ks}, A={A}, ab값={sorted(abs_)}, B={B}, A+B={got}")


def p11():
    f = x**3 + 3 * x**2 - 9 * x
    A = B = C = 0
    for k in range(-60, 61):
        rr = set(sp.Poly(f - k, x).real_roots())  # 서로 다른 실근
        a = sum(1 for r in rr if r > 0)
        b = sum(1 for r in rr if r < 0)
        if a == 0 or b == 0:
            continue
        if a == b:
            A += 1
        elif a < b:
            B += 1
        else:
            C += 1
    got = A + B - C
    report(11, sp.simplify(got - load_answer(11)) == 0, f"A={A}, B={B}, C={C}, A+B-C={got} (k 전수 -60..60, 서로 다른 실근 기준)")


def p12():
    N = 15
    a = [None, sp.Integer(4)]
    # 조건식의 n 번째 식에서 n-1 번째 식을 빼면 a_n/(n+1) - a_{n+1}/(n+2) = -2  (n>=2), n=1 식에서 a_2 결정
    for m in range(1, N + 3):
        # sum_{k=1}^{m} (a_k/(k+1) - a_{k+1}/(k+2)) = -2m  → a_{m+1} 결정
        s = sum(a[k] / (k + 1) for k in range(1, m + 1)) - sum(a[k + 1] / (k + 2) for k in range(1, m))
        am1 = sp.symbols("am1")
        sol = sp.solve(sp.Eq(s - am1 / (m + 2), -2 * m), am1)[0]
        a.append(sol)
    val = sum(sp.Rational(20) / a[k] + sp.Rational(24) / a[k + 2] for k in range(1, 10))
    report(12, sp.simplify(val - load_answer(12)) == 0, f"a_1..a_5={a[1:6]}, 합={val}")


def p13():
    f = sp.Abs(sp.sin(sp.pi * x))
    g = sp.sqrt(3) * sp.Abs(sp.cos(sp.pi * x))
    # 교점: 한 주기 [0,1) 안에서 sin^2 = 3cos^2 → tan^2(pi x)=3, 각 [m, m+1) 에서 정확히 풀기
    A = []
    for m in range(0, 6):
        sols = sp.solveset(sp.Eq(sp.sin(sp.pi * x)**2, 3 * sp.cos(sp.pi * x)**2), x, sp.Interval.Ropen(m, m + 1))
        for s0 in sorted(sols, key=lambda e: float(e)):
            if s0 > 0:
                assert sp.simplify(f.subs(x, s0) - g.subs(x, s0)) == 0
                A.append((s0, sp.simplify(f.subs(x, s0))))
    Bx = set()
    for m in range(0, 6):
        Bx |= {s0 for s0 in sp.solveset(sp.sin(sp.pi * x), x, sp.Interval.Ropen(m, m + 1)) if s0 > 0}
        Bx |= {s0 for s0 in sp.solveset(sp.cos(sp.pi * x), x, sp.Interval.Ropen(m, m + 1)) if s0 > 0}
    Bx = sorted(Bx, key=lambda e: float(e))
    tot = 0
    for i in range(9):
        (x1, y1), (x2, y2) = A[i], A[i + 1]
        x3, y3 = Bx[i], 0
        area = sp.Abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
        tot += sp.simplify(area)
    tot = sp.simplify(tot)
    report(13, sp.simplify(tot - load_answer(13)) == 0, f"A_x={[a[0] for a in A[:4]]}..., B_x={Bx[:4]}..., 합={tot}")


def p14():
    a = {1: 1}
    for m in range(1, 7):
        a[m + 1] = (3 * m - 1) * a[m]
    for k in range(8, 85):
        a[k] = a[k - 7] + (k - 7)
    S = lambda N: sum(a[i] for i in range(1, N + 1))  # noqa: E731
    diff = S(84) - S(77)
    got = diff % 110
    report(14, got == load_answer(14), f"a_1..a_7={[a[i] for i in range(1, 8)]}, S84-S77={diff}, mod 110 = {got}")


def p15():
    xs = sp.symbols("xs", real=True)

    # 구간별 닫힌 식(적분 직접 계산)
    f1 = sp.integrate(3 * t**2 - 2 * xs * t, (t, 0, 1))                    # x<=0: 피적분 >=0
    c = sp.Rational(2, 3) * xs
    f2 = sp.integrate(-(3 * t**2 - 2 * xs * t), (t, 0, c)) + sp.integrate(3 * t**2 - 2 * xs * t, (t, c, 1))
    f3 = sp.integrate(-(3 * t**2 - 2 * xs * t), (t, 0, 1))                  # x>=3/2
    crit = [r for r in sp.solve(sp.diff(f2, xs), xs) if r.is_real and 0 < r <= sp.Rational(3, 2)]
    cands = [f2.subs(xs, r) for r in crit] + [f2.subs(xs, sp.Rational(3, 2)), f1.subs(xs, 0)]
    m = sp.simplify(min(cands, key=lambda e: float(e)))
    # 수치 확인: 격자에서 실제 적분값이 m 이상
    import mpmath as mp
    grid_ok = all(mp.quad(lambda tt: abs(3 * tt**2 - 2 * xv * tt), [0, min(max(2 * xv / 3, 0), 1), 1]) >= float(m) - 1e-12
                  for xv in [i / 200 for i in range(-400, 801)])
    report(15, grid_ok and sp.simplify(m - load_answer(15)) == 0,
           f"f2={sp.expand(f2)}, 임계점={crit}, 최솟값={m} (격자 1201점 수치 확인)")


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
