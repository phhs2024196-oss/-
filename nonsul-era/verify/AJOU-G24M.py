#!/usr/bin/env python3
"""AJOU-G24M (아주대 2024학년도 논술 자연계열 의학과 문항 1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "AJOU-G24M"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


import ast
from fractions import Fraction

import mpmath as mp
mp.mp.dps = 40


def lattice_count(nn: int) -> int:
    """A_n 내부·경계 정수점: p^2/n <= q <= p^2/(2n)+1 (정수 산술)."""
    R = int((2 * nn) ** 0.5) + 3
    return sum(1 for p in range(-R, R + 1) for q in range(0, 6) if p * p <= nn * q and 2 * nn * (q - 1) <= p * p)


def count_interval(k: int, a: Fraction) -> int:
    """[k,k+1) 에서 cos(pi u) = a(k+u) (u=x-k) 의 근 개수. φ 는 (0,u1)↑ (u1,u2)↓ (u2,1)↑ (a<0)."""
    am = mp.mpf(a.numerator) / a.denominator
    phi = lambda u: mp.cos(mp.pi * u) - am * (k + u)
    u1 = mp.asin(-am / mp.pi) / mp.pi
    pts = [mp.mpf(0), u1, 1 - u1, mp.mpf(1)]
    v = [phi(p) for p in pts]
    v[0] = 0 if a * k == 1 else v[0]                       # u=0 정확 판정(유리수)
    tol = mp.mpf(10)**-30
    v = [0 if (i in (1, 2) and abs(val) < tol) else val for i, val in enumerate(v)]
    cnt = 1 if v[0] == 0 else 0
    for i in range(3):
        right_closed = i < 2                               # u=1 은 구간 밖
        if v[i] * v[i + 1] < 0 or (right_closed and v[i + 1] == 0):
            cnt += 1
    return cnt


def total_count(a: Fraction) -> int:
    K = int(1 / abs(a)) + 3
    return sum(count_interval(k, a) for k in range(-K, K + 1))


def p01():
    # [1-1](1) 넓이를 정적분으로 계산해 n 을 탐색
    nn = sp.symbols("nn", positive=True)
    S = sp.integrate(x**2 / (2 * nn) + 1 - x**2 / nn, (x, -sp.sqrt(2 * nn), sp.sqrt(2 * nn)))
    Sf = sp.lambdify(nn, S)
    nmax = max(v for v in range(1, 5000) if Sf(v) <= 100 + 1e-12)
    ok1 = nmax == int(md_answer(1, "1-1"))
    # [1-1](2) 정수점 전수 세기 (제시문 예시 A_2 → 6 확인)
    assert lattice_count(2) == 6
    ns = [v for v in range(1, 3000) if lattice_count(v) == 12]
    ok2 = ns == sorted(ast.literal_eval(md_answer(1, "1-2")))
    # [1-2](1) x=0(정수점의 x좌표가 정수 → g(k)=f(0)=log2 p) 이 정수, |.|<=20 인 p 의 합
    ps = [sp.Integer(2)**e for e in range(-40, 41) if abs(sp.log(sp.Integer(2)**e, 2)) <= 20]
    ok3 = sp.simplify(sum(ps) - sp.sympify(md_answer(1, "2-1"))) == 0
    # [1-2](2) 미분가능 반복함수: f(0)=f(1), f'(0)=f'(1), f(0)=g(3)=sqrt3
    b, c, d = sp.symbols("b c d")
    f = x**3 + b * x**2 + c * x + d
    sol = sp.solve([f.subs(x, 0) - f.subs(x, 1), sp.diff(f, x).subs(x, 0) - sp.diff(f, x).subs(x, 1),
                    f.subs(x, 0) - sp.sqrt(3)], [b, c, d], dict=True)[0]
    f = f.subs(sol)
    cands = [r for r in sp.solve(sp.diff(f, x), x) if 0 <= r < 1] + [sp.Integer(0)]
    gmax = sp.nsimplify(sp.simplify(max((f.subs(x, r) for r in cands), key=lambda e: float(e))), [sp.sqrt(3)])
    ok4 = sp.simplify(gmax - sp.sympify(md_answer(1, "2-2"))) == 0
    # [1-2](3) 교점 개수: a=-1/2024, 열린구간 내 표본, 접하는 a* (k=-2024 에서 φ=φ'=0)
    counts = {total_count(Fraction(-1, 2024))}
    lo, hi = Fraction(-1, 2023), Fraction(-1, 2024)
    for i in range(1, 12):
        counts.add(total_count(lo + (hi - lo) * Fraction(i, 12)))
    k0 = -2024
    F = lambda A, U: [mp.cos(mp.pi * U) - A * (k0 + U), -mp.pi * mp.sin(mp.pi * U) - A]
    A0, U0 = mp.findroot(F, (mp.mpf(-1) / 2023.5, mp.mpf(1) / 2023.5 / mp.pi**2))
    assert lo < Fraction(str(mp.nstr(A0, 35))) < hi
    a_star = Fraction(str(mp.nstr(A0, 35)))
    counts.add(total_count(a_star))
    counts.add(total_count((a_star + hi) / 2))          # a* < a < -1/2024 의 좁은 구간(두 점에서 만남)
    ok5 = counts == set(ast.literal_eval(md_answer(1, "2-3")))
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"n_max={nmax} n={ns} Σp={sum(ps)} max g={gmax} 교점수={sorted(counts)} a*≈{mp.nstr(A0, 15)}")


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
