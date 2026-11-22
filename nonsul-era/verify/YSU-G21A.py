#!/usr/bin/env python3
"""YSU-G21A (연세대 2021학년도 수시 논술 자연계열 오전(자연1) 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "YSU-G21A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/YSU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 100명 중 k명 선정 → 회장 2, 부회장 5, 나머지 위원: 전수 계산
    f = {k: math.comb(100, k) * math.comb(k, 2) * math.comb(k - 2, 5) for k in range(10, 101)}
    m = max(f.values())
    best = {k for k, v in f.items() if v == m}
    md = {int(s) for s in re.findall(r"\d+", md_answer(1, "1"))}
    report(1, best == md, f"argmax k = {sorted(best)}")


def p02():
    a, b, k = sp.symbols("a b k", real=True)
    g = lambda u: sp.cos(sp.pi * u) + k * sp.sin(sp.pi * u)       # k = g(1/2) 인 해의 족
    eq_ok = sp.simplify(sp.expand_trig(g(a + b) + g(a - b) - 2 * g(a) * sp.cos(b * sp.pi))) == 0
    cond_ok = sp.simplify(g(2020) - 1) == 0
    # 대입식 ①②③의 일차결합이 유일성을 준다: 임의 연속 g 에 대해 ①+②-③ 으로 g 가 결정됨(기호 확인)
    G = sp.Function("G")
    e1 = G(x) + G(-x) - 2 * sp.cos(sp.pi * x)                      # ①(g(0)=1 사용)
    e2 = G(x + 1) + G(x)                                           # ②
    e3 = G(x + 1) + G(-x) + 2 * G(sp.Rational(1, 2)) * sp.sin(sp.pi * x)   # ③
    comb = sp.expand(e1 + e2 - e3)
    uniq_ok = sp.simplify(comb - (2 * G(x) - 2 * sp.cos(sp.pi * x) - 2 * G(sp.Rational(1, 2)) * sp.sin(sp.pi * x))) == 0
    I = sp.integrate(g(x), (x, -sp.Rational(1, 2), sp.Rational(1, 2)))
    ok1 = sp.simplify(I - sp.sympify(md_answer(2, "1"))) == 0 and k not in I.free_symbols
    sol = sp.solve(sp.Eq(g(sp.Rational(1, 3)) * g(-sp.Rational(1, 3)), sp.Rational(1, 10)), k)
    sq = {sp.nsimplify(sp.simplify(s**2)) for s in sol}
    ok2 = len(sq) == 1 and sp.simplify(sq.pop() - sp.sympify(md_answer(2, "2"))) == 0
    report(2, eq_ok and cond_ok and uniq_ok and ok1 and ok2, f"int={I} g(1/2)^2 from k={sol}")


def p03():
    # t=x-2020 은 정수 계수 사이의 전단사이므로 p(t)=a t^2+b t+c 를 넓은 상자에서 전수 조사
    def maxabs(a, b, c):
        pts = [sp.Integer(-1), sp.Integer(1)]
        if a != 0:
            v = sp.Rational(-b, 2 * a)
            if -1 <= v <= 1:
                pts.append(v)
        return max(abs(a * t**2 + b * t + c) for t in pts)
    R = range(-6, 7)
    cnt = sum(1 for a in R for b in R for c in R if maxabs(a, b, c) == 1)
    # 상자 경계 밖 해가 없음을 확인: 상자를 넓혀도 개수 불변
    R2 = range(-9, 10)
    cnt2 = sum(1 for a in R2 for b in R2 for c in R2 if maxabs(a, b, c) == 1)
    # 원래 좌표에서 표본 점검: 해 하나를 원 계수로 되돌려 [2019,2021] 위에서 수치 확인
    import numpy as np
    xs = np.linspace(2019, 2021, 20001)
    A, B, C = 2, -2 * 2 * 2020, 2 * 2020**2 - 1                     # p(t)=2t^2-1
    chk = abs(np.max(np.abs(A * (xs - 2020) ** 2 - 1)) - 1) < 1e-9 and A * 2020**2 + B * 2020 + C == -1
    ok = cnt == cnt2 == int(md_answer(3, "1")) and chk
    report(3, ok, f"count={cnt} (wider box {cnt2})")


def p04():
    from fractions import Fraction
    def fg(n):
        e = list(sp.factorint(n).values())
        return sum(e) == math.prod(e)
    p1 = Fraction(sum(fg(n) for n in range(2, 21)), 19)
    S = [n for n in range(2, 2022) if fg(n)]
    P = [n for n in S if sp.isprime(n)]
    p2 = Fraction(len(P), len(S))
    ok = p1 == Fraction(md_answer(4, "1")) and p2 == Fraction(md_answer(4, "2")) and len(P) == 306
    report(4, ok, f"P1={p1} |S|={len(S)} primes={len(P)} P2={p2}")


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
