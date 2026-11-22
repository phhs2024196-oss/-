#!/usr/bin/env python3
"""AJOU-M24 (아주대 2024학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "AJOU-M24"
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
import math
from fractions import Fraction


def S(a: int, b: int, upto: int) -> set:
    return {a * m + b * k for m in range(upto // a + 1) for k in range(upto // b + 1) if a * m + b * k <= upto}


def p01():
    # [1-1](1) A∩S(10,15), A = {0..2024}
    c1 = len(S(10, 15, 2024))
    ok1 = c1 == int(md_answer(1, "1-1"))
    # [1-1](2) 'a 이상의 모든 자연수가 S(3,4)' 를 1..2000 범위에서 판정
    s34 = S(3, 4, 2000)
    amin = min(a for a in range(1, 1990) if all(v in s34 for v in range(a, 2001)))
    ok2 = amin == int(md_answer(1, "1-2"))
    # [1-1](3) '모든 양의 7의 배수가 S(2,b)' 인 b>2 (7의 배수 7..1400 확인)
    bs = [b for b in range(3, 200) if all(7 * j in S(2, b, 1400) for j in range(1, 201))]
    ok3 = bs == sorted(ast.literal_eval(md_answer(1, "1-3")))
    # [1-2](1) 증명: X_n (n=1..5) 의 모든 함수 중 f∘f=f 이면 치역 = 고정점 집합 (반례 탐색)
    bad = 0
    for N in range(1, 6):
        for fv in itertools.product(range(N), repeat=N):
            if all(fv[fv[i]] == fv[i] for i in range(N)) and set(fv) != {i for i in range(N) if fv[i] == i}:
                bad += 1
    ok4 = bad == 0 and md_answer(1, "2-1") == "None"
    # [1-2](2) X_3 에서 f∘f=f 인 함수 개수
    c5 = sum(1 for fv in itertools.product(range(3), repeat=3) if all(fv[fv[i]] == fv[i] for i in range(3)))
    ok5 = c5 == int(md_answer(1, "2-2"))
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"n={c1} a={amin} b={bs} [증명-수치확인] n<=5 반례 {bad}개 idempotent(X3)={c5}")


def play(frac: Fraction, rounds: int = 50, start: Fraction = Fraction(100)) -> list:
    """매 게임 가진 점수의 frac 을 제시, 홀수 번째 앞면(제시분 2배), 짝수 번째 뒷면(제시분 반)."""
    v, out = start, []
    for k in range(1, rounds + 1):
        bet = v * frac
        v = (v - bet) + (2 * bet if k % 2 == 1 else bet / 2)
        out.append(v)
    return out


def p02():
    # 제시문 예시: 100점 중 50점 제시 → 앞면 150, 뒷면 75
    assert (Fraction(50) + 2 * 50, Fraction(50) + Fraction(50, 2)) == (150, 75)
    xs = play(Fraction(1))
    ok1 = xs[-1] == int(md_answer(2, "1-1"))
    ys = play(Fraction(1, 2))
    r1 = {ys[2 * m - 1] / ys[2 * m - 2] for m in range(1, 26)}          # y_{2n}/y_{2n-1}
    r2 = {ys[2 * m] / ys[2 * m - 1] for m in range(1, 25)}              # y_{2n+1}/y_{2n}
    parts = dict(seg.split("=") for seg in md_answer(2, "1-2").replace(" ", "").split(";"))
    ok2 = r1 == {Fraction(parts["y2n/y2n_1"])} and r2 == {Fraction(parts["y2n1/y2n"])}
    ok3 = sp.Rational(ys[-1].numerator, ys[-1].denominator) == sp.sympify(md_answer(2, "1-3"))
    # (4) α 를 1/1000 격자로 훑어 z_50 최대
    best = max((Fraction(i, 1000) for i in range(1001)), key=lambda al: play(al)[-1])
    ok4 = best == Fraction(sp.sympify(md_answer(2, "1-4")).p, sp.sympify(md_answer(2, "1-4")).q)
    # 2-2: 실제 게임당 평균 수익률 r = (y50/100)^(1/50) - 1, 소수점 셋째 자리에서 버림
    r = (float(ys[-1]) / 100) ** (1 / 50) - 1
    r_trunc = math.floor(r * 100) / 100
    # 원 해설 방식 재현: log(1+r) = (2log3-3log2)/2, 상용로그표 1.06=0.0253, 1.07=0.0294
    lg = (2 * 0.4771 - 3 * 0.3010) / 2
    ok5 = abs(r_trunc - float(md_answer(2, "2"))) < 1e-12 and 0.0253 < lg < 0.0294
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"x50={xs[-1]} 비={r1},{r2} y50=100*(9/8)^25:{ys[-1] == 100 * Fraction(9, 8)**25} α={best} r={r:.5f}→{r_trunc} log(1+r)={lg:.4f}")


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
