#!/usr/bin/env python3
"""SGU-M24A (서강대 2024학년도 모의논술 1차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M24A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(str(s["answer"]["value"]) for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    import itertools
    from fractions import Fraction as F
    # (1-1) 운행 = (출발, 도착) 쌍, 다른 역에 도착
    st2 = ["A", "B"]
    legs2 = [(a, b) for a in st2 for b in st2 if a != b]
    c1 = len(list(itertools.product(legs2, repeat=2)))
    # (1-2) 허용 노선: A->B, C->B, B->A, B->C
    allowed = {("A", "B"), ("C", "B"), ("B", "A"), ("B", "C")}
    st3 = ["A", "B", "C"]
    legs3 = [(a, b) for a in st3 for b in st3 if a != b and (a, b) in allowed]
    c2 = len(list(itertools.product(legs3, repeat=3)))
    ok1 = c1 == int(md_answer(1, "1"))
    ok2 = c2 == int(md_answer(1, "2"))

    def f(v):
        v = F(v) - (F(v).numerator // F(v).denominator)   # 주기 1
        return F(2, 3) if v < F(1, 3) else (F(1) if v < F(2, 3) else F(4, 3))

    def fn(k, v):
        for _ in range(k):
            v = f(v)
        return v
    # (1-3) 주기 1 확인(격자) + f^(n)(1/2)=1 인 n
    grid = [F(i, 97) for i in range(-200, 200)]
    okp = all(fn(k, g + 1) == fn(k, g) for k in range(1, 7) for g in grid)
    ns = [k for k in range(1, 61) if fn(k, F(1, 2)) == 1]
    ok3 = okp and ns == [k for k in range(1, 61) if k % 3 == 1] and "3*m + 1" in md_answer(1, "3")
    # (1-4) f^(3n+1) 를 구간 [n+1/3, n+2/3) 의 점에서 직접 계산, 적분·급수
    X = sp.symbols("X", real=True)
    total = 0
    terms = []
    for nn in range(0, 6):
        mids = {fn(3 * nn + 1, F(nn) + F(1, 3) + F(j, 300)) for j in range(0, 100)}
        assert len(mids) == 1
        val = sp.Rational(list(mids)[0].numerator, list(mids)[0].denominator)
        terms.append(sp.integrate((nn + 1) * val * (X - nn)**nn, (X, nn + sp.Rational(1, 3), nn + sp.Rational(2, 3))))
    N = sp.symbols("N", integer=True, nonnegative=True)
    gen = sp.Rational(2, 3)**(N + 1) - sp.Rational(1, 3)**(N + 1)
    okgen = all(sp.simplify(terms[k] - gen.subs(N, k)) == 0 for k in range(6))
    total = sp.summation(gen, (N, 0, sp.oo))
    ok4 = okgen and sp.simplify(total - sp.sympify(md_answer(1, "4"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"c1={c1} c2={c2} n={ns[:4]}.. sum={total}")


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
