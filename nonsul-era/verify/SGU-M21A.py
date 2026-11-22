#!/usr/bin/env python3
"""SGU-M21A (서강대 2021학년도 모의논술 1차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M21A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    a = sp.symbols("a", positive=True)
    k, N = sp.symbols("k N", integer=True, positive=True)
    # (1) [증명-수치확인] (1+a)^{n+1} >= n^2 a^2/2 격자 검사 + 극한
    ok1 = all((1 + av) ** (nv + 1) >= nv**2 * av**2 / 2 for nv in range(1, 201)
              for av in [sp.Rational(i, 50) for i in range(1, 501, 7)])
    lim1 = sp.limit(n / 2**n, n, sp.oo)
    ok1 = ok1 and lim1 == 0 and md_answer(1, "1").replace(" ", "").endswith("=0")
    # (2) 미분한 부분합 = 주어진 식, 급수의 합
    r = sp.symbols("r")
    S = sum(r**j for j in range(1, 9))
    lhs = sp.diff(S, r)
    rhs = (1 - 9 * r**8 + 8 * r**9) / (1 - r) ** 2
    ok2a = sp.simplify(lhs - rhs) == 0
    tot = sp.summation(k / 2**k, (k, 1, sp.oo))
    ok2 = ok2a and sp.simplify(tot - sp.sympify(md_answer(1, "2"))) == 0
    # (3) 등식 수치 확인 (여러 r, n)
    import mpmath as mp
    ok3 = all(abs(sum(rv**j / j for j in range(1, nv + 2)) -
                  (-mp.log(1 - rv) - mp.quad(lambda tt: tt**(nv + 1) / (1 - tt), [0, rv]))) < 1e-12
              for rv in (mp.mpf("0.1"), mp.mpf("0.5"), mp.mpf("0.93")) for nv in (1, 3, 10))
    # (4)
    tot4 = sp.summation(1 / (k * 2**k), (k, 1, sp.oo))
    ok4 = sp.simplify(tot4 - sp.sympify(md_answer(1, "4"))) == 0
    rem = mp.quad(lambda tt: tt**41 / (1 - tt), [0, 0.5])
    ok4 = ok4 and 0 <= rem <= 0.5**41 * mp.log(2)
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] lim n/2^n={lim1} sum n/2^n={tot} sum 1/(n2^n)={sp.simplify(tot4)}")


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
