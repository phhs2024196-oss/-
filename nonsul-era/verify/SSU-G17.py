#!/usr/bin/env python3
"""숭실대 SSU-G17-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G17"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answers(seq):
    """소문항 id → answer.value(문자열). 증명 소문항은 None."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return {str(s["id"]): s["answer"]["value"] for s in d["subparts"]}


def p01():
    """(a,b,c) 자연수 전수 열거로 두 조건을 직접 판정(정적분·도함수는 sympy 로 한 번 기호 계산 후 대입)."""
    ans = load_answers(1)
    X, a_, b_, c_ = sp.symbols("X a b c")
    f = a_ * X**2 + b_ * X + c_
    I = sp.lambdify((a_, b_, c_), sp.integrate(f, (X, -sp.sqrt(3), sp.sqrt(3))) / sp.sqrt(3))   # 정적분/√3
    D = sp.lambdify((a_, b_, c_), sp.diff(f, X).subs(X, sp.Rational(-1, 2)))
    c1 = c2 = 0
    for a in range(1, 99):
        for b in range(1, 100 - a):
            c = 100 - a - b
            if abs(I(a, b, c) - 40) < 1e-9:
                c1 += 1
            if D(a, b, c) < 0:
                c2 += 1
    ok = c1 == int(ans["1"]) and c2 == int(ans["2"])
    report(1, ok, f"(1) {c1} md={ans['1']} | (2) {c2} md={ans['2']}")


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
