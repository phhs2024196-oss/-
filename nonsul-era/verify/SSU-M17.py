#!/usr/bin/env python3
"""숭실대 SSU-M17-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M17"
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
    """A: 자연수 해 전수 열거, B: sympy 정적분."""
    ans = load_answers(1)
    A = [(X, Y, 100 - 2*X - 2*Y) for X in range(1, 50) for Y in range(1, 50) if 100 - 2*X - 2*Y >= 1]
    k1 = sum(1 for (X, Y, Z) in A if X + Y + Z == 60)
    k2 = len(A) - k1
    W = 10 + sp.integrate(4*(t+2)*sp.log(t+2) - 2*sp.log(t+1)/(t+1), (t, 0, 2))
    okB = sp.simplify(sp.expand_log(W - sp.sympify(ans["B"]), force=True)) == 0 or abs(sp.N(W - sp.sympify(ans["B"]), 30)) < 1e-25
    ok = k1 == int(ans["A1"]) and k2 == int(ans["A2"]) and okB
    report(1, ok, f"A1={k1} md={ans['A1']} | A2={k2} md={ans['A2']} | B={sp.N(W,15)} md={ans['B']}")


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
