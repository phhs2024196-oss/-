#!/usr/bin/env python3
"""<학교>-<유형> 세트 <n> (<ID 범위>) — sympy 검산. 파일명: verify/<학교>-<유형>-s<n>.py
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "SSU-T03"          # ← 학교-유형 (기출이면 "SSU-G26", 세종 A형이면 "SJU-G26A")
WIDTH = 3                   # ← 순번 자릿수 (자체 문항 3, 기출 2)
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def example_p11():   # ← 예시. 실제 함수는 이름을 p11 처럼 바꿔야 실행된다(예시 그대로 두면 실행 안 됨)
    """예시: 문제 설정으로부터 다시 계산해 md 의 정답과 비교한다."""
    f = 2 * x**3 + 3
    ok = sp.simplify(f.subs(x, 2) - 19) == 0          # md 정답 19
    report(11, ok, f"f(2)={f.subs(x, 2)}")


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
