#!/usr/bin/env python3
"""숭실대 SSU-M18-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M18"
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
    """(1) 넓이를 적분으로 다시 세워 h 의 최댓값·위치를 solve, (2) 정삼각형 배열 공을 n=1..30 에 대해 직접 셈."""
    ans = load_answers(1)
    tt = sp.Symbol("tt", positive=True)
    f = sp.integrate(2 * x, (x, 1, tt)); g = sp.integrate(1 / x, (x, 1, tt))
    h = g / (f + 1)
    crit = [c for c in sp.solve(sp.diff(h, tt), tt) if c.is_real and c > 1]
    hmax = h.subs(tt, crit[0])
    a1 = sp.sympify(ans["1"])
    ok1 = len(crit) == 1 and sp.simplify(hmax - a1[0]) == 0 and sp.simplify(crit[0] - a1[1]) == 0 \
        and sp.limit(h, tt, sp.oo) == 0 and h.subs(tt, 1) == 0
    a2 = sp.sympify(ans["2"], locals={"n": n})
    ok2 = True
    for N in range(1, 31):
        layer = lambda k: sum(1 for i in range(k) for j in range(k) if i + j < k)   # 한 변 k개 정삼각형 배열
        ok2 &= layer(N) == a2[0].subs(n, N) and sum(layer(k) for k in range(1, N + 1)) == a2[1].subs(n, N)
    report(1, ok1 and ok2, f"(1) t*={crit}, max={sp.simplify(hmax)} md={ans['1']} | (2) n=1..30 직접 셈 일치={ok2}")


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
