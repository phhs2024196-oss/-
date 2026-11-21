#!/usr/bin/env python3
"""숭실대 SSU-M19-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M19"
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
    """원 위 격자 2e5점에서 |..|/5 합을 직접 최소화(절댓값 유지) → 최소점과 md 좌표 비교, 기호 최솟값도 확인."""
    import mpmath as mp
    ans = sp.sympify(load_answers(1)["1"])
    d = lambda X, Y: abs(3*X + 4*Y - 8) / 5 + abs(4*X - 3*Y + 6) / 5
    N = 200000
    best = min((d(mp.cos(2*mp.pi*k/N), mp.sin(2*mp.pi*k/N)), k) for k in range(N))
    th = mp.findroot(lambda s: mp.diff(lambda u: d(mp.cos(u), mp.sin(u)), s), 2*mp.pi*best[1]/N)
    px, py = mp.cos(th), mp.sin(th)
    ok = abs(px - float(ans[0])) < 1e-9 and abs(py - float(ans[1])) < 1e-9
    dmin = sp.nsimplify(d(sp.sympify(ans[0]), sp.sympify(ans[1])))
    ok &= abs(float(sp.Rational(14, 5) - sp.sqrt(2)) - d(px, py)) < 1e-9
    report(1, ok, f"수치 최소점=({mp.nstr(px,12)}, {mp.nstr(py,12)}), md={ans}, 최솟값={mp.nstr(d(px,py),12)}")


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
