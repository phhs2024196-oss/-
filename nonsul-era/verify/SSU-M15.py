#!/usr/bin/env python3
"""숭실대 SSU-M15-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M15"
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
    """(1) 기울기 m<0 을 격자 탐색해 최소 넓이와 2ab 비교, (2) 곡선 위 점 격자에서 F 최대."""
    import mpmath as mp
    ans = load_answers(1)
    a_, b_ = sp.symbols("a b", positive=True)
    Fmd = sp.sympify(ans["1"], locals={"a": a_, "b": b_})
    import random
    random.seed(3); ok1 = True
    for _ in range(20):
        av, bv = random.uniform(0.2, 5), random.uniform(0.2, 5)
        S = lambda m: 0.5 * (av - bv/m) * (bv - av*m)
        mbest = mp.findroot(lambda m: mp.diff(S, m), -bv/av * 1.3)
        smin = min(S(-10**(k/500)) for k in range(-2000, 2001))
        ok1 &= abs(S(mbest) - float(Fmd.subs({a_: av, b_: bv}))) < 1e-9 and smin >= S(mbest) - 1e-9
    Fl = sp.lambdify((a_, b_), Fmd, "mpmath")
    G = lambda av: Fl(av, 2*mp.sqrt(1 - av**2/9))
    a0 = mp.findroot(lambda s: mp.diff(G, s), 2.0)
    P = sp.sympify(ans["2"])
    ok2 = abs(a0 - float(P[0])) < 1e-9 and abs(2*mp.sqrt(1 - a0**2/9) - float(P[1])) < 1e-9
    report(1, ok1 and ok2, f"(1) 무작위 (a,b) 20개 최소넓이=2ab {ok1} | (2) a*={mp.nstr(a0,12)}, F={mp.nstr(G(a0),12)}, md={P}")


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
