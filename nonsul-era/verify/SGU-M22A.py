#!/usr/bin/env python3
"""SGU-M22A (서강대 2022학년도 모의논술 1차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M22A"
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
    N, K = sp.symbols("n k", positive=True)
    loc = {"n": N, "k": K, "L_k": sp.Symbol("L_k")}
    # (1) 전개도: 반지름 3, 중심각 = 2π·1/3 ; OP_j = 3 - 2j/n
    ang = 2 * sp.pi * 1 / 3
    op = lambda j: 3 - 2 * j / N
    L2 = sp.expand(op(K - 1)**2 + op(K)**2 - 2 * op(K - 1) * op(K) * sp.cos(ang))
    md1 = sp.sympify(md_answer(1, "1").split("=", 1)[1], locals=loc)
    ok1 = sp.simplify(L2 - md1) == 0
    # (2)
    j = sp.symbols("j", integer=True, positive=True)
    S = sp.summation(L2.subs(K, j) / N, (j, 1, N))
    lim2 = sp.limit(S, N, sp.oo)
    ok2 = sp.simplify(lim2 - sp.sympify(md_answer(1, "2"))) == 0
    # (3) 평면 좌표로 O 에서 선분 P0P1 까지 최소 거리(격자 + 정확 공식) → 높이 = (3-s)*2√2/3
    import mpmath as mp
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "3").replace(" ", "").split(";"))
    hmd = sp.sympify(parts["h_n"], locals=loc)
    def h_num(nv):
        P0 = mp.matrix([3, 0]); P1 = (3 - mp.mpf(2) / nv) * mp.matrix([mp.cos(2 * mp.pi / 3), mp.sin(2 * mp.pi / 3)])
        best = min(mp.norm(P0 + (P1 - P0) * (i / mp.mpf(4000))) for i in range(4001))
        return (3 - best) * 2 * mp.sqrt(2) / 3
    ok3 = all(abs(h_num(nv) - float(hmd.subs(N, nv))) < 1e-5 for nv in (2, 3, 7, 50))
    lim3 = sp.limit(hmd, N, sp.oo)
    ok3 = ok3 and sp.simplify(lim3 - sp.sympify(parts["limit"])) == 0
    # (4) [증명-수치확인] a_n = OQ_n * h_n^2 감소 (n=2..300)
    b = 3 - hmd * 3 / (2 * sp.sqrt(2))
    a = sp.lambdify(N, b * hmd**2)
    ok4 = all(a(nv) > a(nv + 1) for nv in range(2, 301))
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] L^2 일치, lim sum={lim2}, lim h={lim3}, a_n 감소 n=2..300")


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
