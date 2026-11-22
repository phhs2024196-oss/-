#!/usr/bin/env python3
"""SGU-M23B (서강대 2023학년도 모의논술 2차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M23B"
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
    import itertools
    from fractions import Fraction
    N, K = sp.symbols("n k", integer=True, positive=True)
    loc = {"n": N, "k": K}

    def dist(nv):
        """전수 열거: 가운데 사람 2..n-1 의 '하나!' 방향(R/L) 2^(n-2) 가지, 균등."""
        cnt = {}
        for mids in itertools.product("RL", repeat=nv - 2):
            one = ["R"] + list(mids) + ["L"]
            two = ["R"] + [("L" if d == "R" else "R") for d in mids] + ["L"]
            ok = sum(1 for i in range(nv - 1) if (one[i], one[i + 1]) == ("R", "L") or (two[i], two[i + 1]) == ("R", "L"))
            cnt[ok] = cnt.get(ok, 0) + 1
        tot = 2 ** (nv - 2)
        return {c: Fraction(v, tot) for c, v in cnt.items()}

    E = lambda d: sum(c * p for c, p in d.items())
    ok1 = sp.Rational(str(E(dist(4)))) == sp.sympify(md_answer(1, "1"))
    e2 = sp.sympify(md_answer(1, "2"), locals=loc)
    ok2 = all(sp.Rational(str(E(dist(nv)))) == e2.subs(N, nv) for nv in range(3, 13))
    p3 = sp.sympify(md_answer(1, "3"), locals=loc)
    ok3 = all(sp.Rational(str(dist(nv).get(nv - 1, 0))) == p3.subs(N, nv) for nv in range(3, 13))
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "4").replace(" ", "").split(";"))
    m_md = sp.sympify(parts["m"], locals=loc)
    P_md = sp.sympify(parts["P"], locals=loc)
    ok4 = True
    for nv in range(4, 13):
        d = dist(nv)
        ok4 &= max(d) == m_md.subs(N, nv)
        ok4 &= all(sp.Rational(str(d.get(kv, 0))) == P_md.subs({N: nv, K: kv}) for kv in range(2, nv))
    n2 = dist(2)
    report(1, ok1 and ok2 and ok3 and ok4,
           f"전수 열거 n=3..12 일치, E(4)={E(dist(4))}; 참고 n=2 → P(모두 성공)={n2.get(1)} (공식식 (1/2)^(n-3)=2, n>=3 에서만 유효)")


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
