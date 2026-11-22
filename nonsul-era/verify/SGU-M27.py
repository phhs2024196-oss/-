#!/usr/bin/env python3
"""SGU-M27 (서강대 2027학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M27"
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
    X = sp.symbols("X", real=True)
    # (1-1) 넓이: |f-g| 를 수치 적분(교점 자동) + 기호값 비교
    import mpmath as mp
    mp.mp.dps = 30
    fa = lambda s: s - mp.sqrt(s)
    ga = lambda s: mp.sin(mp.pi / 2 * s + mp.pi / 2)
    S = mp.quad(lambda s: abs(fa(s) - ga(s)), [0, 1, 3, 4])
    ok1 = abs(S - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "1")), 30)))) < 1e-20
    # (1-2) 격자점 전수: 1<=a<=100, 0<=b<=a-sqrt(a)
    cnt = sum(1 for a in range(1, 101) for b in range(0, 101) if a - b >= 0 and (a - b) ** 2 >= a)   # b <= a-√a 를 정수로 정확히
    ok2 = cnt == int(md_answer(1, "2"))
    # (1-3) f=x^4+6x^3+cx+d (변곡점 조건) 에서 정수 격자 c,d 전수: 단순근(부호 바뀌는 근)이 정확히 1개이고 양수
    hits = []
    for c in range(-80, 21):
        for d in range(-150, 51):
            P = sp.Poly(X**4 + 6 * X**3 + c * X + d, X)
            simple = [fac for fac, mlt in P.sqf_list()[1] if mlt == 1]
            roots = [r for fac in simple for r in fac.real_roots()]
            if len(roots) == 1 and roots[0] > 0:
                hits.append((c, d, roots[0]))
    f2 = sp.diff(X**4 + 6 * X**3, X, 2)
    infl = sorted(sp.solve(f2, X))
    ok3 = infl == [-3, 0] and len(hits) == 1 and hits[0][2] == sp.sympify(md_answer(1, "3"))
    # (1-4) 교점 개수 (서로 다른 실근)
    c0, d0 = hits[0][0], hits[0][1]
    F = X**4 + 6 * X**3 + c0 * X + d0
    k = sp.symbols("k")
    def ncount(kv):
        return len(set(sp.Poly(sp.expand(F - kv * X - F.subs(X, 0)), X).real_roots()))
    ks = [sp.Rational(i, 4) for i in range(-300, 101)]
    def expect(kv):
        if kv <= -54 or kv > -22:
            return 2
        if kv == -22:
            return 3
        return 4
    okk = all(ncount(kv) == expect(kv) for kv in ks)
    ans = md_answer(1, "4").replace(" ", "")
    ok4 = okk and ans == "k<=-54ork>-22:2;-54<k<-22:4;k=-22:3"
    report(1, ok1 and ok2 and ok3 and ok4, f"S={mp.nstr(S, 15)} count={cnt} (c,d,m)={hits[0]} k-grid ok={okk}")


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
