#!/usr/bin/env python3
"""HIU-M27 (홍익대 2027학년도 모의논술 서울 자연) — sympy 검산.
규칙: 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "HIU-M27"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def parts(s: str) -> dict:
    """'a = 1; b = 2' 형식 → {'a': '1', 'b': '2'}"""
    return {kk.strip(): v.strip() for kk, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    r, C, T = sp.symbols("r C T", positive=True)
    h = 1 / (1 + r**2) ** sp.Rational(3, 2) - C            # d = sqrt(1+r^2), 높이 1/d^3 - C
    c8 = sp.Rational(1, 8)
    hs = h.subs(C, c8)
    dec = sp.simplify(sp.diff(hs, r)) .subs(r, sp.Rational(1, 2)) < 0
    tmax = hs.subs(r, 0)
    R = [s for s in sp.solve(sp.Eq(hs, 0), r) if s.is_positive][0]
    a1 = parts(md_answer(1, "1"))
    ok1 = dec and sp.simplify(tmax - sp.sympify(a1["t_max"])) == 0 and sp.simplify(R - sp.sympify(a1["R"])) == 0
    # (2) h(r)=T 를 r^2 에 대해 풀어 단면 원 넓이
    u = sp.symbols("u", positive=True)                     # u = r^2
    r2 = [w for w in sp.solve(sp.Eq((1 + u) ** sp.Rational(-3, 2) - c8, T), u)
          if abs(sp.im(sp.N(w.subs(T, sp.Rational(3, 10))))) < 1e-12][0]   # 실근 선택
    S = sp.pi * r2
    md_S = sp.sympify(md_answer(1, "2"), locals={"t": T})
    ok2 = all(abs(float((S - md_S).subs(T, v))) < 1e-12 for v in (0.01, 0.3, 0.8))
    # (3) 부피: 단면 적분 + 독립적으로 원통 껍질 적분(2πr h(r))
    V = sp.integrate(S, (T, 0, tmax))
    Vshell = sp.integrate(2 * sp.pi * r * hs, (r, 0, R))
    ok3 = sp.simplify(V - Vshell) == 0 and sp.simplify(V - sp.sympify(md_answer(1, "3"))) == 0
    # (4) 일반 C 부피 → C→0+, 비율
    VC = sp.integrate(2 * sp.pi * r * h, (r, 0, sp.sqrt(C ** sp.Rational(-2, 3) - 1)))
    lim = sp.limit(VC, C, 0, "+")
    total = sp.integrate(2 * sp.pi * r / (1 + r**2) ** sp.Rational(3, 2), (r, 0, sp.oo))   # 총 뿌린 양
    a4 = parts(md_answer(1, "4"))
    ok4 = sp.simplify(lim - total) == 0 and sp.simplify(lim - sp.sympify(a4["limit"])) == 0 and \
        sp.simplify(V / lim - sp.sympify(a4["ratio"])) == 0 and \
        abs(float(VC.subs(C, c8) - V)) < 1e-12
    report(1, ok1 and ok2 and ok3 and ok4, f"tmax={tmax} R={R} V={sp.simplify(V)} lim={lim} ratio={sp.simplify(V/lim)}")


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
