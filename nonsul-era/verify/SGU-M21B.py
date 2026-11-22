#!/usr/bin/env python3
"""SGU-M21B (서강대 2021학년도 모의논술 2차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M21B"
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
    h = sp.symbols("h", positive=True)
    k, N = sp.symbols("k N", integer=True, positive=True)
    X = sp.symbols("x", positive=True)
    loc = {"x": X}
    # (1) 극한식을 수치로 평가해 md 의 f 의 적분과 비교 (h 여러 값, n 큰 값)
    import mpmath as mp
    f_md = sp.sympify(md_answer(1, "1"), locals=loc)
    def riem(hv, nv):
        return mp.fsum(kk**2 * (hv / nv)**3 * mp.e**(-kk * hv / nv) for kk in range(1, nv + 1))
    ok1 = all(abs(riem(hv, 200000) - float(sp.integrate(f_md, (X, 0, hv)))) < 1e-4 for hv in (0.5, 1.7, 3))
    # (2) V(x)=∫0^x f
    V = sp.integrate(f_md.subs(X, t), (t, 0, X))
    ok2 = sp.simplify(V - sp.sympify(md_answer(1, "2"), locals=loc)) == 0
    # (3) dV/dt = f(x) dx/dt = -2x  → |dx/dt| = 2x/f(x)
    sp_ = sp.simplify(2 * X / f_md)
    ok3 = sp.simplify(sp_ - sp.sympify(md_answer(1, "3"), locals=loc)) == 0
    # (4) 최소
    crit = sp.solve(sp.diff(sp_, X), X)
    xm = [c for c in crit if c.is_positive][0]
    vm = sp.simplify(sp_.subs(X, xm))
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "4").replace(" ", "").split(";"))
    ok4 = sp.simplify(xm - sp.sympify(parts["x"])) == 0 and sp.simplify(vm - sp.sympify(parts["speed"])) == 0 \
        and sp.limit(sp_, X, 0, "+") == sp.oo
    report(1, ok1 and ok2 and ok3 and ok4, f"V={sp.simplify(V)} speed={sp_} min at x={xm} -> {vm}")


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
