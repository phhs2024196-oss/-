#!/usr/bin/env python3
"""SJU-M26F (세종대 2026학년도 모의논술 자유전공학부 수학문항) — sympy/수치 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-M26F"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {k.strip(): sp.sympify(v.strip()) for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    F1 = sp.symbols("F1", positive=True)
    f1 = sp.solve(sp.Eq(F1, sp.E * sp.sqrt(F1)), F1)[0]                  # 조건 (3) 에 x=1
    g1_1 = sp.Integer(1) / f1                                              # g'(1)=f'(1)/f(1), f'(1)=1
    ok1 = sp.simplify(g1_1 - sp.sympify(md_answer(2, "1"))) == 0
    # (2)(3): 일차함수 g=cx+d 가 로그 관계식 g((x+1)/2)=1+g(x)/2 와 f'(1)=1 을 만족하도록 결정
    c, d = sp.symbols("c d", real=True)
    gl = c * x + d
    eqs = sp.Poly(sp.expand(gl.subs(x, (x + 1) / 2) - 1 - gl / 2), x).coeffs()
    sol = sp.solve(eqs + [sp.diff(sp.exp(gl), x).subs(x, 1) - 1], [c, d], dict=True)
    sol = [s for s in sol if s[c].is_real]
    gs = gl.subs(sol[0])
    f = sp.exp(gs)
    ok2 = sp.simplify(f.subs(x, (x + 1) / 2) - sp.E * sp.sqrt(f)) == 0 and sp.simplify(sp.diff(f, x).subs(x, 1) - 1) == 0
    g1 = sp.diff(gs, x)
    ok2 &= all(abs(float((g1.subs(x, v) - g1.subs(x, 1 + (v - 1) / sp.Integer(2)**n)))) < 1e-15 for v in (-3, 0.5, 9) for n in range(1, 25))
    md3 = dict(seg.split("=", 1) for seg in md_answer(2, "3").replace(" ", "").split(";"))
    ok3 = sp.simplify(sp.sympify(md3["g1(x)"]) - g1) == 0 and sp.simplify(sp.sympify(md3["g(x)"], locals={"x": x}) - gs) == 0
    report(2, ok1 and ok2 and ok3, f"f(1)={f1} g'(1)={g1_1} g={gs} [증명-수치확인]")


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
