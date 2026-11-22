#!/usr/bin/env python3
"""SGU-M25B (서강대 2025학년도 모의논술 2차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-M25B"
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
    import random
    import mpmath as mp
    X, T = sp.symbols("X T", real=True)
    # (1-1) [증명-수치확인] g<=h 인 무작위 연속함수 쌍에서 적분 대소
    random.seed(1)
    ok1 = True
    for _ in range(20):
        c = [random.uniform(-2, 2) for _ in range(3)]
        gfun = lambda s, c=c: c[0] + c[1] * mp.cos(s) + c[2] * s
        hfun = lambda s, c=c: gfun(s) + (s - 0.3)**2 * abs(c[2])
        a0, b0 = sorted((random.uniform(-3, 3), random.uniform(-3, 3)))
        ok1 &= mp.quad(gfun, [a0, b0]) <= mp.quad(hfun, [a0, b0]) + 1e-15
    # f(x) 정의 그대로 수치 적분
    fn = lambda xv: xv - mp.mpf(1) / 2 * mp.quad(lambda s: s**2 * mp.cos(xv - s)**3, [0, xv])
    # (1-2) 홀함수
    pts = [random.uniform(-10, 10) for _ in range(20)]
    ok2 = all(abs(fn(p) + fn(-p)) < 1e-12 for p in pts)
    # (1-3) |f(x)-x| <= |x|^3/6 격자
    grid = [-8 + 16 * i / 400 for i in range(401)]
    ok3 = all(abs(fn(p) - p) <= abs(p)**3 / 6 + 1e-12 for p in grid)
    # (1-4) 기호 적분으로 f 를 닫힌꼴로 구해 세 번 미분
    fs = X - sp.Rational(1, 2) * sp.integrate(T**2 * sp.cos(X - T)**3, (T, 0, X))
    d3 = sp.simplify(sp.diff(fs, X, 3).subs(X, 0))
    ok4 = sp.simplify(d3 - sp.sympify(md_answer(1, "4"))) == 0 and abs(float(fs.subs(X, 1.3)) - float(fn(1.3))) < 1e-10
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] (1)(2)(3) 격자·무작위 확인, f'''(0)={d3}")


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
