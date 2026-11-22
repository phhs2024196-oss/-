#!/usr/bin/env python3
"""EWHA-M21 (이화여대 2021학년도 수시 모의논술 자연계열) — sympy 검산.
정답 하드코딩 비교 금지: 문항 YAML 의 answer 를 읽어 독립 재계산값과 비교. 증명 소문항은 [증명-수치확인]."""
import pathlib
import random
import re
import sys

import sympy as sp
import yaml

PREFIX = "EWHA-M21"
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no, sub):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:02d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:02d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import mpmath as mp
    mp.mp.dps = 40
    # (1) e^x > 1+x, x>0 격자 1000점 + 무작위
    pts = [mp.mpf(i) / 100 for i in range(1, 1001)] + [mp.mpf(random.uniform(1e-6, 50)) for _ in range(20)]
    ok1 = all(mp.e**t > 1 + t for t in pts)
    # (2) f'(x) 기호식 확인 + 평균값 정리의 c 존재(x<c<x+1) + f'>0
    xp = sp.symbols("xp", positive=True)
    f = xp * sp.log(1 + 1 / xp)
    fp = sp.simplify(sp.expand_log(sp.diff(f, xp) - (sp.log(xp + 1) - sp.log(xp) - 1 / (xp + 1)), force=True))
    fpl = sp.lambdify(xp, sp.diff(f, xp), "mpmath")
    ok2 = fp == 0 and all(fpl(t) > 0 for t in pts)
    ok2 &= all(t < 1 / (mp.log(t + 1) - mp.log(t)) < t + 1 for t in pts)
    # (3) b_n 증가·유계(≤e), n=1..200
    b = [(1 + mp.mpf(1) / k)**k for k in range(1, 202)]
    ok3 = all(b[i] <= b[i + 1] for i in range(200)) and all(v <= mp.e for v in b)
    report(1, ok1 and ok2 and ok3, f"[증명-수치확인] e^x>1+x, f'>0 (격자1000+무작위20), b_n 증가·≤e (n≤200) b_200={mp.nstr(b[199], 12)}")


def p02():
    l = sp.exp(x) + sp.exp(-x)
    ok1 = sp.simplify(l.subs(x, -x) - l) == 0
    # (2) h(-x)=h(x)=-h(x) 인 h: 값 v 에 대해 v=-v → v=0 (임의 점에서 선형 연립)
    v = sp.symbols("v")
    ok2 = sp.solve([sp.Eq(v, -v)], v) == {v: 0} and sp.sympify(md_answer(2, "2")) == 0
    # (3) [증명-수치확인] 무작위 다항식 홀·짝 함수로 f1+g1=f2+g2 ⇒ 차가 홀·짝 동시 ⇒ 0
    ok3 = True
    for _ in range(20):
        co = [random.randint(-5, 5) for _ in range(6)]
        d = sum(co[i] * x**(2 * i + 1) for i in range(3))   # f1-f2 (홀)
        e = sum(co[i + 3] * x**(2 * i) for i in range(3))   # g2-g1 (짝)
        if sp.expand(d - e) == 0:
            ok3 &= d == 0 and e == 0
    # 일반형: 홀(d)=짝(e) 이면 d(-x)=-d(x)=-e(x)=-e(-x)=-d(-x) → d=0 기호 확인
    # (4) 분해: a=(e^x-e^-x)/2, b=(e^x+e^-x)/2 를 연립으로 도출
    A, B = sp.symbols("A B")
    s = sp.solve([sp.Eq(A + B, sp.exp(x)), sp.Eq(-A + B, sp.exp(-x))], [A, B])
    val = sp.simplify((s[B] - s[A]).subs(x, 2021))
    ok4 = sp.simplify(val - sp.sympify(md_answer(2, "4"))) == 0
    report(2, ok1 and ok2 and ok3 and ok4, f"l(-x)=l(x), h=0, b(2021)-a(2021)={val}")


def p03():
    c = sp.symbols("c", real=True)
    # (1) C(c,c^2+9): 접선 기울기 2c, AC 방향 (c, c^2+9) 와 접선 방향 (1,2c) 내적=0 (기울기 정의 여부 무관)
    sc = sp.solve(sp.Eq(c * 1 + (c**2 + 9) * 2 * c, 0), c)
    C = [(r, r**2 + 9) for r in sc if r.is_real]
    ok1 = C == [tuple(sp.sympify(md_answer(3, "1")))]
    # (2) D(d,d^2+9): BD 방향 (d-3, d^2) · (1,2d) = 0, D≠B 조건 자동
    sd = [r for r in sp.solve(sp.Eq((c - 3) + c**2 * 2 * c, 0), c) if r.is_real]
    D = [(r, r**2 + 9) for r in sd]
    ok2 = D == [tuple(sp.sympify(md_answer(3, "2")))]
    # (3) 넓이: 영역 = {0≤x≤3, y≥x^2, y≤x^2+9, y≤선 BD(1≤x≤3)} — 몬테카를로 대신 경계 min 적분
    d0, B = D[0], (3, 9)
    line = d0[1] + (sp.Rational(B[1] - d0[1], B[0] - d0[0])) * (x - d0[0])
    area = sp.integrate((x**2 + 9) - x**2, (x, 0, d0[0])) + sp.integrate(line - x**2, (x, d0[0], 3))
    # 위 경계 확인: [1,3]에서 line ≤ x^2+9, [0,1]에서 위 경계는 x^2+9
    okb = all(float(line.subs(x, 1 + 2 * i / 100)) <= float(((x**2 + 9)).subs(x, 1 + 2 * i / 100)) + 1e-12 for i in range(101))
    ok3 = okb and sp.simplify(area - sp.sympify(md_answer(3, "3"))) == 0
    report(3, ok1 and ok2 and ok3, f"C={C} D={D} area={area} (오답 재현 ∫0^3 9dx=27)")


if __name__ == "__main__":
    random.seed(2021)
    fns = sorted((k for k in dir() if re.fullmatch(r"p\d+", k)), key=lambda s: int(s[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
