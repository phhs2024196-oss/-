#!/usr/bin/env python3
"""SMWU-G26 (숙명여대 2026학년도 논술우수자전형 자연계열(약학부 제외)) — sympy 검산. 정답 하드코딩 금지(YAML answer.value 를 읽어 독립 재계산과 비교)."""
import re
import sys

import sympy as sp

PREFIX = "SMWU-G26"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
n = sp.symbols("n", positive=True, integer=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    """'b_n = ...; c_n = ...' 형식을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a = sp.symbols("a", real=True)
    # (1-1) 교점 조건 ⇔ a = √(x+1) - x² 가 (1,3) 에서 취하는 값. 단조성 확인 후 양 끝 극한
    k = sp.sqrt(x + 1) - x**2
    import numpy as np
    kp = sp.lambdify(x, sp.diff(k, x))
    mono = all(kp(v) < 0 for v in np.linspace(1, 3, 2001))
    lo, hi = sp.limit(k, x, 3, "-"), sp.limit(k, x, 1, "+")
    ans = sp.sympify(md_answer(1, "1"))            # 튜플 (lo, hi) 로 해석 → 열린구간
    ok1 = mono and sp.simplify(ans[0] - lo) == 0 and sp.simplify(ans[1] - hi) == 0
    # (1-2) [증명-수치확인] |f'|<1 인 무작위 함수들에서 f(c)=2c 의 해가 [0,1] 에서 1개 이하
    import random
    random.seed(3)
    ok2 = True
    for _ in range(200):
        c0, c1, c2 = random.uniform(-2, 3), random.uniform(-0.5, 0.5), random.uniform(-0.4, 0.4)
        g = lambda u: c0 + c1 * u + c2 * np.sin(u * 1.0) - 2 * u        # f'(u)=c1+c2 cos u, |f'|<1
        vals = g(np.linspace(0, 1, 2001))
        ok2 &= int(np.sum(np.diff(np.sign(vals)) != 0)) <= 1 and np.all(np.diff(vals) < 0)
    report(1, ok1 and ok2, f"a in ({lo}, {hi}), k monotone={mono}; [증명-수치확인] 200 cases")


def p02():
    an, nn = sp.symbols("a_n nn", positive=True)
    # (2-1) 법선이 (0,1) 을 지남
    normal0 = -1 / (2 * nn * an) * (0 - an) + nn * an**2
    sol = [s for s in sp.solve(sp.Eq(normal0, 1), an) if s.is_positive is not False]
    asol = sp.simplify(sol[0]); bsol = sp.simplify(nn * asol**2)
    P = parts(md_answer(2, "1"))
    loc = {"n": nn}
    ok1 = len(sol) == 1 and all(
        abs(float((asol - sp.sympify(P["a_n"], locals=loc)).subs(nn, v))) < 1e-12 and
        abs(float((bsol - sp.sympify(P["b_n"], locals=loc)).subs(nn, v))) < 1e-12 for v in range(1, 30))
    # (2-2) 넓이: 곡선 아래 [0,a] 에서 접선 아래(접선>0 부분) 넓이를 뺀다
    X = sp.symbols("X", real=True)
    tl = 2 * nn * an * (X - an) + nn * an**2
    xi = sp.solve(sp.Eq(tl, 0), X)[0]
    A = sp.integrate(nn * X**2, (X, 0, an)) - sp.integrate(tl, (X, xi, an))
    lim = sp.limit(sp.simplify((A / an).subs(an, asol)), nn, sp.oo)
    ok2 = sp.simplify(lim - sp.sympify(md_answer(2, "2"))) == 0
    report(2, ok1 and ok2, f"a_n={asol} b_n={bsol} A_n={sp.factor(A)} lim={lim}")


def p03():
    f = lambda u: u**3 + 2 * u + 4
    ok_inv = sp.solve(sp.Eq(sp.diff(f(x), x), 0), x) == [] or all(not r.is_real for r in sp.solve(sp.diff(f(x), x), x))
    # (3-1) f^{-1}(t²+5t+1)=t-1 ⇔ f(t-1)=t²+5t+1
    s1 = set(sp.solve(sp.Eq(f(t - 1), t**2 + 5 * t + 1), t))
    ok1 = ok_inv and s1 == set(sp.sympify(md_answer(3, "1")))
    # (3-2) 수치: f^{-1} 을 직접 계산해 t 격자에서 근 찾기 + 기호 확인
    a = sp.symbols("a", real=True)
    ar = [r for r in sp.solve(sp.Eq(f(a) - f(-a), 4), a) if r.is_real]
    s2 = set()
    for r in ar:
        s2 |= {v for v in sp.solve(sp.Eq(t**2 + 2, f(r)), t) if v.is_real}
    s2 = {sp.nsimplify(sp.simplify(v)) for v in s2}
    import mpmath as mp
    finv = lambda yv: mp.findroot(lambda u: u**3 + 2 * u + 4 - yv, 0)
    H = lambda tv: finv(tv**2 + 2) + finv(tv**2 - 2)
    num_ok = all(abs(H(float(v))) < 1e-12 for v in s2) and abs(H(1.0)) > 1e-3 and abs(H(3.0)) > 1e-3
    ok2 = s2 == set(sp.sympify(md_answer(3, "2"))) and num_ok
    report(3, ok1 and ok2, f"f'>0; t(3-1)={sorted(s1)} t(3-2)={sorted(s2)}")


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
