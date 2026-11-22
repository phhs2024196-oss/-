#!/usr/bin/env python3
"""AJOU-G22B (아주대 2022 실전 자연 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G22B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(no: int, sub: str) -> dict:
    """'a=..; b=..' 꼴 answer.value 를 사전으로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

import itertools
import math


def _count(length, maxrun):
    """L 이 maxrun 개 넘게 연속하지 않는 길이 length 모스부호 개수(전수 열거)."""
    bad = "L" * (maxrun + 1)
    return sum(1 for w in itertools.product("SL", repeat=length) if bad not in "".join(w))


def _count_dp(length, maxrun):
    """같은 개수를 끝에 붙은 L 의 개수 상태로 세기(긴 길이용, 전수 열거와 대조)."""
    st = [1] + [0] * maxrun
    for _ in range(length):
        st = [sum(st)] + st[:-1]
    return sum(st)


def p01():
    a = {k: _count(k, 1) for k in range(1, 15)}
    b = {k: _count(k, 2) for k in range(1, 15)}
    ok0 = all(_count_dp(k, 1) == a[k] and _count_dp(k, 2) == b[k] for k in range(1, 15))
    # (1-1)(1) [증명-수치확인] (3/2)^n <= a_n <= 2^n, n=1..200
    ok1 = all(sp.Rational(3, 2)**k <= _count_dp(k, 1) <= 2**k for k in range(1, 201))
    # (1-1)(2) A = lim a_{n+1}/a_n
    A = sp.Rational(_count_dp(301, 1), _count_dp(300, 1))
    ok2 = abs(float(A) - float(sp.sympify(md_answer(1, "1-2")))) < 1e-12
    # (1-2)(1) 관계식: md 의 Eq(b(n+3), ...) 를 실제 개수에 대입
    bf = sp.Function("b")
    rel = sp.sympify(md_answer(1, "2-1"), locals={"b": bf, "n": n})
    ok3 = all(bool(rel.subs(n, k).replace(bf, lambda m: sp.Integer(b[int(m)]))) for k in range(1, 12))
    # (1-2)(2) [증명-수치확인] B 근사 및 3/2 < B < 2, x^3-x^2-x-1 의 실근 1개
    B = _count_dp(401, 2) / _count_dp(400, 2)
    f = x**3 - x**2 - x - 1
    roots = sp.real_roots(sp.Poly(f, x))
    ok4 = 1.5 < B < 2 and len(roots) == 1 and abs(float(roots[0]) - B) < 1e-12
    report(1, ok0 and ok1 and ok2 and ok3 and ok4,
           f"a={[a[k] for k in range(1, 7)]} b={[b[k] for k in range(1, 7)]} A≈{float(A):.12f} "
           f"[증명-수치확인] B≈{B:.10f}")


def p02():
    al = sp.symbols("alpha", positive=True)
    nn = sp.symbols("n", positive=True, integer=True)
    ok = True
    vals = []
    for k in range(1, 7):
        f = sp.sin(k * x) / (k + 1)
        Sk = sp.integrate(f, (x, 0, sp.pi / k))
        ok = ok and sp.simplify(Sk - sp.sympify(md_answer(2, "1-1"), locals={"n": nn}).subs(nn, k)) == 0
        # (2-2)(1) 세 점을 지나는 이차함수(일반형 계수 연립) 넓이
        p, q, c = sp.symbols("p q c")
        g = p * x**2 + q * x + c
        sol = sp.solve([g.subs(x, 0), g.subs(x, al) - f.subs(x, al), g.subs(x, sp.pi / k)], [p, q, c], dict=True)[0]
        Tk = sp.integrate(g.subs(sol), (x, 0, sp.pi / k))
        mdT = sp.sympify(md_answer(2, "2-1"), locals={"n": nn, "alpha": al}).subs(nn, k)
        ok = ok and all(abs(float((Tk - mdT).subs(al, v * sp.pi / k))) < 1e-12 for v in (0.2, 0.5, 0.9))
        # (2-2)(2) 극한
        L = sp.limit(Tk, al, 0, "+")
        ok = ok and sp.simplify(L - sp.sympify(md_answer(2, "2-2"), locals={"n": nn}).subs(nn, k)) == 0
        # (2-2)(4) 최댓값(격자) 및 (3) 최대점 alpha=pi/(2k)
        Tf = sp.lambdify(al, Tk, "math")
        grid = [(i / 20000) * math.pi / k for i in range(1, 20000)]
        vmax, amax = max((Tf(v), v) for v in grid)
        mdM = float(sp.sympify(md_answer(2, "2-4"), locals={"n": nn}).subs(nn, k))
        ok = ok and abs(vmax - mdM) < 1e-9 and abs(amax - math.pi / (2 * k)) < 1e-3
        vals.append(round(vmax, 6))
    # (2-1)(2) 급수
    ssum = sp.summation(sp.sympify(md_answer(2, "1-1"), locals={"n": nn}), (nn, 1, sp.oo))
    Sdirect = sp.summation(sp.integrate(sp.sin(nn * x) / (nn + 1), (x, 0, sp.pi / nn)), (nn, 1, sp.oo))
    ok = ok and sp.simplify(ssum - sp.sympify(md_answer(2, "1-2"))) == 0 and sp.simplify(Sdirect - ssum) == 0
    # (2-2)(3) [증명-수치확인] k(x)=4/pi^2 x(pi-x) - sin x >= 0 on (0, pi/2], k''=0 근 1개
    kf = lambda v: 4 / math.pi**2 * v * (math.pi - v) - math.sin(v)
    ok3 = min(kf(i * (math.pi / 2) / 100000) for i in range(1, 100001)) >= -1e-15
    k2 = lambda v: -8 / math.pi**2 + math.sin(v)
    sign_changes = sum(1 for i in range(1, 100000) if k2(i * math.pi / 2e5) * k2((i + 1) * math.pi / 2e5) < 0)
    ok = ok and ok3 and sign_changes == 1
    report(2, ok, f"max T_n(n=1..6)={vals} [증명-수치확인] k(x)>=0, k''=0 근 {sign_changes}개")


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
