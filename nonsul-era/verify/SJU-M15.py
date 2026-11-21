#!/usr/bin/env python3
"""SJU-M15 (세종대 2015학년도 신입학 관련 모의평가 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-M15"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    t, s = sp.symbols("t s", positive=True)
    dist = sp.integrate(2 * s, (s, 0, t))
    a = [v for v in sp.solve(sp.Eq(dist, 1), t)][0]
    b = [v for v in sp.solve(sp.Eq(dist, 9), t)][0]
    h = sp.integrate(2 * s, (s, a, t))
    P1 = parts(md_answer(1, "1"))
    ok1 = sp.sympify(P1["a"]) == a and sp.simplify(sp.sympify(P1["h(t)"], locals={"t": t}) - h) == 0
    V = -(5 * t**2 - 16 * t - 4) * (t - 1)**4
    ok_bd = V.subs(t, a) == 0                                   # 닿는 순간 부피 0
    S = sp.simplify(sp.diff(V, t) / sp.diff(h, t))
    P2 = parts(md_answer(1, "2"))
    ok2 = sp.simplify(sp.sympify(P2["S(t)"], locals={"t": t}) - S) == 0
    # 역검산: ∫_0^{h(t)} A(x)dx 가 V 와 같은지(A(h(t))=S(t) ⇒ A(x)=S(sqrt(x+1)))
    x = sp.symbols("x", positive=True)
    A = S.subs(t, sp.sqrt(x + 1))
    ok2 &= abs(float(sp.integrate(A, (x, 0, h.subs(t, sp.Rational(5, 2)))) - V.subs(t, sp.Rational(5, 2)))) < 1e-9
    crit = [c for c in sp.solve(sp.diff(S, t), t) if a <= c <= b]
    best = max(crit + [a, b], key=lambda v: S.subs(t, v))
    ok3 = best == sp.sympify(md_answer(1, "3"))
    report(1, ok1 and ok_bd and ok2 and ok3, f"a={a} b={b} h={h} S={sp.factor(S)} argmax={best} Smax={S.subs(t, best)}")


def p02():
    import mpmath as mp
    # (2-1) √x-ln x 최솟값 > 0 (격자+정밀)
    g = lambda u: mp.sqrt(u) - mp.log(u)
    xs = [mp.mpf(10)**(k / 200) for k in range(-1600, 1601)]
    ok1 = min(g(u) for u in xs) > 0 and abs(mp.findroot(lambda u: mp.diff(g, u), 3) - 4) < 1e-12
    # (2-2) 극한
    x = sp.symbols("x", positive=True)
    ok2 = sp.limit(sp.log(x) / x, x, sp.oo) == 0
    f = sp.log(x) / x
    cmax = sp.solve(sp.diff(f, x), x); infl = sp.solve(sp.diff(f, x, 2), x)
    P3 = parts(md_answer(2, "3"))
    ok3 = cmax == [sp.sympify(P3["local_max_x"])] and infl == [sp.sympify(P3["inflection_x"])] and \
        sp.diff(f, x, 2).subs(x, sp.E) < 0
    # (2-4) 1≤m<n≤2000 전수: m^n = n^m
    sols = [(m, n) for m in range(1, 60) for n in range(m + 1, 2000) if abs(n * mp.log(m) - m * mp.log(n)) < 1e-12]
    sols = [(m, n) for (m, n) in sols if m**n == n**m]
    ok4 = sols == [(2, 4)]
    report(2, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] min(√x-lnx)>0, lim=0, 극대 x={cmax}, 변곡 x={infl}, 해={sols}")


def p03():
    B = sp.Matrix([[0, 1, 2], [1, 0, 1], [1, 0, -1]])
    A = sp.Matrix(sp.sympify(md_answer(3, "1")))
    ok1 = A * B == sp.eye(3) and A == B.inv()
    x = sp.symbols("x")
    a, b, c, d = sp.symbols("a b c d")
    M = sp.Matrix([[a, b], [c, d]])
    row = sp.Matrix([[x, 3 * x - 2]]) * M - sp.Matrix([[2, 4]])
    eqs = [co for e in row for co in sp.Poly(sp.expand(e), x).all_coeffs()]
    sol = sp.solve(eqs, [a, b, c, d], dict=True)[0]
    ok2 = M.subs(sol) == sp.Matrix(sp.sympify(md_answer(3, "2")))
    # (3-3) [증명-수치확인] 무작위 정수 2x2 B 에서 AB=E 인 A 가 있으면 BA=E; det=0 이면 AB=E 해 없음
    import random
    ok3 = True
    p, q, r, s = sp.symbols("p q r s")
    X = sp.Matrix([[p, q], [r, s]])
    for _ in range(60):
        Bv = sp.Matrix(2, 2, [random.randint(-3, 3) for _ in range(4)])
        sl = sp.solve(list(X * Bv - sp.eye(2)), [p, q, r, s], dict=True)
        if Bv.det() == 0:
            ok3 &= sl == []
        else:
            ok3 &= len(sl) == 1 and Bv * X.subs(sl[0]) == sp.eye(2)
    report(3, ok1 and ok2 and ok3, f"A=B^-1={A.tolist()} (3-2) A={M.subs(sol).tolist()} [증명-수치확인] 60개 B")


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
