#!/usr/bin/env python3
"""숭실대 SSU-G19-01·02 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G19"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answers(seq):
    """소문항 id → answer.value(문자열). 증명 소문항은 None."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return {str(s["id"]): s["answer"]["value"] for s in d["subparts"]}


def p01():
    """(1) |t-1/2| 구간 분할 적분, (2) 무작위 다항식 f 에 대해 F'(1/2) 를 수치미분과 A-B 비교."""
    ans = load_answers(1)
    tt = sp.Symbol("tt", real=True)
    F1 = sp.integrate(2 * tt * sp.Abs(tt - sp.Rational(1, 2)), (tt, 0, 1))
    ok1 = sp.simplify(F1 - sp.sympify(ans["1"])) == 0
    import random
    A_, B_ = sp.symbols("A B")
    expr = sp.sympify(ans["2"], locals={"A": A_, "B": B_})
    ok2 = True
    random.seed(1)
    for _ in range(20):
        cs = [sp.Rational(random.randint(-9, 9), random.randint(1, 5)) for _ in range(4)]
        f = sum(c * tt**i for i, c in enumerate(cs))
        Fx = sp.integrate(f * (x - tt), (tt, 0, x)) + sp.integrate(f * (tt - x), (tt, x, 1))   # 0<x<1
        d = sp.diff(Fx, x).subs(x, sp.Rational(1, 2))
        Av = sp.integrate(f, (tt, 0, sp.Rational(1, 2))); Bv = sp.integrate(f, (tt, sp.Rational(1, 2), 1))
        ok2 &= sp.simplify(d - expr.subs({A_: Av, B_: Bv})) == 0
    report(1, ok1 and ok2, f"(1) F(1/2)={F1} md={ans['1']} | (2) 무작위 다항식 20개 F'(1/2)=A-B 일치={ok2}")


def p02():
    """문제 1-B: 평면(등거리 조건)·수선의 발·단면 원·거리 최솟값을 독립 재계산. (3)은 원 위 점을 매개변수로 두고 수치 최소화로 교차 확인."""
    ans = load_answers(2)
    X = sp.Matrix(sp.symbols("X Y Z", real=True))
    A, B = sp.Matrix([1, -2, 3]), sp.Matrix([3, 2, -1])
    C1, r1 = sp.Matrix([3, 1, -2]), 5
    P = sp.Matrix([4, -3, 8])
    plane = sp.expand((X - A).dot(X - A) - (X - B).dot(X - B))          # = 0 이 S2
    nvec = sp.Matrix([sp.diff(plane, v) for v in X])                      # 법선
    s = sp.Symbol("s", real=True)
    def foot(Q):
        L = Q + s * nvec
        sol = sp.solve(plane.subs(dict(zip(X, L))), s)[0]
        return L.subs(s, sol)
    T = foot(P)
    ok1 = list(T) == list(sp.sympify(ans["1"]))
    C = foot(C1); d = (C1 - C).norm(); r = sp.sqrt(r1**2 - d**2)
    c_ans, r_ans = sp.sympify(ans["2"])
    ok2 = list(C) == list(c_ans) and sp.simplify(r - r_ans) == 0
    # (3) 원 위의 점: C + r(cos u e1 + sin u e2), e1,e2 는 평면 위 정규직교
    nn = nvec / nvec.norm()
    e1 = sp.Matrix([2, -1, 0]); e1 = e1 - e1.dot(nn) * nn; e1 = e1 / e1.norm()
    e2 = nn.cross(e1)
    u = sp.Symbol("u", real=True)
    Xu = C + r * (sp.cos(u) * e1 + sp.sin(u) * e2)
    assert sp.simplify(plane.subs(dict(zip(X, Xu)))) == 0 and sp.simplify((Xu - C1).dot(Xu - C1) - r1**2) == 0
    D2 = sp.lambdify(u, (Xu - P).dot(Xu - P))
    import math
    mn = min(math.sqrt(D2(2 * math.pi * k / 200000)) for k in range(200000))
    exact = sp.sqrt((P - T).norm()**2 + ((T - C).norm() - r)**2)
    ok3 = sp.simplify(exact - sp.sympify(ans["3"])) == 0 and abs(mn - float(exact)) < 1e-6
    report(2, ok1 and ok2 and ok3, f"(1) T={tuple(T)} | (2) C={tuple(C)}, r={r} | (3) 정확값 {exact}, 수치최소 {mn:.8f}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
