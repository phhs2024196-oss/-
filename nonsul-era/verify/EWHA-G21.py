#!/usr/bin/env python3
"""EWHA-G21 (이화여대 2021학년도 수시 논술 자연계열) — sympy 검산.
정답 하드코딩 비교 금지: 문항 YAML 의 answer 를 읽어 문제 설정으로부터 독립 재계산한 값과 비교."""
import itertools
import pathlib
import random
import re
import sys

import sympy as sp
import yaml

PREFIX = "EWHA-G21"
RESULTS = []
x, n = sp.symbols("x n", real=True)


def md_answer(no, sub):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:02d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:02d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) x^2 <= x+1 의 해 구간과 [0,β]에서 f 최댓값
    r = sorted(sp.solve(x**2 - x - 1, x), key=lambda v: float(v))
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "1").replace(" ", "").split(";"))
    al, be, mx = (sp.sympify(parts[k]) for k in ("alpha", "beta", "max"))
    ok_ab = sp.simplify(al - r[0]) == 0 and sp.simplify(be - r[1]) == 0
    f = x + 1 / (x + 1)
    crit = [c for c in sp.solve(sp.diff(f, x), x) if c.is_real and 0 < c < r[1]]
    cand = [f.subs(x, 0), f.subs(x, r[1])] + [f.subs(x, c) for c in crit]
    fmax = sp.nsimplify(sp.simplify(max(cand, key=lambda v: float(v))))
    ok1 = ok_ab and sp.simplify(fmax - mx) == 0 and not crit
    # (2)(3) [증명-수치확인] n=1..200 고정밀 반복
    import mpmath as mp
    mp.mp.dps = 50
    B = (1 + mp.sqrt(5)) / 2
    a, b = mp.mpf(2), mp.mpf(1)
    ok2 = ok3 = True
    for _ in range(200):
        a2, b2 = mp.sqrt(a + 1), 2 - 1 / (b + 1)
        ok2 &= a >= B - mp.mpf(10)**-40 and b <= B + mp.mpf(10)**-40
        ok3 &= a2 <= a and b2 >= b
        a, b = a2, b2
    # (3) 근거 부등식 격자 확인
    ok3 &= all(t * t >= t + 1 - 1e-12 for t in [float(B) + i / 100 for i in range(1000)])
    ok3 &= all(t + 1 / (t + 1) <= 2 + 1e-12 for t in [float(B) * i / 1000 for i in range(1001)])
    report(1, ok1 and ok2 and ok3, f"alpha,beta={r} max f={fmax} [증명-수치확인 n<=200] a_200={mp.nstr(a, 12)} b_200={mp.nstr(b, 12)}")


def p02():
    X, Y, Z = sp.symbols("X Y Z", real=True)
    O, A, B = sp.Matrix([0, 0, 0]), sp.Matrix([2, -sp.sqrt(2), 0]), sp.Matrix([2, sp.sqrt(2), 0])
    P = sp.Matrix([X, Y, 0])
    d2 = lambda U, V: ((U - V).T * (U - V))[0]
    sol = sp.solve([d2(P, O) - d2(P, A), d2(P, O) - d2(P, B)], [X, Y], dict=True)
    C = sp.Matrix([sol[0][X], sol[0][Y], 0])
    ok1 = len(sol) == 1 and sp.simplify(C - sp.Matrix(sp.sympify(md_answer(2, "1")))) == sp.zeros(3, 1)
    # (2) 임의의 z 에서 등거리점의 x,y 가 C 와 같음 [증명-수치확인: 기호적으로 z 소거]
    Dg = sp.Matrix([X, Y, Z])
    sol2 = sp.solve([d2(Dg, O) - d2(Dg, A), d2(Dg, O) - d2(Dg, B)], [X, Y], dict=True)
    ok2 = len(sol2) == 1 and sol2[0][X] == C[0] and sol2[0][Y] == C[1]
    # (3) 법선벡터로 이면각, cos2θ=-1/5, z>0
    z = sp.symbols("z", positive=True)
    D = sp.Matrix([C[0], C[1], z])
    n1 = A.cross(D); n2 = sp.Matrix([0, 0, 1])
    cth = sp.Abs(n1.dot(n2)) / sp.sqrt(n1.dot(n1))
    zs = sp.solve(sp.Eq(2 * cth**2 - 1, sp.Rational(-1, 5)), z)
    OD = [sp.simplify(sp.sqrt(D.dot(D)).subs(z, v)) for v in zs]
    ok3 = len(OD) == 1 and sp.simplify(OD[0] - sp.sympify(md_answer(2, "3"))) == 0
    # 대표 오답(∠DOC 를 θ 로) 재현: t^2=45/8
    t = sp.symbols("t", positive=True)
    wrong = sp.solve(sp.Eq(2 * (sp.Rational(3, 2) / t)**2 - 1, sp.Rational(-1, 5)), t)
    report(2, ok1 and ok2 and ok3, f"C={list(C)} z={zs} OD={OD} (오답 재현 t^2={[w**2 for w in wrong]})")


def p03():
    ns = sp.symbols("n", positive=True, integer=True)
    ok1 = ok2 = ok3 = True
    msg = []
    for N in range(3, 9):
        # (1) f(k)=k
        ok1 &= sp.Rational(N * N - 1, 2) == sp.sympify(md_answer(3, "1"), locals={"n": ns}).subs(ns, N)
        # (2)(3) 전수 열거: 꺾은선 함수의 정적분 = 구간별 sympy 적분
        best, cnt = None, 0
        for perm in itertools.permutations(range(1, N + 1)):
            val = sum(sp.integrate(perm[k] + (perm[k + 1] - perm[k]) * (x - (k + 1)), (x, k + 1, k + 2))
                      for k in range(N - 1)) if N <= 5 else \
                sum(sp.Rational(perm[k] + perm[k + 1], 2) for k in range(N - 1))
            if best is None or val < best:
                best, cnt = val, 1
            elif val == best:
                cnt += 1
        ok2 &= best == sp.sympify(md_answer(3, "2"), locals={"n": ns}).subs(ns, N)
        ok3 &= cnt == sp.sympify(md_answer(3, "3"), locals={"n": ns}).subs(ns, N)
        msg.append(f"n={N}:min={best},cnt={cnt}")
    report(3, ok1 and ok2 and ok3, " ".join(msg[:4]) + " …(n=3..8 전수)")


if __name__ == "__main__":
    fns = sorted((k for k in dir() if re.fullmatch(r"p\d+", k)), key=lambda s: int(s[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
