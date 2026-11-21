#!/usr/bin/env python3
"""KHU-M25 (경희대 2025학년도 모의논술 자연계, KHU-M25-01~03) — sympy 검산.
규칙: 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import random
import re
import sys

import sympy as sp

PREFIX = "KHU-M25"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def shoelace(P, Q, R):
    return sp.Rational(1, 2) * sp.Abs((Q[0] - P[0]) * (R[1] - P[1]) - (Q[1] - P[1]) * (R[0] - P[0]))


def p01():
    a, b, t = sp.symbols("a b t", positive=True)
    c = sp.sqrt(a**2 + b**2)
    # (1) 무작위 점으로 공식 확인
    random.seed(3)
    ok1 = True
    for _ in range(20):
        av, bv = sp.Rational(random.randint(1, 9), 3), sp.Rational(random.randint(1, 9), 4)
        y1, y2 = sp.Rational(random.randint(1, 30), 7), sp.Rational(random.randint(1, 30), 7)
        x1 = av * sp.sqrt(1 + y1**2 / bv**2); x2 = -av * sp.sqrt(1 + y2**2 / bv**2)
        cv = sp.sqrt(av**2 + bv**2)
        F, Fp = (cv, 0), (-cv, 0)
        qff = shoelace((x2, y2), F, Fp)
        pqf = shoelace((x1, y1), (x2, y2), F)
        ok1 &= sp.simplify(qff - cv * y2) == 0
        ok1 &= abs(float(pqf - sp.Rational(1, 2) * sp.Abs(x1 * y2 - x2 * y1 + cv * (y1 - y2)))) < 1e-12
    # (2)
    x1 = a * sp.sqrt(1 + t**2 / b**2); x2 = -a * sp.sqrt(1 + 1 / b**2)
    P, Q, F, Fp = (x1, t), (x2, 1), (c, 0), (-c, 0)
    inner = (Q[0] - P[0]) * (F[1] - P[1]) - (Q[1] - P[1]) * (F[0] - P[0])
    # t 가 클 때 부호 확인 후 절댓값 제거
    sgn = sp.sign(inner.subs({a: 2, b: 3, t: 1000}).evalf())
    R = (sgn * inner / 2) / (c * t)
    lim = sp.limit(R, t, sp.oo)
    expect = sp.Rational(1, 2) + a / (2 * b * c) + a / (2 * c) * sp.sqrt(1 + 1 / b**2)
    ok2 = sp.simplify(lim - expect) == 0
    report(1, ok1 and ok2, f"(1) 넓이 공식 20점 일치, (2) lim R = {sp.simplify(lim)}")


def p02():
    s3 = sp.sqrt(3)
    B = sp.Matrix([0, 0, 0]); C = sp.Matrix([10, 0, 0]); D = sp.Matrix([5, 5 * s3, 0])
    H = (B + C + D) / 3
    h = sp.symbols("h", positive=True)
    A = H + sp.Matrix([0, 0, h])
    n1 = (B - A).cross(C - A); n2 = sp.Matrix([0, 0, 1])
    cos_th = sp.Abs(n1.dot(n2)) / n1.norm()
    hs = sp.solve(sp.Eq(cos_th**2, sp.Rational(1, 3)), h)
    hv = hs[0]
    Av = A.subs(h, hv)
    AB = sp.simplify((Av - B).norm())
    # 외접구 중심
    X = sp.Matrix(sp.symbols("u v w", real=True))
    eqs = [sp.expand((X - P).dot(X - P) - (X - B).dot(X - B)) for P in (C, D, Av)]
    sol = sp.solve(eqs, list(X), dict=True)[0]
    Rr = sp.simplify((X.subs(sol) - B).norm())
    ok = sp.simplify(AB - 5 * sp.sqrt(2)) == 0 and sp.simplify(Rr - 5 * sp.sqrt(6) / 2) == 0
    report(2, ok, f"높이 AH={sp.simplify(hv)}, AB={AB}, R={Rr}")


def p03():
    a = sp.symbols("a", positive=True)
    x, X = sp.symbols("x X", real=True)
    f = x**2
    # (1)
    tan = lambda p: sp.diff(f, x).subs(x, p) * (X - p) + p**2
    m_line = -1 / (2 * a) * (X - a) + a**2
    xs = [s for s in sp.solve(sp.Eq(m_line, X**2), X) if sp.simplify(s - a) != 0]
    p2 = xs[0]
    def area(p):
        xi = sp.solve(sp.Eq(tan(p), 0), X)[0]
        lo, hi = (0, p) if sp.simplify(p).is_positive else (p, 0)
        whole = sp.integrate(f, (x, lo, hi))
        tri = sp.integrate(sp.Abs(tan(p).subs(X, x)), (x, *sorted([xi, p], key=lambda v: float(v.subs(a, 1)))))
        return sp.simplify(whole - tri)
    Aa = area(a)
    Bb = area(p2)
    lim = sp.limit(Aa * Bb, a, 0)
    ok1 = lim == sp.Rational(1, 1152)
    # (2) 수치·기호: 법선이 P1 을 지나는 b<0 두 개
    av = sp.symbols("av", positive=True)
    bb = sp.symbols("bb", real=True)
    normal_through = sp.Eq(-1 / (2 * bb) * (av - bb) + bb**2, av**2)
    roots = [r for r in sp.solve(normal_through, bb) if sp.simplify(r - av) != 0]
    bexpr, cexpr = roots
    dot = sp.simplify((bexpr - av) * (cexpr - av) + (bexpr**2 - av**2) * (cexpr**2 - av**2))
    asol = [s for s in sp.solve(sp.Eq(dot, sp.Rational(51, 4)), av) if s.is_positive]
    ok2 = False
    msg2 = ""
    for s in asol:
        bv, cv = sp.simplify(bexpr.subs(av, s)), sp.simplify(cexpr.subs(av, s))
        if bv.is_real and cv.is_real and bv < 0 and cv < 0 and bv != cv:
            ok2 = sp.simplify(bv + cv + 2) == 0
            msg2 = f"a={s}, b,c={bv},{cv}"
    report(3, ok1 and ok2, f"(1) A={Aa}, B={sp.factor(Bb)}, lim AB={lim}; (2) {msg2}")


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
