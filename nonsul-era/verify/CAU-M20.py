#!/usr/bin/env python3
"""CAU-M20 (2020학년도 중앙대 모의논술 자연계열 수학 문제 1~3) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
PREFIX = "CAU-M20"
import re
import sys
import itertools
from fractions import Fraction as Fr

import sympy as sp

WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """동전 전수: 단계 I 값 분포, 단계 II 종료확률(반복은 기하분포) → E[점수] (답 23/4, 5.8)."""
    stage1 = {}
    for c in itertools.product("HT", repeat=2):
        v = 1 if c == ("H", "H") else 2 if c == ("T", "T") else 3
        stage1[v] = stage1.get(v, 0) + Fr(1, 4)
    stop = Fr(sum(1 for c in itertools.product("HT", repeat=4) if c.count("H") != 2), 16)
    # 단계 II 가 k번째에 끝날 확률의 합 = sum stop*(1-stop)^(k-1) (sympy 무한급수)
    k = sp.symbols("k", integer=True, positive=True)
    pend = sp.summation(sp.Rational(stop.numerator, stop.denominator) * (1 - sp.Rational(stop.numerator, stop.denominator)) ** (k - 1), (k, 1, sp.oo))
    E = sum(sp.Rational(p.numerator, p.denominator) * v**2 for v, p in stage1.items()) * pend
    report(1, E == sp.Rational(23, 4) and round(float(E), 1) == 5.8, f"P(종료)={stop}, 종료확률합={pend}, E={E}≈{float(E):.2f}")


def p02():
    """(2-1) fff(x)=x 항등식 조건 → d=-c^2-c-1, d 최댓값 -3/4 / (2-2) g(x) 닫힌식·극값점·급수 1/(2(e^pi+1))."""
    c, d = sp.symbols("c d", real=True)
    f = lambda u: (c * u + 1) / (d * u + 1)
    num = sp.expand(sp.numer(sp.together(f(f(f(x))) - x)))
    eqs = sp.Poly(num, x).coeffs()
    sols = sp.solve(eqs, [d], dict=True)
    ok1 = any(sp.simplify(s[d] - (-c**2 - c - 1)) == 0 for s in sols)
    # 실제로 fff=x 가 성립하는 (c,d) 에서 d 의 최댓값: d=-c^2-c-1 의 최댓값
    dexpr = -c**2 - c - 1
    cc = sp.solve(sp.diff(dexpr, c), c)[0]
    dmax = dexpr.subs(c, cc)
    # 수치 확인: c=-1/2, d=-3/4 에서 fff(x)=x
    chk = all(abs(float(f(f(f(x))).subs({c: cc, d: dmax}).subs(x, xv)) - xv) < 1e-9 for xv in (0.3, 2.0, -5.0))
    ok2 = dmax == sp.Rational(-3, 4) and chk
    # (2-2)
    X = sp.symbols("X", positive=True)
    G = sp.integrate(sp.sin(sp.log(t)), (t, 1, X))
    closed = X / 2 * (sp.sin(sp.log(X)) - sp.cos(sp.log(X))) + sp.Rational(1, 2)
    ok3 = sp.simplify(G - closed) == 0
    # 극값점: g'(x)=sin(ln x)=0, 0<x<1 → x=e^{-n pi}, 부호 변화 확인 (n=1..5)
    pts = [sp.exp(-k * sp.pi) for k in range(1, 6)]
    sgn = all(sp.sin(sp.log(p * sp.exp(sp.Rational(1, 10)))) * sp.sin(sp.log(p * sp.exp(-sp.Rational(1, 10)))) < 0 for p in pts)
    k = sp.symbols("k", integer=True, positive=True)
    term = sp.simplify(closed.subs(X, sp.exp(-k * sp.pi)) - sp.Rational(1, 2))
    r_ = -sp.exp(-sp.pi)  # 항 = -(1/2) r_^k 꼴인지 확인 후 등비급수
    assert sp.simplify(term - (-(r_**k) / 2)) == 0
    S = sp.simplify(-sp.Rational(1, 2) * r_ / (1 - r_))
    num_S = sum(float(closed.subs(X, sp.exp(-j * sp.pi))) - 0.5 for j in range(1, 40))
    ok4 = sgn and abs(num_S - float(1 / (2 * (sp.exp(sp.pi) + 1)))) < 1e-12 and sp.simplify(S - 1 / (2 * (sp.exp(sp.pi) + 1))) == 0
    report(2, ok1 and ok2 and ok3 and ok4, f"(2-1) d=-c^2-c-1 → d_max={dmax}  (2-2) g 닫힌식 일치, 항={term}, 합={sp.simplify(S)}")


def p03():
    """(3-1) 포물선-원 접함 조건 b=a+1/(4a)+1, 넓이 h(a) 최솟값 9/2 / (3-2) A,B 쌍 전수(회전으로 B 생성)."""
    A_, b = sp.symbols("a b", positive=True)
    y = sp.symbols("y", real=True)
    quad = sp.expand((b - y) / A_ + y**2 - 2 * y)  # x^2 = (b-y)/a 대입
    disc = sp.discriminant(quad * A_, y)
    bsol = [s for s in sp.solve(disc, b)]
    bexpr = [s for s in bsol if sp.simplify(s - (A_ + 1 / (4 * A_) + 1)) == 0]
    ok1 = len(bexpr) == 1
    bb = bexpr[0]
    r = sp.sqrt(bb / A_)
    h = sp.integrate(bb - A_ * x**2, (x, -r, r))
    hmin_pts = [s for s in sp.solve(sp.diff(h, A_), A_) if s.is_positive]
    hmin = min(sp.nsimplify(sp.simplify(h.subs(A_, s))) for s in hmin_pts)
    # 수치: 격자 최소
    grid = min(float(h.subs(A_, 0.05 * i)) for i in range(1, 200))
    # 접점이 실제 원 위 (y 중근이 0<=y<=2, x^2>=0) 확인 a=1
    ycon = sp.solve(quad.subs({A_: 1, b: bb.subs(A_, 1)}), y)
    ok2 = hmin == sp.Rational(9, 2) and abs(grid - 4.5) < 1e-3 and all(0 <= yy <= 2 for yy in ycon)
    # (3-2)
    a = sp.symbols("a", real=True)
    C = sp.Matrix([-sp.sqrt(3), 0]); P = sp.Matrix([a, 3 - a**2])
    pairs = set()
    for sg in (1, -1):
        th = sg * sp.pi / 3
        R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
        B = P + R * (C - P)
        eq = sp.expand(B[1] - (3 - a**2) + 2 * a * (B[0] - a))
        for rr in sp.Poly(eq, a).nroots(n=30):
            if abs(sp.im(rr)) > 1e-12:
                continue
            rr = sp.nsimplify(sp.re(rr), [sp.sqrt(3)])
            if sp.simplify(rr + sp.sqrt(3)) == 0:
                continue  # A=C 퇴화
            Bv = sp.simplify(B.subs(a, rr))
            pairs.add(((rr, 3 - rr**2), (sp.nsimplify(Bv[0], [sp.sqrt(3)]), sp.nsimplify(Bv[1]))))
    want = {((0, 3), (-2 * sp.sqrt(3), 3)), ((7 * sp.sqrt(3) / 6, sp.Rational(-13, 12)), (5 * sp.sqrt(3) / 8, sp.Rational(65, 24)))}
    ok3 = {tuple(map(lambda q: tuple(sp.simplify(v) for v in q), pr)) for pr in pairs} == want
    report(3, ok1 and ok2 and ok3, f"(3-1) b={bb}, h_min={hmin} (격자 {grid:.4f})  (3-2) 쌍 {len(pairs)}개: {sorted(map(str, pairs))}")


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

