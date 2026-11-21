#!/usr/bin/env python3
"""KHU-M24 (경희대 2024학년도) — sympy 검산
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "KHU-M24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import itertools
    p = sp.symbols("p", positive=True)
    PA = PAB = 0
    EX = 0
    for seq in itertools.product([1, 0], repeat=3):   # 1=흰 공
        pr = sp.Mul(*[p if s else 1 - p for s in seq])
        money = 300 + sum(100 if s else -100 for s in seq)
        EX += money * pr
        if money == 400:
            PA += pr
            if 300 + (100 if seq[0] else -100) == 400:
                PAB += pr
    cond = sp.simplify(PAB / PA)
    psol = sp.solve(sp.Eq(sp.expand(EX), 200), p)
    nsol = [60 * v for v in psol if 0 < 60 * v < 60 and (60 * v).is_integer]
    # 공식 예시답안의 오기식 6p^3+12p^2(1-p)+6p(1-p)-2 은 p=1/3 에서 0 이 아님을 확인
    typo = (6 * p**3 + 12 * p**2 * (1 - p) + 6 * p * (1 - p) - 2).subs(p, sp.Rational(1, 3))
    ok = cond == sp.Rational(2, 3) and nsol == [20] and typo != 0
    report(1, ok, f"(1) P(B|A)={cond}  (2) E(X)={sp.factor(sp.expand(EX))} → n={nsol} [오기식 값 {typo}≠0]")


def p02():
    C = sp.symbols("C", real=True)
    I1 = sp.integrate(x * sp.sin(x), x); I2 = sp.integrate(x * sp.cos(x), x)
    ok1 = sp.simplify(I1 - (-x * sp.cos(x) + sp.sin(x))) == 0 and sp.simplify(I2 - (x * sp.sin(x) + sp.cos(x))) == 0
    G = sp.integrate(x**2 * sp.sin(x), x)
    ok2 = sp.simplify(G - ((2 - x**2) * sp.cos(x) + 2 * x * sp.sin(x))) == 0
    crit = sp.solveset(sp.diff(G, x), x, sp.Interval(0, 2 * sp.pi))
    vals = [sp.simplify(G.subs(x, v)) for v in sorted(crit, key=float)]
    mx, mn = max(vals, key=float), min(vals, key=float)
    region = sp.solveset((mx + C) * (mn + C) > 0, C, sp.S.Reals)
    target = sp.Union(sp.Interval.open(-sp.oo, 2 - sp.pi**2), sp.Interval.open(4 * sp.pi**2 - 2, sp.oo))
    # 수치: 경계 근처 C 에서 x축과 만나는지 확인
    import math
    def meets(cv):
        ys = [float(G.subs(x, 2 * math.pi * k / 2000)) + cv for k in range(2001)]
        return min(ys) <= 0 <= max(ys)
    num = (not meets(2 - math.pi**2 - 0.01)) and meets(2 - math.pi**2 + 0.01) and (not meets(4 * math.pi**2 - 2 + 0.01)) and meets(4 * math.pi**2 - 2 - 0.01)
    ok = ok1 and ok2 and region == target and num
    report(2, ok, f"(1) {I1}, {I2}  (2) G={G}, 임계값 {vals}, C∈{region} [경계 수치확인]")


def p03():
    import math
    a, b, p_, q = sp.symbols("a b p q", positive=True)
    c = sp.sqrt(a**2 + b**2)
    l = sp.symbols("l", real=True)
    p_expr = a * sp.sqrt(1 + q**2 / b**2)
    lsol = sp.solve(sp.Eq(c**2 + l**2, (l - q)**2 + p_expr**2), l)[0]
    ok1 = sp.simplify(lsol - (c**2 * q**2 - b**4) / (2 * b**2 * q)) == 0
    thr = sp.solve(sp.Eq(lsol, 0), q)
    ok1 = ok1 and any(sp.simplify(t - b**2 / c) == 0 for t in thr)
    # (2) 각을 직접 계산해 q 가 커질 때 cos(2α+β) 의 극한을 수치로 확인
    def cos2ab(av, bv, qv):
        cv = math.sqrt(av * av + bv * bv); pv = av * math.sqrt(1 + qv * qv / (bv * bv))
        P = (pv, qv); F = (cv, 0); Fp = (-cv, 0)
        def ang(V, A, B):
            u = (A[0] - V[0], A[1] - V[1]); w = (B[0] - V[0], B[1] - V[1])
            return math.acos((u[0] * w[0] + u[1] * w[1]) / (math.hypot(*u) * math.hypot(*w)))
        al = ang(Fp, P, F); be = ang(P, Fp, F)
        return math.cos(2 * al + be)
    t = sp.symbols("t", positive=True)
    f = (t**2 - 1) / (t**2 + 1)
    good = True
    for av, bv in [(2, 1), (3, 2), (5, 1), (1.5, 1.2)]:
        lim = cos2ab(av, bv, 1e7)
        good &= abs(lim - float(f.subs(t, av / bv))) < 1e-6
    F = sp.integrate(t * f, t)
    ok2 = good and sp.simplify(sp.diff(t**2 / 2 - sp.log(1 + t**2), t) - t * f) == 0 and sp.simplify(sp.diff(F - (t**2 / 2 - sp.log(1 + t**2)), t)) == 0
    report(3, ok1 and ok2, f"(1) l={sp.simplify(lsol)}, l>0 ⇔ q>b^2/c  (2) 수치 극한 일치, f(t)={f}, ∫tf(t)dt={F}+C")


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
