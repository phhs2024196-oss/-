#!/usr/bin/env python3
"""SYU-G24N (2024학년도 삼육대 논술 자연계열1 수학 7~15) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<문항번호>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "SYU-G24N"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"
x, t, n, k = sp.symbols("x t n k", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq: int) -> str:
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return str(d["subparts"][0]["answer"]["value"])


def load_pairs(seq: int) -> dict:
    """'a=-1; b=0' 형식."""
    return {kk.strip(): sp.sympify(v) for kk, v in (p.split("=") for p in load_answer(seq).split(";"))}


def p7():
    # 일반 사차식(최고차 1, 원점 통과)에서 조건 (가)(나)를 연립
    b3, b2, b1 = sp.symbols("b3 b2 b1")
    f = x**4 + b3 * x**3 + b2 * x**2 + b1 * x
    sym = sp.Poly(sp.expand(f.subs(x, 1 - x) - f.subs(x, 1 + x)), x).all_coeffs()
    eqs = list(sym) + [sp.diff(f, x).subs(x, 2)]
    sols = sp.solve(eqs, [b3, b2, b1], dict=True)
    vals = []
    for s in sols:
        g = f.subs(s)
        g2 = sp.diff(g, x, 2)
        if g2.subs(x, 2) <= 0:      # x=2 가 극소여야
            continue
        crit = sp.solve(sp.diff(g, x), x)
        maxima = [c for c in crit if sp.diff(g, x, 2).subs(x, c) < 0]
        vals += [g.subs(x, c) for c in maxima]
    ok = len(set(vals)) == 1 and sp.simplify(vals[0] - sp.sympify(load_answer(7))) == 0
    report(7, ok, f"해 {sols}, 극댓값 {set(vals)}")


def p8():
    nn = sp.symbols("nn", positive=True, integer=True)
    r = 1 / 2**nn
    kk = sp.symbols("kk", positive=True)
    kval = sp.solve(sp.Eq(kk / sp.sqrt(5), r), kk)[0]          # 2x-y+k=0 과 원점 거리 = r
    ytop, xint = kval, kval / 2                                 # 꼭짓점 (0,±k), (±k/2,0)
    S = (2 * ytop) * (2 * xint) / 2
    T = sp.pi * r**2
    total = sp.nsimplify(sum(sp.simplify((S - 2 / sp.pi * T).subs(nn, i)) for i in range(1, 6)))
    ok = sp.simplify(total - sp.sympify(load_answer(8))) == 0
    report(8, ok, f"S_n={sp.simplify(S)}, 합={total}")


def p9():
    vP = sp.Abs(t**2 - 3 * t)
    vQ = t
    ts = sorted({r for r in sp.solve(t**2 - 3 * t - t, t) + sp.solve(-(t**2 - 3 * t) - t, t) if r > 0})
    t1, t2 = ts

    def dist(v, a, b):
        return sp.integrate(sp.Piecewise((-(t**2 - 3 * t), t <= 3), (t**2 - 3 * t, True)) if v is vP else v, (t, a, b))

    d1 = sp.Abs(dist(vP, 0, t1) - dist(vQ, 0, t1))
    d2 = sp.Abs(dist(vP, t1, t2) - dist(vQ, t1, t2))
    ok = len(ts) == 2 and sp.simplify(d1 + d2 - sp.sympify(load_answer(9))) == 0
    report(9, ok, f"t1={t1}, t2={t2}, d1={d1}, d2={d2}, 합={d1 + d2}")


def p10():
    a = [sp.log((m + 1)**2, 2) - sp.log(m * (m + 2), 2) for m in range(1, 11)]
    slopes = [ai / ((m + 1)**2 - m * (m + 2)) for m, ai in zip(range(1, 11), a)]
    val = sp.nsimplify(2**sp.simplify(sum(slopes)))
    pq = sp.fraction(sp.Rational(val))
    ok = sp.gcd(*pq) == 1 and pq[0] + pq[1] == sp.sympify(load_answer(10))
    report(10, ok, f"2^합={val}, q={pq[0]}, p={pq[1]}, p+q={pq[0] + pq[1]}")


def p11():
    sol_a = [s for s in sp.solveset(sp.Eq(sp.sin(2 * x - sp.pi / 2), sp.Rational(1, 2)), x, sp.Interval.Ropen(0, 2 * sp.pi))]
    sol_b = [s for s in sp.solveset(sp.Eq(sp.cos(3 * x + sp.pi / 2), sp.Rational(1, 3)), x, sp.Interval.Ropen(0, 2 * sp.pi))]
    al, be = sp.nsimplify(sum(sol_a)), sp.simplify(sum(sol_b))
    # 수치 확인(해 개수와 합)
    import math
    grid = [i * 2 * math.pi / 200000 for i in range(200000)]
    cnt_b = sum(1 for g0, g1 in zip(grid, grid[1:]) if (math.cos(3 * g0 + math.pi / 2) - 1 / 3) * (math.cos(3 * g1 + math.pi / 2) - 1 / 3) < 0)
    diff = sp.simplify(be - al)
    ok = len(sol_a) == 4 and cnt_b == 6 and abs(float(diff - sp.sympify(load_answer(11)))) < 1e-9
    report(11, ok, f"α={al} ({len(sol_a)}개), β={be} ({cnt_b}개), β-α={diff}")


def p12():
    a, b = sp.symbols("a b")
    g = a * x**2 + b * x + 1
    f_l, f_r = 2**(x - 2), sp.log(x + 1, 2)                       # x<1, x>=1
    fm_l, fm_r = sp.log(-x + 1, 2), 2**(-x - 2)                   # f(-x): x<=-1, x>-1
    e1 = sp.Eq(sp.limit(g / f_l, x, 1, "-"), sp.limit(g / f_r, x, 1, "+"))
    e2 = sp.Eq(sp.limit(g / fm_l, x, -1, "-"), sp.limit(g / fm_r, x, -1, "+"))
    sol = sp.solve([e1, e2], [a, b], dict=True)
    ans = load_pairs(12)
    ok = len(sol) == 1 and sol[0][a] == ans["a"] and sol[0][b] == ans["b"]
    report(12, ok, f"해 {sol}")


def p13():
    # 미분가능한 일반 함수: 테일러 계수로 모델링 (f(2)=c0,f'(2)=c1,f(3)=d0,f'(3)=d1)
    c0, c1, d0, d1, h = sp.symbols("c0 c1 d0 d1 h")
    # (가): x=4+h → f(3+h) ≈ d0 + d1 h
    # 극한 존재: 1/h 계수 0, 극한값 -4
    eqA = [sp.Eq(d0 + 2, 0), sp.Eq(d1, -4)]
    # (나): x=2+h
    num = 2 * (c0 + c1 * h) + 3 * (d0 + d1 * h) + (2 + h)
    eqB = [sp.Eq(num.subs(h, 0), 0), sp.Eq(sp.diff(num, h), 1)]
    sol = sp.solve(eqA + eqB, [c0, c1, d0, d1], dict=True)[0]
    val = sol[c0] + sol[c1]
    # 구체적 예시 함수로 재확인: f(x)=다항식 보간
    A0, A1, A2, A3 = sp.symbols("A0:4")
    fx = A0 + A1 * x + A2 * x**2 + A3 * x**3
    s2 = sp.solve([fx.subs(x, 2) - sol[c0], sp.diff(fx, x).subs(x, 2) - sol[c1], fx.subs(x, 3) - sol[d0], sp.diff(fx, x).subs(x, 3) - sol[d1]], [A0, A1, A2, A3])
    fx = fx.subs(s2)
    la = sp.limit((fx.subs(x, x - 1) + 2) / (x - 4), x, 4)
    lb = sp.limit((2 * fx + 3 * fx.subs(x, x + 1) + x) / (x - 2), x, 2)
    ok = la == -4 and lb == 1 and val == sp.sympify(load_answer(13))
    report(13, ok, f"f(2)={sol[c0]}, f'(2)={sol[c1]}, 합={val}; 예시함수 극한 (가)={la}, (나)={lb}")


def p14():
    found = []
    for a in range(1, 20):
        f = 5 * a * sp.sin(a * x + sp.pi / 8) + 3 * a * sp.cos(-a * x + 3 * sp.pi / 8) + 3
        sols = sp.solveset(sp.Eq(sp.simplify(f - 3), 0), x, sp.Interval.Ropen(0, 2 * sp.pi))
        if len(list(sols)) == 6:
            M = sp.maximum(f, x, sp.Interval(0, 2 * sp.pi))
            found.append((a, sp.simplify(M)))
    ok = len(found) == 1 and found[0][0] + found[0][1] == sp.sympify(load_answer(14))
    report(14, ok, f"(a, M)={found}")


def p15():
    a = sp.symbols("a")
    f1 = a * (x**2 + 3 * x)
    f2 = -x**3 + a * x
    # x=1 연속: 좌극한 f2(1) = f(1) = f(-1)+2
    aval = sp.solve(sp.Eq(f2.subs(x, 1), f1.subs(x, -1) + 2), a)
    assert len(aval) == 1
    aval = aval[0]
    # x=0 연속 확인
    cont0 = f1.subs({a: aval, x: 0}) == f2.subs({a: aval, x: 0})
    base = sp.Piecewise((f1.subs(a, aval), x < 0), (f2.subs(a, aval), True))   # [-1,1)

    total = 0
    for j in range(0, 13):          # 구간 [2j-1, 2j+1) 위의 조각
        lo, hi = max(2 * j - 1, 0), min(2 * j + 1, 24)
        if lo >= hi:
            continue
        piece = base.subs(x, x - 2 * j) + 2 * j
        total += sp.integrate(piece, (x, lo, hi))
    ok = cont0 and sp.simplify(total - sp.sympify(load_answer(15))) == 0
    report(15, ok, f"a={aval}, x=0 연속 {cont0}, ∫0^24 f={total}")


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
