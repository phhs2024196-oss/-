#!/usr/bin/env python3
"""KNU 2026학년도 논술(AAT) 모의고사 자연계열Ⅰ 수학 (KNU-M26A-01~02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-M26A"
WIDTH = 2
RESULTS = []
x, h = sp.symbols("x h", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())


def load_answer(seq, i=0):
    return load_item(seq)["subparts"][i]["answer"]["value"]



def p01():
    import math
    msgs, ok = [], True
    r = sp.Symbol("r", positive=True)
    # 1-1: (ⅰ)로 a_n 을 만들고 (ⅱ)에서 b_n 을 순차로 풀어 md 식과 비교(n=1..12)
    a = [None, sp.Integer(1)]
    for n in range(1, 13):
        a.append(sp.simplify((r - sp.Rational(1, n)) * a[n]))
    b = {}
    for n in range(1, 13):
        prod = sp.Mul(*[b[k] for k in range(1, n)])
        b[n] = sp.simplify(a[n + 1] / (r**n * prod))
    nn = sp.Symbol("n", positive=True)
    e11 = sp.sympify(load_answer(1, 0), locals={"r": r, "n": nn})
    ok11 = all(sp.simplify(b[n] - e11.subs(nn, n)) == 0 for n in range(1, 13))
    ok &= ok11
    msgs.append(f"1-1 b_n 12항 일치={ok11}")
    # 1-2 [증명-수치확인]: r∈(1,20] 격자 200 × n=2..300 에서 b_n >= (1+1/(r(n-1)))^-1
    bad = 0
    for i in range(1, 201):
        rv = 1 + i * 0.1
        for n in range(2, 301):
            if (1 - 1 / (rv * n)) < 1 / (1 + 1 / (rv * (n - 1))) - 1e-15:
                bad += 1
    ok &= bad == 0
    msgs.append(f"1-2 [증명-수치확인] 반례 {bad}개/59800")
    # 1-3 [증명-수치확인]: (가) 부등식 k=2..2000, 그리고 r^n/a_{n+1} >= (n+1)^{1/r} 및 증가 무한대(r=1.5,3,10)
    ok_ga = all(sum(1 / j for j in range(2, k + 1)) <= math.log(k) <= sum(1 / j for j in range(1, k)) for k in range(2, 2001))
    div = True
    for rv in (1.5, 3.0, 10.0):
        la = 0.0                      # log(a_{n+1}/r^n) = sum log(1 - 1/(r k))
        for n in range(1, 200001):
            la += math.log(1 - 1 / (rv * n))
            if n in (10, 1000, 200000):
                if -la < math.log(n + 1) / rv - 1e-12:
                    div = False
        div &= -la > 1.0 / rv * math.log(200001) - 1e-9 and -la > 1     # 하한 (n+1)^{1/r} → ∞
    e13 = sp.sympify(load_answer(1, 2))
    ok &= ok_ga and div and e13 == sp.oo
    msgs.append(f"1-3 [증명-수치확인] (가) k≤2000 성립={ok_ga}, r^n/a_(n+1)≥(n+1)^(1/r)→∞ ={div}, md={e13}")
    # 1-4: r=3 에서 log(a_{n+1}/3^n) 의 log n 에 대한 기울기 → 지수 s, 극한 0 ⇔ p < -s
    la, pts = 0.0, {}
    for n in range(1, 10**6 + 1):
        la += math.log(1 - 1 / (3 * n))
        if n in (10**4, 10**6):
            pts[n] = la
    slope = (pts[10**6] - pts[10**4]) / (math.log(10**6) - math.log(10**4))
    thr = sp.nsimplify(round(-slope, 4), tolerance=1e-3, rational=True)
    e14 = sp.sympify(load_answer(1, 3))
    ok &= e14 == sp.Interval.open(-sp.oo, thr)
    msgs.append(f"1-4 기울기={slope:.6f} → p<{thr}, md={e14}")
    report(1, ok, "; ".join(msgs))


def p02():
    msgs, ok = [], True
    # 2-1: 부분집합 직접 세기
    S = {n: sum(1 for _ in itertools.combinations(range(n, 2 * n + 2), n)) for n in range(1, 12)}
    b = {1: S[1]}
    for n in range(2, 12):
        b[n] = S[n] - S[n - 1]
    P, Q = sp.symbols("P Q")
    pq = sp.solve([P * 2 + Q - b[2], P * 3 + Q - b[3]], [P, Q])
    assert all(pq[P] * n + pq[Q] == b[n] for n in range(2, 12))
    got21 = (S[5] + b[1], pq[P], pq[Q])
    e21 = tuple(sp.sympify(load_answer(2, 0)))
    ok &= got21 == e21
    msgs.append(f"2-1 (S5+b1,p,q)={got21} md={e21}")
    # 2-2: a1 = 급수, 이웃 비 후보 = cos 방정식의 해 중 조건 (4) 범위, n=1 은 조건 (3)
    k = sp.Symbol("k", integer=True, positive=True)
    bb = lambda m: b[1] if m == 1 else pq[P] * m + pq[Q]
    a1 = 1 / sp.Integer(b[1] * bb(3)) + sp.summation(1 / ((pq[P] * k + pq[Q]) * (pq[P] * (k + 2) + pq[Q])), (k, 2, sp.oo))
    a1 = sp.nsimplify(a1)
    rho = sp.Symbol("rho", real=True)
    cond5 = sp.sqrt(2) * sp.cos((8 * rho + 1) / 8 * sp.pi) ** 2 + 3 * sp.cos((8 * rho - 3) / 8 * sp.pi) - 2 * sp.sqrt(2)
    cand = sorted({sp.nsimplify(c) for c in sp.solve(cond5, rho)} | set())
    # sympy 가 주기해를 다 주지 않을 수 있으므로 격자 확인으로 보충: rho = 3/8 ± 1/4 + 2j
    cand = sorted(set(cand) | {sp.Rational(3, 8) + s * sp.Rational(1, 4) + 2 * j for s in (1, -1) for j in range(-3, 4)})
    cand = [c for c in cand if abs(float(cond5.subs(rho, c))) < 1e-12]
    def ratio(n):
        lo, hi = a1 / bb(n), sp.Rational(bb(n + 1), 2 * bb(n))
        rs = [c for c in cand if lo < c < hi]
        assert len(rs) == 1, (n, rs)
        return rs[0]
    rat = {n: ratio(n) for n in range(2, 40)}
    al = sp.Symbol("al", real=True)
    # 급수: a1 + a1*al + a1*al*rat2 * sum_{m>=0} (1/8)^m … 비가 n>=3 에서 상수임을 확인 후 등비급수
    assert len({rat[n] for n in range(3, 40)}) == 1
    rc = rat[3]
    tot = a1 + a1 * al + a1 * al * rat[2] / (1 - rc)
    alphas = [c for c in cand if 0 < tot.subs(al, c) < sp.Rational(1, 2)]
    assert len(alphas) == 1
    a2 = a1 * alphas[0]; a3 = a2 * rat[2]; a4 = a3 * rc
    alpha_q = sp.nsimplify(a4 / a3)
    assert sp.simplify(15 * alpha_q**3 - a3) == 0 and sp.simplify(15 * alpha_q**4 - a4) == 0
    got22 = (a1 + a2, alpha_q)
    e22 = tuple(sp.sympify(load_answer(2, 1)))
    ok &= got22 == e22
    msgs.append(f"2-2 a1={a1} a2={a2} a3={a3} (a1+a2, alpha)={got22} md={e22}")
    # 2-3: 조건을 만족하는 일반 다항식(임의 계수 3세트)으로 극한을 직접 계산 → 값이 같고 md 와 일치
    t = sp.Symbol("t", positive=True)
    vals = set()
    for k1, k2, m1, m2, m3 in ((1, 2, 3, -1, 0), (-2, 5, 0, 4, 1), (0, 0, 1, 1, 1)):
        f = (x - 2) ** 2 * (k1 * x + k2) + 8 * x - 16
        g = 3 + (x - 2) * (m1 * x**2 + m2 * x + m3)
        expr = sum(f.subs(x, 2 + an * t) / t / sp.exp(g.subs(x, 2 + an * t)) for an in (a1, a2, a3))
        vals.add(sp.simplify(sp.limit(expr, t, 0, "+")))
    e23 = sp.sympify(load_answer(2, 2))
    ok &= len(vals) == 1 and sp.simplify(vals.pop() - e23) == 0
    msgs.append(f"2-3 일반 f,g 3세트 극한 일치, md={e23}")
    report(2, ok, "; ".join(msgs))


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
