#!/usr/bin/env python3
"""KHU-M27 (경희대 2027학년도 모의논술 자연계, KHU-M27-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import re
import sys
from fractions import Fraction as Fr

import sympy as sp

PREFIX = "KHU-M27"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    c, N = sp.symbols("c N", positive=True)
    # (1) h 미분가능 조건 f'(0)=g'(0) 으로 c 결정
    f = x * (x + N) * (x - c)
    g = x * (x - N)
    csol = sp.solve(sp.Eq(sp.diff(f, x).subs(x, 0), sp.diff(g, x).subs(x, 0)), c)
    ok = csol == [1]
    f = f.subs(c, 1)
    k = sp.symbols("k", integer=True, positive=True)
    an = sp.simplify(sp.summation(f.subs(x, -k) - 1, (k, 1, N - 1)))
    bn = sp.simplify(sp.summation(-g.subs(x, k) - 1, (k, 1, N - 1)))
    # 격자점 전수 세기와 닫힌 식 비교
    for nn in range(2, 25):
        fn = sp.lambdify(x, f.subs(N, nn)); gn = sp.lambdify(x, g.subs(N, nn))
        A2 = sum(1 for X in range(-nn + 1, 0) for Y in range(1, int(fn(X)) + 2) if 0 < Y < fn(X))
        B2 = sum(1 for X in range(1, nn) for Y in range(1, int(-gn(X)) + 2) if 0 < Y < -gn(X))
        ok &= A2 == an.subs(N, nn) and B2 == bn.subs(N, nn)
    lim = sp.limit(an / (N * bn), N, sp.oo)
    ok &= lim == sp.Rational(1, 2)
    # (2) 기울기 a 에 따른 교점 개수
    al = sp.symbols("al", positive=True)
    tl = (n - 2 * al) * (x - al) + (-al**2 + n * al)   # -g 의 접선 (n 기호)
    asol = [s for s in sp.solve(tl.subs(x, -n).subs(n, N).subs(al, al), al)]
    a1 = sp.simplify((N - 2 * asol[0]))
    a2 = sp.diff(f, x).subs(x, -N)
    ok &= sp.simplify(a1 - (3 - 2 * sp.sqrt(2)) * N) == 0 and sp.expand(a2 - N**2 - N) == 0

    def count(nn, a):
        a = sp.nsimplify(a)
        L = a * (x + nn)
        pts = set()
        fn = f.subs(N, nn); gn = g.subs(N, nn)
        pieces = [(-fn, None, -nn), (fn, -nn, 0), (-gn, 0, nn), (gn, nn, None)]  # |h| 조각
        for expr, lo, hi in pieces:
            for r in sp.solve(sp.expand(expr - L), x):
                if abs(sp.im(sp.N(r, 40))) > 1e-25:
                    continue
                rv = sp.re(sp.N(r, 40))
                if (lo is None or rv >= lo - 1e-20) and (hi is None or rv <= hi + 1e-20) and \
                        expr.subs(x, rv) >= -1e-20:
                    pts.add(rv)
        return len({round(float(p), 9) for p in pts})

    table = {}
    for nn in (2, 3, 5):
        A1 = float(a1.subs(N, nn)); A2 = float(a2.subs(N, nn))
        res = (count(nn, sp.Rational(int(A1 * 500), 1000)), count(nn, a1.subs(N, nn)),
               count(nn, sp.Rational(int((A1 + A2) / 2 * 100), 100)), count(nn, A2), count(nn, A2 + 3))
        table[nn] = res
        ok &= res == (5, 4, 3, 2, 2)
    report(1, ok, f"c=1, a_n/(n b_n)->{lim}, 격자점 n=2..24 일치, a1={a1}, a2={sp.expand(a2)}, 개수표 {table[3]}")


def p02():
    def bonus(path):
        s, rec = Fr(16), []
        for u in path:
            s = s * 2 if u else s / 2
            rec.append(s)
        m = sum(rec) / 3
        return max(Fr(0), rec[2] - m)
    dist = {}
    for p in itertools.product([1, 0], repeat=3):
        b = bonus(p); dist[b] = dist.get(b, 0) + Fr(1, 8)
    E1 = sum(k * v for k, v in dist.items())
    E2 = sum(max(bonus(p), bonus(q)) * Fr(1, 64)
             for p in itertools.product([1, 0], repeat=3) for q in itertools.product([1, 0], repeat=3))
    ok = E1 == Fr(55, 6) and E2 == Fr(261, 16)
    # 대표 오답: 음수 차이를 그대로 평균
    wrong = sum((lambda p: (lambda r: r[2] - sum(r) / 3)([16 * Fr(2) ** sum(1 if u else -1 for u in p[:i + 1]) for i in range(3)]))(p) * Fr(1, 8)
                for p in itertools.product([1, 0], repeat=3))
    report(2, ok, f"(1) E={E1}, (2) E[max]={E2}, 오답(음수 포함) {wrong}")


def p03():
    s2 = sp.sqrt(2)
    P1 = sp.Matrix([0, 0, 0]); P2 = sp.Matrix([2 * s2, 0, 0])
    u, v = sp.symbols("u v", real=True)
    sol = [s for s in sp.solve([u**2 + v**2 - (4**2 - 2**2), (u - 2 * s2)**2 + v**2 - (5**2 - 1)], [u, v]) if s[1] > 0][0]
    P3 = sp.Matrix([sol[0], sol[1], 0])
    O1 = P1 + sp.Matrix([0, 0, 1]); O2 = P2 + sp.Matrix([0, 0, 2]); O3 = P3 + sp.Matrix([0, 0, 3])
    ok = sp.simplify((O1 - O2).norm() - 3) == 0 and sp.simplify((O1 - O3).norm() - 4) == 0 and sp.simplify((O2 - O3).norm() - 5) == 0
    area = lambda A, B, C: sp.sqrt(sp.simplify(((B - A).cross(C - A)).dot((B - A).cross(C - A)))) / 2
    nrm = (O2 - O1).cross(O3 - O1)
    cos_t = sp.simplify(abs(nrm[2]) / nrm.norm())
    ok &= sp.simplify(cos_t - sp.sqrt(23) / 6) == 0
    def hit(Pa, Pb):
        tt = sp.solve(Pa[2] + t * (Pb[2] - Pa[2]), t)[0]
        return Pa + tt * (Pb - Pa)
    A = hit(O1, O2); B = hit(O2, O3)
    S1 = sp.simplify(area(O3, A, B)); S2 = sp.simplify(area(P3, A, O2))
    ratio = sp.nsimplify(sp.simplify(S2 / S1))
    ok &= sp.simplify(S1 - 36) == 0 and sp.simplify(ratio - sp.sqrt(3) / 6) == 0
    report(3, ok, f"cosθ={cos_t}, S1={S1}, S2={S2}, S2/S1={ratio}")


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
