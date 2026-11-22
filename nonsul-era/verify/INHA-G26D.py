#!/usr/bin/env python3
"""INHA-G26D (2026학년도 인하대 논술 자연계열 의예과; [문제 1]은 INHA-G26B-03 과 동일해 입력 생략) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "INHA-G26D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    """(2-1) P 가 직선 CD·곡선 위 (2-2) A(3,8) 에서 AC, BD (2-3) 평행사변형 넓이 — C 를 교점으로 직접 수치 계산."""
    from scipy import optimize as so
    a, b = sp.symbols("a b", real=True)
    CD = -(x - a) + (b + 5)
    ok1 = sp.simplify(CD.subs(x, a + 1) - (b + 4)) == 0 and sp.simplify((2 ** (x - 1) + 4).subs(x, a + 1) - (b + 4)).subs(b, 2**a) == 0

    def pts(av):
        bv = 2.0**av
        # C: 직선 y=-(x-a)+b+5 와 y=log2(x-4)+1 의 교점(단조 → 유일), D: y=x 에 내린 수선의 발
        F = lambda u: -(u - av) + bv + 5 - (np.log2(u - 4) + 1)
        cx = so.brentq(F, 4 + 1e-12, 1e6)
        C = np.array([cx, np.log2(cx - 4) + 1])
        A = np.array([av, bv]); B = np.array([bv, av])
        m = (C[0] + C[1]) / 2; D = np.array([m, m])
        return A, B, C, D

    A, B, C, D = pts(3.0)
    AC = np.linalg.norm(C - A); BD = np.linalg.norm(D - B)
    ok2 = abs(AC - np.sqrt(97)) < 1e-9 and abs(BD - 5) < 1e-9
    # (2-3) 평행사변형 ⇔ AC, BD 중점 일치 → a 에 대한 방정식 수치 풀이(근 모두)
    def mid_gap(av):
        A, B, C, D = pts(av)
        return ((A + C) / 2 - (B + D) / 2)[0]
    grid = np.linspace(-10, 6, 1601)
    vals = [mid_gap(v) for v in grid]
    roots = [so.brentq(mid_gap, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]
    areas = []
    for r in roots:
        A, B, C, D = pts(r)
        ys = [A, B, C, D]
        areas.append(0.5 * abs(sum(ys[i][0] * ys[(i + 1) % 4][1] - ys[(i + 1) % 4][0] * ys[i][1] for i in range(4))))
    ok3 = len(roots) == 2 and all(abs(ar - 15) < 1e-8 for ar in areas)
    report(2, ok1 and ok2 and ok3, f"(2-2) AC={AC:.6f}(√97), BD={BD:.6f} (2-3) a={['%.5f'%r for r in roots]}, 넓이={['%.6f'%v for v in areas]}")


def p03():
    """(3-1) 조건으로 p 결정 후 g(t) 수치 재현 (3-2) 계수 연립 + h, q∘h 미분가능 수치 확인."""
    A_, C_ = sp.symbols("A C", real=True)
    pp = sp.integrate(4 * x * (x - A_) * (x - 3), x) + C_
    asol = sp.solve(sp.Eq(pp.subs(x, 0), pp.subs(x, 2)), A_)
    asol = [v for v in asol if 0 < v < 3]
    p = pp.subs(A_, asol[0]).subs(C_, 0)
    val1 = sp.nsimplify(p.subs(x, 1) - p.subs(x, 0))
    pf = sp.lambdify(x, p, "numpy")
    def g_num(tv):
        xs = np.linspace(min(-5, tv - 5), tv, 200001)
        ys = pf(xs); mn = ys.min()
        return xs[np.where(ys <= mn + 1e-9)[0].max()]
    tests = {-1.0: -1.0, 0.5: 0.0, 1.9: 0.0, 2.5: 2.5, 4.0: 3.0}
    okg = all(abs(g_num(k) - v) < 1e-3 for k, v in tests.items())
    ok1 = len(asol) == 1 and val1 == sp.Rational(13, 5) and okg
    # (3-2) q = c4 x^4 + c3 x^3 + c2 x^2 + 2 (q'(0)=0), 임계점 al<0<be 에서 q=4, q(1)=23/8
    # q'(0)=0 이므로 q'=c x (x-al)(x-be) (al<0<be 는 [0,∞)·(-∞,0] 의 최대점), q(0)=2
    cc, al, be = sp.symbols("c al be", real=True)
    q = cc * sp.integrate(x * (x - al) * (x - be), (x, 0, x)) + 2
    # 주의: 문항카드 예시답안은 -(c/12)(α-β)(α+β)^3 로 적었으나 실제 인수분해는 -(c/12)(α-β)^3(α+β) (결론 α=-β 는 같음)
    diffq = sp.factor(q.subs(x, al) - q.subs(x, be))
    sym_ok = sp.simplify(diffq + cc / 12 * (al - be) ** 3 * (al + be)) == 0
    q2 = sp.expand(q.subs(al, -be))
    sols = sp.solve([q2.subs(x, be) - 4, q2.subs(x, 1) - sp.Rational(23, 8)], [cc, be], dict=True)
    cands = set()
    for so_ in sols:
        if so_[be].is_real and so_[be] > 0 and so_[cc] < 0:
            cands.add(sp.expand(q2.subs(so_)))
    # 수치: h(t) 를 정의대로 계산해 q∘h 의 좌·우 차분 미분계수 비교
    def qh_ok(qe):
        qf = sp.lambdify(x, qe, "numpy")
        def h(tv):
            if tv == 0: return 0.0
            xs = np.linspace(0, tv, 200001) if tv > 0 else np.linspace(tv, 0, 200001)
            fv = -np.cos(np.pi * qf(xs) / 4); mx = fv.max()
            idx = np.where(fv >= mx - 1e-10)[0]
            return xs[idx.min()] if tv > 0 else xs[idx.max()]
        comp = lambda tv: qf(h(tv))
        worst = 0
        for t0 in np.linspace(-3, 3, 25):
            e = 1e-3
            dl = (comp(t0) - comp(t0 - e)) / e; dr = (comp(t0 + e) - comp(t0)) / e
            worst = max(worst, abs(dl - dr))
        return worst
    w_good = [qh_ok(c) for c in cands]
    w_bad = qh_ok(-x**4 / 8 + 7 * x**3 / 8 + 2)          # 대조: q(0)=2,q(1)=23/8 이지만 조건 불충족
    expect = {sp.expand(-x**4 / 8 + x**2 + 2), sp.expand(-sp.Rational(49, 8) * x**4 + 7 * x**2 + 2)}
    ok2 = sym_ok and cands == expect and max(w_good) < 0.05 and w_bad > 0.5
    report(3, ok1 and ok2, f"(3-1) a={asol}, p(1)-p(0)={val1} (3-2) {cands} 차분오차={['%.3g'%w for w in w_good]} 대조={w_bad:.3g}")


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
