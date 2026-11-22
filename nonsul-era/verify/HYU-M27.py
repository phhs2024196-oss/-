#!/usr/bin/env python3
"""HYU-M27 (한양대 2027학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "HYU-M27"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def bisect(F, a_, b_, it=400):
    """부호가 바뀌는 구간 [a_, b_] 에서 이분법(mpmath 수)."""
    fa = F(a_)
    for _ in range(it):
        mid = (a_ + b_) / 2
        fm = F(mid)
        if fa * fm <= 0:
            b_ = mid
        else:
            a_, fa = mid, fm
    return (a_ + b_) / 2


def p01():
    import mpmath as mp
    mp.mp.dps = 50
    # (1) h(a) = m(g(c)-f(a)) + (c-a), m=f'(a), c=(m+8)/3 의 근을 정의역 안에서 전부 찾기
    X = sp.symbols("X", real=True)
    f = sp.sqrt(-X**2 - 4 * X + 1)
    fp = sp.diff(f, X)
    g = sp.Rational(3, 2) * X**2 - 8 * X + 12
    m = fp; c = (m + 8) / 3
    h = sp.lambdify(X, m * (g.subs(X, c) - f) + (c - X), "mpmath")
    lo, hi = -2 - mp.sqrt(5), -2 + mp.sqrt(5)
    grid = [lo + (hi - lo) * (i + 0.5) / 4000 for i in range(4000)]
    roots = []
    for u, v in zip(grid, grid[1:]):
        if h(u) * h(v) < 0:
            a_, b_ = u, v
            for _ in range(200):
                mid = (a_ + b_) / 2
                if h(a_) * h(mid) <= 0:
                    b_ = mid
                else:
                    a_ = mid
            roots.append((a_ + b_) / 2)
    sols = [((r, f.subs(X, r)), (c.subs(X, r), g.subs(X, c.subs(X, r)))) for r in roots]
    md = sp.sympify(md_answer(1, "1"))
    ok1 = len(sols) == 1 and all(abs(sp.N(sols[0][i][j]) - md[i][j]) < 1e-12 for i in range(2) for j in range(2))
    # (2) 좌석 7개, 5명(서준 0, 도윤 1, 민서 2, 기타 3, 4) 전수
    n2 = sum(1 for p in itertools.permutations(range(7), 5) if abs(p[0] - p[1]) == 1 and abs(p[1] - p[2]) != 1)
    ok2 = n2 == int(sp.sympify(md_answer(1, "2")))
    # (3) ① 식이 정적분 직접 계산과 같은지(여러 a) + g→∞ 극한(ln a = ln g / g, 1<f<e)
    ok3a = True
    for av in [mp.mpf("1.2"), mp.mpf("1.3"), mp.mpf("1.4")]:
        F = lambda x: av**x - x
        fa = bisect(F, mp.mpf("0.5"), mp.e)
        ga = bisect(F, mp.e, mp.mpf(10**4))
        R = 2 * mp.quad(lambda x: x - av**x, [fa, ga])
        S = 2 * (ga - fa) ** 2
        formula = (1 + fa / ga - 2 / (ga * mp.log(av))) / (2 * (1 - fa / ga))
        ok3a = ok3a and abs(R / S - formula) < 1e-30
    G, Fv = sp.symbols("G F", positive=True)
    expr = (1 + Fv / G - 2 / sp.log(G)) / (2 * (1 - Fv / G))     # g ln a = ln g
    L = [sp.limit(expr.subs(Fv, b), G, sp.oo) for b in (1, sp.E)]
    ok3 = ok3a and all(sp.simplify(l - sp.sympify(md_answer(1, "3"))) == 0 for l in L)
    report(1, ok1 and ok2 and ok3, f"(1) 근 {len(sols)}개 P=({mp.nstr(sols[0][0][0], 8)},{sp.N(sols[0][0][1], 8)}) Q=({sp.N(sols[0][1][0], 8)},{sp.N(sols[0][1][1], 8)})  (2) {n2}  (3) lim={L[0]}")


def p02():
    t, s, x = sp.symbols("t s x", positive=True)
    # (1) 접선·x절편·교점으로 넓이 F(t,s) 계산 → s(1/2), ds/dt
    def tan(u):
        return sp.diff(x + 1 / x, x).subs(x, u) * (x - u) + u + 1 / u
    X = sp.symbols("X", real=True)
    xi1 = sp.solve(tan(t).subs(x, X), X)[0]
    xi2 = sp.solve(tan(s).subs(x, X), X)[0]          # 0<t<1 이면 xi1>0, s>1 이면 xi2<0
    xm = sp.solve(sp.Eq(tan(t).subs(x, X), tan(s).subs(x, X)), X)[0]
    ym = tan(t).subs(x, xm)
    area = sp.simplify(sp.Rational(1, 2) * (xi1 - xi2) * ym)
    Fts = sp.together(area - sp.Rational(32, 15))
    num = sp.numer(Fts)
    sv = [r for r in sp.solve(num.subs(t, sp.Rational(1, 2)), s) if r.is_real and r > 1]
    assert sp.simplify(area.subs({t: sp.Rational(1, 3), s: 3}) - sp.Rational(1, 2) * abs(xi1 - xi2).subs({t: sp.Rational(1, 3), s: 3}) * ym.subs({t: sp.Rational(1, 3), s: 3})) == 0
    s0 = sv[0]
    dsdt = -sp.diff(num, t) / sp.diff(num, s)
    val = sp.simplify(dsdt.subs({t: sp.Rational(1, 2), s: s0}))
    ok1 = len(sv) == 1 and sp.simplify(val - sp.sympify(md_answer(2, "1"))) == 0
    # (2) 접선과 준선 교점 → 다음 점, 등비중항 조건 풀기
    p, x1 = sp.symbols("p x1", positive=True)
    def nxt(xk):
        yk = 2 * sp.sqrt(p * xk)
        yH = yk + (2 * p / yk) * (-p - xk)
        return sp.simplify(yH**2 / (4 * p))
    x2 = nxt(x1); x3 = nxt(x2)
    cand = sp.solve(sp.numer(sp.together((x2 + p) ** 2 - (x1 + p) * (x3 + p))), x1)
    cand = [r for r in cand if sp.simplify(r - p) != 0 and sp.simplify(nxt(r)) != 0]
    ok2 = len(cand) == 1 and sp.simplify(cand[0] - sp.sympify(md_answer(2, "2"), locals={"p": p})) == 0
    # (3) R=(r,0), 중심 O=(r,y), 반지름 y 인 원 위의 P,Q 가 x^2+y^2=4 위 → y(r) 수치로 구해 넓이 적분
    import mpmath as mp
    mp.mp.dps = 30
    def yof(r):
        # 원 (X-r)^2+(Y-y)^2=y^2 와 X^2+Y^2=4 의 공통현이 O 를 지나야(지름) → 수치 탐색
        def cond(y):
            # 공통현 직선: 두 원 식의 차 → 2rX + 2yY = 4 + r^2  (O 가 이 직선 위)
            return 2 * r * r + 2 * y * y - (4 + r * r)
        return bisect(cond, mp.mpf(0), mp.mpf(2))
    A = mp.quad(lambda r: yof(r), [-2, 2])
    ok3 = abs(A - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "3")), 30)))) < 1e-20
    report(2, ok1 and ok2 and ok3, f"(1) s={s0}, ds/dt={val}  (2) x1={cand}  (3) 넓이={mp.nstr(A, 15)}")


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
