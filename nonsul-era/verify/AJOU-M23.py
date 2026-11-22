#!/usr/bin/env python3
"""AJOU-M23 (아주대 2023학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-M23"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import random
    a = sp.symbols("a", real=True)
    # [1-1](1): 구간 [a-2,a+2] 에서 최솟값 = 20 — 세 경우를 모두 풀어 실제 최솟값이 20 인 a 만 남김
    y = 2 * x**2 - 4 * x + a
    cands = set()
    for xs in (1, a - 2, a + 2):
        for r in sp.solve(sp.Eq(y.subs(x, xs), 20), a):
            if r.is_real:
                cands.add(sp.nsimplify(r))
    def minval(av):
        lo, hi = av - 2, av + 2
        pts = [lo, hi] + ([1] if lo <= 1 <= hi else [])
        return min(sp.simplify(y.subs(a, av).subs(x, p_)) for p_ in pts)
    good = {r for r in cands if sp.simplify(minval(r) - 20) == 0}
    md = {sp.nsimplify(sp.sympify(v)) for v in md_answer(1, "1-1").split(",")}
    ok1 = {sp.simplify(g) for g in good} == {sp.simplify(m) for m in md} or \
        all(any(sp.simplify(g - m) == 0 for m in md) for g in good) and len(good) == len(md)
    # [1-1](2): 임계점·끝값 비교
    f = x + sp.sin(4 * x) - 1
    crit = [c for c in sp.solve(sp.diff(f, x), x) if 0 <= float(c) <= float(sp.pi / 2)]
    k = sp.symbols("k", integer=True)
    crit = [sp.acos(-sp.Rational(1, 4)) / 4, (2 * sp.pi - sp.acos(-sp.Rational(1, 4))) / 4]
    vals = [f.subs(x, v) for v in crit + [0, sp.pi / 2]]
    mn = min(vals, key=lambda v: float(v))
    ok2 = sp.simplify(mn - sp.sympify(md_answer(1, "1-2"))) == 0
    # [1-2](1): 접점 t, 접선이 (-1,-2) 지남 → 두 접선과 곡선 사이 넓이
    g = 2 * x**2 - 3 * x - 5
    ts = sorted(sp.solve(sp.Eq(-2 - g.subs(x, t), sp.diff(g, x).subs(x, t) * (-1 - t)), t))
    l1 = g.subs(x, ts[0]) + sp.diff(g, x).subs(x, ts[0]) * (x - ts[0])
    l2 = g.subs(x, ts[1]) + sp.diff(g, x).subs(x, ts[1]) * (x - ts[1])
    area = sp.integrate(g - l1, (x, ts[0], -1)) + sp.integrate(g - l2, (x, -1, ts[1]))
    ok3 = sp.simplify(area - sp.sympify(md_answer(1, "2-1"))) == 0
    # [1-2](2): 무작위 삼차함수에서 두 접선이 일치하는 p≠q 없음 [증명-수치확인]
    p, q = sp.symbols("p q", real=True)
    A, B, C, D = sp.symbols("A B C D", real=True)
    F = A * x**3 + B * x**2 + C * x + D
    dF = sp.diff(F, x)
    e1 = sp.expand((dF.subs(x, p) - dF.subs(x, q)) / (p - q))   # 기울기 같음(약분)
    e2 = sp.expand(sp.cancel((F.subs(x, q) - F.subs(x, p) - dF.subs(x, p) * (q - p)) / (q - p)**2))  # 절편 같음
    random.seed(7)
    ok4 = True
    for _ in range(30):
        cf = {A: random.choice([-3, -1, 1, 2, 5]), B: random.randint(-5, 5), C: random.randint(-5, 5), D: random.randint(-5, 5)}
        sols = sp.solve([e1.subs(cf), e2.subs(cf)], [p, q], dict=True)
        ok4 = ok4 and all(sp.simplify(s_[p] - s_[q]) == 0 or not (s_[p].is_real and s_[q].is_real) for s_ in sols)
    # [1-2](3): 자연수 x0,y0 에서 서로 다른 세 접선(접점 3개) 조건 전수 → x0+y0 최소
    h = x**3 - 3 * x**2 - 2
    best = None
    for X0 in range(1, 30):
        for Y0 in range(1, 30):
            eq = sp.Poly(sp.expand(Y0 - h.subs(x, t) - sp.diff(h, x).subs(x, t) * (X0 - t)), t)
            if len(set(eq.real_roots())) == 3:
                best = X0 + Y0 if best is None else min(best, X0 + Y0)
    ok5 = best == int(md_answer(1, "2-3"))
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] a={good}; 최솟값={mn}; 넓이={area}; 삼차 접선 일치 반례 없음={ok4}; 최소 x0+y0={best}")


def _g_count(k, S, nmax):
    import itertools
    return [sum(1 for fv in itertools.product(range(1, nn + 1), repeat=k)
                if all(fv[a_ - 1] != fv[b_ - 1] for a_, b_ in S)) for nn in range(0, nmax + 1)]


def p02():
    import itertools
    nn = sp.symbols("nn")
    # [2-1](1): k=3, S=∅ → 함수 개수 직접 세어 다항식 보간 후 합
    vals = _g_count(3, [], 4)
    g = sp.expand(sp.interpolate(list(zip(range(0, 5), vals)), nn))
    tot = sum(g.subs(nn, v) for v in range(1, 25))
    ok1 = tot == int(md_answer(2, "1-1"))
    # [2-1](2): md 의 S 로 g(n) 을 직접 세어 n(n-1)(n-2)^2 확인 + 원소 5개인 모든 S 도 같은지
    pairs = [tuple(map(int, m)) for m in re.findall(r"\((\d+)\s*,\s*(\d+)\)", md_answer(2, "1-2"))]
    target = nn * (nn - 1) * (nn - 2)**2
    def poly_of(S):
        v = _g_count(4, S, 5)
        return sp.expand(sp.interpolate(list(zip(range(0, 6), v)), nn))
    allp = [(a_, b_) for a_ in range(1, 5) for b_ in range(a_ + 1, 5)]
    ok2 = sp.expand(poly_of(pairs) - target) == 0 and all(sp.expand(poly_of(list(S)) - target) == 0
                                                         for S in itertools.combinations(allp, 5))
    # [2-1](3): f1, f2 전수
    S1 = [(1, 2), (2, 3), (3, 4)]; S2 = [(1, 2), (2, 3), (3, 4), (1, 4)]
    F1 = [f for f in itertools.product(range(1, 5), repeat=4) if all(f[a_ - 1] != f[b_ - 1] for a_, b_ in S1)]
    F2 = [f for f in itertools.product(range(1, 5), repeat=4) if all(f[a_ - 1] != f[b_ - 1] for a_, b_ in S2)]
    cnt = sum(1 for f1 in F1 for f2 in F2 if all(f1[f2[i] - 1] == 1 for i in range(4)))
    ok3 = cnt == int(md_answer(2, "1-3"))
    # [2-2](1): 등차수열 c_m 에 대해 c_m^2 - c_{m-1}c_{m+1} 정리 [증명-수치확인]
    a, d, m = sp.symbols("a d m")
    c = lambda j: a + (j - 1) * d
    diff = sp.expand(c(m)**2 - c(m - 1) * c(m + 1))
    ok4 = diff == d**2
    # [2-2](2): g(0)=g(1)=0, g(2)=24, g(3)=120, 최고차계수 1 인 4차 → 계수 절댓값이 (*) 위반
    b3, b2, b1, b0 = sp.symbols("b3 b2 b1 b0")
    G = nn**4 + b3 * nn**3 + b2 * nn**2 + b1 * nn + b0
    so = sp.solve([G.subs(nn, 0), G.subs(nn, 1), G.subs(nn, 2) - 24, G.subs(nn, 3) - 120], [b3, b2, b1, b0], dict=True)[0]
    G = sp.expand(G.subs(so))
    cs = [abs(G.coeff(nn, j)) for j in range(5)]
    viol = [j for j in range(1, 4) if cs[j - 1] * cs[j + 1] > cs[j]**2]
    mdG = sp.sympify(md_answer(2, "2-2").split(";")[0].split("=")[1].replace("n", "nn"), locals={"nn": nn})
    ok5 = sp.expand(G - mdG) == 0 and viol != []
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] 합={tot}; S 다항식 일치={ok2}; (f1,f2)={cnt}; 차={diff}; g={G} 위반 m={viol}")


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
