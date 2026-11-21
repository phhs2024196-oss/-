#!/usr/bin/env python3
"""KGU-M25 (경기대 2025학년도 모의논술 수리 문항 1~3) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer 와 비교."""
import random
import re
import sys

import sympy as sp

PREFIX = "KGU-M25"
WIDTH = 2
RESULTS = []
A, B, a, b = sp.symbols("A B a b", positive=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub="1"):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KGU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    s = next(s for s in d["subparts"] if str(s["id"]) == str(sub))
    return sp.sympify(s["answer"]["value"], locals={"A": A, "B": B, "a": a, "b": b})


def p01():
    """격자(한 칸 50 m)에서 포물선을 세 점으로 결정 → 간척 면적을 x 적분으로 계산 → 이익 최대 h (수치 격자 탐색으로도 확인)."""
    x, h, c2 = sp.symbols("x h c2", real=True)
    c = sp.solve(sp.Eq(c2 * 100 ** 2, 100), c2)[0]          # 꼭짓점 (0,0), 끝점 (±100,100)
    curve = c * x ** 2
    assert curve.subs(x, 50) == 25                           # 중간 세로선에서 높이 반 칸
    # 시설 [-100,100]×[100-h,100], 0<h<=100: 간척 = 직사각형 중 곶 밖(y<curve) 부분 = ∫ (min(curve,100) - (100-h))^+ dx
    xb = sp.sqrt((100 - h) / c)                              # curve = 100-h 인 x
    reclaim = 2 * sp.integrate(curve - (100 - h), (x, xb, 100))
    P = 200 * A * h - B * reclaim
    crit = [s for s in sp.solve(sp.diff(P, h), h)]
    ans = load_answer(1)
    ok = False
    for s0 in crit:
        if sp.simplify(s0 - ans) == 0:
            ok = True
    # 수치: 무작위 0<A<B 에서 h∈(0,150] 격자 최대 위치와 비교(h>100 은 전부 간척)
    def profit(hv, Av, Bv):
        if hv <= 100:
            return float(P.subs({h: hv, A: Av, B: Bv}))
        return float(P.subs({h: 100, A: Av, B: Bv})) + (Av - Bv) * 200 * (hv - 100)
    for _ in range(5):
        Bv = random.uniform(1, 10); Av = random.uniform(0.05, 0.95) * Bv
        grid = [i * 0.05 for i in range(1, 3001)]
        hbest = max(grid, key=lambda hv: profit(hv, Av, Bv))
        ok &= abs(hbest - float(ans.subs({A: Av, B: Bv}))) < 0.1
    report(1, ok, f"P'(h)=0 해={crit}, 격자 최대 위치 일치, md={ans}")


def p02():
    """자르는 길이 8x, 이익 x(a-2x)^2 - 8x*5a^2/96 을 0<x<a/2 에서 최대화 (기호 + 수치)."""
    x = sp.symbols("x", positive=True)
    P = x * (a - 2 * x) ** 2 - 8 * x * sp.Rational(5, 96) * a ** 2
    cands = [s for s in sp.solve(sp.diff(P, x), x) if (s / a).is_positive and (s / a) < sp.Rational(1, 2)]
    best = max(cands + [sp.Integer(0)], key=lambda s: P.subs(x, s).subs(a, 1))
    got = sp.Tuple(sp.simplify(best), sp.simplify(P.subs(x, best)))
    ans = load_answer(2)
    ok = sp.simplify(got[0] - ans[0]) == 0 and sp.simplify(got[1] - ans[1]) == 0
    f = sp.lambdify(x, P.subs(a, 12))
    xs = [i * 6 / 60000 for i in range(1, 60000)]
    xm = max(xs, key=f)
    ok &= abs(xm - float(ans[0].subs(a, 12))) < 1e-3
    report(2, ok, f"정의역 내 임계점={cands}, (x,P)={tuple(got)}, a=12 수치 최대 x={xm:.4f}, md={ans}")


def p03():
    """연도별 전입 a*1.21^n, 전출 b*1.1^n 을 정확한 유리수로 누적해 인구 최소 해를 직접 찾고 공식과 비교(무작위 정수 a<b 300쌍 + 경계 사례)."""
    ans = load_answer(3)
    ok = True
    cases = [(random.randint(1, 5000), None) for _ in range(300)]
    cases = [(av, av + random.randint(1, 20000)) for av, _ in cases]
    cases += [(100, 110), (100, 121), (1000, 1331)]          # log_{1.1}(b/a) = 1, 2, 3 (경계: 두 해 동률)
    for av, bv in cases:
        pop, best, bestyrs = sp.Integer(0), None, []
        for nn in range(0, 400):
            pop += sp.Integer(av) * sp.Rational(121, 100) ** nn - sp.Integer(bv) * sp.Rational(11, 10) ** nn
            if best is None or pop < best:
                best, bestyrs = pop, [2020 + nn]
            elif pop == best:
                bestyrs.append(2020 + nn)
        pred = int(ans.subs({a: av, b: bv}))
        ok &= pred in bestyrs
    report(3, ok, f"{len(cases)}개 (a,b) 직접 누적 최소 해와 공식 일치(동률 사례 포함), md={ans}")


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
