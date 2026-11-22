#!/usr/bin/env python3
"""서울과학기술대 2021학년도 수시모집 논술전형(자연계열 1차) 문제 1~3 (SNUT-G21A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교."""
import sys

import sympy as sp

PREFIX = "SNUT-G21A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi, "oo": sp.oo}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """1.1 구(반지름 3)에 내접하는 원뿔: 꼭짓점 (0,0,3), 밑면 z=3-h 에서 r^2=9-(3-h)^2 → S^2 최대. 1.2 두 원 교점으로 OPQ 넓이 최대 α. 1.3 두 선수 위치 함수 비교."""
    # 1.1
    h = sp.symbols("h", positive=True)
    r2 = 9 - (3 - h) ** 2
    l2 = r2 + h ** 2
    S2 = sp.pi ** 2 * r2 * l2                      # S=pi r l
    crit = [c for c in sp.solve(sp.diff(S2, h), h) if 0 < c < 6]
    val = max(sp.simplify(S2.subs(h, c)) for c in crit)
    ends = [sp.limit(S2, h, 0), sp.limit(S2, h, 6)]
    ans1 = load_answer(1, 0)
    ok1 = sp.simplify(val - ans1) == 0 and all(e < val for e in ends)
    # 1.2
    al = sp.symbols("alpha", positive=True)
    X, Y = sp.symbols("X Y", real=True)
    def inter(R2):
        sol = sp.solve([X ** 2 - X + Y ** 2, X ** 2 + Y ** 2 - R2], [X, Y], dict=True)
        return sol
    Pv = (sp.cos(al) ** 2, sp.sin(al) * sp.cos(al))
    Qv = (sp.sin(al) ** 2, sp.sin(al) * sp.cos(al))
    # 교점이 원 조건을 만족하는지 직접 확인
    ok_pt = all(sp.simplify(c[0] ** 2 - c[0] + c[1] ** 2) == 0 for c in (Pv, Qv)) and \
        sp.simplify(Pv[0] ** 2 + Pv[1] ** 2 - sp.cos(al) ** 2) == 0 and sp.simplify(Qv[0] ** 2 + Qv[1] ** 2 - sp.sin(al) ** 2) == 0
    area = sp.Abs(Pv[0] * Qv[1] - Qv[0] * Pv[1]) / 2   # 외적 = 삼각형 넓이(원점 기준)
    import numpy as np
    xs = np.linspace(1e-6, np.pi / 2 - 1e-6, 2000001)
    f = np.abs(np.cos(xs) ** 2 * np.sin(xs) * np.cos(xs) - np.sin(xs) ** 2 * np.sin(xs) * np.cos(xs)) / 2
    mx = f.max()
    arg = sorted(set(round(float(v), 3) for v in xs[f > mx - 1e-12]))
    expect = sorted(round(float(v), 3) for v in sp.sympify(load_answer(1, 1)))
    ok2 = ok_pt and abs(mx - 1 / 8) < 1e-9 and arg == expect and sp.simplify(area.subs(al, sp.pi / 8) - sp.Rational(1, 8)) == 0
    # 1.3  P에서 결승선까지 90. A: a t^2/2, B: 20 + ∫(s+2)ds
    s = sp.symbols("s", positive=True)
    posB = 20 + sp.integrate(s + 2, (s, 0, t))
    tB = [r for r in sp.solve(sp.Eq(posB, 90), t) if r > 0][0]
    av = sp.symbols("a", positive=True)
    tA = sp.sqrt(180 / av)
    cond = sp.solve_univariate_inequality(tA < tB, av, relational=False)
    ans3 = load_answer(1, 2)
    exp3 = sp.solve_univariate_inequality(av > ans3.rhs, av, relational=False) if hasattr(ans3, "rhs") else None
    ok3 = tB == 10 and cond == sp.Interval.open(sp.Rational(9, 5), sp.oo) and ans3.lhs == a and ans3.rhs == sp.Rational(9, 5)
    report(1, ok1 and ok2 and ok3, f"[1.1] max S^2={val} (yaml {ans1}), [1.2] max넓이={mx:.9f} α={arg}, [1.3] tB={tB}, A승리 a∈{cond}")


def p02():
    """2.1 1000~1999 중 자리합 10 전수. 2.2 비복원 무작위 입력 X 의 분포·기댓값(Fraction). 2.3 5번 실패마다 20초."""
    from fractions import Fraction
    cands = [m for m in range(1000, 2000) if sum(map(int, str(m))) == 10]
    N = len(cands)
    ans1 = load_answer(2, 0)
    # X: i번째에 처음 맞힐 확률 = prod_{j<i}(1-1/(N-j)) * 1/(N-i+1)
    pX = []
    surv = Fraction(1)
    for i in range(1, N + 1):
        p = surv * Fraction(1, N - i + 1)
        pX.append(p)
        surv *= Fraction(N - i, N - i + 1)
    EX = sum(i * p for i, p in zip(range(1, N + 1), pX))
    EY = sum((5 * i + 20 * ((i - 1) // 5)) * p for i, p in zip(range(1, N + 1), pX))
    ans2, ans3 = load_answer(2, 1), load_answer(2, 2)
    ok = N == ans1 and sum(pX) == 1 and EX == ans2 and EY == ans3
    report(2, ok, f"[2.1] N={N}, [2.2] E(X)={EX}, [2.3] E(Y)={EY} (yaml {ans1},{ans2},{ans3})")


def p03():
    """3.1 접선 기울기와 tan^2θ 직접 계산, 3.2 A1(t)/t^k 극한(k=1..5), 3.3 S/A2."""
    xx = sp.symbols("xx", real=True)
    tt = sp.symbols("tt", positive=True)
    roots = sp.solve(sp.Eq(tt - xx ** 2, tt * xx), xx)
    al_, be_ = sorted(roots, key=lambda r: sp.N(r.subs(tt, 3)), reverse=True)      # al>0(P1), be<0(P2)
    m1, m2 = (sp.diff(tt - xx ** 2, xx).subs(xx, r) for r in (al_, be_))
    tan2 = sp.simplify(((m1 - m2) / (1 + m1 * m2)) ** 2)
    ans1 = load_answer(3, 0).subs(t, tt)
    ok1 = sp.simplify(tan2 - ans1) == 0 and sp.simplify(ans1 - (4 * tt ** 2 + 16 * tt) / (16 * tt ** 2 - 8 * tt + 1)) == 0
    # 접선: y = f(r) + m (x - r); 절편으로 삼각형 넓이
    def area(r, m):
        y0 = (tt - r ** 2) + m * (0 - r)
        x0 = r - (tt - r ** 2) / m
        return sp.simplify(sp.Abs(x0 * y0) / 2)
    A1 = sp.simplify(area(al_, m1))
    A2 = sp.simplify(area(be_, m2))
    S = sp.simplify(sp.integrate(tt - xx ** 2 - tt * xx, (xx, be_, al_)))
    lims = {}
    for kk in range(1, 6):
        lims[kk] = sp.limit(A1 / tt ** kk, tt, sp.oo)
    finite = {v for v in lims.values() if v.is_finite}
    ans2 = load_answer(3, 1)
    L3 = sp.limit(S / A2, tt, sp.oo)
    ans3 = load_answer(3, 2)
    ok = ok1 and finite == set(ans2) and sp.simplify(L3 - ans3) == 0
    report(3, ok, f"[3.1] tan^2={sp.simplify(tan2)}, [3.2] 극한 k=1..5 {lims} → 수렴값 {finite}, [3.3] S/A2→{L3} (yaml {ans2}, {ans3})")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G21A: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
