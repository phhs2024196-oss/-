#!/usr/bin/env python3
"""서울과학기술대 2024학년도 자연계열 오후 수리논술 (SNUT-G24B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import sys

import sympy as sp

PREFIX = "SNUT-G24B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi}


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


def eq(u, v):
    return sp.simplify(sp.nsimplify(u) - sp.nsimplify(v)) == 0


def p01():
    """[1.1] 공통접선 전수(수치·기호), [1.2] 역함수 교점 전수, [1.3] 최솟값 격자."""
    import numpy as np
    f = x**2 / 4 + x + 3
    g = -f.subs(x, x - 2)                              # x축 대칭 후 x방향 +2 평행이동
    A_, B_ = sp.symbols("A_ B_", real=True)
    disc = lambda h: sp.discriminant(sp.expand(h - (A_ * x + B_)), x)
    sols = sp.solve([disc(f), disc(g)], [A_, B_], dict=True)
    lines = sorted((s_[A_], s_[B_]) for s_ in sols)
    ans1 = load_answer(1, 0, raw=True).replace(" ", "")
    exp = {(-1, -1), (2, 2)}
    ok1 = set(lines) == exp and "y=-x-1" in ans1 and "y=2*x+2" in ans1 and sp.simplify(g + sp.Rational(1, 4) * x**2 + 2) == 0
    # 수치 확인: 직선과 곡선의 최소 거리(수직 방향 차이의 최솟값) = 0, 다른 직선은 접하지 않음
    xs = np.linspace(-30, 30, 600001)
    tang = lambda m, c, h: np.min(np.abs((sp.lambdify(x, h)(xs)) - (m * xs + c)))
    ok1 = ok1 and all(tang(m_, c_, f) < 1e-6 and tang(m_, c_, g) < 1e-6 for m_, c_ in exp)
    # [1.2] a^x + b 와 역함수가 x=1, 2 에서 만난다: (i) y=x 위의 교점, (ii) 대칭쌍 f(1)=2,f(2)=1 도 검토
    a_, b_ = sp.symbols("a_ b_", real=True)
    s_on = sp.solve([a_ + b_ - 1, a_**2 + b_ - 2], [a_, b_], dict=True)
    s_on = [s_ for s_ in s_on if s_[a_] > 0]
    s_pair = [s_ for s_ in sp.solve([a_ + b_ - 2, a_**2 + b_ - 1], [a_, b_], dict=True) if s_[a_].is_real and s_[a_] > 0]
    ans2 = load_answer(1, 1)
    val = sp.simplify(s_on[0][a_] - s_on[0][b_])
    # 수치: 해당 (a,b) 에서 f 와 역함수의 교점(= f(x)=x 해, a>1 이므로 증가)
    fa = sp.lambdify(x, s_on[0][a_] ** x + s_on[0][b_])
    xs2 = np.linspace(0.2, 3, 2800001) + 1.234e-7   # 격자가 정확히 근에 걸리지 않게 이동
    hh = fa(xs2) - xs2
    rts = xs2[:-1][hh[:-1] * hh[1:] < 0]
    ok2 = len(s_on) == 1 and len(s_pair) == 0 and eq(val, ans2) and len(rts) == 2 and abs(rts[0] - 1) < 1e-5 and abs(rts[1] - 2) < 1e-5
    # [1.3] min over [-π/4, π/4]
    th = np.linspace(-np.pi / 4, np.pi / 4, 2000001)
    fv = (2 + np.tan(th) ** 2) / (2 - np.tan(th))
    ans3 = load_answer(1, 2)
    ok3 = abs(fv.min() - float(ans3)) < 1e-9
    report(1, ok1 and ok2 and ok3, f"[1.1] 공통접선 (기울기,절편)={lines} / [1.2] a-b={val}, 교점 x≈{[round(float(r),4) for r in rts]} / [1.3] min≈{fv.min():.9f} vs {float(ans3):.9f}")


def p02():
    """[2.1] 격자 사각형 전수 열거, [2.2] 넓이 조건, [2.3] 귀납 명제 수치 확인(정확한 유리수). [증명-수치확인]"""
    from fractions import Fraction
    from itertools import combinations
    def rects(n):
        out = []
        for x1, x2 in combinations(range(n + 1), 2):
            for y1, y2 in combinations(range(n + 1), 2):
                out.append(Fraction((x2 - x1) * (y2 - y1), n * n))
        return out
    r2 = rects(2)
    ok0 = len(r2) == 9
    r9 = rects(9)
    ans1 = load_answer(2, 0)
    ok1 = ok0 and len(r9) == ans1
    a8 = Fraction(1, 64)
    cnt = sum(1 for r in r9 if r > a8)
    ans2 = load_answer(2, 1)
    ok2 = cnt == ans2 and min(r9) == Fraction(1, 81)
    # [2.3] 모든 2 ≤ m ≤ 300 에서 두 번째로 큰 넓이 > S_m (엄밀한 Fraction), a_k = 1/k^2 확인
    ok3 = True
    for m in range(2, 301):
        rs = sorted(set(rects(m)) if m <= 40 else {Fraction(1), Fraction(m * (m - 1), m * m)}, reverse=True)
        second = rs[1]
        am = min(rects(m)) if m <= 40 else Fraction(1, m * m)
        S = sum(Fraction(1, k * k) for k in range(2, m + 1))
        ok3 = ok3 and second == 1 - Fraction(1, m) and am == Fraction(1, m * m) and second > S
    report(2, ok1 and ok2 and ok3, f"[2.1] n=2 → {len(r2)}개, n=9 → {len(r9)}개 / [2.2] 넓이>1/64: {cnt}개 / [증명-수치확인] [2.3] 2≤m≤300 에서 1-1/m > Σ1/k² (m≤40 은 사각형 전수 열거)")


def p03():
    """[3.1] 미분 확인, [3.2] 넓이 수치적분, [3.3] 리만합 극한."""
    import mpmath as mp
    mp.mp.dps = 30
    F = sp.exp(x) * (sp.sin(x) - sp.cos(x))
    ans1 = sp.sympify(load_answer(3, 0, raw=True).replace("+C", ""), locals=SYM)
    ok1 = sp.simplify(sp.diff(ans1, x) - 2 * sp.exp(x) * sp.sin(x)) == 0
    # [3.2] A_n 수치적분 (구간별 |.|)
    def A_num(nv):
        h = lambda xv: abs(2 * mp.e**xv * mp.sin(xv))
        return mp.quad(h, [2 * nv * mp.pi, (2 * nv + 1) * mp.pi, (2 * nv + 2) * mp.pi])
    ans2 = sp.sympify(load_answer(3, 1, raw=True), locals={**SYM, "n": n})
    ok2 = all(abs(A_num(nv) - sp.N(ans2.subs(n, nv), 30)) < mp.mpf("1e-15") * A_num(nv) for nv in (1, 2, 3, 5))
    # [3.3] 리만합 수열을 큰 n 에서 계산 → 극한 외삽
    def term_sum(N):
        s = mp.mpf(0)
        for k in range(1, N + 1):
            Ak = mp.e ** (2 * k * mp.pi) * (1 + mp.e**mp.pi) ** 2
            s += (mp.log(Ak) - 2 * mp.log(1 + mp.e**mp.pi)) ** 7 / mp.mpf(N) ** 8
        return s
    ans3 = load_answer(3, 2)
    vals = [term_sum(N) for N in (200, 400, 800)]
    # 리처드슨 외삽: S(N) = L + c1/N + c2/N² → 3점으로 L 추정
    s1_, s2_, s3_ = vals
    r1 = 2 * s2_ - s1_
    r2 = 2 * s3_ - s2_
    L = (4 * r2 - r1) / 3
    ok3 = abs(L - sp.N(ans3, 30)) < mp.mpf("1e-6") * sp.N(ans3, 30)
    report(3, ok1 and ok2 and ok3, f"[3.1] 도함수 일치 / [3.2] A_n 수치적분 일치 / [3.3] 리만합 외삽 L≈{mp.nstr(L, 10)} vs 16π⁷={sp.N(ans3, 10)}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G24B: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
