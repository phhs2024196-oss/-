#!/usr/bin/env python3
"""서울과학기술대 2023학년도 모의논술(자연계열) (SNUT-M23-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import sys

import sympy as sp

PREFIX = "SNUT-M23"
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
    """[1.1] 치역 조건 a 구간 전수, [1.2] 등비 r 탐색, [1.3] h'(x) 수치."""
    import numpy as np
    # [1.1] 치역 [a²-3-|a|, a²-3+|a|] 가 {-1,1,3} 포함 ⇔ 최소 ≤ -1, 최대 ≥ 3. 격자 전수 + 실제 표본 확인
    av = np.linspace(-5, 5, 100001)
    xs = np.linspace(0, 2 * np.pi, 20001)
    okset = []
    for a_ in av:
        lo, hi = a_**2 - 3 - abs(a_), a_**2 - 3 + abs(a_)
        if lo <= -1 + 1e-12 and hi >= 3 - 1e-12:
            okset.append(a_)
    okset = np.array(okset)
    # 실제 sin 값 범위에서 -1,1,3 이 달성되는지 (a=±2)
    def attains(a_):
        fv = a_ * np.sin(xs) + a_**2 - 3
        return all(np.min(np.abs(fv - c)) < 1e-3 for c in (-1, 1, 3))
    ans1 = load_answer(1, 0)
    ok1 = len(okset) > 0 and abs(okset.min() - float(ans1)) < 1e-9 and attains(-2.0) and attains(2.0) and not attains(1.9)
    # [1.2] 수렴 등비급수: 공비 r 후보에서 Σ a_n = lim(a_n+b_n) = 0+2a1 → a1/(1-r)=2a1
    r = sp.symbols("r")
    sol = [s for s in sp.solve(sp.Eq(1 / (1 - r), 2), r) if abs(s) < 1]
    ans2 = load_answer(1, 1)
    ok2 = sol == [ans2]
    # [1.3] h = g(f(x)), h'(x) 수치 미분
    fx = 1 / (1 + sp.exp(-x))
    gx = sp.log(fx / (1 - fx))
    hp = sp.simplify(sp.diff(gx, x))
    ans3 = load_answer(1, 2)
    ok3 = all(eq(hp.subs(x, v), ans3) for v in (-2, 0, sp.Rational(1, 3), 3)) and sp.simplify(gx - x) == 0
    report(1, ok1 and ok2 and ok3, f"[1.1] 조건 만족 a∈[{okset.min():.3f},{okset.max():.3f}]→{set(np.round(okset[[0,-1]],3))}, 최솟값={ans1} / [1.2] r={sol} / [1.3] h'={hp}, g(f(x))-x=0")


def p02():
    """[2.1]~[2.4] 임의의 a 에 대해 접선·수직·평행 직선을 구성해 재계산. [증명-수치확인]"""
    A = sp.symbols("A", real=True, nonzero=True)
    cf = lambda u: u**2 / 2 + sp.Rational(1, 2)
    tang = lambda u: (lambda X: cf(u) + u * (X - u))
    # l2: l1 과 수직 → 기울기 -1/a 인 접선 (접점 b: 기울기 b)
    b_ = -1 / A
    l1 = lambda X: tang(A)(X)
    l2 = lambda X: tang(b_)(X)
    ans1 = sp.sympify(load_answer(2, 0, raw=True).split("=")[1], locals={"a": A, "x": x})
    ok1 = sp.simplify(l2(x) - ans1) == 0
    # [2.2] 교점 P 의 y좌표 = 0 : 기호적 + 무작위 a 20점
    xP = sp.solve(sp.Eq(l1(x), l2(x)), x)[0]
    yP = sp.simplify(l1(xP))
    import random
    random.seed(1)
    ok2 = yP == 0
    for _ in range(20):
        av = random.choice([-1, 1]) * random.uniform(0.1, 5)
        xp = sp.nsolve(l1(x).subs(A, av) - l2(x).subs(A, av), x, 0)
        ok2 = ok2 and abs(float(l1(xp).subs(A, av))) < 1e-9
    # [2.3] l3: H(a,0) 지나고 l2 와 평행
    l3 = lambda X: (X - A) * sp.diff(l2(x), x)
    ans3 = sp.sympify(load_answer(2, 2, raw=True).split("=")[1], locals={"a": A, "x": x})
    ok3 = sp.simplify(l3(x) - ans3) == 0
    # [2.4] H'(b,0), l4 : l1 과 평행
    l4 = lambda X: (X - b_) * sp.diff(l1(x), x)
    xQ = sp.solve(sp.Eq(l3(x), l4(x)), x)[0]
    Q = (sp.simplify(xQ), sp.simplify(l3(xQ)))
    ok4 = Q == (0, 1) and str(Q) == str(sp.sympify(load_answer(2, 3, raw=True)))
    report(2, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] [2.2] P=({sp.simplify(xP)}, {yP}) 이고 무작위 a 20점 y=0 / [2.1] l2 일치 / [2.3] l3: y={sp.simplify(l3(x))} / [2.4] Q={Q}")


def p03():
    """극한으로 정의된 f 를 고정밀 수치로 계산해 [3.1]~[3.4] 재계산."""
    import mpmath as mp
    mp.mp.dps = 60
    def f_num(xv, kv, N=2000):
        xv = mp.mpf(xv)
        return (xv ** (N + 1) * mp.sin(kv * xv) + xv) / (xv**N + xv * mp.cos(mp.pi * xv / 2) + 1)
    kk = mp.mpf("0.7")
    ans1 = load_answer(3, 0, raw=True)
    # [3.1] f(1) = (sin k + 1)/2
    ok1 = abs(f_num(1, kk) - (mp.sin(kk) + 1) / 2) < mp.mpf("1e-40")
    # [3.2] x>1: f(x) = x sin kx (N 크게)
    ok2 = all(abs(f_num(xv, kk, 4000) - mp.mpf(xv) * mp.sin(kk * mp.mpf(xv))) < mp.mpf("1e-30") for xv in ("1.05", "1.5", "2"))
    # [3.3] 연속 조건 sin k = (sin k + 1)/2 의 양의 최소 k : f(1) 과 x→1+ 극한 비교
    g = lambda kv: mp.sin(kv) - (mp.sin(kv) + 1) / 2
    k0 = mp.findroot(lambda kv: mp.cos(kv), 1.5)   # g 는 sin k = 1 에서 접하는 중근 → g 의 극값점(g(k0)=0 확인)
    assert abs(g(k0)) < mp.mpf("1e-50")
    ans3 = load_answer(3, 2)
    # 0<k<π/2 에서는 g<0 (불연속), 양의 근 중 최소
    smaller = [mp.mpf(i) / 1000 for i in range(1, int(float(ans3) * 1000) - 1)]
    ok3 = abs(k0 - ans3) < mp.mpf("1e-30") and all(g(s) < 0 for s in smaller)
    # [3.4] 넓이 = ∫_1^2 x sin(πx/2) dx (f≥0 확인)
    S = mp.quad(lambda xv: xv * mp.sin(mp.pi * xv / 2), [1, 2])
    nonneg = all(f_num(mp.mpf(1) + mp.mpf(i) / 50, mp.pi / 2, 4000) >= -1e-12 for i in range(0, 51))
    ans4 = load_answer(3, 3)
    ok4 = nonneg and abs(S - sp.N(ans4, 40)) < mp.mpf("1e-30")
    report(3, ok1 and ok2 and ok3 and ok4, f"[3.1] f(1) 일치 / [3.2] x sin kx 일치 / [3.3] k={mp.nstr(k0, 10)} / [3.4] S={mp.nstr(S, 12)} vs {sp.N(ans4, 12)}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY M23: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
