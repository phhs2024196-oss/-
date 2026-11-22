#!/usr/bin/env python3
"""서울과학기술대 2024학년도 모의논술(자연계열) (SNUT-M24-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import sys

import sympy as sp

PREFIX = "SNUT-M24"
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
    """[1.1] 함수 전수 열거, [1.2] 기하 구성 후 급수, [1.3] 부피 방정식."""
    import itertools
    import math
    X = range(1, 7)
    Y = range(2, 9)
    prime = lambda m: m > 1 and all(m % d for d in range(2, int(m**0.5) + 1))
    cnt = 0
    for vals in itertools.product(Y, repeat=6):
        f = dict(zip(X, vals))
        if all(prime(f[i]) for i in X if i % 2 == 1) and all(f[i] % 2 == 0 for i in X if i % 2 == 0) and all(f[i] < f[i + 1] for i in range(1, 6)):
            cnt += 1
    ans1 = load_answer(1, 0)
    ok1 = cnt == ans1
    # [1.2] 좌표로 직접 구성: O=원점, OQ1 = x축 방향 단위, OP1 = 45° 방향
    import numpy as np
    O = np.zeros(2)
    Q = np.array([1.0, 0.0])
    P = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    total = 0.0
    areas = []
    for _ in range(60):
        d = P / np.linalg.norm(P)
        Pn = O + d * ((Q - O) @ d)                     # Q 에서 OP 에 내린 수선의 발 = P_{n+1}
        area = 0.5 * abs((Pn - P)[0] * (Q - P)[1] - (Pn - P)[1] * (Q - P)[0])   # 삼각형 Q P P_{n+1}
        areas.append(area)
        total += area
        r = np.linalg.norm(Pn)                          # OQ_{n+1} = OP_{n+1}
        Q = np.array([r, 0.0])
        P = Pn
    ans2 = load_answer(1, 1)
    ok2 = abs(areas[0] - (math.sqrt(2) - 1) / 4) < 1e-12 and abs(areas[1] / areas[0] - 0.5) < 1e-12 and abs(total - float(ans2)) < 1e-12
    # [1.3] V1 = ∫ π(a²-x) dx (0..a²), V2 = ∫ π(a-x)^4 dx (0..a)
    aa = sp.symbols("aa", positive=True)
    V1 = sp.integrate(sp.pi * (aa**2 - x), (x, 0, aa**2))
    V2 = sp.integrate(sp.pi * (aa - x) ** 4, (x, 0, aa))
    sol = sp.solve(sp.Eq(V1, V2), aa)
    ans3 = load_answer(1, 2)
    ok3 = sol == [ans3]
    report(1, ok1 and ok2 and ok3, f"[1.1] 전수 열거 {len(Y) ** 6}가지 중 {cnt}개 / [1.2] a_1={areas[0]:.9f}, 공비={areas[1]/areas[0]:.6f}, 합≈{total:.9f} / [1.3] V1={V1}, V2={V2}, a={sol}")


def p02():
    """[2.1] 교점 개수 격자 확인, [2.2] 넓이, [2.3] 최솟값 격자."""
    import numpy as np
    # [2.1] 여러 a 에서 c>0 인 값 중 교점이 2개인 c 를 판별식 방식으로 수치 탐색
    ans1 = sp.sympify(load_answer(2, 0, raw=True), locals=SYM)
    ok1 = True
    nreal = lambda cv, av: len(set(sp.Poly(x**3 - 3 * av * x - cv, x).real_roots()))
    for av in (sp.Rational(1, 4), 1, sp.Rational(9, 4), 4, 9):       # √a 가 유리수가 되는 a
        exact = ans1.subs(a, av)
        ok1 = ok1 and nreal(exact, av) == 2 and nreal(exact * sp.Rational(99, 100), av) == 3 and nreal(exact * sp.Rational(101, 100), av) == 1
    # [2.2] 넓이 기호 적분과 수치 적분
    A_ = sp.symbols("A_", positive=True)
    c = 2 * A_ * sp.sqrt(A_)
    fac = sp.simplify(x**3 - 3 * A_ * x - c - (x + sp.sqrt(A_)) ** 2 * (x - 2 * sp.sqrt(A_)))   # 인수분해 확인 → 교점 -√a(중근), 2√a
    S = sp.integrate(c - (x**3 - 3 * A_ * x), (x, -sp.sqrt(A_), 2 * sp.sqrt(A_)))
    ans2 = sp.sympify(load_answer(2, 1, raw=True), locals=SYM)
    ok2 = eq(S, ans2.subs(a, A_)) and fac == 0
    import mpmath as mp
    for av in (1.0, 2.5, 4.0):
        sv = mp.quad(lambda xv: 2 * av * av**0.5 - (xv**3 - 3 * av * xv), [-av**0.5, 2 * av**0.5])
        ok2 = ok2 and abs(sv - float(ans2.subs(a, av))) < 1e-9
    # [2.3] 4S/(27(a-2)²(6-a)) 의 [3,5] 최솟값
    aa = np.linspace(3, 5, 2000001)
    Sv = 27 * aa**2 / 4
    val = 4 * Sv / (27 * (aa - 2) ** 2 * (6 - aa))
    ans3 = load_answer(2, 2)
    ok3 = abs(val.min() - float(ans3)) < 1e-9 and abs(aa[val.argmin()] - 4) < 1e-5
    report(2, ok1 and ok2 and ok3, f"[2.1] c=2a√a (a=1/4,1,9/4,4,9 에서 교점 2개, ±1% 에서 3·1개) / [2.2] S={sp.simplify(S)} / [2.3] min≈{val.min():.9f} at a≈{aa[val.argmin()]:.5f}")


def p03():
    """[3.1]~[3.4] 곡선 위 점·접선을 수치로 구성."""
    import mpmath as mp
    mp.mp.dps = 40
    f = lambda xv: mp.cos(xv) / xv**2
    ans1 = sp.sympify(load_answer(3, 0, raw=True), locals={**SYM, "n": n})
    # [3.1] f 의 양의 영점 (수치 근) 처음 12개
    zs = []
    for k_ in range(1, 13):
        z = mp.findroot(f, (2 * k_ - 1) * mp.pi / 2 + mp.mpf("0.01"))
        zs.append(z)
    ok1 = all(abs(zs[i - 1] - sp.N(ans1.subs(n, i), 40)) < mp.mpf("1e-30") for i in range(1, 13))
    # [3.2] y_n : 접선의 y절편 = f(x_n) - x_n f'(x_n)
    ys = [f(z) - z * mp.diff(f, z) for z in zs]
    ans2 = sp.sympify(load_answer(3, 1, raw=True), locals={**SYM, "n": n})
    ok2 = all(abs(ys[i - 1] - sp.N(ans2.subs(n, i), 40)) < mp.mpf("1e-25") for i in range(1, 13))
    # [3.3] A_n = 삼각형 P_n P_{n+1} Q_n 의 넓이 (좌표), 합
    A = []
    for i in range(11):
        P1 = (zs[i], 0); P2 = (zs[i + 1], 0); Q = (0, ys[i])
        A.append(abs((P2[0] - P1[0]) * (Q[1] - P1[1]) - (P2[1] - P1[1]) * (Q[0] - P1[0])) / 2)
    s10 = sum(A[i] * A[i + 1] for i in range(10))
    ans3 = load_answer(3, 2)
    ok3 = abs(s10 - sp.N(ans3, 40)) < mp.mpf("1e-25")
    # [3.4] 무한급수: 부분합이 1/2 로 수렴
    nn = sp.symbols("nn", integer=True, positive=True)
    kk = sp.symbols("kk", integer=True, positive=True)
    Sn = sp.summation(1 / ((2 * kk - 1) * (2 * kk + 1)), (kk, 1, nn))
    lim = sp.limit(Sn, nn, sp.oo)
    ans4 = load_answer(3, 3)
    partial = sum(1 / mp.mpf((2 * k_ - 1) * (2 * k_ + 1)) for k_ in range(1, 200001))
    ok4 = eq(lim, ans4) and abs(partial - sp.N(ans4, 30)) < mp.mpf("1e-5")
    report(3, ok1 and ok2 and ok3 and ok4, f"[3.1] x_1..x_12 일치 / [3.2] y_1..y_12 일치(교대 부호) / [3.3] Σ_(n≤10)={mp.nstr(s10, 12)} vs 10/21 / [3.4] 극한={lim}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY M24: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
