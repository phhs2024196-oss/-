#!/usr/bin/env python3
"""숭실대 2021학년도 논술 자연2 (SSU-G21B-01·02·04; 3번 확통 제외) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수(번호 = 원문 문제 번호)를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G21B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, locals_=None):
    """yaml 의 subparts[sub].answer.value 를 sympy 로 읽는다."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return sp.sympify(d["subparts"][sub]["answer"]["value"], locals=locals_ or {})


def close(a, b, tol=1e-9):
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(1.0, abs(b))


def p01():
    """(1) x=k 의 좌·우극한을 가우스 기호 값과 주기 f 로 직접 계산해 a 결정(k=1..6).
    (2) 그 a 로 ∫_0^n g|sin πx| 를 구간별 수치적분(n=1..6)해 md 식과 비교."""
    import mpmath as mp
    mp.mp.dps = 25
    a, h = sp.symbols("a h", real=True)
    fper = lambda X, A: A - sp.Abs(2 * (X - sp.floor(X)) - 1)
    sols = set()
    for k in range(1, 7):
        eps = sp.Rational(1, 10**6)       # 가우스 기호 값은 (k-1, k) 에서 상수
        left = sp.limit(fper(k - h, a), h, 0, "+") / (sp.floor(k - eps + 1) * sp.floor(k - eps + 2))
        right = fper(sp.Integer(k), a) / ((k + 1) * (k + 2))
        sols |= set(sp.solve(sp.Eq(left, right), a))
    ans1 = load_answer(1, 0)
    ok1 = sols == {ans1}
    A0 = float(list(sols)[0])
    f = lambda X: A0 - abs(2 * (X - mp.floor(X)) - 1)
    g = lambda X: f(X) / (mp.floor(X + 1) * mp.floor(X + 2))
    ans2 = load_answer(1, 1, {"n": n})
    errs = []
    for N in range(1, 7):
        val = mp.fsum(mp.quad(lambda X: g(X) * abs(mp.sin(mp.pi * X)), [j - 1 + mp.mpf("1e-30"), j - mp.mpf(0.5), j - mp.mpf("1e-30")]) for j in range(1, N + 1))
        errs.append(abs(val - float(ans2.subs(n, N))))
    ok2 = max(errs) < 1e-12
    report(1, ok1 and ok2, f"(1) a={sols} (2) n=1..6 최대오차={mp.nstr(max(errs), 3)}, md={ans1}, {ans2}")


def p02():
    """접점 x=t(<0 이면 b>0) 의 접선과 곡선의 교점, 중점 b, 넓이 K·L 을 기호 적분."""
    a = sp.symbols("a", real=True)
    T = sp.symbols("T", negative=True)
    b = sp.symbols("b", positive=True)
    C = x ** 3 + a * x
    line = sp.diff(C, x).subs(x, T) * (x - T) + C.subs(x, T)
    roots = sp.solve(sp.Eq(C, line), x)
    Q = [r for r in roots if sp.simplify(r - T) != 0][0]
    R = sp.simplify((T + Q) / 2)
    Tb = [v for v in sp.solve(sp.Eq(R.subs(T, sp.Symbol("TT")), b), sp.Symbol("TT"))][0]
    P_b, Q_b = sp.simplify(T.subs(T, Tb)), sp.simplify(Q.subs(T, Tb))
    ans1 = load_answer(2, 0, {"b": b})
    ok1 = sp.simplify(P_b - ans1[0]) == 0 and sp.simplify(Q_b - ans1[1]) == 0
    diff = sp.expand((line - C).subs(T, Tb))
    K = sp.integrate(diff, (x, P_b, b))
    L = sp.integrate(diff, (x, b, Q_b))
    lim = sp.limit(K / L, b, sp.oo)
    ans2 = load_answer(2, 1)
    nonneg = all(diff.subs({b: 1, a: av, x: xv}) >= 0 for av in (-5, 0, 3) for xv in (-2, -1, 0, 1, 2, 3, 4))
    ok2 = sp.simplify(lim - ans2) == 0 and nonneg
    report(2, ok1 and ok2, f"(1) P={P_b}, Q={Q_b} (2) K={sp.factor(K)}, L={sp.factor(L)}, lim={lim}, md={ans1},{ans2}")


def p04():
    """직사각형의 네 꼭짓점이 S(볼록) 안에 있을 조건으로 격자 전수 최대화 → M(a) 를 md 식과 비교, 다시 a 로 최대화."""
    import numpy as np
    ans1 = load_answer(4, 0, {"a": sp.Symbol("a")})
    Mexpr = sp.lambdify(sp.Symbol("a"), ans1, "numpy")

    def inside(px, py):
        return (py <= 6 - px ** 2 + 1e-12) & (py >= 6 - 2 * px - 1e-12)

    def brute(a):
        Y = 6 - a * a
        xs = np.linspace(a * a / 2 - 0.2, a, 4001)[:, None]
        ys = np.linspace(0, Y, 4001)[None, :]
        ok = inside(xs, ys) & inside(np.full_like(ys, a), ys) & inside(xs, np.full_like(xs, Y))
        area = np.where(ok, (a - xs) * (Y - ys), 0.0)
        return area.max()

    agrid = [0.3, 0.7, 1.0, 1.4, 1.8]
    err = max(abs(brute(av) - Mexpr(av)) for av in agrid)
    fine = np.linspace(0.01, 1.99, 199)
    best = fine[int(np.argmax([brute(av) for av in fine]))]
    ans2 = load_answer(4, 1)
    ok = err < 2e-3 and abs(best - float(ans2)) < 0.011
    report(4, ok, f"(1) 격자 최대넓이와 M(a) 최대오차={err:.2e} (2) 격자 argmax a≈{best:.3f}, md={ans1}, {ans2}")


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
