#!/usr/bin/env python3
"""서울과기대 2022학년도 논술전형고사 3차 (SNUT-G22C-01~03) — sympy/mpmath 검산. 정답 하드코딩 비교 없이 문제 조건에서 독립 재계산해 YAML answer 와 비교."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SNUT-G22C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, d, k = sp.symbols("a b d k", real=True)
theta, theta_k, alpha = sp.symbols("theta theta_k alpha", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "d": d, "k": k, "theta": theta,
       "theta_k": theta_k, "alpha": alpha, "pi": sp.pi, "E": sp.E}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d_ = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d_["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def same(u, v, tol=1e-9):
    """수치 동일성(상대오차 tol)."""
    return abs(complex(sp.N(u, 30)) - complex(sp.N(v, 30))) <= tol * max(1, abs(complex(sp.N(v, 30))))

def p01():
    # 1.1
    X, Y = sp.symbols("X Y", real=True)
    F = sp.cos(X - Y) + X ** 2 + Y ** 2 - 9
    xs = sp.solve(F.subs(Y, X), X)
    xs = sorted([s for s in xs if s.is_real])
    slope = lambda px, py: sp.simplify(sp.idiff(F, Y, X).subs({X: px, Y: py}))
    parts = []
    for s in xs:
        m = slope(s, s)
        c = s - m * s
        parts.append(f"({s},{s}): y={sp.simplify(m)}x{'+' if c >= 0 else ''}{c}".replace("-1x", "-x").replace("1x", "x"))
    got = ", ".join(reversed(parts))
    ans = load_answer(1, 0, raw=True)
    report(1, got.replace(" ", "") == ans.replace(" ", ""), f"1.1 교점 {xs}, 접선 {got}, yaml={ans}")
    # 1.2 y=(1/1024)4^{-x}+3/2 = a^{x-m}+n 으로 읽기
    base = sp.Rational(1, 4)                       # 4^{-x} = (1/4)^x
    m = sp.symbols("m", real=True)
    ms = sp.solve(sp.Eq(base ** (-m), sp.Rational(1, 1024)), m)[0]
    n_ = sp.Rational(3, 2)
    bsol = sp.solve(sp.Eq(3 * sp.log(base) / sp.log(sp.symbols("bb", positive=True)), 4 / (4 * base - 3)), sp.symbols("bb", positive=True))[0]
    g = lambda X_: ms * n_ * sp.log(X_) / sp.log(bsol)
    val = sp.simplify(sp.expand_log(g(base ** 5), force=True))
    chk = all(abs(sp.N((sp.Rational(1, 1024) * 4 ** (-xv) + sp.Rational(3, 2)) - (base ** (xv - ms) + n_))) < 1e-12 for xv in [0, 1, 2.5, -3])
    ans = load_answer(1, 1)
    report(1, chk and sp.simplify(val - ans) == 0, f"1.2 m={ms}, n={n_}, b={bsol}, (g∘f)(5)={val}, yaml={ans}")
    # 1.3 교점 개수(수치) → a_n 공식 확인 후 합
    mp.mp.dps = 20
    def count(nn):
        c = (2 * nn - 1) * mp.pi
        h = lambda X: mp.sin(X) - X / c
        N = 40000 * nn
        grid = [mp.mpf(i) * c / N + mp.mpf('1e-7') for i in range(0, N + 1)]
        vals = [h(g_) for g_ in grid]
        pos = sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
        return 2 * pos + 1                         # 원점대칭 + 원점
    an = [count(nn) for nn in range(1, 5)]
    formula_ok = an == [4 * nn - 1 for nn in range(1, 5)]
    S = sp.summation(1 / ((4 * n - 1) * (4 * n + 3)), (n, 1, sp.oo))
    ans = load_answer(1, 2)
    report(1, formula_ok and sp.simplify(S - ans) == 0, f"1.3 a_1..a_4(수치)={an}, Σ1/(a_n a_(n+1))={S}, yaml={ans}")


def p02():
    V1 = sp.integrate(sp.pi ** 2 - 2 * sp.pi * x, (x, 0, sp.pi / 2))
    ans = load_answer(2, 0)
    report(2, sp.simplify(V1 - ans) == 0, f"2.1 아래쪽 부피={V1}, yaml={ans}")
    V2 = sp.integrate(sp.pi * (x - sp.pi / 2) * (1 + sp.cos(2 * x)), (x, sp.pi / 2, sp.pi))
    ans = load_answer(2, 1)
    report(2, sp.simplify(V2 - ans) == 0, f"2.2 위쪽 부피={V2}, yaml={ans}")
    h = sp.symbols("h", real=True)
    hs = [s for s in sp.solve(sp.Eq(sp.integrate(sp.pi ** 2 - 2 * sp.pi * x, (x, 0, h)), 4 * sp.pi ** 3 / 25), h) if 0 <= sp.N(s) <= sp.N(sp.pi / 2)]
    remain = sp.N(V2 - 4 * sp.pi ** 3 / 25) > 0 and sp.N(V1 - 4 * sp.pi ** 3 / 25) > 0   # 위쪽에 물이 남아 있고 아래쪽은 넘치지 않음
    ans = load_answer(2, 2)
    report(2, len(hs) == 1 and remain and sp.simplify(hs[0] - ans) == 0, f"2.3 h={hs}, 4분 후 위쪽에 물 잔존={remain}, yaml={ans}")


def p03():
    tt = sp.symbols("tt", real=True)
    def crit(tv):
        r = sp.solve(3 * x ** 2 + 2 * tv * x + 9, x)
        return sorted(r, key=lambda z: sp.N(z)) if all(sp.im(sp.N(z)) == 0 for z in r) and len(set(r)) == 2 else None
    # 3.1 t 를 훑어 조건(극대 α<0, 극소 β<0)을 만족하는 t 의 범위를 확인
    good = []
    for tv in [sp.Rational(i, 4) for i in range(-80, 81)]:
        r = crit(tv)
        if r is None:
            continue
        al, be = r                                   # 삼차항 계수 +: 작은 근이 극대, 큰 근이 극소
        if sp.N(al) < 0 and sp.N(be) < 0:
            good.append(tv)
    lo = min(good)
    bound = 3 * sp.sqrt(3)
    ans = load_answer(3, 0, raw=True)
    ok = all(sp.N(g_) > sp.N(bound) for g_ in good) and lo - sp.Rational(1, 4) <= sp.N(bound) and ans.replace(" ", "") == "t>3*sqrt(3)" and \
        all((sp.N(tv) > sp.N(bound)) == (tv in good) for tv in [sp.Rational(i, 4) for i in range(-80, 81)])
    report(3, ok, f"3.1 격자 t∈[-20,20]에서 조건 만족 최소 t={lo} (3√3≈{sp.N(bound, 6)}), yaml={ans}")
    # 3.2 α²-β²
    ex = load_answer(3, 1)
    ok = True
    for tv in [6, 7.5, 9, 12, 20]:
        al, be = [sp.N(z, 30) for z in crit(sp.nsimplify(tv))]
        ok = ok and abs((al ** 2 - be ** 2) - sp.N(ex.subs(t, sp.nsimplify(tv)), 30)) < 1e-20
    report(3, ok, "3.2 α²-β² (수치 근) = yaml 식, t=6,7.5,9,12,20")
    # 3.3 tanθ1-tanθ2 = f(α)/α - f(β)/β
    ex = load_answer(3, 2)
    ok = True
    for tv in [6, 7.5, 9, 12, 20]:
        f = lambda X, T=sp.nsimplify(tv): X ** 3 + T * X ** 2 + 9 * X + 15
        al, be = [sp.N(z, 30) for z in crit(sp.nsimplify(tv))]
        ok = ok and abs((f(al) / al - f(be) / be) - sp.N(ex.subs(t, sp.nsimplify(tv)), 30)) < 1e-20
    report(3, ok, "3.3 f(α)/α-f(β)/β (수치 근) = yaml 식")
    # 3.4 정의로부터 수치 최대화
    def gnum(tv):
        al, be = [sp.N(z, 30) for z in crit(sp.nsimplify(tv))]
        f = lambda X: X ** 3 + sp.nsimplify(tv) * X ** 2 + 9 * X + 15
        return f(al) / al - f(be) / be
    best = max((gnum(sp.Rational(i, 20)), sp.Rational(i, 20)) for i in range(int(5.2 * 20) + 1, 40 * 20))
    ans = load_answer(3, 3)
    report(3, abs(best[0] - sp.N(ans)) < 1e-4 and abs(best[1] - 9) <= sp.Rational(1, 20), f"3.4 격자 최대 g={sp.N(best[0], 8)} at t={best[1]}, yaml={ans}≈{sp.N(ans, 8)}")


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
