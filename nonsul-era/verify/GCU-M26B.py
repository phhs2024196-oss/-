#!/usr/bin/env python3
"""가천대 2026학년도 논술 모의고사 B형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-M26B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)




def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p05():
    # (가) f(x)+x^3 은 일차 이하, x 의 계수 k → f = -x^3 + p x + q 로 두고 극한 조건을 sympy 로 풀기
    p, q, k = sp.symbols("p q k", real=True)
    f = -x**3 + p * x + q
    eq1 = sp.Eq(sp.limit((f + x**3) / (x + 1), x, sp.oo), k)
    num = f - x**3
    eq0 = sp.Eq(num.subs(x, -1), 0)              # 분모→0 이므로 분자→0
    s0 = sp.solve([eq1, eq0], [p, q], dict=True)[0]
    num2 = sp.factor(num.subs(s0))
    L = sp.limit(num2 / (x + 1), x, -1)
    ksol = sp.solve(sp.Eq(L, f.subs(s0).subs(x, 1)), k)[0]
    fk = f.subs(s0).subs(k, ksol).subs(x, ksol)
    report(5, fk == 31, f"f(x)={sp.expand(f.subs(s0))}, 극한={L}, k={ksol}, f(k)={fk} (YAML 31)")


def p06():
    a, th = sp.symbols("a theta", real=True)
    amin = sp.solve(sp.Eq(-a + sp.Rational(1, 2), -1), a)[0]   # a>0 이면 최솟값 -a+1/2
    s = -amin / 3                                               # sin+cos
    # 실제 각 θ 를 구해 수치 대입(독립 확인)
    sols = sp.solve(sp.Eq(sp.sin(th) + sp.cos(th), s), th)
    vals = {sp.nsimplify(sp.N(sp.cos(t)**3 * (sp.tan(t)**3 + 1), 30), [sp.Integer(1)]) for t in sols}
    report(6, amin == sp.Rational(3, 2) and vals == {sp.Rational(-11, 16)}, f"a={amin}, sinθ+cosθ={s}, 값={vals} (YAML -11/16)")


def p07():
    a, b, c, t, X = sp.symbols("a b c t X", real=True)
    f = lambda u: u**3 + a * u**2 + b * u + c
    I = sp.integrate(f(t), (t, -X, X))
    s1 = sp.solve(sp.Poly(sp.expand(I), X).coeffs(), [a, c], dict=True)[0]
    fs = lambda u: f(u).subs(s1)
    G = sp.integrate((3 * X - t) * fs(t), (t, 2, X))
    L = sp.limit(G / (X**2 - 4), X, 2)
    bsol = sp.solve(sp.Eq(L, 12), b)[0]
    val = sp.integrate(fs(x).subs(b, bsol), (x, s1[a], bsol))
    report(7, val == 8, f"a=c=0: {s1}, 극한={sp.expand(L)}, b={bsol}, 적분={val} (YAML 8)")


def p08():
    def seq(a1):
        a = [None, a1]
        for n in range(1, 7):
            an = a[n]
            a.append(-3 * an + 8 if an < 5 else an - n + 3)
        return a
    good = [a1 for a1 in range(-200000, 200001) if seq(a1)[7] == 15]
    a4s = {seq(a1)[4] for a1 in good}
    g1 = -4 in a4s            # ㄱ
    g2 = sum(good) == 14      # ㄴ
    chosen = [nm for nm, ok in (("ㄱ", g1), ("ㄴ", g2)) if ok]
    report(8, chosen == ["ㄴ"], f"a1 후보={good}, a4 값={sorted(a4s)}, 옳은 것={chosen} (YAML ㄴ)")


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
