#!/usr/bin/env python3
"""DGU-G21 (2021학년도 동국대 논술 실전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-G21"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1+a/100)^5=2, log2=0.3, log1.15=0.06 → a=15 (실제 근 ≈14.87 와 근접 확인)."""
    L = sp.symbols("L")                       # L = log(1+a/100)
    Lsol = sp.solve(sp.Eq(5 * L, sp.Rational(3, 10)), L)[0]
    given = {sp.Rational(6, 100): sp.Rational(115, 100)}   # 주어진 근삿값표: log1.15=0.06
    base = given[Lsol]
    a = sp.nsimplify((base - 1) * 100)
    true_a = float((2 ** sp.Rational(1, 5) - 1) * 100)
    ok = a == 15 and abs(true_a - 15) < 0.2
    report(1, ok, f"log(1+a/100)={Lsol} → a={a} (참값 {true_a:.4f}%)")


def p02():
    """(1) P(X≥68)=0.1 (2) B(400,0.1): 평균 40, 표준편차 6 (3) P(Y≥52)≈0.02 (표 사용)."""
    from scipy.stats import norm, binom
    table = {sp.Rational(52, 100): sp.Rational(20, 100), sp.Rational(84, 100): sp.Rational(30, 100),
             sp.Rational(128, 100): sp.Rational(40, 100), sp.Integer(2): sp.Rational(48, 100)}
    z1 = (sp.Rational(68) - sp.Rational(552, 10)) / 10
    p1 = sp.Rational(1, 2) - table[z1]
    n = 400
    m = n * p1
    sd = sp.sqrt(n * p1 * (1 - p1))
    z3 = (52 - m) / sd
    p3 = sp.Rational(1, 2) - table[z3]
    exact1 = 1 - norm.cdf(float(z1))
    exact3 = binom.sf(51, n, 0.1)
    ok = p1 == sp.Rational(1, 10) and m == 40 and sd == 6 and p3 == sp.Rational(2, 100) and abs(exact1 - 0.1) < 0.001
    report(2, ok, f"(1) z={z1}, p={p1} (참 {exact1:.4f}) (2) m={m}, σ={sd} (3) z={z3}, p={p3} (이항 정확값 {exact3:.4f})")


def p03():
    """∫3^t (s-3)^2 ds + ∫0^{f(t)} (√s+3) ds = t f(t) → f=(x-3)^2, 넓이 1/3, 부피 1/5."""
    A, B = sp.symbols("A B", real=True)       # A=t-3≥0, B=√f≥0
    s, T = sp.symbols("s T", positive=True)
    lhs = sp.integrate((s - 3)**2, (s, 3, A + 3)) + sp.integrate(sp.sqrt(s) + 3, (s, 0, B**2))
    eq = sp.expand(sp.simplify(lhs.subs(sp.sqrt(B**2), B) - (A + 3) * B**2))
    sols = sp.solve(eq, B)
    nonneg = [r for r in sols if all((r.subs(A, av) >= 0) for av in (1, 2, 5))]
    f = (x - 3)**2
    chk = sp.simplify(sp.integrate((s - 3)**2, (s, 3, T)) + sp.integrate(sp.sqrt(s) + 3, (s, 0, f.subs(x, T))) - T * f.subs(x, T))
    area = sp.integrate(f, (x, 3, 4))
    vol = sp.integrate(f**2, (x, 3, 4))
    ok = nonneg == [A] and sp.simplify(chk.subs(T, 5)) == 0 and sp.simplify(chk.subs(T, sp.Rational(37, 7))) == 0 \
        and area == sp.Rational(1, 3) and vol == sp.Rational(1, 5)
    report(3, ok, f"식 {sp.factor(eq)}=0 의 B≥0 근 {nonneg} → f=(x-3)^2, 넓이 {area}, 부피 {vol}")


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
