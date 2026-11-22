#!/usr/bin/env python3
"""가천대 2026학년도 수시 논술(실전) 자연계열 F세트 수학(문제 7~15, 보고서 문항카드 82~90) — sympy 검산.
정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산한 뒤, 그 값을 YAML answer.value 문자열과 대조한다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G26F"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items/GCU/PAST"
x, t, h, n = sp.symbols("x t h n", real=True)


def ans(seq):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return str(d["subparts"][0]["answer"]["value"])


def tex(v):
    v = sp.nsimplify(v)
    if v.is_Rational and v.q != 1:
        s = f"\\frac{{{abs(v.p)}}}{{{v.q}}}"
        return ("-" if v < 0 else "") + s
    return str(v)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def has(seq, *vals):
    a = ans(seq).replace(" ", "")
    return all(tex(v) in a for v in vals)


def p07():
    d = sp.symbols("d")
    a = lambda k: 1 + (k - 1) * d  # noqa: E731
    dv = sp.solve(sp.Eq(sum(a(k) for k in range(1, 11)), 100), d)[0]
    s = sum(sp.Rational(2) / (a(k) * a(k + 1)).subs(d, dv) for k in range(1, 11))
    report(7, has(7, s), f"d={dv}, 합={s}")


def p08():
    p, q, r, a = sp.symbols("p q r a")
    f = p * x ** 2 + q * x + r
    L1 = 9 * sp.limit(sp.cancel((f - x) / (x ** 3 - 27)), x, 3)
    # 일반 이차식에서 극한 존재 조건: 분자(x=3)=0, 분자(x=2)=0
    c1 = sp.Eq((f - x).subs(x, 3), 0)
    c2 = sp.Eq((f - f.subs(x, x + 1)).subs(x, 2), 0)
    sols = sp.solve([c1, c2], [q, r], dict=True)[0]
    fs = f.subs(sols)
    l1 = 9 * sp.limit((fs - x) / (x ** 3 - 27), x, 3)
    l2 = sp.limit((fs - fs.subs(x, x + 1)) / (x ** 2 - 4), x, 2)
    pv = sp.solve(sp.Eq(l1, l2), p)
    pv = [v for v in pv if v != 0]
    assert len(pv) == 1
    av = sp.simplify(l1.subs(p, pv[0]))
    f7 = sp.simplify(fs.subs({p: pv[0], x: 7}))
    a_ = ans(8).replace(" ", "")
    ok = f"a={tex(av)}" in a_ and f"f(7)={f7}" in a_
    report(8, ok, f"p={pv[0]}, a={av}, f(7)={f7}")


def p09():
    def step(v, k):  # a_{k+1} from a_k
        return -2 * v + 7 if v < 4 else v - k + 5
    found = []
    for a1 in range(-200, 201):
        seq = [a1]
        for k in range(1, 10):
            seq.append(step(seq[-1], k))
        if seq[5] == 11:
            found.append(seq)
    assert len(found) == 1
    s = found[0]
    a_ = ans(9).replace(" ", "")
    ok = f"a_1={s[0]}" in a_ and f"a_4={s[3]}" in a_ and f"={sum(s)}" in a_
    report(9, ok, f"수열={s}, 합={sum(s)} (정수 초항 -200..200 전수)")


def p10():
    a, b, c = sp.symbols("a b c", positive=True)
    bv = sp.solve(sp.Eq(sp.pi / b, 2 * sp.pi), b)[0]   # 최대~최소 간격 = 반주기
    f = a * sp.sin(bv * x) + c
    sol = [s for s in sp.solve([sp.Eq(a + c, 10), sp.Eq(f.subs(x, sp.pi / 3), 8)], [a, c], dict=True) if s[a] < s[c]][0]
    fs = f.subs(sol)
    al, be = sp.pi, 3 * sp.pi  # 0<=x<=4pi 에서 최대·최소 지점
    assert sp.simplify(fs.subs(x, al) - 10) == 0 and sp.diff(fs, x).subs(x, be) == 0
    m = fs.subs(x, be)
    g = 10 + (m - 10) / (be - al) * (x - al)
    area = sp.integrate(g, (x, al, be))  # g>0 on [al,be]
    report(10, ans(10).replace(" ", "") == sp.latex(area).replace(" ", ""), f"b={bv}, a={sol[a]}, c={sol[c]}, m={m}, 넓이={area}")


def p11():
    def cnt(n, v):  # 실수 v 의 n제곱근 중 실수의 개수
        if n % 2:
            return 1
        return 2 if v > 0 else (1 if v == 0 else 0)
    vals = []
    for n in range(2, 12):
        for a in [sp.Rational(-7, 3), -2, -1, sp.Rational(-1, 2), 0, sp.Rational(1, 3), 1, 3]:
            e1 = 2 * cnt(n + 1, a ** n) + (-1) ** n * cnt(n, cnt(n + 1, a))
            e2 = cnt(n, a) - 3 * cnt(n + 1, a ** n) + cnt(n + 1, abs(a) ** n - a ** n)
            vals.append((e1, e2))
    mn = min(v[0] for v in vals)
    targets = {v[1] for v in vals if v[0] == mn}
    assert len(targets) == 1
    tv = targets.pop()
    report(11, ans(11).strip() == str(tv), f"첫 식 최솟값={mn}, 그때 둘째 식 값={tv}")


def p12():
    def f(v):
        return (v + 5) * (v - 1) if v < 0 else v - 4

    def g(v, k):
        return f(v) * f(k - v)
    eps = sp.Rational(1, 10 ** 9)
    good = []
    for k in [sp.Rational(i, 2) for i in range(-30, 31)]:
        okk = True
        for a0 in {sp.Integer(0), k}:   # f(x), f(k-x) 가 끊길 수 있는 점
            L, R = g(a0 - eps, k), g(a0 + eps, k)
            if abs(L - R) > sp.Rational(1, 10 ** 6):
                okk = False
        if okk:
            good.append(k)
    a_ = ans(12).replace(" ", "")
    ok = good == [-5, 0, 4] and all(f"k={v}" in a_ for v in good)
    report(12, ok, f"극한 존재 k (반정수 격자 -15..15, 좌우 1e-9 비교)={good}")


def p13():
    a = sp.symbols("a")
    best = None
    for deg in range(0, 6):
        cs = sp.symbols(f"c0:{deg + 1}")
        f = sum(cs[i] * x ** i for i in range(deg + 1))
        C = sp.symbols("C")
        F = sp.integrate(f, x) + C
        eqs = sp.Poly(sp.expand(F - (x ** 2 * f - 2 * x ** 4 + a * x + 1)), x).coeffs()
        sol = sp.solve(eqs, list(cs) + [C, a], dict=True)
        if sol:
            best = (deg, sol[0], f, F)
            break
    deg, s, f, F = best
    av = s[a]
    fp0 = sp.diff(f, x).subs(x, 0).subs(s)
    X = 2 / (av * fp0)
    val = F.subs(s).subs(x, X)
    report(13, ans(13).strip() == str(val), f"deg f={deg}, a={av}, f'(0)={fp0}, F({X})={val}")


def p14():
    a = sp.symbols("a", positive=True)
    # A: log_a x + 3 = -log_a x + 1  ->  log_a x = -1
    Lx = sp.symbols("Lx")
    assert sp.solve(sp.Eq(Lx + 3, -Lx + 1), Lx) == [-1]   # A(1/a, 2)
    sols = []
    for L in [sp.Rational(k, 6) for k in range(1, 60)]:        # L = log_a t > 0 (t>1, a>1)
        beta = sp.Rational(16, 3) / (L + 1)                      # 넓이: (2L+2)(t-1/a)/2 = 16/3
        if beta ** 2 + (L + 1) ** 2 != sp.Rational(100, 9):      # AB^2
            continue
        By, Cy = L + 3, -L + 1
        if not (By.is_integer and Cy.is_integer):
            continue
        av = [r for r in sp.solve(sp.Eq(a ** L - 1 / a, beta), a) if r.is_real and r > 1]
        for r in av:
            sols.append((r, r ** L))
    a_ = ans(14).replace(" ", "")
    ok = sols == [(3, 3)] and "a=3" in a_ and "t=3" in a_
    report(14, ok, f"(a,t)={sols}")


def p15():
    c = sp.symbols("c", real=True)
    h = sp.symbols("h")
    # 다항식 f 를 미정계수 삼차로 두고 lim (1/h)∫_1^{1+h} f^2 = f(1)^2 를 sympy 로 확인
    p = sp.symbols("p0:4")
    f = sum(p[i] * x ** i for i in range(4))
    tt = sp.symbols("tt")
    lim = sp.limit(sp.integrate(f.subs(x, tt) ** 2, (tt, 1, 1 + h)) / h, h, 0)
    assert sp.simplify(lim - f.subs(x, 1) ** 2) == 0
    sol = sp.solve(sp.Poly(sp.expand(4 * f - (-2 * x ** 3 + 6 * x + c)), x).coeffs(), p, dict=True)[0]
    fs = f.subs(sol)
    cv = sp.solve(sp.Eq(c, fs.subs(x, 1) ** 2), c)
    assert len(set(cv)) == 1
    fs = fs.subs(c, cv[0])
    crit = sp.solve(sp.diff(fs, x), x)
    mx = [fs.subs(x, r) for r in crit if sp.diff(fs, x, 2).subs(x, r) < 0][0]
    mn = [fs.subs(x, r) for r in crit if sp.diff(fs, x, 2).subs(x, r) > 0][0]
    a_ = ans(15).replace(" ", "")
    ok = f"극댓값${mx}$" in a_ and f"극솟값${mn}$" in a_
    report(15, ok, f"f={sp.expand(fs)}, 극댓값={mx}, 극솟값={mn}")


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
