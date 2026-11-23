#!/usr/bin/env python3
"""KW-G22C (광운대 2022학년도 논술고사 자연계열 3교시) — 검산. 기대값은 V.ans() 로만 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: F401,F403
import math
import itertools
import fractions
import numpy as np


def brentq(f, a, b):
    fa = f(a)
    for _ in range(200):
        m = (a + b) / 2
        fm = f(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return (a + b) / 2

V = Verifier("KW-G22C", "items/KW/PAST")
X = sp.Symbol("X")
A_ = sp.Symbol("a")


def p01():
    # 1-1, 1-2 : a_2=92, a_{n-1}=245, S_n=2022
    found = []
    for n in range(3, 400):
        # a_1+a_n = a_2+a_{n-1}=337 → S_n = n*337/2
        if fractions.Fraction(n * 337, 2) == 2022:
            found.append(n)
    V.check(1, "1-1", found[0] if found else None, V.ans(1, "1-1"))
    n = found[0]
    d = fractions.Fraction(245 - 92, n - 3)
    V.check(1, "1-2", sp.Rational(d.numerator, d.denominator), V.ans(1, "1-2"))
    # 2-1 : 부호 조합 전수
    a, c, d_ = sp.symbols("a c d")
    f = lambda x: a * (x - c)**2 + d_
    sols = []
    for sg in itertools.product((10, -10), repeat=4):
        r = sp.solve([f(i + 1) - sg[i] for i in range(4)], [a, c, d_], dict=True)
        for s in r:
            if s[a] > 0 and s[a] != 0:
                # 4개 모두 만족 (3개 방정식 미지수 3 + 1 초과) 확인
                if all(sp.simplify(f(i + 1).subs(s) - sg[i]) == 0 for i in range(4)):
                    sols.append(s)
    # 네 방정식 동시 만족: 위 solve 는 과결정이라 빈 경우가 있음 → 개별 3개 + 검사
    if not sols:
        for sg in itertools.product((10, -10), repeat=4):
            r = sp.solve([f(i + 1) - sg[i] for i in range(3)], [a, c, d_], dict=True)
            for s in r:
                if s[a] > 0 and sp.simplify(f(4).subs(s) - sg[3]) == 0:
                    sols.append(s)
    exp = {k: S(V.ans(1, "2-1", k)) for k in "acd"}
    ok = len(sols) == 1 and all(sp.simplify(sols[0][s] - exp[str(s)]) == 0 for s in (a, c, d_))
    V.report(1, ok, f"2-1 해 {sols}")
    # 2-2 : |f(x)|=k 근의 개수
    fa, fc, fd = (sp.Rational(10), sp.Rational(5, 2), sp.Rational(-25, 2))
    xs = np.linspace(-6, 11, 3400001)
    fx = np.abs(10 * (xs - 2.5)**2 - 12.5)

    def count(k):
        g = fx - k
        return int(np.sum(g[:-1] * g[1:] < 0) + np.sum(g == 0))
    parts = {}
    for seg in V.ans(1, "2-2").split(";"):
        key, cond = seg.split(":")
        parts[key.strip()] = cond.strip()
    cnt = {"none": 0, "two": 2, "three": 3, "four": 4}
    ok = True
    bad = []
    for k in [-1, 0, 1e-3, 1, 6, 12.4, 12.5, 12.6, 13, 20, 100]:
        kk = k
        # k=0 : 근 2개(중근 아님: f(x)=0 두 근), k=25/2: 세 개
        c_ = count(k) if k not in (0, 12.5) else (2 if k == 0 else 3)
        match = [nm for nm, cond in parts.items()
                 if eval(cond.replace(" or ", " or ").replace("k", "kk").replace("25/2", "12.5").replace("<", "<").replace("=", "==").replace("<==", "<=").replace(">==", ">="), {"kk": kk})]
        if len(match) != 1 or cnt[match[0]] != c_:
            bad.append((k, c_, match))
    V.report(1, not bad, f"2-2 k 격자 불일치 {bad}")
    # 3-1, 3-2
    a_ = sp.Symbol("a", real=True)
    peri = 2 * (-2 * a_ + 2 * (2 * a_**3 - 8 * a_))
    V.report(1, sp.simplify(peri - S(V.ans(1, "3-1")).subs(sp.Symbol("a"), a_)) == 0, "3-1 둘레")
    crit = [r for r in sp.solve(sp.diff(peri, a_), a_) if -2 < r < 0]
    mx = peri.subs(a_, crit[0])
    V.report(1, len(crit) == 1 and sp.simplify(mx - S(V.ans(1, "3-2", "max"))) == 0 and sp.simplify(crit[0] - S(V.ans(1, "3-2", "a"))) == 0, f"3-2 a={crit} max={sp.simplify(mx)} (경계 포함 확인 {sp.limit(peri, a_, -2, '+')}, {sp.limit(peri, a_, 0, '-')})")
    # 4-1
    xs2 = np.linspace(3 * math.pi / 4 + 1e-9, 5 * math.pi / 4 - 1e-9, 200001)
    s = np.sin(xs2)
    lo, hi = V.ans(1, "4-1").split("<sin(x)<")
    V.report(1, abs(s.min() - float(S(lo))) < 1e-6 and abs(s.max() - float(S(hi))) < 1e-6, "4-1 sin 범위")
    # 4-2 : f(x)=1 근 (sec2x 는 x=π 주변 정의, 2x=3π/2·5π/2 에서 불연속)
    g = lambda x: math.sin(x) + 1 / math.cos(2 * x) - 1
    xs3 = np.linspace(3 * math.pi / 4 + 1e-7, 5 * math.pi / 4 - 1e-7, 400001)
    vals = np.sin(xs3) + 1 / np.cos(2 * xs3) - 1
    pole = (np.abs(np.cos(2 * xs3)) < 1e-3)
    roots = []
    for i in range(len(xs3) - 1):
        if vals[i] * vals[i + 1] < 0 and abs(vals[i]) < 1e3 and abs(vals[i + 1]) < 1e3 and not (pole[i] or pole[i + 1]):
            roots.append(brentq(g, xs3[i], xs3[i + 1]))
    roots += [math.pi] if abs(g(math.pi)) < 1e-12 and not any(abs(r - math.pi) < 1e-6 for r in roots) else []
    roots = sorted(roots)
    V.check(1, "4-2 근의 개수", len(roots), V.ans(1, "4-2"))
    # 4-3
    al, be = roots[0], roots[-1]
    A = al - math.pi / 2
    B = be - math.pi
    C = math.pi - A - B
    R = math.sqrt(6) / (2 * math.sin(C))
    V.check(1, "4-3 R", R, V.ans(1, "4-3"), 1e-9)


def p02():
    n = sp.Symbol("n", positive=True, integer=True)
    i = sp.Symbol("i", integer=True)
    V.check(2, "1-1", sp.integrate(X**2, (X, 0, 2)), V.ans(2, "1-1"))
    # 1-2 급수
    lim = sp.limit(sp.summation((2 * i / n)**2 * (2 / n), (i, 1, n)), n, sp.oo)
    V.check(2, "1-2", lim, V.ans(2, "1-2"))
    # 2
    I = sp.Symbol("I", positive=True)
    L = 10 * sp.log(I / 10**-12, 10)
    I1 = 10**5 * 10**-12
    exact = sp.diff(L, I).subs(I, I1) * 2
    h = sp.Rational(1, 10**30)
    num = (L.subs(I, I1 + h) - L.subs(I, I1 - h)) / h
    V.check(2, "2 (도함수)", sp.simplify(exact), V.ans(2, "2"))
    # 3 : 전수
    sols = [(x, y, z, 21 - x - y - z) for x in range(22) for y in range(22 - x) for z in range(22 - x - y)]
    tot = len(sols)
    V.check(2, "3-1", tot, V.ans(2, "3-1"))
    ev = [s for s in sols if s[0] % 2 == 0 and s[1] % 2 == 0 and s[2] % 2 == 0]
    V.check(2, "3-2", sp.Rational(len(ev), tot), V.ans(2, "3-2"))
    ev2 = [s for s in sols if s[0] % 6 == 0 and s[1] % 2 == 0 and s[2] % 2 == 0]
    V.check(2, "3-3", sp.Rational(len(ev2), tot), V.ans(2, "3-3"))
    ev3 = [s for s in ev if s[0] % 3 == 0]
    V.check(2, "3-4", sp.Rational(len(ev3), len(ev)), V.ans(2, "3-4"))
    # 4-1
    P = lambda m: (sum(fractions.Fraction(1, 2**k) for k in range(1, m + 1)), sum(fractions.Fraction(1, 2**(k - 1)) + 0 for k in range(1, m + 1)) * 0 + sum(fractions.Fraction(1, 2**(k - 1)) for k in range(1, m + 1)))
    pm = P(60)
    x, y = float(pm[0]), float(pm[1])
    V.report(2, abs(x - float(S(V.ans(2, "4-1", "x")))) < 1e-12 and abs(y - float(S(V.ans(2, "4-1", "y")))) < 1e-12, f"4-1 극한 ({x},{y})")
    # 4-2
    ok = True
    vals = []
    for m in range(2, 12):
        p0, p1, p2 = [sp.Matrix([sp.Rational(P(k)[0].numerator, P(k)[0].denominator), sp.Rational(P(k)[1].numerator, P(k)[1].denominator)]) for k in (m - 1, m, m + 1)]
        t = sp.Symbol("t")
        r = sp.solve(list(p0 + t * (p2 - p0) - p1), t)
        tv = r[t]
        vals.append(tv / (1 - tv))
    V.report(2, all(v == S(V.ans(2, "4-2")) for v in vals) and all(0 < t for t in vals), f"4-2 α/β {set(vals)}")
    # 4-3
    xx, yy = sp.symbols("xx yy")
    Q = sp.Matrix([xx, yy])
    Aa = sp.Matrix([1, 0])
    Pp = sp.Matrix([1, 2])
    v = (Aa - Q) + (Pp - Q)
    eq = sp.expand(v.dot(v) - 1)
    ans = V.ans(2, "4-3").replace("x", "xx").replace("y", "yy")
    lhs, rhs = ans.split("=")
    V.report(2, sp.expand(sp.expand(S(lhs) - S(rhs)) * 4 - eq) == 0, f"4-3 원 {sp.factor(eq)}")
    # 4-4
    def cosR(m):
        r = lambda k: np.array([1 - 2.0**-k, 2.0**(1 - k) - 1])
        a_, b_ = r(m), r(m + 1)
        return abs(np.dot(a_, b_)) / (np.linalg.norm(a_) * np.linalg.norm(b_)), np.dot(a_, b_)
    cs = [(cosR(m)[0], m) for m in range(1, 60)]
    acute = all(cosR(m)[1] > 0 for m in range(1, 60))
    mn = min(cs)
    V.report(2, acute and abs(mn[0] - float(S(V.ans(2, "4-4")))) < 1e-9 and mn[1] == 1, f"4-4 최소 {mn} 예각 {acute}")


if __name__ == "__main__":
    V.run(globals())
