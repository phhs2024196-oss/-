#!/usr/bin/env python3
"""SSWU-M21 (성신여대 2021학년도 자연계열 모의논술 문제 1~4) 검산. 기대값은 V.ans() 로 YAML 에서 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("SSWU-M21", "items/SSWU/PAST")
a_, n_, h_ = sp.symbols("a n h", positive=True)


def ev(s, **kw):
    return sp.sympify(s).subs({sp.Symbol(k): v for k, v in kw.items()})


def Mbrute(a):
    a = sp.nsimplify(a)
    f = lambda t: t ** 2 - 2 * a * t + 2 * a
    cand = [1, 2] + ([a] if 1 <= a <= 2 else [])
    return max(abs(f(c)) for c in cand)


def p01():
    ok1 = all(sp.simplify(Mbrute(a) - ev(V.ans(1, "1"), a=a)) == 0 for a in [-3, -1, 0, sp.Rational(1, 2), 1])
    V.report(1, ok1, "1-1 a<=1")
    m1, m2, m3 = (V.ans(1, "2", k) for k in ("M1", "M2", "M3"))
    def Mf(a):
        return ev(m1, a=a) if a <= sp.Rational(3, 2) else (ev(m2, a=a) if a <= sp.Rational(5, 2) else ev(m3, a=a))
    pts = [sp.Rational(i, 20) for i in range(21, 120)]
    V.report(1, all(sp.simplify(Mbrute(a) - Mf(a)) == 0 for a in pts), "1-2 구간 전수(a=1.05~5.95)")
    nn = sp.Symbol("n")
    for n in range(10, 40):
        tot = sum(Mbrute(k - 7) for k in range(1, n + 1))
        if tot != ev(V.ans(1, "3"), n=n):
            V.report(1, False, f"1-3 n={n}"); return
    V.report(1, True, "1-3 n=10..39")


def p02():
    a, h = a_, h_
    c = sp.sympify(V.ans(2, "1"), locals={"a": a, "h": h})
    f = sp.sqrt
    # 평균값 정리: f'(c)=기울기, 0<a<c<b
    ok = True
    for av, hv in [(1, 3), (2, 1), (sp.Rational(1, 3), 5), (7, sp.Rational(1, 2))]:
        cv = c.subs({a: av, h: hv})
        ok &= sp.simplify(1 / (2 * sp.sqrt(cv)) - (f(av + hv) - f(av)) / hv) == 0 and av < cv < av + hv
    V.report(2, ok, "2-1 f'(c)=기울기, a<c<b")
    V.check(2, "2-2", sp.limit((c - a) / h, h, 0, "+"), V.ans(2, "2"))
    xx = sp.Symbol("xx", positive=True)
    b = 4 * a
    cc = sp.simplify(c.subs(h, b - a))
    S = sp.integrate(sp.sqrt(xx), (xx, a, b))
    S1 = (f(a) + f(cc)) / 2 * (cc - a); S2 = (f(cc) + f(b)) / 2 * (b - cc)
    r = sp.simplify(S / (S1 + S2))
    p, q = [int(t) for t in V.ans(2, "3").split(":")]
    V.check(2, "2-3 c=9a/4", cc, sp.Rational(9, 4) * a)
    V.check(2, "2-3 비", r, sp.Rational(p, q))


def p03():
    mp.mp.dps = 40
    ang = lambda d: mp.mpc(mp.cos(mp.radians(d)), mp.sin(mp.radians(d)))
    P0 = [ang(240 + 60 * i) for i in range(6)]  # A0..F0 반시계
    def inter(p1, p2, p3, p4):
        d1, d2 = p2 - p1, p4 - p3
        M = mp.matrix([[d1.real, -d2.real], [d1.imag, -d2.imag]])
        rhs = mp.matrix([(p3 - p1).real, (p3 - p1).imag])
        t = mp.lu_solve(M, rhs)[0]
        return p1 + t * d1
    def nxt(P):
        A, B, C, D, E, F = P
        return [inter(A, C, B, D), inter(B, D, C, E), inter(C, E, D, F), inter(D, F, E, A), inter(E, A, F, B), inter(F, B, A, C)]
    A0, B0, C0, D0, E0, F0 = P0
    d = lambda p, q: abs(p - q)
    exp = dict(kv.split("=") for kv in V.ans(3, "1").replace(" ", "").split(";"))
    ok = abs(d(A0, B0) - mp.mpf(str(sp.N(sp.sympify(exp["A0B0"]), 30)))) < 1e-25
    ok &= abs(d(A0, C0) - mp.mpf(str(sp.N(sp.sympify(exp["A0C0"]), 30)))) < 1e-25
    ok &= abs(d(A0, D0) - mp.mpf(str(sp.N(sp.sympify(exp["A0D0"]), 30)))) < 1e-25
    tri = abs(((C0 - A0) * mp.conj(E0 - A0)).imag) / 2
    ok &= abs(tri - mp.mpf(str(sp.N(sp.sympify(exp["area"]), 30)))) < 1e-25
    V.report(3, ok, "3-1 길이·넓이")
    hexa = lambda P: sum(((P[i] * mp.conj(P[(i + 1) % 6])).imag for i in range(6))) / 2
    levels = [P0]
    for _ in range(40):
        levels.append(nxt(levels[-1]))
    diff = abs(hexa(levels[0])) - abs(hexa(levels[1]))
    V.report(3, abs(diff - mp.mpf(str(sp.N(sp.sympify(V.ans(3, "2")), 30)))) < 1e-25, f"3-2 넓이 차 {mp.nstr(diff,10)}")
    s = mp.fsum(d(L[0], levels[i + 1][0]) for i, L in enumerate(levels[:-1]))
    V.report(3, abs(s - mp.mpf(str(sp.N(sp.sympify(V.ans(3, "3")), 30)))) < 1e-8, f"3-3 합(41항) {mp.nstr(s,12)}")
    # 규칙 A_kA_{k+1}=2/sqrt3^(k+1)
    V.report(3, all(abs(d(levels[k][0], levels[k + 1][0]) - 2 / mp.sqrt(3) ** (k + 1)) < 1e-25 for k in range(30)), "3-3 일반항")


def p04():
    def edges(n):
        H = lambda x, y: 0 <= x < n and 0 <= y <= n
        Vv = lambda x, y: (0 <= y < n) if x in (0, n) else (0 < x < n and y + 1 <= n - x)
        return H, Vv
    def sq(n):
        H, Vv = edges(n); out = {}
        for s in range(1, n + 1):
            out[s] = sum(1 for i in range(n - s + 1) for j in range(n - s + 1)
                         if all(H(i + t, j) and H(i + t, j + s) and Vv(i, j + t) and Vv(i + s, j + t) for t in range(s)))
        return out
    # 4-1 오른쪽/위 이동 최단 경로 수
    ok = True
    for n in range(1, 11):
        H, Vv = edges(n)
        w = {(0, 0): 1}
        for tot in range(1, 2 * n + 1):
            for x in range(0, n + 1):
                y = tot - x
                if 0 <= y <= n:
                    w[(x, y)] = (w.get((x - 1, y), 0) if H(x - 1, y) and x > 0 else 0) + (w.get((x, y - 1), 0) if Vv(x, y - 1) and y > 0 else 0)
        ok &= w[(n, n)] == ev(V.ans(4, "1"), n=n)
    V.report(4, ok, "4-1 n=1..10")
    s6 = sq(6)
    exp = dict(kv.split("=") for kv in V.ans(4, "2").replace(" ", "").split(";"))
    V.report(4, all(s6[k] == int(exp[f"a{k}"]) for k in range(1, 7)), f"4-2 n=6 {list(s6.values())}")
    ev_, od_ = (V.ans(4, "3", k) for k in ("even", "odd"))
    ok = True
    for n in range(1, 21):
        tot = sum(sq(n).values())
        ok &= tot == ev(ev_ if n % 2 == 0 else od_, n=n)
    V.report(4, ok, "4-3 n=1..20 전수 정사각형 개수(홀짝)")
    # 보충: 홀수 n 의 a_k=(n-2k+1)(n-2k+2)/2+1 (k<=m)
    okk = all(sq(n)[k] == (n - 2 * k + 1) * (n - 2 * k + 2) // 2 + 1 for n in (5, 7, 9) for k in range(1, n // 2 + 1))
    V.report(4, okk, "4-3 홀수 n 의 a_k 식(보충)")


if __name__ == "__main__":
    V.run(globals())
