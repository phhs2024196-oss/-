#!/usr/bin/env python3
"""KU-M25A (고려대 2023.8.11 시행 모의논술, 2025학년도 대비 자연계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산 후 YAML 정답과 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction

import mpmath as mp
import sympy as sp

PREFIX = "KU-M25A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    h = sp.symbols("h", positive=True)
    p, c, d = sp.symbols("p c d", real=True)
    f = p * x**2 + c * x + d
    # (가)(i): lim f(h)(ln(h+1)-ln h)/h
    L1 = sp.limit((f.subs(x, h) * sp.log(1 + 1 / h) / h).subs({c: 0, d: 0}), h, sp.oo)   # = p
    psol = sp.solve(sp.Eq(L1, 1), p)[0]
    f = f.subs(p, psol)
    # (가)(ii): 적분 후 h->0 급수 전개; 발산 방지 → 음수 차수 계수 0, 상수항 = 3
    I = sp.integrate(f, (x, h, h + sp.sin(h))) / h**2
    ser = sp.series(I, h, 0, 1).removeO()
    eqs = [ser.coeff(h, -1), ser.coeff(h, 0) - 3]
    sol = sp.solve(eqs, [c, d], dict=True)[0]
    fx = sp.expand(f.subs(sol))
    ok1 = sp.simplify(fx - sp.sympify(md_answer(1, "1"), locals={"x": x})) == 0
    # (나): 넓이 → β-α ; (다): 근과 계수
    al, a, b = sp.symbols("alpha a b", real=True)
    for D in sp.solve(sp.Eq(sp.integrate(-(x - al) * (x - al - sp.Symbol("D")), (x, al, al + sp.Symbol("D"))), sp.Rational(1, 6)), sp.Symbol("D")):
        if D.is_real and D > 0:
            dif = D
    be = al + dif
    par = -(x - a)**2 + b
    sol2 = sp.solve([(fx - par).subs(x, al), (fx - par).subs(x, be)], [al, b], dict=True)
    bexpr = sp.simplify(sol2[0][b])
    md_b = sp.sympify(md_answer(1, "2"), locals={"a": a})
    ok2 = sp.simplify(bexpr - md_b) == 0 and all(sp.simplify(s[b] - md_b) == 0 for s in sol2)
    # 넓이 직접 확인(직선 k 사용) 수치
    av = sp.Rational(3, 7); alv = sp.solve((fx - par).subs({a: av, b: md_b.subs(a, av)}), x)
    alv = sorted(alv, key=lambda r: float(r))
    k = fx.subs(x, alv[0]) + (fx.subs(x, alv[1]) - fx.subs(x, alv[0])) / (alv[1] - alv[0]) * (x - alv[0])
    area = sp.simplify(sp.integrate(k - fx, (x, alv[0], alv[1])))
    ok3 = area == sp.Rational(1, 6)
    report(1, ok1 and ok2 and ok3, f"f={fx}, beta-alpha={dif}, b={bexpr}, area(a=3/7)={area}")


def p02():
    a = sp.symbols("a", real=True)
    f = x**3 - x
    m, q = sp.symbols("m q")
    g = m * x + q
    s = sp.solve([sp.Eq(f.subs(x, a) - g.subs(x, a), a), sp.Eq(sp.diff(f, x).subs(x, a) - m, 0)], [m, q], dict=True)[0]
    k = sp.expand(f - g.subs(s))
    # (2-1)(2-2) [증명-수치확인]: a 격자에서 실근 1개, -a h(a) > a^2
    ok1 = ok2 = True
    grid = [sp.Rational(i, 20) for i in range(-200, 201)]
    for av in grid:
        poly = sp.Poly(k.subs(a, av), x)
        if len(set(sp.real_roots(poly))) != 1:          # 서로 다른 실근 개수(정확 계산)
            ok1 = False
            break
        if av != 0:
            r = poly.count_roots(-sp.oo, -av)             # 근이 (-inf, -a] 에 있는 개수
            r_le = r == 1
            # a>0 이면 h<-a, a<0 이면 h>-a 이어야 -a h > a^2 ; k(-a)!=0 확인
            ok2 = ok2 and k.subs({a: av, x: -av}) != 0 and (r_le if av > 0 else not r_le)
    # (2-3) a 크게 → h(a)/a 고정밀 수치
    mp.mp.dps = 50
    vals = []
    for A in (mp.mpf(10)**6, mp.mpf(10)**9):
        kk = sp.lambdify(x, k.subs(a, sp.Integer(int(A))), "mpmath")
        r = mp.findroot(kk, (-3 * A, -A), solver='illinois')   # k(-3A)<0<k(-A)
        vals.append(r / A)
    lim = sp.Float(str(vals[-1]), 30)
    ok3 = abs(lim - sp.sympify(md_answer(2, "3"))) < 1e-12 and abs(vals[0] - vals[1]) < 1e-9
    report(2, ok1 and ok2 and ok3, f"g={sp.factor(g.subs(s))}, k={k}, roots=1 on grid [증명-수치확인], h(a)/a(a=1e9)={mp.nstr(vals[-1], 15)}")


def p03():
    A = sp.Matrix([-1, 0])
    B = [sp.Matrix([sp.cos(sp.pi - j * sp.pi / 4), sp.sin(sp.pi - j * sp.pi / 4)]) for j in range(1, 5)]
    pts = [A] + B
    chords = [sp.simplify((pts[i + 1] - pts[i]).norm()) for i in range(4)]
    okc = all(sp.simplify(ch - chords[0]) == 0 for ch in chords) and B[3] == sp.Matrix([1, 0])
    u = B[0] - A
    t = sp.symbols("t", real=True)
    # 대칭이동 정확값: B3를 직선 A + s u 에 대해 반사
    un = u / u.norm()
    v3 = B[2] - A
    v3r = 2 * (v3.dot(un)) * un - v3
    exact = sp.nsimplify(sp.simplify(((B[1] - A) - v3r).norm()**2))
    # 독립 수치 최소화(격자 + 황금분할)
    mp.mp.dps = 40
    P = [(mp.cos(mp.pi - j * mp.pi / 4) + 1, mp.sin(mp.pi - j * mp.pi / 4)) for j in range(1, 4)]   # A 기준 벡터 AB1..AB3
    F = lambda tv: (mp.hypot(P[1][0] - tv * P[0][0], P[1][1] - tv * P[0][1]) +
                    mp.hypot(P[2][0] - tv * P[0][0], P[2][1] - tv * P[0][1]))**2
    t0 = min((mp.mpf(i) / 1000 for i in range(-3000, 3001)), key=F)
    tm = mp.findroot(lambda s: mp.diff(F, s), t0)
    num = F(tm)
    md = sp.sympify(md_answer(3, "1"))
    ok = okc and sp.simplify(exact - md) == 0 and abs(num - float(md)) < 1e-12
    wrong = sp.simplify((B[1] - B[2]).norm()**2)
    report(3, ok, f"chords equal={okc}, reflect={exact}, numeric min={mp.nstr(num, 15)} at t={mp.nstr(tm, 8)}, 오답(B2B3^2)={wrong}")


def p04():
    pdef = {"A": Fraction(1, 5), "B": Fraction(1, 4), "C": Fraction(1, 10)}
    fail = Fraction(0); afail = Fraction(0)
    for st in itertools.product([True, False], repeat=3):   # True = 정상
        pr = Fraction(1)
        for name, ok in zip("ABC", st):
            pr *= (1 - pdef[name]) if ok else pdef[name]
        works = (st[0] or st[1]) and st[2]
        if not works:
            fail += pr
            if not st[0]:
                afail += pr
    ans = afail / fail
    ok = ans == Fraction(md_answer(4, "1"))
    report(4, ok, f"P(작동X)={fail}, P(A불량∩작동X)={afail}, 조건부={ans}")


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
