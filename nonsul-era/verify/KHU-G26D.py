#!/usr/bin/env python3
"""경희대 2026학년도 논술(의·약학계 수학) KHU-G26D-01~02 — sympy 검산. 정답 하드코딩 비교 금지, 문제 설정에서 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KHU-G26D"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I: 직사각형 ABCD(AB=2, AD=2√3)를 EB, EC 로 접어 A, D 가 P 에서 만남 — 좌표로 P 를 구함."""
    r3 = sp.sqrt(3)
    B = sp.Matrix([0, 0, 0]); C = sp.Matrix([2*r3, 0, 0]); E = sp.Matrix([r3, 2, 0])
    px, py, pz = sp.symbols("px py pz", real=True)
    P = sp.Matrix([px, py, pz])
    d2 = lambda U, V: (U - V).dot(U - V)
    sols = sp.solve([d2(P, E) - 3, d2(P, B) - 4, d2(P, C) - 4], [px, py, pz], dict=True)
    sol = [s for s in sols if s[pz] > 0][0]
    P = P.subs(sol)
    # 이면각: EB 위 수선의 발 H, 반평면 PEB 와 반평면 EBC(C 쪽) 의 방향
    u = (B - E) / sp.sqrt(d2(B, E))
    H = E + u*(P - E).dot(u)
    v1 = P - H
    w = C - E; v2 = w - u*w.dot(u)
    cos_t = sp.nsimplify(sp.simplify(v1.dot(v2)/sp.sqrt(v1.dot(v1)*v2.dot(v2))))
    # (2) 넓이·부피
    area = sp.sqrt(((B - P).cross(C - P)).dot((B - P).cross(C - P)))/2
    vol = abs((B - E).cross(C - E).dot(P - E))/6
    ok = cos_t == sp.Rational(3, 4) and sp.simplify(area - r3) == 0 and sp.simplify(vol - 1) == 0
    report(1, ok, f"P={list(P)}, cosθ={cos_t}, △PBC={sp.simplify(area)}, V={sp.simplify(vol)}")


def p02():
    """논제 II: f(x)=a(bx+1) [0,1], -a(b+1)(x-2) (1,2]."""
    x, a, b, d = sp.symbols("x a b d", real=True)
    f1 = a*(b*x + 1); f2 = -a*(b + 1)*(x - 2)
    asol = sp.solve(sp.integrate(f1, (x, 0, 1)) + sp.integrate(f2, (x, 1, 2)) - 1, a)[0]
    ok = sp.simplify(asol - 2/(2*b + 3)) == 0
    c = sp.simplify((sp.integrate(x**2*f1, (x, 0, 1)) + sp.integrate(x**2*f2, (x, 1, 2))).subs(a, asol))
    # f>=0 인 b 의 범위: 격자로 확인 (b<-1 이면 음수 구간 존재)
    def valid(bv):
        av = asol.subs(b, bv)
        if av <= 0:
            return False
        return all(f1.subs({a: av, b: bv, x: sp.Rational(k, 20)}) >= 0 for k in range(21)) and \
            all(f2.subs({a: av, b: bv, x: 1 + sp.Rational(k, 20)}) >= 0 for k in range(1, 21))
    ok &= valid(-1) and not valid(sp.Rational(-11, 10)) and valid(5)
    lo = c.subs(b, -1); hi = sp.limit(c, b, sp.oo)
    mono = sp.simplify(sp.diff(c, b)) 
    ok &= lo == sp.Rational(1, 6) and hi == sp.Rational(7, 6) and all(mono.subs(b, bv) > 0 for bv in [-1, 0, 3, 100])
    # (2) g(b)
    def g(bv):
        av = asol.subs(b, bv)
        F = lambda dd: (sp.integrate(f1, (x, 0, sp.Min(dd, 1))) if True else 0)
        if sp.integrate(f1.subs({a: av, b: bv}), (x, 0, 1)) >= sp.Rational(3, 5):
            sols = sp.solve(sp.integrate(f1.subs({a: av, b: bv}), (x, 0, d)) - sp.Rational(3, 5), d)
            return [s for s in sols if s.is_real and 0 < s <= 1][0]
        sols = sp.solve(sp.integrate(f1.subs({a: av, b: bv}), (x, 0, 1)) + sp.integrate(f2.subs({a: av, b: bv}), (x, 1, d)) - sp.Rational(3, 5), d)
        return [s for s in sols if s.is_real and 1 < s <= 2][0]
    val = sp.nsimplify(sp.simplify(g(sp.Rational(1, 2)) + g(39)))
    ok &= sp.simplify(val - (2*sp.sqrt(sp.Rational(11, 5)) - sp.Rational(9, 10))) == 0
    report(2, ok, f"a=2/(2b+3), c={c}, 범위 [{lo},{hi}); g(1/2)+g(39)={val}")


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
