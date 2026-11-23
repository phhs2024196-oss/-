#!/usr/bin/env python3
"""KUT-G24A-01~10 (한국기술교육대 2024학년도 수리논술 오전) — sympy/전수 검산. 기대값은 YAML(V.ans)에서 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import sympy as sp

V = Verifier("KUT-G24A", "items/KUT/PAST")


def p01():
    X = sp.symbols("X")
    r1, r2 = sp.solve(X**2 - 8*X + 4, X)          # 두 근 = log2 a, log2 b
    a, b = 2**r1, 2**r2
    val = sp.log(b, a) + sp.log(a, b)
    V.check(1, "log_a b + log_b a", sp.N(val, 30), V.ans(1, "1"))


def p02():
    a, b, c = sp.symbols("a b c")
    f = x**3 + a*x**2 + b*x + c
    sa = sp.solve(sp.limit((f - x**3)/(2*x**2 - x), x, sp.oo) + sp.Rational(1, 2), a)[0]
    f1 = f.subs(a, sa)
    sc = sp.solve(f1.subs(x, 2), c)[0]               # f(2)=0 이어야 극한 존재
    f2 = f1.subs(c, sc)
    sb = sp.solve(sp.limit(sp.cancel(f2/(x - 2)), x, 2) - 10, b)[0]
    F = f2.subs(b, sb)
    V.check(2, "f(1)", F.subs(x, 1), V.ans(2, "1"))


def p03():
    k = sp.symbols("k")
    g = -2 + 3*(x + 1) + k*(x + 1)**2                 # g(-1)=-2, g'(-1)=3 (k 임의)
    assert sp.simplify(sp.limit(sp.cancel((g + 2)/(x**2 + 3*x + 2)), x, -1) - 3) == 0
    f = (2*x**2 + x + 3)*(3*x - 2)
    h = f*g
    V.check(3, "h'(-1)", sp.diff(h, x).subs(x, -1), V.ans(3, "1"))


def p04():
    s = sp.symbols("s", positive=True)                # s = arcsin(-k) ∈ (0, π/2): 해집합 [0,s) ∪ (π-s, π]
    sols = sp.solveset(sp.Eq(sp.tan(sp.pi - 2*s), sp.sqrt(3)), s, sp.Interval.open(0, sp.pi/2))
    assert len(sols) == 1
    s0 = list(sols)[0]
    al, be = s0, sp.pi - s0
    # 격자 확인: k=-sin(s0) 에서 {x: sin(x-π)>k} 의 경계가 α, β
    kk = -sp.sin(s0)
    N = 6000
    xs = [sp.pi*i/N for i in range(N + 1)]
    inside = [float(sp.sin(v - sp.pi)) > float(kk) for v in xs]
    a_f, b_f = float(al), float(be)
    ref = [(float(v) < a_f) or (float(v) > b_f) for v in xs]
    assert sum(1 for p, q in zip(inside, ref) if p != q) <= 2   # 경계 근방 수치 오차만 허용
    V.check(4, "alpha", al, V.ans(4, "1", "alpha"))
    V.check(4, "beta", be, V.ans(4, "1", "beta"))


def p05():
    def S(n):
        r = 3**(sp.Rational(1, n))
        d = sp.Rational(1, 2)*3**(sp.Rational(1, n) + sp.Rational(1, 2))
        th = sp.acos(d/r)                              # 중심각의 절반
        seg = r**2/2*(2*th - sp.sin(2*th))             # 활꼴 넓이
        return sp.pi*r**2 - 2*seg
    V.check(5, "S2/S4", sp.simplify(S(2)/S(4)), V.ans(5, "1"))


def p06():
    a, b, c = sp.symbols("a b c")
    sa = sp.solve(sp.Eq((0 - 1)**2 + a, 0), a)[0]      # 기함수: f(0)=0
    N = x**3 - b*x**2 + c*x - 5
    sc = sp.solve(N.subs(x, 1), c)[0]                  # 분자 0
    Nb = N.subs(c, sc)
    sb = sp.solve(sp.limit(sp.cancel(Nb/(x - 1)), x, 1, "+") - sa, b)[0]   # 우극한 = f(1) = a
    f3 = (N.subs({c: sc}).subs(b, sb)/(x - 1)).subs(x, 3)
    V.check(6, "f(3)", f3, V.ans(6, "1"))


def p07():
    a = sp.symbols("a", positive=True)
    # f 가 길이 3 의 열린구간에서 감소 ⇔ 감소구간 (-2a/3, 0) 의 길이 2a/3 ≥ 3
    amin = sp.solve(sp.Eq(2*a/3, 3), a)[0]
    def ok(av):                                          # 격자 전수: 어떤 k 에 대해 (k,k+3) 에서 f'<=0
        fp = lambda t: 3*t**2 + 2*av*t
        ks = [sp.Rational(i, 100) for i in range(-1000, 100)]
        return any(all(fp(k0 + sp.Rational(j, 20)) <= 0 for j in range(0, 61)) for k0 in ks)
    assert ok(amin) and not ok(amin - sp.Rational(1, 10))
    V.check(7, "a (f(1)=a-2 최소)", amin, V.ans(7, "1"))


def p08():
    a, b = sp.symbols("a b")
    f = x**2 + a*x + b
    sa = sp.solve(f.subs(x, 2) - f.subs(x, -2) - 8, a)[0]
    f = f.subs(a, sa)
    I2 = sp.integrate(f, (x, 0, 2))
    blo = sp.solve(I2 + 2, b)[0]
    bhi = sp.solve(I2 - 2, b)[0]
    J = sp.integrate(f, (x, -2, 1))                     # b 에 대해 증가 → 양 끝점에서 최대·최소
    V.check(8, "max+min", J.subs(b, blo) + J.subs(b, bhi), V.ans(8, "1"))


def p09():
    A = {3*n - 2 for n in range(1, 4000)}
    B = {4*n - 2 for n in range(1, 4000)}
    C = sorted(A & B)
    V.check(9, "9-1 sum c", sum(C[:10]), V.ans(9, "9-1"))
    D = sorted(v for v in A if v not in B and v < 12000)
    V.check(9, "d100", D[99], V.ans(9, "9-2", "d100"))
    V.check(9, "d200", D[199], V.ans(9, "9-2", "d200"))
    V.check(9, "d300", D[299], V.ans(9, "9-2", "d300"))
    V.check(9, "d100+d200+d300", D[99] + D[199] + D[299], V.ans(9, "9-2", "sum"))


def p10():
    a, t = sp.symbols("a t", positive=True)
    # B = (2,2): |OB|=2√2, 예각 π/4, 제1사분면
    Bx, By = sp.symbols("Bx By", positive=True)
    solB = sp.solve([Bx**2 + By**2 - 8, By - Bx], [Bx, By], dict=True)[0]
    assert solB == {Bx: 2, By: 2}
    f = a*x*(x - t)**2 + x/2                           # 접점 A(t, t/2)
    Ax, Ay = t, t/2
    area = sp.Abs(Ax*2 - 2*Ay)/2                       # 삼각형 AOB
    tt = sp.solve(sp.Eq(area, 2), t)
    assert tt == [4]
    aa = sp.solve(f.subs({t: 4}).subs(x, 2) - 2, a)[0]
    F = f.subs({t: 4, a: aa})
    V.check(10, "10-1 f(1)", F.subs(x, 1), V.ans(10, "10-1"))
    h = sp.expand(8*F - 4*x)
    crit = sp.solve(sp.diff(h, x), x)
    vals = sorted(h.subs(x, c) for c in crit)           # [극솟값, 극댓값]
    al, be = vals[0], vals[1]
    # 서로 다른 실근 3개인 k 범위 = (극솟값, 극댓값): 표본 k 로 전수 확인
    def nroots(kv):
        return len(set(sp.Poly(h - kv, x).real_roots()))
    mid = (al + be)/2
    assert nroots(mid) == 3 and nroots(al) == 2 and nroots(be) == 2 and nroots(al - 1) == 1 and nroots(be + 1) == 1
    V.check(10, "alpha", al, V.ans(10, "10-2", "alpha"))
    V.check(10, "beta", be, V.ans(10, "10-2", "beta"))
    V.check(10, "alpha+beta", al + be, V.ans(10, "10-2", "alpha+beta"))


if __name__ == "__main__":
    V.run(globals())
