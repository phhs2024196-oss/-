#!/usr/bin/env python3
"""CUK-G23A (가톨릭대 2023 수시 논술 자연·공학/간호 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
V = Verifier("CUK-G23A", "items/CUK/PAST")


def p01():
    X, kk = sp.symbols("X kk")
    f = 2*X**3 - 2*(kk+2)*X**2 + (kk**2+4*kk-6)*X - 2*kk**2 + 12
    al, be = sp.symbols("al be", real=True)
    z = al + be*sp.I
    w = 3*z - 1
    eq = sp.expand(sp.conjugate(w) - (2*z - sp.Rational(5, 3)*sp.I))
    s = sp.solve([sp.re(eq), sp.im(eq)], [al, be], dict=True)[0]
    wv = sp.simplify(w.subs(s))
    kv = [r for r in sp.solve(sp.expand(f.subs(X, wv)), kk) if r.is_real]
    V.check(1, "k", kv, [sp.sympify(V.ans(1, "1").split("=")[1])])
    k4 = f.subs(kk, 4)
    V.report(1, sp.simplify(k4.subs(X, wv)) == 0 and any(abs(sp.im(sp.N(r))) > 1e-9 for r in sp.Poly(k4, X).nroots()), "k=4에서 3z-1=2+i가 허근")
    fp = sp.diff(k4, X)
    # S: 모든 x에 대하여 |ax^2-2(a+3)x+11| <= f'(x)/2 인 a 의 집합 (격자 + 경계)
    xs = np.linspace(-30, 30, 120001)
    fpn = sp.lambdify(X, fp, "numpy")(xs)
    okset = []
    for a in np.arange(-8, 8.001, 0.25):
        okk = np.all(np.abs(a*xs**2 - 2*(a+3)*xs + 11) <= fpn/2 + 1e-9)
        if okk:
            okset.append(float(a))
    V.check(1, "S(격자 a 중 성립)", okset, [float(sp.sympify(V.ans(1, "2").split("=")[1].strip("{}")))])
    a_ = sp.symbols("a_")
    V.report(1, sp.simplify(sp.diff(k4, X) - (6*(X-2)**2 + 2)) == 0, "f'(x)=6(x-2)^2+2>0")


def p02():
    X = sp.symbols("X")
    c3, c2, c1, c0 = sp.symbols("c3 c2 c1 c0")
    f = X**4 + c3*X**3 + c2*X**2 + c1*X + c0
    t = sp.symbols("t")
    cond = sp.expand(sp.integrate(f.subs(X, 5+t), (t, 0, X)) - sp.integrate(f.subs(X, 5-t), (t, 0, X)))
    eqs = sp.Poly(cond, X).coeffs() + [f.subs(X, 0), sp.diff(f, X).subs(X, 10)]
    sol = sp.solve(eqs, [c3, c2, c1, c0], dict=True)
    V.report(2, len(sol) == 1, "f 유일")
    fx = sp.expand(f.subs(sol[0]))
    V.check(2, "f=x^2(x-10)^2", sp.expand(fx - X**2*(X-10)**2), 0)
    nposroots = lambda kv: len([r for r in sp.Poly(fx - kv, X).nroots(n=30) if abs(sp.im(r)) < 1e-12 and sp.re(r) > 1e-12])
    Mv = sp.sympify(V.ans(2, "1").split("=")[1])
    ok = all(nposroots(kv) == 3 for kv in (1, Mv/2, Mv-1)) and all(nposroots(kv) == 1 for kv in (Mv+1, 2*Mv)) \
        and max(fx.subs(X, c) for c in sp.solve(sp.diff(fx, X), X)) == Mv
    V.report(2, ok, "n(k)=3 (0<k<M), 1 (k>M) 및 M=극댓값")
    # 논제 2: 양의 세 근 a<b<c 에서 c-a=sqrt70 인 k 를 이분법으로
    def gap(kv_):
        r = sorted(z.real for z in np.roots([1, -20, 100, 0, -kv_]) if abs(z.imag) < 1e-9 and z.real > 0)
        return r[-1] - r[0] - np.sqrt(70)
    lo, hi = 1e-6, float(Mv) - 1e-6
    for _ in range(200):
        mid = (lo+hi)/2
        if gap(mid) > 0: lo = mid
        else: hi = mid
    kk = (lo+hi)/2
    V.check(2, "f(a)=k", sp.Integer(round(kk)), sp.sympify(V.ans(2, "2").split("=")[1]))
    V.report(2, abs(kk - round(kk)) < 1e-6, "k 이분법 정수 일치")


def p03():
    A = sp.Point(-5, 0); B = sp.Point(-4, 3)
    cx, cy = sp.symbols("cx cy", real=True)
    pts = sp.solve([cx**2 + cy**2 - 9, -5*cx - 9], [cx, cy], dict=True)   # 접선 cx*x+cy*y=9 가 A 통과
    P = [sp.Point(s[cx], s[cy]) for s in pts]
    C, D = sorted(P, key=lambda q: q.distance(B))
    area = lambda U, W, Z: abs(sp.Polygon(U, W, Z).area)
    S = max(area(A, B, C), area(A, B, D))
    V.check(3, "S", S, sp.sympify(V.ans(3, "1").split("=")[1]))
    # 논제 2: 원 x^2+y^2=18 위 현 PQ=6 의 넓이 최댓값 (각도 스캔)
    best = 0
    r = np.sqrt(18)
    half = np.arccos(3/r)          # 현 6 → 중심각 2*half
    thetas = np.linspace(0, 2*np.pi, 2000001)
    Pp = np.stack([r*np.cos(thetas-half), r*np.sin(thetas-half)], 1)
    Qq = np.stack([r*np.cos(thetas+half), r*np.sin(thetas+half)], 1)
    ar = 0.5*np.abs((Pp[:, 0]+5)*(Qq[:, 1]) - (Qq[:, 0]+5)*(Pp[:, 1]))
    ok = ar > 1e-9
    mx = ar[ok].max()
    V.report(3, abs(mx - float(sp.sympify(V.ans(3, "2").split("=")[1]))) < 1e-4, f"M 수치 {mx:.6f}")
    V.check(3, "M", sp.Integer(round(mx)), sp.sympify(V.ans(3, "2").split("=")[1]))


if __name__ == "__main__":
    V.run(globals())
