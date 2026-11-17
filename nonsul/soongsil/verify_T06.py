#!/usr/bin/env python3
"""S06 치환적분·부분적분과 넓이 — 검산 스크립트"""
import sys
import numpy as np
import sympy as sp
import mpmath as mp

x, t, y, a = sp.symbols('x t y a', real=True)
n = sp.symbols('n', integer=True, positive=True)
mp.mp.dps = 30
results = []


def check(pid, cond, msg):
    results.append(bool(cond))
    print(f"{pid} {'PASS' if cond else 'FAIL'}  {msg}")


def close(u, v, rel=1e-9):
    u, v = mp.mpf(u), mp.mpf(sp.N(v, 40))
    return abs(u - v) <= rel * max(1, abs(v))


# S06-01: ∫_{-2}^{2} (x^2+1)/(1+e^{x^3}) dx
def p01():
    f = (x ** 2 + 1) / (1 + sp.exp(x ** 3))
    ok = sp.simplify(f + f.subs(x, -x) - (x ** 2 + 1)) == 0
    ans = sp.integrate(x ** 2 + 1, (x, 0, 2))
    num = mp.quad(lambda s: (s ** 2 + 1) / (1 + mp.e ** (s ** 3)), [-2, 0, 2])
    ok &= ans == sp.Rational(14, 3) and close(num, ans)
    check("S06-01", ok, f"f(x)+f(-x)=x^2+1, 값={ans} (수치 {mp.nstr(num, 12)})")


# S06-02: f(x)e^{f(x)}=x 일 때 ∫_0^e f(x) dx
def p02():
    inv = y * sp.exp(y)
    ok = inv.subs(y, 0) == 0 and inv.subs(y, 1) == sp.E
    ok &= sp.simplify(sp.diff(inv, y) - (y + 1) * sp.exp(y)) == 0  # y>=0 에서 증가
    ans = sp.E * 1 - sp.integrate(inv, (y, 0, 1))
    num = mp.quad(lambda s: mp.lambertw(s).real, [0, mp.e])  # 검산 전용 수치
    ok &= sp.simplify(ans - (sp.E - 1)) == 0 and close(num, ans)
    check("S06-02", ok, f"e·1-∫_0^1 ye^y dy = {sp.simplify(ans)} (수치 {mp.nstr(num, 12)})")


# S06-03: ∫_0^{π/2} cos^3 x/(sin x+cos x) dx
def p03():
    K = mp.quad(lambda s: mp.cos(s) ** 3 / (mp.sin(s) + mp.cos(s)), [0, mp.pi / 2])
    Kp = mp.quad(lambda s: mp.sin(s) ** 3 / (mp.sin(s) + mp.cos(s)), [0, mp.pi / 2])
    tot = sp.integrate(1 - sp.sin(x) * sp.cos(x), (x, 0, sp.pi / 2))
    ans = (sp.pi - 1) / 4
    ok = sp.simplify(tot - 2 * ans) == 0 and close(K, ans) and close(Kp, ans)
    check("S06-03", ok, f"K=K'=(π-1)/4={sp.N(ans, 12)}, 오답 π/4={sp.N(sp.pi / 4, 6)}")


# S06-04: y=a sin x 가 y=sin2x 아래 영역을 이등분
def p04():
    al = sp.symbols('al', positive=True)
    upper = sp.integrate(sp.sin(2 * x) - 2 * sp.cos(al) * sp.sin(x), (x, 0, al))
    ok = sp.simplify(upper - (1 - sp.cos(al)) ** 2) == 0
    total = sp.integrate(sp.sin(2 * x), (x, 0, sp.pi / 2))
    ok &= total == 1
    av = 2 - sp.sqrt(2)
    alpha = mp.acos(mp.mpf(sp.N(av / 2, 40)))
    part = mp.quad(lambda s: mp.sin(2 * s) - mp.mpf(sp.N(av, 40)) * mp.sin(s), [0, alpha])
    ok &= close(part, sp.Rational(1, 2))
    check("S06-04", ok, f"위쪽 넓이=(1-cosα)^2=1/2 → a=2-√2={sp.N(av, 10)}")


# S06-05: I_n=∫_1^e (ln x)^n dx, lim n I_n
def p05():
    I = lambda m: sp.integrate(sp.log(x) ** m, (x, 1, sp.E))
    ok = all(sp.simplify(I(m) + m * I(m - 1) - sp.E) == 0 for m in range(1, 7))
    vals = []
    for m in [10, 100, 1000, 10000]:
        Im = mp.quad(lambda s: s ** m * mp.e ** s, [0, 1 - mp.mpf(10) / m, 1])
        lo, hi = mp.e * m / (m + 2), mp.e * m / (m + 1)
        ok &= lo <= m * Im <= hi
        vals.append(m * Im)
    ok &= abs(vals[-1] - mp.e) < 1e-3
    check("S06-05", ok, f"I_n+nI_(n-1)=e, en/(n+2)<=nI_n<=en/(n+1), 10000·I=={mp.nstr(vals[-1], 8)} → e")


# S06-06: f=(e^x-1)/(e-1) 와 역함수로 둘러싸인 넓이
def p06():
    # S06-06: y=ln(1+x^2) 와 두 변곡점을 지나는 직선으로 둘러싸인 넓이
    f = sp.log(1 + x ** 2)
    infl = sorted(sp.solve(sp.diff(f, x, 2), x))
    P, Q = [(r, f.subs(x, r)) for r in infl]
    line = P[1] + (Q[1] - P[1]) / (Q[0] - P[0]) * (x - P[0])
    pts = sorted(sp.solve(sp.Eq(f, line), x))
    ans = sp.integrate(line - f, (x, pts[0], pts[-1]))
    ok = infl == [-1, 1] and sp.simplify(line - sp.log(2)) == 0 and sp.simplify(ans - (4 - sp.pi)) == 0
    num = mp.quad(lambda s: mp.log(2) - mp.log(1 + s * s), [-1, 1])
    ok &= close(num, 4 - sp.pi)
    # 대표 오답: ∫_0^1 1/(1+x^2) 를 ln2 로 착각 -> 넓이 4-4ln2
    wrong = 2 * (sp.log(2) - (sp.log(2) - 2 * (1 - sp.log(2))))
    ok &= sp.simplify(wrong - (4 - 4 * sp.log(2))) == 0
    check("S06-06", ok, f"변곡점 x=±1, 직선 y=ln2, 넓이=4-π={sp.N(ans, 10)}, 오답 {sp.N(wrong, 6)}")


# S06-07: y=e^x 의 접선과 곡선, x=0, x=2 사이 넓이 최소
def p07():
    tan = sp.exp(t) + sp.exp(t) * (x - t)
    S = sp.integrate(sp.exp(x) - tan, (x, 0, 2))
    ts = sp.solve(sp.diff(S, t), t)
    Smin = sp.simplify(S.subs(t, ts[0]))
    ok = ts == [1] and sp.simplify(Smin - (sp.E ** 2 - 2 * sp.E - 1)) == 0
    ok &= sp.diff(S, t, 2).subs(t, 1) > 0
    check("S06-07", ok, f"S(t)'=0 → t={ts}, 최솟값 e^2-2e-1={sp.N(Smin, 10)}")


# S06-08: 단면 부피 V(반원, x축 수직), W(정사각형, y축 수직)
def p08():
    V = sp.integrate(sp.pi / 8 * sp.log(x) ** 2, (x, 1, sp.E))
    W = sp.integrate((sp.E - sp.exp(y)) ** 2, (y, 0, 1))
    V2 = sp.pi / 8 * sp.integrate(2 * y * (sp.E - sp.exp(y)), (y, 0, 1))  # 이중세기
    ok = sp.simplify(V - sp.pi * (sp.E - 2) / 8) == 0
    ok &= sp.simplify(W - (4 * sp.E - sp.E ** 2 - 1) / 2) == 0
    ok &= sp.simplify(V - V2) == 0
    check("S06-08", ok, f"V=π(e-2)/8={sp.N(V, 8)}, W=(4e-e^2-1)/2={sp.N(W, 8)}")


# S06-09: ∫_0^π x sin x/(1+sin x) dx
def p09():
    J = mp.quad(lambda s: s * mp.sin(s) / (1 + mp.sin(s)), [0, mp.pi / 2, mp.pi])
    half = sp.integrate(sp.Rational(1, 2) / sp.cos(x / 2) ** 2, (x, 0, sp.pi / 2))
    ok = sp.simplify(half - 1) == 0
    ans = sp.pi ** 2 / 2 - sp.pi
    ok &= close(J, ans)
    wrong = sp.pi * (sp.pi - 2)
    check("S06-09", ok, f"J=π^2/2-π={sp.N(ans, 12)} (수치 {mp.nstr(J, 12)}), 오답 π(π-2)={sp.N(wrong, 6)}")


# S06-10: ∫_0^{π/2} e^x (1+sin x)/(1+cos x) dx
def p10():
    F = sp.exp(x) * sp.tan(x / 2)
    d = sp.lambdify(x, sp.diff(F, x) - sp.exp(x) * (1 + sp.sin(x)) / (1 + sp.cos(x)), 'mpmath')
    ok = all(abs(d(mp.mpf(s))) < 1e-25 for s in np.random.default_rng(1).uniform(0, 1.5, 20))
    ans = F.subs(x, sp.pi / 2) - F.subs(x, 0)
    num = mp.quad(lambda s: mp.e ** s * (1 + mp.sin(s)) / (1 + mp.cos(s)), [0, mp.pi / 2])
    ok &= sp.simplify(ans - sp.exp(sp.pi / 2)) == 0 and close(num, ans)
    check("S06-10", ok, f"(e^x tan(x/2))'=피적분함수, 값 e^(π/2)={sp.N(ans, 12)}")


for p in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        p()
    except Exception as e:  # noqa
        results.append(False)
        print(f"{p.__name__} FAIL  exception {e!r}")

npass = sum(results)
print(f"SUMMARY T06: {npass}/10 PASS")
if npass != 10:
    sys.exit(1)
