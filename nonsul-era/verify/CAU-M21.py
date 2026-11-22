#!/usr/bin/env python3
"""CAU-M21 (2021학년도 중앙대 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "CAU-M21"
WIDTH = 2
RESULTS = []
x, t, n, u = sp.symbols("x t n u", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """보물 사후확률: 그래프의 두 밀도에서 P(X>=2) 를 적분으로 재계산 (답 1/2)."""
    g = sp.Rational(2, 9) * (x - 1)          # 1<=x<=4, g(4)=2/3
    f = sp.Rational(2, 3) - sp.Rational(2, 9) * x   # 0<=x<=3, f(0)=2/3
    ok_pdf = sp.integrate(g, (x, 1, 4)) == 1 and sp.integrate(f, (x, 0, 3)) == 1
    pGB = sp.integrate(g, (x, 2, 4))
    pGBc = sp.integrate(f, (x, 2, 3))
    pB = sp.Rational(1, 9)
    pG = pGB * pB + pGBc * (1 - pB)
    post = pGB * pB / pG
    report(1, ok_pdf and pGB == sp.Rational(8, 9) and pGBc == sp.Rational(1, 9)
           and pG == sp.Rational(16, 81) and post == sp.Rational(1, 2),
           f"P(G|B)={pGB}, P(G|B^c)={pGBc}, P(G)={pG}, P(B|G)={post}")


def p02():
    """(1) 정적분 =2 (기호+수치), (2) f(x)=x sqrt(1-x^2), 기울기 -1 접선 y=-x+3sqrt3/4."""
    ig = (1 + 2 * sp.exp(-x)) / (1 + sp.exp(x) + sp.exp(-x))
    F = x - sp.log(1 + sp.exp(x) + sp.exp(-x))
    ok_anti = sp.simplify(sp.diff(F, x) - ig) == 0
    v = sp.simplify(F.subs(x, 1) - F.subs(x, -1))
    num = mp.quad(lambda s: (1 + 2 * mp.e**(-s)) / (1 + mp.e**s + mp.e**(-s)), [-1, 1])
    ok1 = ok_anti and sp.nsimplify(v) == 2 and abs(num - 2) < 1e-20
    # (2) 2f(cos)+f(sin)=3 sin cos 와 x->pi/2-x 식의 연립
    A, B, s_, c_ = sp.symbols("A B s c")
    sol = sp.solve([2 * A + B - 3 * s_ * c_, A + 2 * B - 3 * s_ * c_], [A, B])  # A=f(cos), B=f(sin)
    fx = x * sp.sqrt(1 - x**2)                     # A=s c, cos=x 에서 sin=sqrt(1-x^2)
    chk = all(abs(float((2 * fx.subs(x, mp.cos(a)) + fx.subs(x, mp.sin(a)) - 3 * mp.sin(a) * mp.cos(a)))) < 1e-12
              for a in [0.1, 0.5, 1.0, 1.4])
    xs = [r for r in sp.solve(sp.Eq(sp.diff(fx, x), -1), x) if r.is_real and 0 <= r <= 1]
    tangents = {sp.nsimplify(sp.simplify(fx.subs(x, r) + r)) for r in xs}   # 절편 = f(r)+r
    ok2 = sol[A] == s_ * c_ and chk and xs == [sp.sqrt(3) / 2] and tangents == {3 * sp.sqrt(3) / 4}
    report(2, ok1 and ok2, f"(1) 적분={v}, 수치={mp.nstr(num, 12)}  (2) 접점 x={xs}, 절편={tangents}")


def p03():
    """(1) 두 원 공통현 길이 2sqrt2/3 를 원 직접 구성으로, (2) lim A(t)/t = 4sqrt2 (최적화 수치)."""
    h, k, r = sp.symbols("h k r", real=True)
    d1 = (h + 2 * k + 1) / sp.sqrt(5)
    d2 = (2 * h + k - 2) / sp.sqrt(5)
    circles = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            sols = sp.solve([sp.Eq(s1 * d1, r), sp.Eq(s2 * d2, r), sp.Eq((h + 2)**2 + (k - 3)**2, r**2)], [h, k, r], dict=True)
            circles += [S for S in sols if S[r].is_real and S[r] > 0]
    circles = {(sp.nsimplify(S[h]), sp.nsimplify(S[k]), sp.nsimplify(S[r])) for S in circles}
    X, Y = sp.symbols("X Y", real=True)
    (h1, k1, r1), (h2, k2, r2) = sorted(circles, key=lambda c: float(c[2]))[:2]
    pts = sp.solve([(X - h1)**2 + (Y - k1)**2 - r1**2, (X - h2)**2 + (Y - k2)**2 - r2**2], [X, Y], dict=True)
    L = sp.simplify(sp.sqrt((pts[0][X] - pts[1][X])**2 + (pts[0][Y] - pts[1][Y])**2))
    ok1 = len(circles) == 2 and sp.simplify(L - 2 * sp.sqrt(2) / 3) == 0
    # (2) a(b)=sqrt2 b/sqrt((b-t)^2-2), 넓이 ab 최소 → A(t)
    b = sp.symbols("b", positive=True)
    tt = sp.symbols("tt", positive=True)
    a_b = sp.sqrt(2) * b / sp.sqrt((b - tt)**2 - 2)
    # 접선 조건 독립 확인: 원 중심 (0,t) 에서 직선 x/a+y/b=1 까지 거리 sqrt2
    tv, bv = 5, 9
    av = a_b.subs({tt: tv, b: bv})
    dist = sp.Abs(tv / bv - 1) / sp.sqrt(1 / av**2 + sp.Rational(1, bv**2))
    ok_tan = abs(float(dist) - float(sp.sqrt(2))) < 1e-12
    bstar = (3 * tt + sp.sqrt(tt**2 + 16)) / 2
    crit = sp.simplify(sp.diff(a_b * b, b).subs(b, bstar))
    vals = []
    for T in (1e3, 1e5, 1e7):
        fobj = lambda bb: mp.sqrt(2) * bb**2 / mp.sqrt((bb - T)**2 - 2)
        lo = T + mp.sqrt(2) + 1e-9
        bm = mp.findroot(lambda bb: mp.diff(fobj, bb), 2 * T)
        vals.append(fobj(bm) / T)
    ok2 = ok_tan and crit == 0 and abs(vals[-1] - 4 * mp.sqrt(2)) < 1e-5
    report(3, ok1 and ok2, f"(1) 원 {len(circles)}개, 공통현={L}  (2) b*에서 미분=0, A(t)/t≈{[mp.nstr(v, 10) for v in vals]}")


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
