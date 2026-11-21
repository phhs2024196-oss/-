#!/usr/bin/env python3
"""DGU-G26 (2026학년도 동국대 실전 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-G26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _cos(u, v):
    return sp.simplify(u.dot(v) / (u.norm() * v.norm()))


def p01():
    """정팔면체·정사면체(모서리 1) 이면각 (1)(2)(3) 과 (4) 붙였을 때 공면 [증명-수치확인]."""
    s = 1 / sp.sqrt(2)
    V = sp.Matrix
    A, F = V([0, 0, s]), V([0, 0, -s])
    B, C, D, E = V([-s, 0, 0]), V([0, -s, 0]), V([s, 0, 0]), V([0, s, 0])
    assert all(sp.simplify((P - Q).norm() - 1) == 0 for P, Q in [(A, B), (B, C), (C, D), (D, E), (E, B), (F, C), (A, D)])
    M = (B + C) / 2
    # (1) 교선 BC 에 수직인 두 반직선 MA, MF 사이 각 (둔각 그대로)
    ca = _cos(A - M, F - M)
    # (2) 두 평면의 법선으로 예각 이면각
    n1 = (B - A).cross(C - A)
    n2 = (E - A).cross(D - A)
    cb = sp.Abs(_cos(n1, n2))
    # (3) 정사면체 JGHI: 정삼각형 GHI 와 꼭짓점 J
    G, H, I = V([0, 0, 0]), V([1, 0, 0]), V([sp.Rational(1, 2), sp.sqrt(3) / 2, 0])
    J = (G + H + I) / 3 + V([0, 0, sp.sqrt(sp.Rational(2, 3))])
    assert sp.simplify((J - G).norm() - 1) == 0
    O = (G + H) / 2
    cg = _cos(J - O, I - O)
    # (4) G=B, H=C, I=F 로 붙임: 꼭짓점 J 는 면 BCF 의 바깥쪽(원점 반대쪽)
    cen = (B + C + F) / 3
    nrm = (C - B).cross(F - B)
    nrm = nrm / nrm.norm()
    if nrm.dot(cen) < 0:
        nrm = -nrm
    J2 = sp.simplify(cen + sp.sqrt(sp.Rational(2, 3)) * nrm)
    edges_ok = all(sp.simplify((J2 - P).norm() - 1) == 0 for P in (B, C, F))
    coplanar = sp.simplify(sp.Matrix.hstack(A - B, C - B, J2 - B).det()) == 0
    other_side = sp.simplify((A - M).dot(J2 - M)) < 0  # A, J 가 BC 양쪽 → 평면 ABC 가 넓게 이어짐
    ang_sum = sp.acos(-ca) == sp.acos(cg) and abs(float(sp.acos(ca) + sp.acos(cg) - sp.pi)) < 1e-12
    ok = (ca == sp.Rational(-1, 3) and cb == sp.Rational(1, 3) and cg == sp.Rational(1, 3)
          and edges_ok and coplanar and ang_sum and bool(other_side))
    report(1, ok, f"(1) cosα={ca} (2) cosβ={cb} (3) cosγ={cg} (4) α+γ=π:{ang_sum}, det(A,B,C,J)=0:{coplanar} [증명-수치확인]")


def p02():
    """등차수열 결정 (1) a_n, b_n (2) min c_n (3) lim S_n/n^2 — 공차 q=1..60 전수, b1 은 구간별 일차방정식."""
    a1, b1 = sp.symbols("a1 b1", real=True)
    found = []
    for q in range(1, 61):
        for p in range(1, 63):
            an = a1 + (n - 1) * p
            bn = b1 + (n - 1) * q
            if sp.limit((an - bn) / n, n, sp.oo) != 2:
                continue
            A1 = sp.solve(sp.Eq(sp.summation(an - bn, (n, 1, 10)), 180), a1)[0]
            # ③ Σ|b_n|=88 : b_n=0 인 b1 값으로 구간을 나눠 일차방정식
            cuts = sorted({-(k - 1) * q for k in range(1, 5)})
            pts = [-10**4] + cuts + [10**4]
            for lo_, hi_ in zip(pts[:-1], pts[1:]):
                mid = sp.Rational(lo_ + hi_, 2)
                expr = sum(sp.sign((b1 + (k - 1) * q).subs(b1, mid)) * (b1 + (k - 1) * q) for k in range(1, 5))
                for r in sp.solve(sp.Eq(expr, 88), b1):
                    if lo_ <= r <= hi_:
                        a_v = A1.subs(b1, r)
                        seq = [a_v + (k - 1) * p + r + (k - 1) * q for k in range(1, 300)]
                        sums = [sum(seq[:m]) for m in range(1, 300)]
                        mn = min(sums)
                        if [i + 1 for i, v in enumerate(sums) if v == mn] == [7]:
                            found.append((p, q, a_v, r))
    found = sorted(set(found))
    ok1 = len(found) == 1
    p_, q_, A_, B_ = found[0]
    an = sp.expand(A_ + (n - 1) * p_)
    bn = sp.expand(B_ + (n - 1) * q_)
    c = [abs(an.subs(n, k)) - abs(bn.subs(n, k)) for k in range(1, 2001)]
    cmin = min(c)
    argmin = [k + 1 for k, v in enumerate(c) if v == cmin]
    k = sp.symbols("k", integer=True, positive=True)
    Sn = sp.simplify(sp.summation(sp.expand((an - bn).subs(n, k)), (k, 14, n)) + sum(c[:13]))  # n≥14 에서 a,b>0
    assert all(an.subs(n, m) > 0 and bn.subs(n, m) > 0 for m in range(14, 50))
    lim = sp.limit(Sn / n**2, n, sp.oo)
    num_chk = abs(float(sum(c[:2000])) / 2000**2 - float(lim)) < 1e-2
    ok = (ok1 and an == 4 * n - 20 and bn == 2 * n - 27 and cmin == -17 and argmin == [5]
          and lim == 1 and num_chk and sp.expand(Sn - n**2 - 8 * n).free_symbols == set())
    report(2, ok, f"(1) a_n={an}, b_n={bn} (해 {len(found)}개) (2) min c_n={cmin} at n={argmin} (3) S_n={sp.expand(Sn)}, lim={lim}")


def p03():
    """삼차함수 g=ax^3+bx: (2) P0 접선 (3) Q0 접선 (4) 수직 접선 존재 조건 → 변곡점 기울기 최댓값 M."""
    a, b = sp.symbols("a b", real=True)
    g = a * x**3 + b * x
    # (1) 일반 삼차 f 를 x 축 방향 평행이동(변곡점 x좌표만큼)하면 이차항, y 방향으로 상수항이 사라짐
    c3, c2, c1, c0, h = sp.symbols("c3 c2 c1 c0 h", real=True)
    f = c3 * x**3 + c2 * x**2 + c1 * x + c0
    hh = sp.solve(sp.Poly(sp.expand(f.subs(x, x + h)), x).coeff_monomial(x**2), h)[0]
    sh = sp.expand(f.subs(x, x + hh) - f.subs(x, hh))
    one_ok = sh.coeff(x, 2) == 0 and sh.subs(x, 0) == 0 and sp.simplify(hh + c2 / (3 * c3)) == 0
    # (2)
    L = sp.expand(g.subs(x, t) + sp.diff(g, x).subs(x, t) * (x - t))
    two_ok = sp.expand(L - ((3 * a * t**2 + b) * x - 2 * a * t**3)) == 0
    # (3)
    roots = sp.roots(sp.Poly(sp.expand(g - L), x))
    xq = [r for r in roots if sp.simplify(r - t) != 0][0]
    Lq = sp.expand(g.subs(x, xq) + sp.diff(g, x).subs(x, xq) * (x - xq))
    three_ok = sp.simplify(xq + 2 * t) == 0 and sp.expand(Lq - ((12 * a * t**2 + b) * x + 16 * a * t**3)) == 0
    # (4) (3u+b)(12u+b)=-1, u=at^2>0 의 해 존재 조건 — u 에 대한 최솟값으로 판정
    u = sp.symbols("u", positive=True)
    prod = (3 * u + b) * (12 * u + b)
    ustar = sp.solve(sp.diff(prod, u), u)
    bs = sp.symbols("bs", negative=True)
    minval = sp.simplify(prod.subs(b, bs).subs(u, sp.solve(sp.diff(prod.subs(b, bs), u), u)[0]))
    thr = sp.solve(sp.Eq(minval, -1), bs)
    # 수치: b 격자에서 양의 근 u 존재 여부
    import numpy as np
    good = []
    for bv in np.linspace(-3, 3, 6001):
        rs = np.roots([36, 15 * bv, bv * bv + 1])
        if any(abs(r.imag) < 1e-12 and r.real > 0 for r in rs):
            good.append(bv)
    infl = sp.solve(sp.diff(g, x, 2), x)
    slope_at_infl = sp.diff(g, x).subs(x, infl[0])
    Mv = thr[0] if len(thr) == 1 else max(thr)
    # 참고사항 예: a=1, b=-4/3 에서 실제로 수직 접선쌍 존재
    tt = sp.solve(sp.Eq((3 * t**2 - sp.Rational(4, 3)) * (12 * t**2 - sp.Rational(4, 3)), -1), t)
    ex_ok = any(r.is_real and r != 0 for r in tt)
    four_ok = (Mv == sp.Rational(-4, 3) and abs(max(good) - (-4 / 3)) < 1e-3 and min(good) == -3
               and slope_at_infl == b and ex_ok)
    ok = one_ok and two_ok and three_ok and four_ok
    report(3, ok, f"(1) 평행이동 h={hh} (2) y={sp.factor(L)} (3) x_Q={xq}, y={Lq} (4) b≤{Mv}, 수치 max b={max(good):.4f}, M={Mv}")


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
