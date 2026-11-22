#!/usr/bin/env python3
"""SGU-G26B (서강대 2026학년도 수시 논술 자연계열 2차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-G26B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1-1) 전수: 판별식 > 0 이고 삼차(2a≠0)
    A, B = sp.symbols("A B")
    disc = sp.discriminant(2 * A * x**3 - 3 * B * x**2 + B**2, x)
    dl = sp.lambdify((A, B), disc)
    cnt = sum(1 for a in range(0, 101) for b in range(0, 101)
              if a + b > 0 and a != 0 and dl(a, b) > 0)
    ok1 = cnt == int(md_answer(1, "1"))
    # (1-2) a_n 직접 적분(n=1..6) 검사 + 급수합 = ∫x^8 Σ(1-x^3)^n dx = ∫x^5(1-x^3)dx (단조수렴)
    k = sp.symbols("k", integer=True, positive=True)
    an = lambda m: sp.integrate(x**8 * (1 - x**3)**m, (x, 0, 1))
    closed = sp.Rational(1, 3) * (1 / (k + 1) - 2 / (k + 2) + 1 / (k + 3))
    okf = all(sp.simplify(an(m) - closed.subs(k, m)) == 0 for m in range(1, 7))
    S = sp.integrate(x**8 * (1 - x**3) / x**3, (x, 0, 1))
    Sn = sp.summation(closed, (k, 1, n))
    lim = sp.limit(Sn, n, sp.oo)
    ok2 = okf and sp.simplify(S - lim) == 0 and sp.simplify(S - sp.Rational(md_answer(1, "2"))) == 0
    # (1-3) 기호 n·a·c
    a, c = sp.symbols("a c", positive=True)
    N = sp.symbols("N", integer=True, positive=True)
    X = sp.symbols("X", positive=True)
    f = c * X**N
    tan = f.subs(X, a) + sp.diff(f, X).subs(X, a) * (X - a)
    x0 = sp.solve(tan, X)[0]
    An = sp.integrate(f, (X, 0, a)) - sp.Rational(1, 2) * (a - x0) * f.subs(X, a)
    Bn = sp.Rational(1, 2) * x0 * (-tan.subs(X, 0))
    ratio = sp.simplify(N**2 * An / Bn)
    l3 = sp.limit(ratio, N, sp.oo)
    ok3 = sp.simplify(l3 - sp.sympify(md_answer(1, "3"))) == 0
    # (1-4)
    kk, tt = sp.symbols("k t", positive=True)
    g = kk * x**3
    l = g.subs(x, tt) + sp.diff(g, x).subs(x, tt) * (x - tt)
    roots = [r for r in sp.solve(sp.expand(g - l), x) if sp.simplify(r - tt) != 0]
    xa = roots[0]
    Apt = (xa, g.subs(x, xa)); Bpt = (0, g.subs(x, xa)); Cpt = (0, l.subs(x, 0))
    dist = lambda P, Q: sp.sqrt((P[0] - Q[0])**2 + (P[1] - Q[1])**2)
    ab, bc, ca = dist(Apt, Bpt), dist(Bpt, Cpt), dist(Cpt, Apt)
    area = sp.Abs((Bpt[0] - Apt[0]) * (Cpt[1] - Apt[1]) - (Cpt[0] - Apt[0]) * (Bpt[1] - Apt[1])) / 2
    r = 2 * area / (ab + bc + ca)
    l4 = sp.limit(2 * sp.pi * r / tt, tt, sp.oo)
    ok4 = sp.simplify(l4 - sp.sympify(md_answer(1, "4"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"count={cnt} sum={S} (a_n 닫힌식 {okf}) lim3={l3} lim4={l4}")


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    # (2-1) 연속·미분가능: b = 우극한, a = 우미분계수 (조임 경계 ±x^{3/2} 로 극한, 진동항 수치 확인)
    xp = sp.symbols("xp", positive=True)
    right = xp * sp.sqrt(xp) * sp.cos(1 / xp) + sp.exp(xp)
    lo, hi = sp.limit(-xp**sp.Rational(3, 2) + sp.exp(xp), xp, 0, "+"), sp.limit(xp**sp.Rational(3, 2) + sp.exp(xp), xp, 0, "+")
    bval = lo if lo == hi else None
    dlo = sp.limit((-xp**sp.Rational(3, 2) + sp.exp(xp) - bval) / xp, xp, 0, "+")
    dhi = sp.limit((xp**sp.Rational(3, 2) + sp.exp(xp) - bval) / xp, xp, 0, "+")
    aval = dlo if dlo == dhi else None
    rf = sp.lambdify(xp, right, "mpmath")
    numok = all(abs((rf(h) - bval) / h - aval) < 2 * mp.sqrt(h) for h in map(mp.mpf, ("1e-4", "1e-6", "1e-8")))
    parts = dict(s.split("=") for s in md_answer(2, "1").replace(" ", "").split(";"))
    ok1 = numok and sp.sympify(parts["a"]) == aval and sp.sympify(parts["b"]) == bval
    # (2-2) k 를 (0,1)에서 훑어 극대점 곱 = -2/3 인 k 를 수치로 찾고 기호해와 비교
    K = sp.symbols("K", positive=True)
    gfun = (1 - K * x)**2 * (1 - x**2)
    gp, gpp = sp.diff(gfun, x), sp.diff(gfun, x, 2)

    def maxprod(kv):
        crit = [complex(r) for r in sp.Poly(gp.subs(K, kv), x).nroots()]
        crit = [c.real for c in crit if abs(c.imag) < 1e-12]
        mx = [c for c in crit if float(gpp.subs({K: kv, x: c})) < 0]
        return mx[0] * mx[1] if len(mx) == 2 else None
    sols = []
    grid = [i / 200 for i in range(1, 200)]
    vals = [(kv, maxprod(sp.Rational(kv).limit_denominator(1000))) for kv in grid]
    for (k1, v1), (k2, v2) in zip(vals, vals[1:]):
        if v1 is not None and v2 is not None and (v1 + 2 / 3) * (v2 + 2 / 3) <= 0:
            lo_, hi_ = k1, k2
            F = lambda kv: maxprod(sp.Float(kv, 30)) + 2 / 3
            for _ in range(60):
                mid = (lo_ + hi_) / 2
                if F(lo_) * F(mid) <= 0:
                    hi_ = mid
                else:
                    lo_ = mid
            sols.append((lo_ + hi_) / 2)
    md2 = sp.sympify(md_answer(2, "2"))
    ok2 = len(sols) == 1 and abs(sols[0] - float(md2)) < 1e-10
    # (2-3) p^2 = 4(x-a)^2 확인 후 적분 방정식
    a = sp.symbols("a", positive=True)
    s1, s2 = sp.sqrt(1 - x**2), sp.sqrt(1 - a**2)
    PA2 = (x - a)**2 + (s1 - s2)**2
    PB2 = (x - a)**2 + (s1 + s2)**2
    okp = sp.simplify(sp.expand(PA2 * PB2) - 4 * (x - a)**2) == 0
    I = sp.integrate(2 * (a - x), (x, 0, a)) + sp.integrate(2 * (x - a), (x, a, 1))
    asol = [s for s in sp.solve(sp.Eq(I, sp.Rational(1, 2)), a) if 0 < s < 1]
    ok3 = okp and asol == [sp.sympify(md_answer(2, "3"))]
    # (2-4) b 마다 q 의 [-1,1] 최댓값을 수치로 구해 조건식 h(b)=0 의 근을 훑기
    def M(bv):
        bv = mp.mpf(bv)
        q = lambda xv: ((xv - bv)**2 + (mp.sqrt(1 - xv**2) + mp.sqrt(1 - bv**2))**2)**2 - \
            ((xv - bv)**2 + (mp.sqrt(1 - xv**2) - mp.sqrt(1 - bv**2))**2)**2
        xs = [-1 + 2 * mp.mpf(i) / 2000 for i in range(2001)]
        x0 = max(xs, key=q)
        lo_, hi_ = max(-1, x0 - mp.mpf(1) / 1000), min(1, x0 + mp.mpf(1) / 1000)
        for _ in range(200):  # 삼분탐색
            m1, m2 = lo_ + (hi_ - lo_) / 3, hi_ - (hi_ - lo_) / 3
            if q(m1) < q(m2):
                lo_ = m1
            else:
                hi_ = m2
        return q((lo_ + hi_) / 2)
    h = lambda bv: M(bv)**2 / ((1 - mp.mpf(bv)**2) * (3 + mp.sqrt(1 + 8 * mp.mpf(bv)**2))**2) - 15
    bgrid = [mp.mpf(i) / 100 for i in range(1, 100)]
    hv = [h(bv) for bv in bgrid]
    roots = []
    for i in range(len(bgrid) - 1):
        if hv[i] * hv[i + 1] <= 0:
            lo_, hi_ = bgrid[i], bgrid[i + 1]
            for _ in range(80):
                mid = (lo_ + hi_) / 2
                if h(lo_) * h(mid) <= 0:
                    hi_ = mid
                else:
                    lo_ = mid
            roots.append((lo_ + hi_) / 2)
    md4 = sp.sympify(md_answer(2, "4"))
    ok4 = len(roots) == 1 and abs(roots[0] - float(md4)) < 1e-12
    report(2, ok1 and ok2 and ok3 and ok4,
           f"a={aval} b={bval} k~{float(sols[0]) if sols else None:.12f} a3={asol} b4~{float(roots[0]) if roots else None:.12f}")


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
