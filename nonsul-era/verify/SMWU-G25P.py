#!/usr/bin/env python3
"""SMWU-G25P (숙명여대 2025학년도 논술우수자전형 자연계열(약학부)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SMWU-G25P"
WIDTH = 2
RESULTS = []
x, t, n, k, a, c, m = sp.symbols("x t n k a c m", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {kk.strip(): sp.sympify(v) for kk, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _count_ffx_quadratic(av):
    f = lambda z: av * (z + 2) * (z - 3)
    poly = sp.Poly(sp.expand(f(f(x))), x)
    return len(set(sp.real_roots(poly)))


def q_composite_quadratic():
    """f(x)=a(x+2)(x-3), f(f(x))=0 의 서로 다른 실근 3개 → a. 근 개수가 바뀌는 a(판별식 0)를 후보로 실근 개수 직접 셈."""
    ap = sp.symbols("ap", positive=True)
    cands = set()
    for r in (3, -2):                      # f 의 근
        disc = sp.discriminant(sp.expand(ap * (x + 2) * (x - 3) - r), x)
        cands |= {s for s in sp.solve(disc, ap) if s.is_positive}
    tests = sorted(cands | {sp.Rational(1, 10), sp.Rational(1, 3), 1, 2})
    good = [av for av in tests if _count_ffx_quadratic(av) == 3]
    return good


def q_composite_sine_count(av):
    """f(x)=a sin x (0<=x<=pi), 카드 해석대로 f(f(x))=0 <=> a sin x = j*pi (j>=0) 의 [0,pi] 서로 다른 해 개수."""
    import math
    sols = set()
    j = 0
    while j * math.pi <= av + 1e-12:
        r = j * math.pi / av
        if r <= 1 + 1e-12:
            r = min(r, 1.0)
            s1 = math.asin(r)
            sols.add(round(s1, 9)); sols.add(round(math.pi - s1, 9))
        j += 1
    return len(sols)


def q_common_tangent_line(f1, f2):
    """두 이차함수에 동시에 접하는 직선 y=m x+n 들(판별식=0 연립)."""
    mm, nn = sp.symbols("mm nn")
    e1 = sp.discriminant(sp.expand(f1 - (mm * x + nn)), x)
    e2 = sp.discriminant(sp.expand(f2 - (mm * x + nn)), x)
    return sp.solve([e1, e2], [mm, nn], dict=True), mm, nn


def q_perp_tangent_a():
    """y=-x^2-2, y=x^2-2ax+a^2+a 의 공통접선 두 개가 수직 → a."""
    av = sp.symbols("av", real=True)
    mm, nn = sp.symbols("mm nn")
    e1 = sp.discriminant(sp.expand(-x**2 - 2 - (mm * x + nn)), x)
    e2 = sp.discriminant(sp.expand(x**2 - 2 * av * x + av**2 + av - (mm * x + nn)), x)
    nsol = sp.solve(e1, nn)[0]
    qm = sp.Poly(sp.expand(e2.subs(nn, nsol)), mm)            # m 에 대한 이차식
    A2, B2, C2 = qm.all_coeffs()
    sols = sp.solve(sp.Eq(C2 / A2, -1), av)                    # 두 기울기 곱 = -1
    ok_two = all(sp.simplify(B2**2 - 4 * A2 * C2).subs(av, s) > 0 for s in sols)
    return sols, ok_two


def q_king_c_values():
    """f(x)+f(c-x)=c, ∫_0^c f = 2 → c. 조건을 만족하는 함수 두 개로 ∫_0^c f 를 직접 계산."""
    cs = sp.symbols("cs", real=True)
    fams = [cs / 2 + (x - cs / 2)**3, cs / 2 + sp.sin(x - cs / 2)]
    vals = []
    for fx in fams:
        assert sp.simplify(fx + fx.subs(x, cs - x) - cs) == 0
        I = sp.simplify(sp.integrate(fx, (x, 0, cs)))
        vals.append(set(sp.solve(sp.Eq(I, 2), cs)))
    return vals


def g_extrema_sum(lo, hi):
    """f(x)=(x^2-1)(x+3k) 의 극댓값+극솟값을 실제 임계점으로 계산 후 [lo,hi] 에서 최대."""
    kk = sp.symbols("kk", real=True)
    f = (x**2 - 1) * (x + 3 * kk)
    cr = sp.solve(sp.diff(f, x), x)
    S = sp.simplify(sum(f.subs(x, r) for r in cr))
    cands = [lo, hi] + [r for r in sp.solve(sp.diff(S, kk), kk) if lo <= r <= hi]
    best = max(cands, key=lambda r: float(S.subs(kk, r)))
    return S, best, sp.nsimplify(sp.simplify(S.subs(kk, best)))


def g_an():
    """넓이 조건으로 a_n 을 직접 결정: F(x)=e^{x^2}/(e^2-1), F(a_n)-F(1)=Σ_{j<n} e^{2j-1}."""
    nn, j = sp.symbols("nn j", integer=True, positive=True)
    X = sp.symbols("X", positive=True)
    f = 2 * x / (sp.E**2 - 1) * sp.exp(x**2)
    F = sp.integrate(f, x)
    rhs = sp.summation(sp.exp(2 * j - 1), (j, 1, nn - 1))
    sol = sp.solve(sp.Eq(F.subs(x, X) - F.subs(x, 1), rhs), X)
    return [sp.simplify(s) for s in sol], nn, f


def g_chord_limit(an_expr, nn):
    """원 (x-r)^2+(y-r)^2=r^2, r=a_n 과 직선 y=tan(1/n)x 의 현 길이 제곱의 극한."""
    r = an_expr
    T = sp.tan(1 / nn)
    xs = sp.solve(sp.Eq((x - r)**2 + (T * x - r)**2, r**2), x)
    l2 = sp.simplify((xs[0] - xs[1])**2 * (1 + T**2))
    vals = [float(l2.subs(nn, v)) for v in (10**4, 10**6)]
    return sp.limit(l2, nn, sp.oo), vals


def g_ivt_poly(nmax=12):
    rows = []
    for nv in range(1, nmax + 1):
        p = sum(x**i for i in range(1, 2 * nv + 1)) - 1
        rr = [r for r in sp.real_roots(sp.Poly(p, x)) if -2 < r < 1]
        rows.append((p.subs(x, -2) > 0, p.subs(x, 0) < 0, p.subs(x, 1) == 2 * nv - 1, len(rr) >= 2,
                     p.subs(x, -2) == sp.Rational(2, 3) * 4**nv - sp.Rational(5, 3)))
    return all(all(r) for r in rows)


def g_contrapositive_examples():
    """f(a)>0,f(b)>0 이고 (a,b) 에 f<=0 인 점이 있는 예에서 (a,b) 에 실근 존재(증명 결론) 확인."""
    exs = [((x - sp.Rational(1, 2))**2 - sp.Rational(1, 100), 0, 1), ((x - sp.Rational(1, 2))**2, 0, 1),
           (sp.cos(3 * x) + sp.Rational(1, 2), 0, 2)]
    ok = True
    for fx, A, B in exs:
        assert fx.subs(x, A) > 0 and fx.subs(x, B) > 0
        roots = [r for r in sp.solveset(fx, x, sp.Interval.open(A, B))]
        ok = ok and len(roots) >= 1
    return ok


def p01():
    kk = sp.symbols("kk", real=True)
    rts = sp.solve(x**2 - (sp.exp(kk) - 2) * x - 2 * sp.exp(kk) + 1, x)
    S3 = sp.simplify(sp.expand(rts[0]**3 + rts[1]**3))
    crit = [r for r in sp.solve(sp.diff(S3, kk), kk) if r.is_real]
    P1 = parts(md_answer(1, "1"))
    ok1 = (crit == [0] and sp.diff(S3, kk, 2).subs(kk, 0) > 0 and sp.simplify(P1["k"] - crit[0]) == 0
           and sp.simplify(S3.subs(kk, crit[0]) - P1["min"]) == 0
           and float(S3.subs(kk, -30)) > -4 and float(S3.subs(kk, 3)) > -4)
    S, best, val = g_extrema_sum(-sp.sqrt(3), sp.sqrt(3))
    P2 = parts(md_answer(1, "2"))
    ok2 = sp.simplify(best - P2["k"]) == 0 and sp.simplify(val - P2["sum"]) == 0
    report(1, ok1 and ok2, f"a^3+b^3={S3} k*={crit} S(k)={S} best k={best} sum={val}")


def p02():
    assert md_answer(2, "1") is None
    ok1 = g_contrapositive_examples()
    import numpy as np
    md = sp.sympify(md_answer(2, "2"))
    b1, b2 = 4 * np.sqrt(6) / 9, np.sqrt(3)
    bad = []
    for kv in np.linspace(0.001, 3, 6000):
        rr = np.roots([1, 0, -2, -kv])
        has = any(abs(r.imag) < 1e-9 and -kv - 1e-12 <= r.real <= kv + 1e-12 for r in rr)
        if min(abs(kv - b1), abs(kv - b2)) < 1e-3:
            continue
        if has != bool(md.contains(sp.Float(kv))):
            bad.append(kv)
    # 경계값 자체도 포함되는지
    edge = all(bool(md.contains(e)) for e in (4 * sp.sqrt(6) / 9, sp.sqrt(3))) and not md.contains(0)
    ok2 = not bad and edge
    report(2, ok1 and ok2, f"grid mismatches={len(bad)} set={md}")


def p03():
    sols, nn, f = g_an()
    an = [s for s in sols if s.subs(nn, 1) == 1 and s.subs(nn, 3).is_positive]
    lim, vals = g_chord_limit(an[0], nn)
    ok1 = len(an) == 1 and sp.simplify(lim - sp.sympify(md_answer(3, "1"))) == 0
    assert md_answer(3, "2") is None
    import math
    a_of = sp.lambdify(nn, an[0], "math")
    s, okn = 0.0, True
    for N in range(1, 41):
        for mv in range((N - 1)**2 + 1, N**2 + 1):
            s += 1 / a_of(mv)
        okn = okn and s >= N - 1e-12
    kq = sp.symbols("kq", positive=True)
    step = (2 * kq + 1)**2 - (2 * (kq + 1)**2 - 1)          # (2k+1)^2 >= 2(k+1)^2-1 ⇔ 2k^2 >= 0
    ok2 = okn and sp.simplify(step - 2 * kq**2) == 0
    report(3, ok1 and ok2, f"a_n={an} lim={lim}; sum_(m<=n^2) 1/a_m >= n for n<=40; step 차={sp.simplify(step)}")


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
