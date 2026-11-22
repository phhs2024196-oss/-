#!/usr/bin/env python3
"""AJOU-M27 (아주대 2027학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "AJOU-M27"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _parts(s):
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def p01():
    N = sp.symbols("N", positive=True, integer=True)
    X = sp.symbols("X", positive=True)
    f = X**N * sp.exp(-X)
    f1, f2 = sp.diff(f, X), sp.diff(f, X, 2)
    crit = sp.solve(sp.simplify(f1 / (X**(N - 1) * sp.exp(-X))), X)
    infl = sorted(sp.solve(sp.simplify(sp.expand(f2 / (X**(N - 2) * sp.exp(-X)))), X), key=lambda e: float(e.subs(N, 9)))
    a_n, b_n, c_n = infl[0], infl[1], crit[0]
    loc = {"n": N, "sqrt": sp.sqrt}
    md1 = _parts(md_answer(1, "1-1"))
    ok1 = all(sp.simplify(sp.sympify(md1[k], locals=loc) - v) == 0 for k, v in (("a_n", a_n), ("b_n", b_n), ("c_n", c_n)))
    # 제시문 오목볼록 확인 (n=2..30)
    for nv in range(2, 31):
        g2 = sp.lambdify(X, f2.subs(N, nv))
        av, bv = float(a_n.subs(N, nv)), float(b_n.subs(N, nv))
        ok1 &= g2((av) / 2) > 0 and g2((av + bv) / 2) < 0 and g2(bv + 1) > 0
    # (2) S_n 발산 + 현이 곡선 아래 (수치) [증명-수치확인]
    fl = lambda nv, xv: xv**nv * math.exp(-xv)  # noqa: E731
    Ss, ok2 = [], True
    for nv in range(2, 60):
        av, bv, cv = nv - math.sqrt(nv), nv + math.sqrt(nv), nv
        S = 0.5 * (fl(nv, av) + 2 * fl(nv, cv) + fl(nv, bv)) * math.sqrt(nv)
        area = sum(fl(nv, av + (bv - av) * (i + 0.5) / 4000) for i in range(4000)) * (bv - av) / 4000
        ok2 &= area > S
        if nv >= 4:
            ok2 &= S >= (4 / 3) ** nv
        Ss.append(S)
    ok2 &= Ss[-1] > 1e50 and sp.sympify(md_answer(1, "1-2")) == sp.oo
    # (3) a_n >= 1 (n>=3), n=2 에서는 거짓 [증명-수치확인]
    ok3 = all(float(a_n.subs(N, nv)) >= 1 for nv in range(3, 2001)) and float(a_n.subs(N, 2)) < 1
    # (4) d_n
    ratio = sp.simplify(f / f1)                              # = X/(N-X)
    d_n = sp.simplify(a_n - ratio.subs(X, a_n))
    ok4 = sp.simplify(d_n - sp.sympify(md_answer(1, "1-4"), locals=loc)) == 0
    # (5) T_n 발산, 접선이 곡선 아래 [증명-수치확인]
    ok5 = sp.sympify(md_answer(1, "1-5")) == sp.oo
    for nv in range(3, 60):
        av = nv - math.sqrt(nv)
        dv = float(d_n.subs(N, nv))
        slope = (nv - av) * av ** (nv - 1) * math.exp(-av)
        T = 0.5 * fl(nv, av) * (av - dv)
        area = sum(fl(nv, av * (i + 0.5) / 4000) for i in range(4000)) * av / 4000
        ok5 &= area > T and T >= 0.5 * math.exp(-1) * (math.sqrt(nv) - 1) - 1e-12
        ok5 &= all(fl(nv, xv) >= fl(nv, av) + slope * (xv - av) - 1e-9 for xv in [av * i / 50 for i in range(1, 50)])
    # 문제 1-2 (1) 기록값·개수: 전수
    import itertools
    md21 = md_answer(1, "2-1").split("N(m)=")[1]
    M = sp.symbols("m")
    cnt_expr = sp.sympify(md21, locals={"n": N, "m": M, "factorial": sp.factorial})
    ok6 = True
    for nv in range(1, 11):
        cnt = {}
        for rec in itertools.product((1, -1), repeat=nv):
            cnt[sum(rec)] = cnt.get(sum(rec), 0) + 1
        ok6 &= sorted(cnt) == list(range(-nv, nv + 1, 2))
        ok6 &= all(cnt_expr.subs({N: nv, M: m}) == c for m, c in cnt.items())
    # 문제 1-2 (2): 판별식 조건으로 m 결정 후 비, 극한
    ok7 = True
    for l in range(1, 41):
        nv = 4 * l * l
        ms = [m for m in range(-nv, nv + 1, 2) if m * m - 4 * nv <= 0]
        p = math.comb(nv, (nv + max(ms)) // 2)
        q = math.comb(nv, (nv + min(ms, key=abs)) // 2)
        r = Fraction(p, q)
        ok7 &= (1 - Fraction(1, l)) ** (2 * l) <= r <= 1 / (1 + Fraction(1, l)) ** (2 * l)
    L = sp.symbols("L", positive=True)
    lim = sp.limit((1 - 1 / L)**(2 * L), L, sp.oo)
    ok7 &= lim == sp.limit((1 + 1 / L)**(-2 * L), L, sp.oo) and sp.simplify(lim - sp.sympify(md_answer(1, "2-2"))) == 0
    big = 4 * 300**2
    ok7 &= abs(math.comb(big, big // 2 + 600) / math.comb(big, big // 2) - math.exp(-2)) < 1e-3
    report(1, all([ok1, ok2, ok3, ok4, ok5, ok6, ok7]),
           f"a,b,c={a_n},{b_n},{c_n} d_n={d_n} S_n·T_n 발산 p/q→{lim} [증명-수치확인] "
           f"({ok1},{ok2},{ok3},{ok4},{ok5},{ok6},{ok7})")


def _g(xv, p):
    fv = (xv - p)**2 - 25
    return -fv if p - 5 < xv < p + 5 else fv


def _a(k, p):
    b = sum(1 for xv in range(21) if _g(xv, p) > k)
    return b - (21 - b), b, 21 - b


def _cover(p, nn):
    """ℓ=1/nn 일 때 S의 덮개 정사각형 수(제1사분면, 내부와 겹치는 것만) — 직접 셈."""
    A, E = Fraction(p - 8), Fraction(p + 8)          # g(x)=39 ⇔ |x-p|=8 (바깥 가지)
    pts = [(A, Fraction(39)), (Fraction(p - 5), Fraction(0)), (Fraction(p), Fraction(25)),
           (Fraction(p + 5), Fraction(0)), (E, Fraction(39))]

    def h(xv):
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if x1 <= xv <= x2:
                return y1 + (y2 - y1) * (xv - x1) / (x2 - x1)
    cnt = 0
    for i in range(max(0, math.floor(A * nn)), math.ceil(E * nn)):
        xa, xb = max(Fraction(i, nn), A), min(Fraction(i + 1, nn), E)
        if xa >= xb:
            continue
        mn = min(h(q) for q in [xa, xb] + [q for q, _ in pts if xa < q < xb])
        cnt += 39 * nn - math.floor(mn * nn)
    return cnt


def p02():
    ok0 = _a(10, 10) == (13, 17, 4) and _cover(10, 1) == 446          # 제시문 예시 재현
    s10 = sum(_a(k, 10)[0] for k in range(1, 31))
    ok1 = s10 == int(md_answer(2, "1-1"))
    sums = {p: sum(_a(k, p)[0] for k in range(1, 31)) for p in range(0, 200)}
    best = max(sums.values())
    pmin = min(p for p in sums if sums[p] == best)
    ok2 = pmin == int(md_answer(2, "1-2"))
    ps = [p for p in range(0, 200) if _a(30, p)[0] < 0]
    ok3 = len(ps) == int(md_answer(2, "2-1"))
    p0 = min(ps)
    c1 = _cover(p0, 1)
    ok4 = c1 == int(md_answer(2, "2-2"))
    # (3) 정적분 값과 실제 개수/n^2 의 수렴
    X = sp.symbols("X")
    hX = sp.Piecewise((-13 * X - 26, X < -2), (5 * X + 10, X < 3), (-5 * X + 40, X < 8), (13 * X - 104, True))
    hp = sp.Piecewise(((39 - 0) / (-3) * (X - (p0 - 5)), X < p0 - 5), (5 * (X - (p0 - 5)), X < p0),
                      (25 - 5 * (X - p0), X < p0 + 5), (13 * (X - (p0 + 5)), True))
    okh = all(sp.simplify(hX.subs(X, v) - hp.subs(X, v)) == 0 for v in range(-5, 12))
    I = sp.integrate(39 - hX, (X, 0, 11))
    md3 = sp.Rational(md_answer(2, "2-3"))
    seq = [Fraction(_cover(p0, nn), nn * nn) for nn in (50, 200, 800)]
    ok5 = okh and I == md3 and abs(float(seq[-1]) - float(I)) < abs(float(seq[0]) - float(I)) < 1 and abs(float(seq[-1]) - float(I)) < 0.1
    report(2, all([ok0, ok1, ok2, ok3, ok4, ok5]),
           f"예시 a10=13·446 재현, Σa_k(p=10)={s10}, p_min(최대)={pmin}, #p(a30<0)={len(ps)}(p={ps[0]}~{ps[-1]}), "
           f"덮개(ℓ=1)={c1}, ∫={I}, d_n/n^2(n=800)={float(seq[-1]):.3f}")


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
