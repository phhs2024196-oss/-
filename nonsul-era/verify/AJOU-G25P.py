#!/usr/bin/env python3
"""AJOU-G25P (아주대 2025학년도 논술 자연계열 약학과) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G25P"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def same(a, b, tol=1e-12):
    return abs(complex(sp.N(a - b, 30))) < tol


def p01():
    oks = []
    import mpmath as mp
    import numpy as np
    mp.mp.dps = 30
    a, X = sp.symbols("a X", positive=True)
    # (1-1)(1): 제1사분면 교점 x좌표 p(a), 최솟값
    pa = [s for s in sp.solve(sp.Eq(a * X**2, -a * X + a**2 + 4), X) if s.subs(a, 1) > 0][0]
    crit = [c for c in sp.solve(sp.diff(pa, a), a) if c > 0]
    pmin = sp.simplify(pa.subs(a, crit[0]))
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "1-1").replace(" ", "").split(";"))
    grid_min = min(float(pa.subs(a, v)) for v in np.linspace(0.05, 20, 4000))
    oks.append(crit == [sp.sympify(parts["a"])] and sp.simplify(pmin - sp.sympify(parts["p_min"])) == 0
               and abs(grid_min - float(pmin)) < 1e-4)
    # (1-1)(2): S = ∫_{-p}^{p}(f2-f1) 를 a 의 함수로 직접 → md 식(a,p) 과 무작위 a 에서 비교
    p = sp.symbols("p", positive=True)
    Sdirect = 2 * sp.integrate(-a * X + a**2 + 4 - a * X**2, (X, 0, pa))
    md2 = sp.sympify(md_answer(1, "1-2"), locals={"a": a, "p": p})
    oks.append(all(abs(float(Sdirect.subs(a, v)) - float(md2.subs({a: v, p: pa.subs(a, v)}))) < 1e-9 for v in (0.3, 1, 2, 3.7, 9)))
    # (1-1)(3): y=4a 가 S 를 이등분하는 a 를 수치로 찾고 4p^3+3p^2 계산
    def lower_area(av):
        pv = float(pa.subs(a, av))
        f2 = lambda x: -av * abs(x) + av**2 + 4
        return mp.quad(lambda x: max(0, min(f2(x), 4 * av) - av * x * x), [-pv, -2, 0, 2, pv] if pv > 2 else [-pv, 0, pv])
    Sf = sp.lambdify(a, Sdirect, "mpmath")
    F = lambda av: lower_area(av) - Sf(av) / 2
    vals = []
    for lo_, hi_ in ((mp.mpf("0.2"), mp.mpf(2)), (mp.mpf(2), mp.mpf(20))):   # p(a) 는 a=2 에서 최소 → 근이 양쪽에 하나씩
        assert F(lo_) * F(hi_) < 0
        for _ in range(90):                                                  # 이분법
            mid = (lo_ + hi_) / 2
            lo_, hi_ = (mid, hi_) if F(lo_) * F(mid) > 0 else (lo_, mid)
        asol = (lo_ + hi_) / 2
        pv = mp.mpf(str(sp.N(pa.subs(a, sp.Float(str(asol), 30)), 30)))
        vals.append(4 * pv**3 + 3 * pv**2)
    val = vals[0]
    oks.append(all(abs(v_ - float(sp.sympify(md_answer(1, "1-3")))) < 1e-8 for v_ in vals))
    # (1-2)(1): e^{x+k} 와 g 의 교점 수, k 격자에서 3개인 구간 = md 구간
    xs = np.linspace(-40, 12, 520001)
    gx = np.abs(2 * np.abs(xs) - 3)
    iv = sp.sympify(md_answer(1, "2-1"))
    lo, hi = float(iv.inf), float(iv.sup)
    bad = []
    for kv in np.linspace(lo - 1.5, hi + 1.5, 601):
        if min(abs(kv - lo), abs(kv - hi)) < 2e-3:
            continue
        d = np.exp(xs + kv) - gx
        cnt = int(np.sum(d[:-1] * d[1:] < 0)) + int(np.sum(d == 0))   # 격자점에 정확히 걸린 근 포함
        if (cnt == 3) != (lo < kv < hi):
            bad.append((kv, cnt))
    oks.append(not bad)
    # (1-2)(2): 원과 g 의 교점 수(정확 계산). 특수값(접함·꼭짓점 통과) 전체에서 3개인 c = md 집합, 격자에서는 3개 없음
    c, x = sp.symbols("c x", real=True)
    pieces = [(-2, -3, -sp.oo, -sp.Rational(3, 2)), (2, 3, -sp.Rational(3, 2), 0), (-2, 3, 0, sp.Rational(3, 2)), (2, -3, sp.Rational(3, 2), sp.oo)]

    def count(cv):
        pts = set()
        for s_, b_, l_, h_ in pieces:
            for r in sp.solve(sp.Eq((x - cv)**2 + (s_ * x + b_ - sp.Rational(3, 2))**2, sp.Rational(9, 4)), x):
                if r.is_real and l_ <= r <= h_:
                    pts.add(sp.nsimplify(sp.simplify(r)))
        return len(pts)
    specials = set()
    for s_, b_, l_, h_ in pieces:
        disc = sp.discriminant(sp.expand((x - c)**2 + (s_ * x + b_ - sp.Rational(3, 2))**2 - sp.Rational(9, 4)), x)
        specials |= set(sp.solve(disc, c))
    for vx, vy in ((-sp.Rational(3, 2), 0), (0, 3), (sp.Rational(3, 2), 0)):
        specials |= set(sp.solve(sp.Eq((vx - c)**2 + (vy - sp.Rational(3, 2))**2, sp.Rational(9, 4)), c))
    three = sp.FiniteSet(*[s for s in specials if count(s) == 3])
    gridthree = [cv for cv in np.linspace(-6, 6, 1201) if count(sp.Rational(str(round(cv, 3)))) == 3]
    gridthree = [cv for cv in gridthree if all(abs(cv - float(s)) > 1e-6 for s in specials)]
    oks.append(sp.simplify(three.symmetric_difference(sp.sympify(md_answer(1, "2-2")))) == sp.EmptySet and not gridthree)
    report(1, all(oks), f"a*={crit} pmin={pmin} a(1-3)={mp.nstr(asol, 10)} 4p^3+3p^2={mp.nstr(val, 12)} k-bad={bad[:3]} c={three} {oks}")


def p02():
    oks = []
    a, p, be, ga, X, Y = sp.symbols("a p beta gamma X Y", real=True)
    # (2-1): 접점 x좌표는 a/2 t^2 - a p t - 1 = 0 의 근(직접 유도), m1·m2 교점
    tt = sp.symbols("t")
    tan_eq = sp.expand(a * tt * (p - tt) + a / 2 * tt**2 - (-1))      # 접선이 P(p,-1) 을 지날 조건
    r = sp.solve(tan_eq, tt)
    m1 = sp.Eq(Y, -1 / (a * be) * (X - be) + a * be**2 / 2)
    m2 = sp.Eq(Y, -1 / (a * ga) * (X - ga) + a * ga**2 / 2)
    Q = sp.solve([m1, m2], [X, Y], dict=True)[0]
    Qx = sp.simplify(Q[X].subs({be: r[0], ga: r[1]})); Qy = sp.simplify(Q[Y].subs({be: r[0], ga: r[1]}))
    md = sp.sympify(md_answer(2, "1"), locals={"a": a, "p": p})
    oks.append(sp.simplify(Qx - md[0]) == 0 and sp.simplify(Qy - md[1]) == 0)
    # (2-2)(1): a=3, Q 가 y=3/2 x^2 위, 제2사분면
    ps = [s for s in sp.solve(sp.Eq(Qy.subs(a, 3), sp.Rational(3, 2) * Qx.subs(a, 3)**2), p)
          if Qx.subs({a: 3, p: s}) < 0 and Qy.subs({a: 3, p: s}) > 0]
    oks.append(ps == [sp.sympify(md_answer(2, "2-1"))])
    # (2-2)(2): sinθ = |PA×PB|/(|PA||PB|)
    pv = ps[0]
    rs = sorted(sp.solve(tan_eq.subs({a: 3, p: pv}), tt))
    A = sp.Matrix([rs[0], sp.Rational(3, 2) * rs[0]**2]); B = sp.Matrix([rs[1], sp.Rational(3, 2) * rs[1]**2]); P = sp.Matrix([pv, -1])
    u, w = A - P, B - P
    sinth = sp.simplify(sp.Abs(u[0] * w[1] - u[1] * w[0]) / (u.norm() * w.norm()))
    oks.append(sp.simplify(sinth - sp.sympify(md_answer(2, "2-2"))) == 0)
    # (2-3): AB, cosθ, Q_x > B_x 로 (a,p) 를 수치로 구해 PQ·비·넓이 확인
    import mpmath as mp
    mp.mp.dps = 30

    def geo(av, pv_):
        s = mp.sqrt(pv_**2 + 2 / av); b_, g_ = pv_ - s, pv_ + s
        A_ = (b_, av / 2 * b_**2); B_ = (g_, av / 2 * g_**2); P_ = (pv_, -1)
        Q_ = (2 * av * pv_, 2 * av * pv_**2 + 1 / av + 1)          # 위 (2-1) 에서 sympy 로 확인한 식
        dist = lambda U, V: mp.sqrt((U[0] - V[0])**2 + (U[1] - V[1])**2)
        AP, BP, AB, PQ, BQ = dist(A_, P_), dist(B_, P_), dist(A_, B_), dist(P_, Q_), dist(B_, Q_)
        cos = ((A_[0] - P_[0]) * (B_[0] - P_[0]) + (A_[1] - P_[1]) * (B_[1] - P_[1])) / AP / BP
        return dict(AB=AB, cos=cos, PQ=PQ, ratio=AP / BQ, area=AP * BP * mp.sqrt(1 - cos**2) / 2, qright=Q_[0] > g_)
    sols = set()
    for a0 in (0.3, 1, 2, 4, 8):
        for p0 in (-3, -1, -0.3, 0.3, 1, 3):
            try:
                rr = mp.findroot(lambda u_, v_: [geo(u_, v_)["AB"] - 12 * mp.sqrt(5) / 5, geo(u_, v_)["cos"] - 2 / mp.sqrt(13)], (a0, p0))
                if rr[0] > 0 and geo(rr[0], rr[1])["qright"]:
                    sols.add((mp.nstr(rr[0], 15), mp.nstr(rr[1], 15)))
            except Exception:  # noqa: BLE001
                pass
    G = [geo(mp.mpf(s0), mp.mpf(s1)) for s0, s1 in sols]
    pq = mp.mpf(str(sp.N(sp.sympify(md_answer(2, "3-1")), 40)))
    ar = mp.mpf(str(sp.N(sp.sympify(md_answer(2, "3-2")), 40)))
    oks.append(len(G) >= 1 and all(abs(g_["PQ"] - pq) < 1e-12 for g_ in G))
    oks.append(any(abs(g_["ratio"] - 3 / mp.sqrt(13)) < 1e-12 and abs(g_["area"] - ar) < 1e-12 for g_ in G))
    report(2, all(oks), f"Q=({Qx}, {Qy}) p={ps} sinθ={sinth} (a,p)={sols} PQ={[mp.nstr(g_['PQ'], 10) for g_ in G]} {oks}")


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
