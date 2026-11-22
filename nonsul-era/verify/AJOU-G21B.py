#!/usr/bin/env python3
"""AJOU-G21B (아주대 2021학년도 논술 자연계열 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

PREFIX = "AJOU-G21B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(str(s["answer"]["value"]) for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def smooth_points(f, g, extra=()):
    a = sp.symbols("a", real=True)
    unk = [a, *extra]
    eqs = [sp.Eq(f.subs(x, a), g.subs(x, a)), sp.Eq(sp.diff(f, x).subs(x, a), sp.diff(g, x).subs(x, a))]
    return sp.solve(eqs, unk, dict=True), a


def p01():
    msgs, oks = [], []
    # 1-1 (1)
    c = sp.symbols("c", real=True)
    sols, a = smooth_points(x**2 - 3 * x - 6, x**3 - 4 * x + c, (c,))
    s = [d for d in sols if d[a].is_real and not d[a].is_integer]
    v = 27 * s[0][c]
    oks.append(len(s) == 1 and sp.simplify(v - sp.sympify(md_answer(1, "1-1"))) == 0); msgs.append(f"(1-1.1)27c={v}")
    # 1-1 (2)
    f = x**4 - 2 * x**2 - 2 * x; g = sp.Rational(3, 2) * x**2 - 5 * x + sp.Rational(1, 2)
    sols, a = smooth_points(f, g)
    aa = [d[a] for d in sols if d[a].is_real]
    I = sp.integrate(f, (x, 0, aa[0])) + sp.integrate(g, (x, aa[0], 2))
    oks.append(len(aa) == 1 and sp.simplify(I - sp.sympify(md_answer(1, "1-2"))) == 0); msgs.append(f"(1-1.2)a={aa},I={I}")
    # 1-2 (1) 접선 y=a_n x+b_n
    k = sp.symbols("k", integer=True, positive=True)
    f = 2**x
    an = sp.diff(f, x).subs(x, k); bn = f.subs(x, k) - an * k
    tot = sp.summation(sp.simplify(bn / an), (k, 1, 100))
    oks.append(sp.simplify(tot - sp.sympify(md_answer(1, "2-1"))) == 0); msgs.append(f"(1-2.1)Σ={sp.nsimplify(tot)}")
    # 1-2 (2) 이차함수 결정, 두 접선의 예각
    p, q, r = sp.symbols("p q r")
    g = p * x**2 + q * x + r; f = sp.exp(x)
    sol = sp.solve([g.subs(x, 0) - 1, sp.diff(g, x).subs(x, 0) - 1, g.subs(x, 2) - f.subs(x, 2)], [p, q, r], dict=True)[0]
    m1 = sp.diff(f, x).subs(x, 2); m2 = sp.diff(g.subs(sol), x).subs(x, 2)
    th = sp.atan(abs((m1 - m2) / (1 + m1 * m2)))
    oks.append(abs(float(sp.tan(th) - sp.sympify(md_answer(1, "2-2")))) < 1e-14); msgs.append(f"(1-2.2)tanθ≈{float(sp.tan(th)):.10f}")
    # 1-3 (1): a 결정 후 b 를 최댓값 20 조건으로 수치 탐색
    b, c = sp.symbols("b c", real=True)
    f = -x**4 - 2 * x**2 + b; g = -sp.Rational(4, 3) * x**3 - 4 * x + c
    a = sp.symbols("a", real=True)
    aa = [s for s in sp.solve(sp.diff(f, x).subs(x, a) - sp.diff(g, x).subs(x, a), a) if s.is_real]
    a0 = aa[0]
    csol = sp.solve(f.subs(x, a0) - g.subs(x, a0), c)[0]
    xs = np.linspace(-6, 8, 140001)
    fl = sp.lambdify((x, b), f, "numpy"); gl = sp.lambdify((x, b), g.subs(c, csol), "numpy")

    def hmax(bv):
        return np.max(np.where(xs <= float(a0), fl(xs, bv), gl(xs, bv)))
    lo, hi = -100.0, 100.0                      # hmax 는 b 에 대해 증가 → 이분법
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if hmax(mid) < 20 else (lo, mid)
    bv = (lo + hi) / 2
    val = 3 * (float(a0) + bv + float(csol.subs(b, bv)))
    oks.append(len(aa) == 1 and abs(val - float(sp.sympify(md_answer(1, "3-1")))) < 1e-6); msgs.append(f"(1-3.1)a={a0},b≈{bv:.6f},값≈{val:.6f}")
    # 1-3 (2): 부드럽게 만나는 모든 a(|a|≤20) → d, 갈아타는 함수 최댓값 M 수치, M 최대인 d
    best = None
    xs = np.linspace(-30, 30, 600001)
    for side, slope in ((-1, 1), (1, -1)):        # a<π/2 : g'=1, a>π/2 : g'=-1
        for kk in range(-8, 9):
            for base in (np.arccos(slope / 2) / 2, -np.arccos(slope / 2) / 2):
                av = base + kk * np.pi
                if (av - np.pi / 2) * side <= 0 or abs(av) > 20:
                    continue
                d = -np.sin(2 * av) - abs(av - np.pi / 2)
                h = np.where(xs <= av, np.sin(2 * xs) + d, -np.abs(xs - np.pi / 2))
                M = h.max()
                if best is None or M > best[0] + 1e-12:
                    best = (M, d, av)
    mdd = float(sp.sympify(md_answer(1, "3-2")))
    oks.append(abs(best[1] - mdd) < 1e-12); msgs.append(f"(1-3.2)a*≈{best[2]:.6f},d≈{best[1]:.6f}")
    report(1, all(oks), " ".join(msgs))


def urn_prob(r0, b0, seq):
    """(가) 시행: seq 는 'R'/'B' 문자열. 그 순서로 꺼낼 확률."""
    r, b, pr = r0, b0, Fraction(1)
    for ch in seq:
        if ch == "R":
            pr *= Fraction(r, r + b); r += 1
        else:
            pr *= Fraction(b, r + b); b += 1
    return pr


def p02():
    msgs, oks = [], []
    # 2-1 (1) 여러 (r,b) 에서 전수
    md = sp.sympify(md_answer(2, "1-1"), locals={"r": sp.Symbol("r"), "b": sp.Symbol("b")})
    ok = True
    for r0 in range(1, 6):
        for b0 in range(1, 6):
            p2 = sum(urn_prob(r0, b0, s + "R") for s in "RB")
            ok &= sp.Rational(p2.numerator, p2.denominator) == md.subs({sp.Symbol("r"): r0, sp.Symbol("b"): b0})
    oks.append(ok); msgs.append("(1)r,b=1..5 일치")
    # (2)
    num = sum(urn_prob(2, 1, s1 + "RR") for s1 in "RB")
    den = sum(urn_prob(2, 1, s1 + s2 + "R") for s1 in "RB" for s2 in "RB")
    pr = num / den
    oks.append(sp.Rational(pr.numerator, pr.denominator) == sp.sympify(md_answer(2, "1-2"))); msgs.append(f"(2)P={pr}")
    # (3) 2021회 후 빨강=파랑 — 고정밀 DP
    import mpmath as mp
    mp.mp.dps = 40
    dist = {2: mp.mpf(1)}                 # 빨간 공 수 분포
    for step in range(2021):
        tot = 3 + step
        nd = {}
        for rr, pv in dist.items():
            nd[rr + 1] = nd.get(rr + 1, 0) + pv * rr / tot
            nd[rr] = nd.get(rr, 0) + pv * (tot - rr) / tot
        dist = nd
    p3 = dist[1012]
    oks.append(abs(p3 - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "1-3")), 45)))) < mp.mpf(10) ** -35); msgs.append(f"(3)P≈{mp.nstr(p3, 12)}")
    # 2-2 (1) a_n=n, r=b=1, w=2, 2회
    EX = EX2 = Fraction(0)
    for seq in itertools.product("CW", repeat=2):
        w, tot, pr = 2, 4, Fraction(1)
        for i, ch in enumerate(seq, 1):
            pr *= Fraction(w, tot) if ch == "W" else Fraction(tot - w, tot)
            if ch == "W":
                w += i
            tot += i
        EX += pr * w; EX2 += pr * w * w
    d = kv(md_answer(2, "2-1"))
    oks.append(sp.Rational(str(EX)) == sp.sympify(d["E"]) and sp.Rational(str(EX2 - EX**2)) == sp.sympify(d["V"])); msgs.append(f"(2-1)E={EX},V={EX2 - EX**2}")
    # 2-2 (2) a_n 결정 후 흰 공 부분집합 전수
    a = {k: (3**k + 2) - (3**(k - 1) + 2) for k in range(1, 11)}
    subs = [S for m in range(11) for S in itertools.combinations(range(1, 11), m) if 1 + sum(a[k] for k in S) == 547]
    w4 = 1 + sum(a[k] for k in subs[0] if k <= 4) if len(subs) == 1 else None
    oks.append(len(subs) == 1 and w4 == int(md_answer(2, "2-2"))); msgs.append(f"(2-2)흰공시행={subs},w4={w4}")
    # 2-2 (3) a_n=n-1+e_n, e 비감소·합 1 → 유일; 빨강 합 1, 파랑 합 11 인 서로소 부분집합 전수
    base = 3 + sum(range(100))
    extra = 2 + 12 + 4940 - base
    seqs = []
    for pos in range(100):                       # e_n = 1 (n>pos) 형태, 합 = 100-pos
        if 100 - pos == extra:
            seqs.append([k - 1 + (1 if k - 1 >= pos else 0) for k in range(1, 101)])
    av = seqs[0]
    small = [k for k in range(100) if av[k] <= 11]
    ms = set()
    for m in range(len(small) + 1):
        for Bset in itertools.combinations(small, m):
            if sum(av[k] for k in Bset) != 11:
                continue
            rest = [k for k in small if k not in Bset]
            for mr in range(len(rest) + 1):
                if any(sum(av[k] for k in R) == 1 for R in itertools.combinations(rest, mr)):
                    ms.add(m); break
    ssum = sum(1 / (sp.sqrt(av[k]) + sp.sqrt(av[k + 1])) for k in range(99))
    d = kv(md_answer(2, "2-3"))
    oks.append(len(seqs) == 1 and ms == set(sp.sympify(d["m"])) and abs(float(ssum - sp.sympify(d["sum"]))) < 1e-12)
    msgs.append(f"(2-3)m={sorted(ms)},합≈{float(ssum):.10f}")
    report(2, all(oks), " ".join(msgs))


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
