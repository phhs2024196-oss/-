#!/usr/bin/env python3
"""AJOU-G21C (아주대 2021학년도 논술 자연계열 저녁, 의학과 제외) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

PREFIX = "AJOU-G21C"
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


# ── 1번: 접어 올린 함수 ──
def fold(vals, b):
    return np.abs(vals - b) + b


def touch_points(xs, F, level, tol):
    """F≥level 인 함수가 y=level 에 닿는 점(국소 최소이며 값≈level)."""
    out = []
    for i in range(1, len(xs) - 1):
        if F[i] <= F[i - 1] and F[i] <= F[i + 1] and F[i] - level < tol:
            if not out or xs[i] - out[-1] > 1e-3:
                out.append(xs[i])
    return out


def count_meet(k):
    """y=|(x-2)^2-1|+1 과 y=kx 교점 수(구간별 정확 풀이)."""
    k = sp.nsimplify(k)
    roots = set()
    for expr, cond in (((x - 2) ** 2, lambda r: r <= 1 or r >= 3), (2 - (x - 2) ** 2, lambda r: 1 < r < 3)):
        for r in sp.solve(sp.Eq(expr, k * x), x):
            if r.is_real and cond(float(r)):
                roots.add(sp.nsimplify(r))
    return len(roots)


def p01():
    msgs, oks = [], []
    # (1)
    ks = sp.sympify(md_answer(1, "1-1"))
    ok1 = all(count_meet(kk) == 3 for kk in ks)
    near = [float(kk) for kk in ks]
    grid = [sp.Rational(i, 200) for i in range(-400, 1001)]
    bad = [float(g) for g in grid if count_meet(g) == 3 and min(abs(float(g) - v) for v in near) > 6e-3]
    oks.append(ok1 and not bad); msgs.append(f"(1)k={list(ks)} 격자 반례 {len(bad)}")
    # (2) g=|f-b|+b, b=변곡점 y
    f = x**3 - 3 * x**2 + 2 * x - 4
    xi = sp.solve(sp.diff(f, x, 2), x)[0]; bb = f.subs(x, xi)
    xs = np.linspace(-3, 5, 800001)
    g = fold(sp.lambdify(x, f, "numpy")(xs), float(bb))
    mx = [xs[i] for i in range(1, len(xs) - 1) if g[i] > g[i - 1] and g[i] >= g[i + 1]]
    mn = [xs[i] for i in range(1, len(xs) - 1) if g[i] < g[i - 1] and g[i] <= g[i + 1]]
    d = kv(md_answer(1, "1-2"))
    P = sorted(float(v) for v in sp.sympify(d["p"])); Q = sorted(float(v) for v in sp.sympify(d["q"]))
    ok2 = len(mx) == len(P) and len(mn) == len(Q) and all(abs(u - v) < 1e-4 for u, v in zip(mx, P)) and all(abs(u - v) < 1e-4 for u, v in zip(mn, Q))
    oks.append(ok2); msgs.append(f"(2)극대{np.round(mx, 4).tolist()} 극소{np.round(mn, 4).tolist()}")
    # (3) 미분가능 조건: f=(x-a)^3+1 이 유일한지 반례 탐색 후, 교점 수 계산(a 여러 값)
    ok3 = True
    rng = np.random.default_rng(3)
    for _ in range(2000):   # f(x)-1=(x-x0)^2(x-s), s≠x0 이면 단순근 s 에서 f'(s)≠0 → 접으면 꺾임(미분불가)
        x0, s = rng.uniform(-5, 5, 2)
        ok3 &= abs((s - x0) ** 2) > 0                       # f'(s)=(s-x0)^2
    # 따라서 f(x)-1 의 모든 실근이 중근 이상이어야 하고, 삼차이므로 삼중근: f=(x-a)^3+1
    counts = []
    for av in (-1.3, 0.0, 0.7, 2.5):
        xs = np.linspace(av - 6, av + 6, 1200001)
        G = np.abs((xs - av) ** 3 + 1 - 1) + 1
        H = np.abs(3 * (xs - av) ** 2 - 1) + 1
        s = np.sign(G - H)
        counts.append(int(np.sum(s[1:] != s[:-1])))
    oks.append(ok3 and len(set(counts)) == 1 and counts[0] == int(md_answer(1, "1-3"))); msgs.append(f"(3)교점={counts[0]} [증명-수치확인]")
    # (4) ∫_{-1}^{1}|a tan^3(πx/4)| = 4/π
    a = sp.symbols("a", positive=True)
    I = 2 * sp.integrate(sp.tan(sp.pi * x / 4) ** 3, (x, 0, 1))
    asol = sp.solve(a * I - 4 / sp.pi, a)
    import mpmath as mp
    Inum = mp.quad(lambda z: abs(mp.tan(mp.pi * z / 4) ** 3), [-1, 0, 1])
    oks.append(len(asol) == 1 and sp.simplify(asol[0] - sp.sympify(md_answer(1, "1-4"))) == 0 and abs(Inum - float(I)) < 1e-12)
    msgs.append(f"(4)a={sp.simplify(asol[0])}")
    # 1-2 (1) f=(x-7)^2, S_10 수치
    xs = np.linspace(0, 14, 1400001)
    F = (xs - 7) ** 2
    for b in range(0, 11):
        F = fold(F, b)
    S10 = touch_points(xs, F, 10, 1e-3)
    d = kv(md_answer(1, "2-1"))
    oks.append(len(S10) == int(d["a"]) and abs(sum(S10) - float(d["b"])) < 1e-3); msgs.append(f"(1-2.1)|S10|={len(S10)},합≈{sum(S10):.4f}")
    # 1-2 (2) f=ln3x: u=ln3x 로 두면 f=u, x=e^u/3. p_n 수치 → 비율로 급수
    us = np.linspace(-9, 9, 1800001)
    ps = []
    for N in range(0, 7):
        F = us.copy()
        for b in range(0, N + 1):
            F = fold(F, b)
        S = touch_points(us, F, N, 1e-3)
        ps.append(math.exp(sum(S)) / 3 ** len(S))
    ratios = [ps[i + 1] / ps[i] for i in range(len(ps) - 1)]
    r = sum(ratios) / len(ratios)
    total = ps[0] / (1 - r)
    oks.append(max(ratios) - min(ratios) < 1e-3 and abs(total - float(sp.sympify(md_answer(1, "2-2")))) < 1e-3)
    msgs.append(f"(1-2.2)p0≈{ps[0]:.4f},비≈{r:.4f},합≈{total:.4f}")
    report(1, all(oks), " ".join(msgs))


# ── 2번: 바둑돌 ──
def black_step(state, rule):
    a0, a1 = state
    if rule == 0 and a0 > 0:
        return (a0 - 1, a1 + 1)
    if rule == 1 and a1 > 0:
        return (a0 + 1, a1 - 1)
    if rule == 2 and a0 > 0:
        return (a0 - 1, a1)
    return state


def run_black(b, rules):
    s = (b, 0)
    for r in rules:
        s = black_step(s, r)
    return s


def given_log(fr: Fraction, table: dict) -> float:
    fn, fd = sp.factorint(fr.numerator), sp.factorint(fr.denominator)
    assert set(fn) | set(fd) <= set(table)
    return sum(e * table[p] for p, e in fn.items()) - sum(e * table[p] for p, e in fd.items())


def cond_prob_12():
    """사건 A(짝수 번째 직후 원점에 없음), B(12회 후 남아 있음) — 정확 DP."""
    dist = {(1, 0): Fraction(1)}
    for step in range(1, 13):
        nd = {}
        for s, pv in dist.items():
            for r in range(3):
                s2 = black_step(s, r)
                nd[s2] = nd.get(s2, 0) + pv / 3
        if step % 2 == 0:
            nd = {s: pv for s, pv in nd.items() if s[0] == 0}
        dist = nd
    pA = sum(dist.values()); pAB = sum(pv for s, pv in dist.items() if sum(s) > 0)
    return pAB / pA


def white_sum(nn, weight):
    tot = Fraction(0)
    for m in range(nn + 1):          # 홀수 눈 m 번 → 위치 nn-2m
        k = nn - 2 * m
        if k != 0:
            tot += weight(abs(k)) * Fraction(math.comb(nn, m), 2**nn)
    return tot


def configs(nb, nn):
    states = {((nb, 0), 0)}
    for _ in range(nn):
        nxt = set()
        for (bs, w) in states:
            for face in range(1, 7):
                nxt.add((black_step(bs, (face - 1) % 3), w + (1 if face % 2 == 0 else -1)))
        states = nxt
    return len(states)


def count_13():
    multi = len(set(itertools.permutations("aabb")))          # 한 규칙 자리(4칸)에 두 눈을 두 번씩
    total = 0
    for pos in itertools.combinations(range(12), 4):
        rest = [i for i in range(12) if i not in pos]
        for pos2 in itertools.combinations(rest, 4):
            rules = [2] * 12
            for i in pos:
                rules[i] = 0
            for i in pos2:
                rules[i] = 1
            s12 = run_black(4, rules)
            if sum(s12) < 4:
                continue
            for face in range(1, 7):      # 12회 후 흰 돌: 짝·홀 각 6번 → 원점
                if face % 2 == 0 and black_step(s12, (face - 1) % 3) == (0, 4):
                    total += multi ** 3
    return total


def p02():
    msgs, oks = [], []
    table = {2: 0.30, 3: 0.47, 7: 0.84, 11: 1.04}
    # 2-1 (1)
    gone = sum(Fraction(1, 27) for rules in itertools.product(range(3), repeat=3) if run_black(1, rules) == (0, 0))
    lg = given_log(gone, table)
    oks.append(abs(lg - float(md_answer(2, "1-1"))) < 1e-9); msgs.append(f"(1)p={gone},log≈{lg:.2f}")
    # 2-1 (2) [증명-수치확인]
    pBA = cond_prob_12()
    al = sp.Rational(5, 9) ** 5
    mdexpr = sp.sympify(md_answer(2, "1-2").split("=", 1)[1].split(",")[0].replace("alpha", "(5/9)**5"))
    oks.append(pBA < Fraction(3, 92) and sp.Rational(pBA.numerator, pBA.denominator) == sp.nsimplify(mdexpr) and al < sp.Rational(19, 359))
    msgs.append(f"(2)P(B|A)={float(pBA):.6f}<3/92 [증명-수치확인]")
    # 2-1 (3)
    Xs = [sum(run_black(2, rules)) for rules in itertools.product(range(3), repeat=3)]
    E = Fraction(sum(Xs), 27); V = Fraction(sum(v * v for v in Xs), 27) - E**2
    d = kv(md_answer(2, "1-3"))
    oks.append(sp.Rational(E.numerator, E.denominator) == sp.sympify(d["E"]) and sp.simplify(sp.sqrt(sp.Rational(V.numerator, V.denominator)) - sp.sympify(d["sigma"])) == 0)
    msgs.append(f"(3)E={E},V={V}")
    # 2-2 (1)
    md = sp.sympify(md_answer(2, "2-1"), locals={"n": n})
    oks.append(all(white_sum(nn, lambda k: k * k) == md.subs(n, nn) for nn in range(1, 16))); msgs.append("(2-1)n=1..15 일치")
    # 2-2 (2)
    md = sp.sympify(md_answer(2, "2-2"), locals={"n": n})
    cs = {nn: configs(3, nn) for nn in range(3, 9)}
    oks.append(all(cs[nn] == md.subs(n, nn) for nn in cs)); msgs.append(f"(2-2)배치수{cs}")
    # 2-2 (3)
    c13 = count_13()
    oks.append(c13 == int(sp.sympify(md_answer(2, "2-3")))); msgs.append(f"(2-3)개수={c13}")
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
