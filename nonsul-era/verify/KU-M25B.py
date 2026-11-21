#!/usr/bin/env python3
"""KU-M25B (고려대 2024.5 시행 모의논술, 2025학년도 대비 자연계) — 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산 후 YAML answer 와 비교."""
import itertools
import math
import re
import sys
from fractions import Fraction as Fr

import numpy as np
import sympy as sp

PREFIX = "KU-M25B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def sig(v):
    return v if v >= 0 else 0 * v


def kinks(F, lo, hi, step):
    """격자 위에서 좌·우 기울기가 다른 점(꺾은선의 미분불가능 점)."""
    n = int((hi - lo) / step)
    xs = [lo + k * step for k in range(n + 1)]
    ys = [F(v) for v in xs]
    sl = [(ys[k + 1] - ys[k]) / step for k in range(n)]
    return [xs[k] for k in range(1, n) if sl[k] != sl[k - 1]]


def p01():
    # (1-1) b, c 를 1/4 격자에서 훑어 미분불가능 점이 정확히 3개인 (b,c) 집합 = {b=c} ?
    q = Fr(1, 4)
    ok1 = True
    for bi in range(-8, 9):
        for ci in range(bi, 9):
            b, c = bi * q, ci * q
            g = lambda t: -sig(-t + b) + 1
            h = lambda t: -sig(t - c) + 2
            F = lambda t: sig(g(t)) + sig(h(t))
            cnt = len(kinks(F, Fr(-8), Fr(8), Fr(1, 8)))
            if (cnt == 3) != (b == c):
                ok1 = False
    ans1 = md_answer(1, "1").replace(" ", "")
    ok1 = ok1 and ans1 in ("b=c", "c=b")

    # (1-2) 괄호 안 함수 H, f=(x-a)(x-b)H 의 좌·우미분계수 비교(수치), a,b 를 1/2 격자에서 전수
    g2 = lambda t: -max(-t - 1, 0) + 1
    H = lambda t: max(g2(t), 0) + max(g2(-t), 0) - 1
    nodes = [k / 4 for k in range(-24, 25)]
    hh = 1e-6

    def nondiff(a, b):
        f = lambda t: (t - a) * (t - b) * H(t)
        out = []
        for x0 in nodes:
            L = (f(x0) - f(x0 - hh)) / hh
            R = (f(x0 + hh) - f(x0)) / hh
            if abs(L - R) > 1e-3:
                out.append(x0)
        return out
    found = set()
    grid = [k / 2 for k in range(-10, 11)]
    for a in grid:
        for b in grid:
            if len(nondiff(a, b)) == 2 and a * b * H(0) > 0:
                found.add((sp.Rational(a), sp.Rational(b)))
    rng = np.random.default_rng(1)
    rand_ok = all(len(nondiff(*rng.uniform(-3, 3, 2))) == 4 for _ in range(50))
    md2 = {tuple(sp.Rational(v) for v in m) for m in re.findall(r"\((-?\d+),(-?\d+)\)", md_answer(1, "2"))}
    ok2 = found == md2 and rand_ok

    # (1-3) 직선 y=ax+b 와 f=H 의 교점 개수(구간별 일차식의 근을 정확히 셈), 영역 넓이 수치 적분
    pts = [-2.0, -1.0, 1.0, 2.0]

    def count(a, b):
        roots = set()
        segs = [(-np.inf, -2.0), (-2.0, -1.0), (-1.0, 1.0), (1.0, 2.0), (2.0, np.inf)]
        samp = [(-3.0, -2.5), (-1.75, -1.25), (-0.5, 0.5), (1.25, 1.75), (2.5, 3.0)]
        for (lo, hi), (t1, t2) in zip(segs, samp):
            F1, F2 = H(t1) - a * t1 - b, H(t2) - a * t2 - b
            s = (F2 - F1) / (t2 - t1)
            if abs(s) < 1e-15:
                if abs(F1) < 1e-15:
                    return 99
                continue
            r = t1 - F1 / s
            if lo - 1e-12 <= r <= hi + 1e-12:
                roots.add(round(r, 9))
        return len(roots)
    N = 800
    A = np.linspace(-3, 3, N, endpoint=False) + 3 / N
    B = np.linspace(-2, 4, N, endpoint=False) + 3 / N
    hit = sum(1 for a in A for b in B if count(a, b) == 3)
    area = hit * (6 / N) ** 2
    ok3 = abs(area - float(sp.sympify(md_answer(1, "3")))) < 0.03
    report(1, ok1 and ok2 and ok3, f"1-1 cnt3⇔b=c, 1-2 {sorted(found)}, 1-3 area≈{area:.4f}")


def p02():
    p, q, r, al = sp.symbols("p q r alpha")
    sols = []
    for pv in range(-60, 61):
        if pv == 0:
            continue
        f = x**3 + pv * x**2 + q * x + r
        a0 = sp.solve(sp.diff(f, x, 2), x)[0]
        s = sp.solve([f.subs(x, a0) + a0, sp.diff(f, x).subs(x, a0) + 1], [q, r], dict=True)[0]
        if s[q].is_integer and s[r].is_integer:
            sols.append((pv, int(s[q]), int(s[r])))
    m = min(a * b * c for a, b, c in sols)
    fs = {sp.expand(x**3 + a * x**2 + b * x + c) for a, b, c in sols if a * b * c == m}
    md1 = {sp.expand(sp.sympify(e, locals={"x": x})) for e in md_answer(2, "1").strip("{}").split(",")}
    ok1 = fs == md1
    # (2-2) 정수 해 수열을 따라 pq/r 를 계산하고, 정수 해들로 보간한 일반형의 극한도 확인
    pos = sorted(t for t in sols if t[0] > 0)
    ratios = [sp.Rational(a * b, c) for a, b, c in pos]
    k = sp.symbols("k", positive=True)
    # 정수 해 (p,q,r) 가 p 에 대한 다항식 관계를 따르는지 보간해 극한 계산
    qp = sp.interpolate([(a, b) for a, b, _ in pos[:4]], p)
    rp = sp.interpolate([(a, c) for a, _, c in pos[:5]], p)
    lim = sp.limit(p * qp / rp, p, sp.oo)
    ok2 = sp.simplify(lim - sp.sympify(md_answer(2, "2"))) == 0 and abs(float(ratios[-1]) - float(lim)) < 0.01 \
        and all(qp.subs(p, a) == b and rp.subs(p, a) == c for a, b, c in pos)
    # (2-3) 자연수 p 에 대해 (-3,0) 을 지나는 접선 개수(서로 다른 접선) 를 실근으로 셈
    t = sp.symbols("t", real=True)
    good = []
    for a, b, c in pos:
        f = x**3 + a * x**2 + b * x + c
        fp = sp.diff(f, x)
        eq = sp.Poly(sp.expand(fp.subs(x, t) * (-3 - t) + f.subs(x, t)), t)
        roots = set(sp.real_roots(eq))
        lines = {(sp.nsimplify(fp.subs(x, rt)), sp.nsimplify(f.subs(x, rt) - fp.subs(x, rt) * rt)) for rt in roots}
        if len(lines) == 3:
            good.append(a)
    md3 = {int(v) for v in re.findall(r"-?\d+", md_answer(2, "3"))}
    ok3 = set(good) == md3
    report(2, ok1 and ok2 and ok3, f"min pqr={m} f={sorted(map(str, fs))} lim={lim} p∈{good}")


def p03():
    def pts(n):
        th = [-sp.pi / 2 + 2 * sp.pi * i / (n + 1) for i in range(n + 1)]
        P = [sp.Matrix([sp.cos(a), sp.sin(a)]) for a in th]
        return P[0], P[1:]
    A, Bs = pts(7)
    b = [Bi - A for Bi in Bs]
    v = sp.nsimplify(sp.simplify(b[1].dot(b[4])))
    ok1 = sp.simplify(v - sp.sympify(md_answer(3, "1"))) == 0
    # 순서 확인: 원 위 같은 간격 (현의 길이 모두 같음)
    import mpmath as mp
    mp.mp.dps = 30

    def S(n):
        A = mp.matrix([0, -1])
        tot = mp.mpf(0)
        B1 = mp.matrix([mp.cos(-mp.pi / 2 + 2 * mp.pi / (n + 1)), mp.sin(-mp.pi / 2 + 2 * mp.pi / (n + 1))]) - A
        for i in range(1, n + 1):
            a = -mp.pi / 2 + 2 * mp.pi * i / (n + 1)
            bi = mp.matrix([mp.cos(a), mp.sin(a)]) - A
            tot += B1[0] * bi[0] + B1[1] * bi[1]
        return tot
    vals = [S(n) for n in (10, 100, 1000, 20000)]
    target = float(sp.sympify(md_answer(3, "2")))
    ok2 = all(abs(vals[i + 1] - target) < abs(vals[i] - target) for i in range(3)) and abs(vals[-1] - target) < 2e-3
    report(3, ok1 and ok2, f"b2·b5={v}  S(n)={[float(u) for u in vals]}")


def p04():
    balls = ["R"] * 4 + ["B"] * 5
    seqs = list(itertools.permutations(range(9), 4))
    tot = len(seqs)
    p4 = Fr(sum(1 for s in seqs if balls[s[3]] == "R"), tot)
    EX = Fr(sum(100 * (balls[s[0]] == "R") + 200 * (balls[s[3]] == "R") for s in seqs), tot)
    ok1 = p4 == Fr(md_answer(4, "1"))
    ok2 = EX == Fr(md_answer(4, "2"))
    report(4, ok1 and ok2, f"P(4th red)={p4} E[X]={EX} (만 원), 순열 {tot}개 전수")


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
