#!/usr/bin/env python3
"""SGU-G22A (서강대 2022학년도 수시 논술 자연계열 1차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from fractions import Fraction
from itertools import combinations, product

import sympy as sp

PREFIX = "SGU-G22A"
WIDTH = 2
RESULTS = []
x, n = sp.symbols("x n", positive=True)


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
    # (1-1) 13명 각각 여/남 1/2 를 전수 열거, 밝힌 여학생 3명 더함
    dist = {}
    for bits in product((0, 1), repeat=13):
        k = 3 + sum(bits)
        dist[k] = dist.get(k, 0) + Fraction(1, 2**13)
    mean = sum(k * p for k, p in dist.items())
    var = sum((k - mean) ** 2 * p for k, p in dist.items())
    a1 = md_answer(1, "1").replace(" ", "")
    pmf_s = a1.split(";")[0].split("=", 1)[1].split("(x=")[0]
    lo, hi = map(int, re.search(r"x=(\d+)\.\.(\d+)", a1).groups())
    V_md = sp.Rational(a1.split("V=")[1])
    xs = sp.Symbol("x")
    pmf = sp.sympify(pmf_s, locals={"x": xs})
    ok1 = (set(dist) == set(range(lo, hi + 1))
           and all(sp.Rational(dist[k].numerator, dist[k].denominator) == pmf.subs(xs, k) for k in dist)
           and sp.Rational(var.numerator, var.denominator) == V_md)
    # (1-2) 2n E(1/X) = H(4n) - H(2n) 의 극한(디감마 표현) + 큰 n 수치
    S = sp.harmonic(4 * n) - sp.harmonic(2 * n)          # = sum_{k=1}^{2n} 1/(2n+k)
    assert all(S.subs(n, m) == sum(sp.Rational(1, 2 * m + j) for j in range(1, 2 * m + 1)) for m in (1, 2, 5))
    lim = sp.limit(S, n, sp.oo)
    import mpmath as mp
    num = mp.harmonic(4 * 10**6) - mp.harmonic(2 * 10**6)
    md2 = sp.sympify(md_answer(1, "2"))
    ok2 = sp.simplify(lim - md2) == 0 and abs(num - float(md2)) < 1e-6
    # (1-3) 전수 열거
    gold = [1] * 6 + [0] * 4
    pairs = list(combinations(range(10), 2))
    pA = pAB = Fraction(0)
    for d in range(1, 7):
        if d <= 3:
            outs = [(d + 1, Fraction(1, 2)), (d + 2, Fraction(1, 2))]
        else:
            outs = [(d - 1 if gold[i] and gold[j] else d - 2, Fraction(1, len(pairs))) for i, j in pairs]
        for r, pr in outs:
            if r >= 4:
                pA += Fraction(1, 6) * pr
                if d % 2 == 0:
                    pAB += Fraction(1, 6) * pr
    ans3 = pAB / pA
    ok3 = sp.Rational(ans3.numerator, ans3.denominator) == sp.sympify(md_answer(1, "3"))
    # (1-4) 전수 열거: B 에서 1개 → A, A 에서 2개 꺼내 던짐
    p = sp.symbols("p")
    faces = {"N": [("R", sp.Rational(1, 2)), ("G", sp.Rational(1, 2))], "U": [("R", p), ("B", 1 - p)]}
    tot = 0
    B = ["N", "U", "U"]
    for c in B:
        A = ["N", "U", c]
        prs = list(combinations(range(3), 2))
        for i, j in prs:
            for (c1, q1), (c2, q2) in product(faces[A[i]], faces[A[j]]):
                if c1 == c2:
                    tot += sp.Rational(1, 3) * sp.Rational(1, len(prs)) * q1 * q2
    sols = [s for s in sp.solve(sp.Eq(tot, sp.Rational(4, 9)), p) if 0 <= s <= 1]
    ok4 = sols == [sp.sympify(md_answer(1, "4"))]
    report(1, ok1 and ok2 and ok3 and ok4, f"V={var} lim={lim} P(B|A)={ans3} p={sols}")


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    # (2-1) n=2..3000 수치 [증명-수치확인]
    ok1 = all(mp.mpf(1 + m) ** (mp.mpf(1) / m) < 1 + mp.sqrt(mp.mpf(2) / (m - 1)) for m in range(2, 3001))
    # (2-2) p, q > 0 : 도함수 부호(기호) + 격자 [증명-수치확인]
    X = sp.symbols("X", positive=True)
    L = sp.log(1 + 1 / X)
    P = L - 1 / (X + 1)
    Q = 1 / sp.sqrt(X * (X + 1)) - L
    dP = sp.simplify(sp.diff(P, X))
    grid = [mp.mpf(10) ** (e / 50) for e in range(-200, 301)]
    Pf, Qf, dQf = (sp.lambdify(X, e_, "mpmath") for e_ in (P, Q, sp.diff(Q, X)))
    ok2 = (sp.simplify(dP + 1 / (X * (X + 1) ** 2)) == 0 and sp.limit(P, X, sp.oo) == 0
           and sp.limit(Q, X, sp.oo) == 0 and all(Pf(t) > 0 and Qf(t) > 0 and dQf(t) < 0 for t in grid))
    # (2-3) f' = P f > 0
    f = (1 + 1 / X) ** X
    ok3 = sp.simplify(sp.diff(f, X) - P * f) == 0
    # (2-4) f <= g : x 격자(1/n 경계 포함)
    ff = sp.lambdify(X, f, "mpmath")
    def g(t):
        return ff(mp.mpf(1) / int(mp.floor(1 / t))) if t <= 1 else 3
    pts = [mp.mpf(i) / 1000 for i in range(1, 5001)] + [mp.mpf(1) / m for m in range(1, 300)] + \
          [mp.mpf(1) / m - mp.mpf(10) ** -12 for m in range(2, 300)]
    ok4 = all(ff(t) <= g(t) + mp.mpf(10) ** -25 for t in pts)
    # (2-5) 극한·치역·볼록성
    lim0 = sp.limit(f, X, 0, "+")
    limoo = sp.limit(f, X, sp.oo)
    f2 = sp.lambdify(X, sp.diff(f, X, 2), "mpmath")
    concave = all(f2(t) < 0 for t in grid[:450])
    md = dict(seg.split("=", 1) for seg in md_answer(2, "5").replace(" ", "").split(";"))
    rng = sp.sympify(md["range"])
    ok5 = (rng == sp.Interval.open(lim0, limoo) and sp.sympify(md["asymptote"]) == limoo
           and md["concave"] == "down" and md["inflection"] == "none" and concave)
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] 2-1 n<=3000, p,q>0 grid, f<=g {len(pts)}pts, lim0+={lim0}, lim∞={limoo}, f''<0")


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
