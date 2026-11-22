#!/usr/bin/env python3
"""AJOU-G21A (아주대 2021학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import random
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "AJOU-G21A"
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


def slopes(xs):
    """B(x)=Σ|x-x_k| 의 그래프에 나타나는 직선 기울기 집합(구간 대표점에서 기울기 계산)."""
    pts = sorted(set(xs))
    reps = [pts[0] - 1] + [(pts[i] + pts[i + 1]) / 2 for i in range(len(pts) - 1)] + [pts[-1] + 1]
    return {sum(1 if r > xk else -1 for xk in xs) for r in reps}


def p01():
    msgs, oks = [], []
    # (1) n=401, x_k=k : 정수 격자에서 최솟값(구간별 일차이므로 꺾인 점에서 최소)
    xs = list(range(1, 402))
    mn = min(sum(abs(c - k) for k in xs) for c in xs)
    oks.append(mn == sp.sympify(md_answer(1, "1-1"))); msgs.append(f"(1)min={mn}")
    # (2) 서로 다른 실수 → 기울기 집합 시뮬레이션으로 a_n 확인 후 닫힌식의 극한
    ok2 = True
    for nn in range(1, 25):
        xs = random.sample(range(-1000, 1000), nn)
        a_sim = sum(s * s for s in slopes(xs))
        i = sp.symbols("i", integer=True)
        a_cf = sp.summation((-n + 2 * i) ** 2, (i, 0, n))
        ok2 &= a_sim == a_cf.subs(n, nn)
    lim = sp.limit(a_cf / n**3, n, sp.oo)
    oks.append(ok2 and sp.simplify(lim - sp.sympify(md_answer(1, "1-2"))) == 0); msgs.append(f"(2)lim={lim}")
    # (3) 짝수 n: 1≤x_1≤…≤x_n≤3 정수 순서쌍 전수 → S 원소 곱 > 0 개수
    ok3 = True
    md3 = sp.sympify(md_answer(1, "1-3"), locals={"n": n})
    for nn in range(2, 31, 2):
        cnt = 0
        for a1 in range(nn + 1):
            for a2 in range(nn + 1 - a1):
                a3 = nn - a1 - a2
                xs = [1] * a1 + [2] * a2 + [3] * a3
                S = slopes(xs)
                prod = 1
                for s in S:
                    prod *= s
                cnt += prod > 0
        ok3 &= cnt == md3.subs(n, nn)
    oks.append(ok3); msgs.append("(3)n=2..30 전수 일치")
    # (4) 증명: 무작위 반례 탐색 — B(-1)-n, B(1)-n 의 부호가 반대이거나 0
    ok4 = True
    for _ in range(3000):
        nn = random.randint(1, 30)
        xs = [random.uniform(-0.999, 0.999) for _ in range(nn)]
        B = lambda z: sum(abs(z - xk) for xk in xs)
        ok4 &= (B(-1) - nn) * (B(1) - nn) <= 1e-12
    oks.append(ok4 and md_answer(1, "1-4").startswith("항상")); msgs.append("(4)[증명-수치확인]")
    # 1-2 (1) f=sec x, x1=0, x2=π/3: 미분해서 C(x)tan x 와 비교
    Cc = sp.symbols("C")
    F = sp.sympify(md_answer(1, "2-1"), locals={"x": x, "C": Cc}).subs(Cc, 0)
    f = sp.sec(x)
    Cx = (f - f.subs(x, 0)) ** 2 + (f - f.subs(x, sp.pi / 3)) ** 2
    d = sp.diff(F, x) - Cx * sp.tan(x)
    ok5 = all(abs(complex(d.subs(x, v)).real) < 1e-10 for v in (-1.2, -0.4, 0.3, 1.1))
    oks.append(ok5); msgs.append("(1-2.1)F'=C·tan")
    # 1-2 (2) f=e^{3x}-cos^2(πx): 극한 존재 ⇒ f(x1)=f(x2)=f(1); x1=x2=1 로 두고 극한 계산
    f2 = sp.exp(3 * x) - sp.cos(sp.pi * x) ** 2
    C2 = 2 * (f2 - f2.subs(x, 1)) ** 2
    L = sp.limit(C2 / sp.log(x) ** 2, x, 1)
    oks.append(sp.simplify(L - sp.sympify(md_answer(1, "2-2"))) == 0); msgs.append(f"(1-2.2)L={L}")
    report(1, all(oks), " ".join(msgs))


def proper_colorings(rows, cols, k):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    cnt = 0
    for col in itertools.product(range(k), repeat=len(cells)):
        g = dict(zip(cells, col))
        if all(g[(r, c)] != g[(r, c + 1)] for r in range(rows) for c in range(cols - 1)) and \
           all(g[(r, c)] != g[(r + 1, c)] for r in range(rows - 1) for c in range(cols)):
            cnt += 1
    return cnt


def p02():
    msgs, oks = [], []
    # 2-1 (1) r(k) 전수(1×3 깃발) → 합
    md = sp.sympify(md_answer(2, "1-1"), locals={"n": n})
    r = {k: proper_colorings(1, 3, k) for k in range(1, 10)}
    ok = all(sum(r[k] for k in range(2, nn + 1)) == md.subs(n, nn) for nn in range(2, 10))
    oks.append(ok); msgs.append(f"(1)r(3)={r[3]}")
    # (2) 5색 삼색기 80개에서 2개 → 색 집합 서로소 확률
    flags = [f for f in itertools.product(range(5), repeat=3) if f[0] != f[1] and f[1] != f[2]]
    pairs = list(itertools.combinations(flags, 2))
    good = sum(1 for a, b in pairs if not set(a) & set(b))
    pr = Fraction(good, len(pairs))
    oks.append(sp.Rational(pr.numerator, pr.denominator) == sp.sympify(md_answer(2, "1-2"))); msgs.append(f"(2)P={pr}")
    # (3) 2×3, 2×5 격자 3색
    A, B = proper_colorings(2, 3, 3), proper_colorings(2, 5, 3)
    d = kv(md_answer(2, "1-3"))
    oks.append(A == int(d["A"]) and B == int(d["B"])); msgs.append(f"(3)A={A},B={B}")
    # 2-2 b_n 망원, a_n=H_n-b_n
    k = sp.symbols("k", integer=True, positive=True)
    bn = sp.summation(1 / (k * (k + 1)), (k, 1, n))
    an = sp.harmonic(n) - bn
    la = sp.limit(an, n, sp.oo)
    lb = sp.limit(bn ** (2 * n + 2), n, sp.oo)
    d = kv(md_answer(2, "2"))
    oks.append(la == sp.sympify(d["a"]) and sp.simplify(lb - sp.sympify(d["b"])) == 0); msgs.append(f"(2-2)a→{la},b^(2n+2)→{lb}")
    # 2-3 (1) 오른쪽 반 위·아래 넓이 같음: ∫_0^1(1-f)=∫_0^1(f+1)
    a = sp.symbols("a", real=True)
    f = sp.cos(sp.pi * x / 2) ** 3 / 2 + a
    asol = sp.solve(sp.integrate(1 - f, (x, 0, 1)) - sp.integrate(f + 1, (x, 0, 1)), a)
    asol = [s for s in asol if abs(float(s)) <= 0.5]
    oks.append(len(asol) == 1 and sp.simplify(asol[0] - sp.sympify(md_answer(2, "3-1"))) == 0); msgs.append(f"(3-1)a={asol}")
    # 2-3 (2) 수치: 정사각형 안 x∈[0,1]에서 포물선 아래 넓이(=y=-1 위, min(f,1)) 가 1 이 되는 √b
    import mpmath as mp
    mp.mp.dps = 30

    def area_below(sb):
        b = sb ** 2
        g = lambda xx: min(max(2 - b * xx ** 2, -1), 1) + 1
        x1, x2 = 1 / sb, mp.sqrt(3) / sb
        return mp.quad(g, [0, x1, x2, 1])
    sb = mp.findroot(lambda s: area_below(s) - 1, 2.8)
    md2 = sp.sympify(md_answer(2, "3-2"))
    oks.append(sb ** 2 > 3 and abs(sb - mp.mpf(str(sp.N(md2, 30)))) < 1e-20); msgs.append(f"(3-2)√b≈{mp.nstr(sb, 12)}")
    report(2, all(oks), " ".join(msgs))


if __name__ == "__main__":
    random.seed(2021)
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
