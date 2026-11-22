#!/usr/bin/env python3
"""HYU-G24C (한양대 2024학년도 논술 자연계열 오후2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction
from math import comb

import mpmath as mp
import sympy as sp

PREFIX = "HYU-G24C"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)
mp.mp.dps = 40


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) 행별 전수: 색 0=빨,1=파,2=노
    def row(m):
        return sum(1 for c in itertools.product(range(3), repeat=m)
                   if c.count(0) > c.count(1) and c.count(2) <= 2)
    n1 = row(5) * row(6)
    ok1 = n1 == int(sp.sympify(md_answer(1, "1")))
    # (2) 작은 높이(세로 h)로 전수 확인한 규칙을 써서 계산: 열마다 (r,b,y) 개수 조합 → 합 집합
    def f_count(n, h=10):
        cols = {(r, b, h - r - b) for r in range(1, h) for b in range(1, h - r) if h - r - b >= 1}
        sums = {(0, 0, 0)}
        for _ in range(n):
            sums = {(s[0] + c[0], s[1] + c[1], s[2] + c[2]) for s in sums for c in cols}
        return len(sums)
    v2 = f_count(5) + f_count(7)
    ok2 = v2 == int(sp.sympify(md_answer(1, "2")))
    # (3) f = r(x+1)(x-3)(x-s): 조건으로 r,s 풀기
    r, s = sp.symbols("r s", real=True)
    f = r * (x + 1) * (x - 3) * (x - s)
    fp = sp.diff(f, x)
    sol = sp.solve([sp.diff(fp, x).subs(x, sp.Rational(2, 3)), fp.subs(x, 0) - 18], [r, s], dict=True)
    sol = [d for d in sol if sp.Poly(fp.subs(d), x).LC() < 0]      # f' 가 최댓값을 가지려면 위로 볼록
    assert len(sol) == 1
    fx = f.subs(sol[0])
    bag = [1, 2, 2, 3, 3, 3]
    EY = sum(sp.integrate(fx, (x, -1, c)) for c in bag) / len(bag)
    ok3 = sp.simplify(EY - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"count={n1} f(5)+f(7)={v2} f={sp.expand(fx)} E(Y)={EY}")


def clip(subject, clipper):
    """Sutherland–Hodgman 볼록 다각형 교집합(반시계 정점)."""
    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inside = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
        def inter(p, q):
            x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
            x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out.append(inter(p, q))
                out.append(q)
            elif inside(p):
                out.append(inter(p, q))
    return out


def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def p02():
    # (1) P=(√3 cos t, √2 sin t), Q=(y_P, x_P): 내적 최대(수치)
    dot = lambda t: (mp.sqrt(3) * mp.cos(t) + 1) * mp.sqrt(2) * mp.sin(t) * 2
    grid = [2 * mp.pi * i / 4000 for i in range(4000)]
    t0 = max(grid, key=dot)
    tm = mp.findroot(lambda t: mp.diff(dot, t), t0)
    m1 = dot(tm)
    ok1 = abs(m1 - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "1")), 40)))) < 1e-30
    # (2) 두 정사각형 겹침 넓이를 다각형 클리핑으로 직접 계산 후 수치 미분
    R = mp.sqrt(2) / 2
    def S(th):
        sq1 = [(R * mp.cos(mp.pi + k * mp.pi / 2), R * mp.sin(mp.pi + k * mp.pi / 2)) for k in range(4)]
        sq2 = [(R * mp.cos(mp.pi + th + k * mp.pi / 2), R * mp.sin(mp.pi + th + k * mp.pi / 2)) for k in range(4)]
        return area(clip(sq2, sq1))
    d2 = mp.diff(S, mp.pi / 3)
    ok2 = abs(d2 - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "2")), 40)))) < 1e-20
    # (3) a 를 접할 조건으로 수치로 찾고, 교점 개수 4n+1 확인 → cos sqrt(2/a-4) 극한 경향
    vals = []
    okcnt = True
    for n in range(1, 6):
        # 접점: h(x)=cos x-1+a x^2, h=0, h'=0 을 (2nπ,(2n+1)π) 에서 풀기
        sol = mp.findroot(lambda X, A: [mp.cos(X) - 1 + A * X**2, -mp.sin(X) + 2 * A * X],
                          ((2 * n + 0.5) * mp.pi, 2 / ((2 * n + 0.5) * mp.pi)**2))
        X, A = sol[0], sol[1]
        okcnt &= (2 * n * mp.pi < X < (2 * n + 1) * mp.pi) and A > 0
        # 양의 실근 개수: 부호 변화(단순근) + 접점(이중근) 1개
        h = lambda t: mp.cos(t) - 1 + A * t**2
        ts = [mp.mpf(i) / 200 for i in range(1, int(float(X + 30) * 200))]
        simple = sum(1 for i in range(len(ts) - 1) if h(ts[i]) * h(ts[i + 1]) < 0)
        okcnt &= (1 + 2 * (simple + 1)) == 4 * n + 1
        vals.append(mp.cos(mp.sqrt(2 / A - 4)))
    # 관계식 x_n^2 = 2/a_n - 4 를 이용한 기호 극한
    t = sp.symbols("t", positive=True)
    L = sp.limit(1 - 2 * t**2 / (t**2 + 4), t, sp.oo)
    mono = all(vals[i + 1] < vals[i] for i in range(len(vals) - 1))
    ok3 = okcnt and mono and L == sp.sympify(md_answer(2, "3")) and abs(vals[-1] - L) < 0.01
    report(2, ok1 and ok2 and ok3,
           f"maxdot={mp.nstr(m1, 15)} S'(pi/3)={mp.nstr(d2, 15)} cos-seq={[mp.nstr(v, 6) for v in vals]} lim={L}")


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
