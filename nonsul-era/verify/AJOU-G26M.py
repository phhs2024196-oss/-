#!/usr/bin/env python3
"""AJOU-G26M (아주대 2026학년도 논술고사 자연계열(의학과) 수학 문항) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "AJOU-G26M"
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


def md_pairs(no: int, sub: str) -> dict:
    """'k=3; limit=1/2' 또는 'radius: ..; circle: ..' 꼴을 사전으로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = re.split(r"[:=]", seg, maxsplit=1)
        out[k.strip()] = v.strip()
    return out


def md_set(no: int, sub: str) -> list:
    return [sp.sympify(s) for s in md_answer(no, sub).strip().strip("{}").split(",")]


def same(a, b, tol=1e-10) -> bool:
    return abs(complex(sp.N(a, 30)) - complex(sp.N(b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


L2 = sp.log(2)
CURVES = [  # (식, 왼끝, 오른끝)
    (-x + 3, 0, 2),
    (x**2 / 4 - sp.log(x / 2) / 2, 2, 4),
    ((16 - L2) / 4 * (x - 2) - 4, 2, 4),
    (-x - 2, -sp.Rational(2, 3), 2),
    (2 * x, -2, -sp.Rational(2, 3)),
    ((L2 - 16) / 4 * (x + 2) - 4, -4, -2),
    (x**2 / 4 - sp.log(-x / 2) / 2, -4, -2),
    (x + 3, -2, 0),
]


def _polyline():
    import numpy as np
    segs = []
    for f, a, b in CURVES:
        X = np.linspace(float(a), float(b), 6001)
        segs.append(np.c_[X, sp.lambdify(x, f, "numpy")(X) * np.ones_like(X)])
    return segs


def _count(segs, F):
    """F(x,y)=0 인 곡선 C 위의 서로 다른 점 개수(부호 변화 + 정확히 0)."""
    import numpy as np
    pts = []
    for S in segs:
        v = F(S[:, 0], S[:, 1])
        for i in np.where(v == 0)[0]:
            pts.append(S[i])
        for i in np.where(v[:-1] * v[1:] < 0)[0]:
            u = v[i] / (v[i] - v[i + 1])
            pts.append(S[i] + u * (S[i + 1] - S[i]))
    out = []
    for p_ in pts:
        if not any(abs(p_[0] - q[0]) < 1e-3 and abs(p_[1] - q[1]) < 1e-3 for q in out):
            out.append(p_)
    return len(out)


def p01():
    import numpy as np
    # (1-1) 길이
    Ln = 0
    for f, a, b in CURVES:
        if f.has(sp.log(x / 2)) or f.has(sp.log(-x / 2)):
            Ln += sp.integrate(sp.sqrt(sp.factor(1 + sp.diff(f, x)**2)), (x, a, b))
        else:
            Ln += sp.sqrt((b - a)**2 + (f.subs(x, b) - f.subs(x, a))**2)
    ok1 = same(Ln, sp.sympify(md_answer(1, "1-1")))
    # (1-2) 넓이 = 위 곡선 - 아래 곡선
    top = {(-4, -2): CURVES[6][0], (-2, 0): CURVES[7][0], (0, 2): CURVES[0][0], (2, 4): CURVES[1][0]}
    bot = {(-4, -2): CURVES[5][0], (-2, -sp.Rational(2, 3)): CURVES[4][0], (-sp.Rational(2, 3), 2): CURVES[3][0], (2, 4): CURVES[2][0]}
    def top_f(lo, hi):
        return next(f for (a, b), f in top.items() if a <= lo and hi <= b)
    def bot_f(lo, hi):
        return next(f for (a, b), f in bot.items() if a <= lo and hi <= b)
    cuts = [-4, -2, -sp.Rational(2, 3), 0, 2, 4]
    def band(lo, hi, upper=None):
        tot = 0
        pts = sorted(set([lo, hi] + [c for c in cuts if lo < c < hi]), key=float)
        for a, b in zip(pts[:-1], pts[1:]):
            up = upper if upper is not None else top_f(a, b)
            tot += sp.integrate(up - bot_f(a, b), (x, a, b))
        return tot
    area = sp.simplify(band(-4, 4))
    ok2 = sp.simplify(area - sp.sympify(md_answer(1, "1-2"))) == 0
    # (1-3) 직선 y=-10(x-k) 왼쪽(아래쪽) 넓이 = 절반. 교점 a(C8), b(C4)
    kk = sp.symbols("k", real=True)
    line = -10 * (x - kk)
    a_ = sp.solve(sp.Eq(line, x + 3), x)[0]
    b_ = sp.solve(sp.Eq(line, -x - 2), x)[0]
    kv = sp.Rational(1, 7)                      # 가정 확인용 대표값: a∈(-2/3,0), b∈(-2/3,2)
    assert -sp.Rational(2, 3) < a_.subs(kk, kv) < 0 < b_.subs(kk, kv) < 2
    left = band(-4, -sp.Rational(2, 3)) + sp.integrate(x + 3 - (-x - 2), (x, -sp.Rational(2, 3), a_)) \
        + sp.integrate(line - (-x - 2), (x, a_, b_))
    ksol = sp.nsolve(left - area / 2, kk, 0.15, prec=40)
    ok3 = 0 < ksol < sp.Rational(3, 10) and same(ksol, sp.sympify(md_answer(1, "1-3")), 1e-25)
    # (2) g(a) 불연속점 → 가장 낮은 차수의 단항 다항식 p, p(0)
    segs = _polyline()
    agrid = np.linspace(-1, 1, 4001)
    cnt = [_count(segs, lambda X_, Y_, a=a: Y_ - a * X_) for a in agrid]
    jumps = []
    for i in range(1, len(agrid) - 1):
        if cnt[i] != cnt[i - 1] or cnt[i] != cnt[i + 1]:
            if not jumps or agrid[i] - jumps[-1][-1] > 0.01:
                jumps.append([agrid[i]])
            else:
                jumps[-1].append(agrid[i])
    disc = [float(np.mean(j)) for j in jumps]
    p0 = float(np.prod([-d_ for d_ in disc]))
    ok4 = len(disc) == 4 and abs(p0 - float(sp.N(sp.sympify(md_answer(1, "2"))))) < 2e-3
    # (3) f(t) 의 불연속점
    tgrid = np.linspace(0.005, 6, 11991)
    ft = [_count(segs, lambda X_, Y_, t_=t_: Y_ - sp.Rational(4, 3).__float__() * (X_ - t_)) for t_ in tgrid]
    tj = []
    for i in range(len(tgrid) - 1):
        if ft[i] != ft[i + 1]:
            if not tj or tgrid[i] - tj[-1][-1] > 0.01:
                tj.append([tgrid[i]])
            else:
                tj[-1].append(tgrid[i])
    tdisc = [float(np.mean(j)) for j in tj]
    md3 = sorted(float(sp.N(v)) for v in md_set(1, "3"))
    ok5 = len(tdisc) == len(md3) and all(abs(u - v) < 3e-3 for u, v in zip(tdisc, md3))
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"len={sp.N(Ln, 12)} area={area} k={sp.N(ksol, 15)} g불연속={[round(d_, 4) for d_ in disc]} p(0)≈{p0:.5f} "
           f"f불연속≈{[round(d_, 4) for d_ in tdisc]} [{ok1},{ok2},{ok3},{ok4},{ok5}] (2)(3)은 격자 수치")


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
