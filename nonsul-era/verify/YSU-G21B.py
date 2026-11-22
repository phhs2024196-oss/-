#!/usr/bin/env python3
"""YSU-G21B (연세대 2021학년도 수시 논술 자연계열 오후(자연2) 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction as Fr

import sympy as sp

PREFIX = "YSU-G21B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/YSU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 원: (x+a/2)^2+(y+b/2)^2=(a^2+b^2-24)/4 > 0, 이등분 <=> 직선이 중심 (-a/2,-b/2) 를 지남
    circ = [(a, b, c) for a, b, c in itertools.product(range(1, 7), repeat=3) if a * a + b * b - 24 > 0]
    good = [(a, b, c) for a, b, c in circ if Fr(-a, 2) + 2 * Fr(-b, 2) + c == 0]
    p = sp.Rational(len(good), len(circ))                     # 조건부확률(원을 나타낼 때)
    p_all = sp.Rational(len(good), 216)                       # 대표 오답: 전체 216 기준
    ok = sp.simplify(p - sp.sympify(md_answer(1, "1"))) == 0
    report(1, ok, f"|원|={len(circ)} |이등분|={len(good)} P={p} (오답 재현 6/216={p_all})")


def p02():
    expr = sp.sympify(md_answer(2, "1"), locals={"k": sp.Symbol("k"), "n": sp.Symbol("n"), "r": sp.Symbol("r")})
    K, N, R = sp.symbols("k n r")
    ok = True
    for nn in range(2, 11):
        for kk in range(2, nn + 1):
            # 양의 정수해 = 1..nn-1 에서 kk-1 개의 칸막이 위치 선택(전수 생성)
            sols = []
            for cut in itertools.combinations(range(1, nn), kk - 1):
                b = (0,) + cut + (nn,)
                sols.append(tuple(b[i + 1] - b[i] for i in range(kk)))
            for rr in range(1, nn - kk + 2):
                cnt = sum(s.count(rr) for s in sols)
                if int(expr.subs({K: kk, N: nn, R: rr})) != cnt:
                    ok = False
    ex = int(expr.subs({K: 3, N: 5, R: 2}))                   # <표 1> 예: 6
    report(2, ok and ex == 6, f"전수 열거 n<=10 일치, (n,k,r)=(5,3,2) -> {ex}")


def p03():
    a, h = sp.symbols("a h", positive=True)
    tt = sp.symbols("t", positive=True)
    # (3-1) 밑변 위 직사각형: 넓이 t(1-t)ah 최대
    S1 = tt * (1 - tt) * a * h
    t1 = sp.solve(sp.diff(S1, tt), tt)[0]
    ratio1 = sp.simplify(S1.subs(tt, t1) / (a * h / 2))          # 직사각형/삼각형
    area1 = sp.solve(sp.Eq(sp.Symbol("T") * (1 - ratio1), 43), sp.Symbol("T"))[0]
    ok1 = sp.simplify(area1 - sp.sympify(md_answer(3, "1"))) == 0
    # (3-2) 배치1: 두 번째 직사각형이 삼각형 APS 안 (넓이 t^2 ah/4)
    S2 = a * h * (tt * (1 - tt) + tt**2 / 4)
    t2 = [s for s in sp.solve(sp.diff(S2, tt), tt) if 0 < s < 1][0]
    r2 = sp.simplify(S2.subs(tt, t2) / (a * h / 2))
    # 배치2: 모서리 삼각형(밑변 비율 lam) 안 — 격자 탐색으로 최대 비율이 배치1 이하인지 확인
    best_alt = max((tv * (1 - tv) + lam * (1 - tv)**2 / 4) / 0.5
                   for tv in [i / 400 for i in range(1, 400)] for lam in [j / 20 for j in range(0, 21)])
    area2 = sp.solve(sp.Eq(sp.Symbol("T") * (1 - r2), 47), sp.Symbol("T"))[0]
    ok2 = sp.simplify(area2 - sp.sympify(md_answer(3, "2"))) == 0 and best_alt <= float(r2) + 1e-9
    report(3, ok1 and ok2, f"(3-1) 최대비 {ratio1}, 넓이 {area1}; (3-2) t={t2}, 최대비 {r2} (모서리배치 최대 {best_alt:.6f}), 넓이 {area2}")


# ---- 문제 4: 정사각형 개수의 정확 계산(분수 산술) ----
_V = [(Fr(-1, 3), Fr(1, 5)), (Fr(-1, 3), Fr(4, 3)), (Fr(2, 3), Fr(4, 3)), (Fr(9, 5), Fr(1, 5))]
_SEGS = [(_V[i], _V[(i + 1) % 4]) for i in range(4)]


def _clip(p, q, x0, x1, y0, y1):
    t0, t1 = Fr(0), Fr(1)
    dx, dy = q[0] - p[0], q[1] - p[1]
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return None
        else:
            r = qq / pp
            if pp < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    return None if t0 > t1 else (t0, t1)


def _inside(px, py):
    return px >= Fr(-1, 3) and Fr(1, 5) <= py <= Fr(4, 3) and px + py <= 2


def _counts(m):
    hh = Fr(1, 2**m)
    f = g = 0
    for i in range(math.floor(Fr(-1, 3) / hh) - 1, math.ceil(Fr(9, 5) / hh) + 1):
        for j in range(math.floor(Fr(1, 5) / hh) - 1, math.ceil(Fr(4, 3) / hh) + 1):
            x0, x1, y0, y1 = i * hh, (i + 1) * hh, j * hh, (j + 1) * hh
            if all(_inside(u, v) for u in (x0, x1) for v in (y0, y1)):
                f += 1
            pts, pos = set(), False
            for p, q in _SEGS:
                c = _clip(p, q, x0, x1, y0, y1)
                if c:
                    pos = pos or c[1] > c[0]
                    pts.add((p[0] + c[0] * (q[0] - p[0]), p[1] + c[0] * (q[1] - p[1])))
            if pos or len(pts) >= 2:
                g += 1
    return f, g


def p04():
    # 넓이 A, 둘레 기여 L 을 도형에서 독립 계산
    X, Y = sp.symbols("X Y")
    A = sp.integrate(sp.integrate(1, (X, -sp.Rational(1, 3), 2 - Y)), (Y, sp.Rational(1, 5), sp.Rational(4, 3)))
    L = (sp.Rational(4, 3) - sp.Rational(1, 5)) + 1 + (sp.Rational(9, 5) + sp.Rational(1, 3)) + (sp.Rational(9, 5) - sp.Rational(2, 3))
    ok_num = True
    errs, fs, gs = [], {}, {}
    for m in range(4, 10):
        f, g = _counts(m)
        fs[m], gs[m] = f, g
        ef = (float(A) - f / 4**m) * 2**m          # 2^n(A-4^{-n}f) 유계
        eg = (float(L) - g / 2**m) * 2**m          # 2^n(L-2^{-n}g) 유계
        errs.append((m, round(ef, 3), round(eg, 3)))
        ok_num = ok_num and 0 <= ef < 10 and abs(eg) < 6
    # alpha: b_n = alpha n + ln f(n), f ~ A 4^n
    p1, p2 = parts(md_answer(4, "1")), parts(md_answer(4, "2"))
    # 수렴하려면 alpha = -lim ln(f(n+1)/f(n)), beta = -lim ln(g(n+1)/g(n)) (계산값과 비교)
    al_num = -math.log(fs[9] / fs[8])
    be_num = -math.log(gs[9] / gs[8])
    al_md, be_md = sp.sympify(p1["alpha"]), sp.sympify(p2["beta"])
    ok1 = abs(float(al_md) - al_num) < 0.01 and sp.simplify(sp.exp(-al_md)).is_Integer \
        and sp.simplify(sp.sympify(p1["limit"]) - sp.log(A)) == 0
    ok2 = abs(float(be_md) - be_num) < 0.01 and sp.simplify(sp.exp(-be_md)).is_Integer \
        and sp.simplify(sp.sympify(p2["limit"]) - sp.log(L)) == 0
    report(4, ok_num and ok1 and ok2, f"A={A} L={L} alpha~{al_num:.5f} beta~{be_num:.5f} 오차*2^n={errs}")


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
