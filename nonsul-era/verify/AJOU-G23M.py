#!/usr/bin/env python3
"""AJOU-G23M (아주대 2023학년도 논술 자연계열 의학과 수학 문항) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G23M"
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


def _cut(poly):
    """y축 위 교점(둘째 꼭짓점)에서 깎아 만든 도형: P=(0,1), Q, R → P, M, N, R."""
    P, Q, R = poly[0], poly[1], poly[2]
    M = ((P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2)
    N = ((Q[0] + R[0]) / 2, (Q[1] + R[1]) / 2)
    return [P, M, N] + poly[2:], (M, N)


def p01():
    import random
    import numpy as np
    # [1-1](1): 무작위 삼각형으로 크기 비교 [증명-수치확인]
    random.seed(3)
    sign = set()
    for _ in range(20000):
        P, Q, R = [np.array([random.uniform(-5, 5), random.uniform(-5, 5)]) for _ in range(3)]
        if abs((Q - P)[0] * (R - P)[1] - (Q - P)[1] * (R - P)[0]) < 1e-6:
            continue
        M, N = (P + Q) / 2, (Q + R) / 2
        d = lambda u, v: float(np.sum((u - v)**2))
        sign.add(np.sign(d(P, Q) + d(Q, R) - (d(P, M) + d(M, N) + d(N, R))))
    ok1 = sign == {1.0} and ">" in md_answer(1, "1-1")
    # [1-1](2): 시행 10회를 정확히(유리수) 시뮬레이션 → 도형과 y=-x+1 사이 넓이(신발끈)
    half = sp.Rational(1, 2)
    poly = [(sp.Integer(0), sp.Integer(1)), (sp.Integer(0), sp.Integer(0)), (sp.Integer(1), sp.Integer(0))]
    slopes = []
    for k in range(1, 61):
        poly, (M, N) = _cut(poly)
        slopes.append((N[1] - M[1]) / (N[0] - M[0]))
        if k == 10:
            pts = poly + [poly[0]]
            area = abs(sum(pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1] for i in range(len(poly)))) * half
    ok2 = sp.simplify(area - sp.sympify(md_answer(1, "1-2"))) == 0
    # [1-1](3): n=1..60 기울기 = -n [증명-수치확인]
    ok3 = all(slopes[k - 1] == -k for k in range(1, 61))
    # [1-2](1): f 결정 후, 기울기 m 를 훑어 (1,1) 지나는 직선과 |f| 교점 3개인 m 찾기 → r 합
    a3 = sp.Rational(2, 25)
    b2, b1 = sp.symbols("b2 b1")
    f = a3 * x**3 + b2 * x**2 + b1 * x
    so = sp.solve([sp.diff(f, x).subs(x, 0) - sp.diff(f, x).subs(x, 7), f.subs(x, 5) - 1], [b2, b1], dict=True)[0]
    f = f.subs(so)
    Mp = ((2 + 0) / sp.Integer(2), (f.subs(x, 2) - sp.Rational(2, 25)) / 2)        # PQ 중점
    fc = [float(v) for v in sp.Poly(f, x).all_coeffs()]
    def count(mv):                              # 직선 y = mv(x-Mx)+My 와 y=|f| 교점 수(수치, 중근 허용오차)
        mvf = float(mv); c0 = float(Mp[1]) - mvf * float(Mp[0])
        pts = []
        for sgn, keep in ((1, lambda r: r >= -1e-9), (-1, lambda r: r <= 1e-9)):
            co = [sgn * fc[0], sgn * fc[1], sgn * fc[2] - mvf, sgn * fc[3] - c0]
            for r in np.roots(co):
                if abs(r.imag) < 1e-5 and keep(r.real):
                    pts.append(r.real)
        pts.sort()
        uniq = [q for i, q in enumerate(pts) if i == 0 or abs(q - pts[i - 1]) > 1e-4]
        return len(uniq)
    # 교점 수가 바뀌는 특수 기울기: 원점 통과, 접선(이중근), 그리고 일반 기울기 표본
    mm = sp.symbols("mm")
    specials = set()
    specials.add((Mp[1] - 0) / (Mp[0] - 0))
    for sgn in (1, -1):
        disc = sp.discriminant(sp.expand(sgn * f - (mm * (x - Mp[0]) + Mp[1])), x)
        for r in sp.Poly(disc, mm).real_roots():
            specials.add(r)
    hits = [s_ for s_ in specials if count(s_) == 3]
    samp = [sp.Rational(k, 40) for k in range(-400, 401)]
    hits_gen = [s_ for s_ in samp if count(s_) == 3 and s_ not in specials]
    rsum = 0
    for mv in hits:          # 선분 PR 의 기울기 = mv (중점연결정리) → r 들
        co = [float(v) for v in sp.Poly(sp.expand(f - f.subs(x, 2)), x).all_coeffs()]
        co[2] -= float(mv); co[3] += 2 * float(mv)
        rr = {round(r.real, 6) for r in np.roots(co) if abs(r.imag) < 1e-5 and r.real > 0 and abs(r.real - 2) > 1e-4}
        rsum += sum(rr)
    ok4 = not hits_gen and abs(rsum - float(sp.sympify(md_answer(1, "2-1")))) < 1e-6
    # [1-2](2): k 를 훑어 y=-|x|/√e+k 와 y=e^{-|x|} 교점이 정확히 2개인 경우의 적분값 최소
    import math
    xs = np.linspace(0, 12, 120001)
    best = None
    for kv in np.linspace(0.85, 3, 4301):
        hv = np.exp(-xs) + xs / math.sqrt(math.e) - kv              # x≥0 에서의 차
        pos = [i for i in range(1, len(xs) - 1) if hv[i] * hv[i + 1] < 0]
        tang = abs(hv.min()) < 1e-7 and hv.argmin() > 0            # 접하는 경우
        tot = 2 * len(pos) + (1 if abs(hv[0]) < 1e-12 else 0) + (2 if tang else 0)
        if tot != 2:
            continue
        pv = xs[hv.argmin()] if tang else xs[pos[0]]
        yv = (1 + math.exp(-pv)) / 2                                # 두 중점을 지나는 직선 y=(1+e^{-p})/2
        beta = -math.log(yv)
        val = 2 * (1 - math.exp(-beta))                             # ∫_{-β}^{β} e^{-|x|}
        best = val if best is None or val < best else best
    # 접선 기울기 조건으로 접점 직접 계산: -e^{-p} = -1/√e
    pt = sp.solve(sp.Eq(-sp.exp(-x), -1 / sp.sqrt(sp.E)), x)[0]
    tangv = sp.simplify(2 * (1 - (1 + sp.exp(-pt)) / 2))
    ok5 = sp.simplify(tangv - sp.sympify(md_answer(1, "2-2"))) == 0 and best >= float(tangv) - 1e-9
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] 비교부호={sign}; 넓이={area}; 기울기 -n (n≤60)={ok3}; 3교점 기울기={[sp.nsimplify(h) if h.is_rational else round(float(h), 6) for h in hits]} r합={rsum:.6f}; 최솟값={tangv} (격자 최소≈{best:.6f})")


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
