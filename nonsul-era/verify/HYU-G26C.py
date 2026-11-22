#!/usr/bin/env python3
"""HYU-G26C (한양대 2026학년도 논술 자연계열 오후2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from fractions import Fraction as Fr

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "HYU-G26C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def parts(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 1. 5점 중 3점 선택(10가지), 무게중심과 직선 2x+y-30=0 사이 거리 ≤ 6√5
    pts = [(-6, -3), (-6, 3), (-3, 6), (6, 3), (6, -3)]
    cnt = 0
    for tri in itertools.combinations(pts, 3):
        gx, gy = sp.Rational(sum(p[0] for p in tri), 3), sp.Rational(sum(p[1] for p in tri), 3)
        if abs(2 * gx + gy - 30) / sp.sqrt(5) <= 6 * sp.sqrt(5):
            cnt += 1
    ok1 = cnt == int(md_answer(1, "1"))
    # 2. 좌표로 작도: A1(-1,1) B1(-1,-1) C1(1,-1) D1(1,1), F1(0,-1), G1(1,0), 내접원 단위원
    mp.mp.dps = 40
    th = mp.acos(mp.sqrt(5) / 5)
    F1, G1 = mp.matrix([0, -1]), mp.matrix([1, 0])

    def ang(phi):
        E = mp.matrix([mp.cos(phi), mp.sin(phi)])
        u, v = E - G1, F1 - G1
        return mp.acos((u[0] * v[0] + u[1] * v[1]) / (mp.norm(u) * mp.norm(v)))
    # 그림처럼 E1 은 왼쪽 위 호(π/2<φ<π)
    phi = mp.findroot(lambda p: ang(p) - th, 2.4)
    E1 = mp.matrix([mp.cos(phi), mp.sin(phi)])
    ok_e = mp.pi / 2 < phi < mp.pi

    def line_x_at_y(P, Q, y):
        return P[0] + (Q[0] - P[0]) * (y - P[1]) / (Q[1] - P[1])

    def line_y_at_x(P, Q, xx):
        return P[1] + (Q[1] - P[1]) * (xx - P[0]) / (Q[0] - P[0])

    def side_err(yc):                       # C2=(yc+1, yc) ∈ F1G1, B2 ∈ E1F1(같은 y), D2 ∈ E1G1(같은 x)
        xc = yc + 1
        xb = line_x_at_y(E1, F1, yc)
        yd = line_y_at_x(E1, G1, xc)
        return (yd - yc) - (xc - xb)        # 정사각형: 세로 = 가로
    yc = mp.findroot(side_err, -0.5)
    side2 = (yc + 1) - line_x_at_y(E1, F1, yc)
    q = side2 / 2                           # 닮음비(같은 각 조건으로 매 단계 같은 비)
    total = 4 / (1 - q**2)
    ok2 = ok_e and abs(total - mp.mpf(sp.N(sp.sympify(md_answer(1, "2")), 40))) < 1e-25
    # 3. 좌표: E 원점, D(3,0,0), F(0,3,0), 위 밑면 높이 6
    tt = sp.symbols("t", positive=True)
    A, B, C = sp.Matrix([3, 0, 6]), sp.Matrix([0, 0, 6]), sp.Matrix([0, 3, 6])
    D, F = sp.Matrix([3, 0, 0]), sp.Matrix([0, 3, 0])
    P = A + sp.Rational(2, 3) * (C - A)
    M = (D + F) / 2
    Q, R = sp.Matrix([tt, 0, 0]), sp.Matrix([0, 3 - tt, 0])
    S = B + sp.Rational(2, 3) * (M - B)                       # BS:SM = 2:1
    nrm = (Q - P).cross(R - P)
    cond_S = sp.expand(nrm.dot(S - P))                         # S 가 평면 PQR 위
    v = sp.symbols("v")
    N = P + v * (S - P)
    N = N.subs(v, sp.solve(N[2], v)[0])                         # 직선 PS 와 평면 DEF(z=0) 의 교점 = QR 위
    cond_perp = sp.expand((N - M).dot(R - Q))
    tsol = [r for r in sp.solve([cond_S, cond_perp], tt, dict=True) if 0 < r[tt] < 3]
    nn = nrm.subs(tsol[0])
    cos = sp.nsimplify(sp.simplify(abs(nn[2]) / sp.sqrt(nn.dot(nn))))
    ok3 = len(tsol) == 1 and sp.simplify(cos - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"count={cnt} phi={mp.nstr(phi, 8)} side2={mp.nstr(side2, 12)} sum={mp.nstr(total, 15)} t={tsol} cos={cos}")


def p02():
    # 1. n=3..20 격자 최댓값·최솟값, 증가 확인
    md = parts(md_answer(2, "1"))
    ok1 = True
    xs = np.linspace(0, 1, 20001)
    for m in range(3, 21):
        fv = ((1 + xs)**m + (1 - xs)**m) / (1 + xs**m)
        mx, mn = sp.sympify(md["max"], locals={"n": n}).subs(n, m), sp.sympify(md["min"], locals={"n": n}).subs(n, m)
        ok1 = ok1 and np.all(np.diff(fv) > -1e-12) and abs(fv.max() - float(mx)) < 1e-9 * float(mx) \
            and abs(fv.min() - float(mn)) < 1e-12
    # 2. k 몇 개에 대해 s 격자에서 조건을 만족하는 s 중 목표식의 최댓값, s=1/k 에서 정확값
    ok2 = True
    kk = sp.symbols("k", positive=True)
    mdexpr = sp.sympify(md_answer(2, "2"), locals={"k": kk})
    for kv in (sp.Rational(3, 2), sp.Integer(2), sp.Integer(3)):
        k = float(kv)
        ss = np.linspace(0, 1, 200001)
        f = -ss**3 + (1 / k + 1) * ss**2 - (1 / k + 1) * ss
        ra = np.clip(1 - ss**8, 0, None)**(1 / 8)
        rb = np.clip(1 - f**8, 0, None)**(1 / 8)
        g, h = (ra + rb) / 2, (ss + f) / 2
        feas = g**8 + h**8 >= 1 - 1 / k**8 - 1e-12
        target = (ra - rb)**8 + (ss - f)**8
        grid_max = target[feas].max()
        # s=1/k 에서 정확 계산
        s0 = 1 / kv
        f0 = -s0**3 + (1 / kv + 1) * s0**2 - (1 / kv + 1) * s0
        A_ = (1 - s0**8)**sp.Rational(1, 8)
        B_ = (1 - f0**8)**sp.Rational(1, 8)
        feas0 = sp.simplify(((A_ + B_) / 2)**8 + ((s0 + f0) / 2)**8 - (1 - 1 / kv**8)) >= 0
        val0 = sp.simplify((A_ - B_)**8 + (s0 - f0)**8)
        mdv = mdexpr.subs(kk, kv)
        ok2 = ok2 and bool(feas0) and sp.simplify(val0 - mdv) == 0 and grid_max <= float(mdv) * (1 + 1e-6)
    # 3. 정확 산술로 h(s) 의 불연속점 개수를 세어 조건(개수 2)을 만족하는 k 탐색
    def count(nn, k, s):
        if s < 0:
            return 0
        c = 0
        for val in ({s - k, -s - k} if s > 0 else {-k}):          # g(x) = ±s - k
            Dl = val - 2                                           # 왼쪽: (x+n)^2+2 = val, x≤0
            if Dl == 0:
                c += 1
            elif Dl > 0:
                c += 1 + (1 if Dl <= nn * nn else 0)
            Dr = val - 2 + 8 * nn * nn                             # 오른쪽: (x-3n)^2+2-8n^2 = val, x>0
            if Dr == 0:
                c += 1
            elif Dr > 0:
                c += 1 + (1 if Dr < 9 * nn * nn else 0)
        return c

    def ndisc(nn, k):
        cands = {Fr(0), abs(Fr(nn * nn + 2) + k), abs(k + 2), abs(k + 2 - 8 * nn * nn)}
        e = Fr(1, 10**9)
        return sum(1 for p in cands if len({count(nn, k, p - e), count(nn, k, p), count(nn, k, p + e)}) > 1)
    an = {}
    for nn in range(1, 11):
        hits = [Fr(j, 4) for j in range(1, 4 * 10 * nn * nn + 1) if ndisc(nn, Fr(j, 4)) == 2]
        an[nn] = hits
        # 찾은 k 의 아주 가까운 이웃은 조건 불만족(고립점)
        for kh in hits:
            if ndisc(nn, kh + Fr(1, 10**6)) == 2 or ndisc(nn, kh - Fr(1, 10**6)) == 2:
                an[nn] = None
    ok3 = all(v is not None and len(v) == 1 for v in an.values())
    total = sum(v[0] for v in an.values()) if ok3 else None
    ok3 = ok3 and total == int(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"max/min ok={ok1} target ok={ok2} a_n={[str(v[0]) for v in an.values()] if ok3 else an} sum={total}")


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
