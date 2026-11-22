#!/usr/bin/env python3
"""PNU-G25 (부산대 2025학년도 수시 논술 자연계 1~3번) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지. 문제 설정으로부터 독립 재계산한 값과 YAML answer 를 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "PNU-G25"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/PNU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    math = [50, 60, 70, 80, 90, 100]
    sci = [70, 50, 60, 70, 100, 90]
    U = [(i, j) for i in range(6) for j in range(6)]
    P = {(i, j) for i, j in U if math[i] > math[j]}
    Q = {(i, j) for i, j in U if sci[i] < sci[j]}
    a1 = kv(md_answer(1, "1"))
    ok1 = (int(a1["n(P)"]), int(a1["n(Q)"]), int(a1["n(PUQ)"])) == (len(P), len(Q), len(P | Q))
    # [1-2] 4명 순열 전수 열거
    cnt = 0
    for perm in itertools.permutations(range(6), 4):
        hi = [math[k] >= 80 for k in perm]
        if not any(hi[k] and hi[k + 1] for k in range(3)):
            cnt += 1
    ok2 = int(md_answer(1, "2")) == cnt
    report(1, ok1 and ok2, f"n(P)={len(P)} n(Q)={len(Q)} n(PUQ)={len(P | Q)} PQ∩={sorted((i+1, j+1) for i, j in P & Q)} arr={cnt}")


def p02():
    # [2-1] 8개 구간 형태 2^8 전수: 0=상수, 1=-cos2x/2+C. f(0)=2 부터 연속으로 상수 결정
    k8 = [sp.pi * k / 4 for k in range(9)]
    sols = []
    for ch in itertools.product([0, 1], repeat=8):
        # 미분가능: 접합점 x=kπ/4 (k=1..7) 에서 좌우 도함수 일치
        okd = True
        for k in range(1, 8):
            dl = sp.sin(2 * k8[k]) if ch[k - 1] else 0
            dr = sp.sin(2 * k8[k]) if ch[k] else 0
            if sp.simplify(dl - dr) != 0:
                okd = False
                break
        if not okd:
            continue
        pieces, val = [], sp.Integer(2)
        for k in range(8):
            if ch[k]:
                C = val + sp.cos(2 * k8[k]) / 2
                expr = -sp.cos(2 * x) / 2 + C
            else:
                expr = val
            pieces.append(expr)
            val = sp.sympify(expr).subs(x, k8[k + 1])
        if sp.simplify(val - 1) != 0:
            continue
        xs = np.linspace(0, 2 * np.pi, 80001)
        fx = np.empty_like(xs)
        fl = [sp.lambdify(x, e, "numpy") for e in pieces]
        idx = np.minimum((xs / (np.pi / 4)).astype(int), 7)
        for k in range(8):
            m = idx == k
            fx[m] = np.broadcast_to(fl[k](xs[m]), xs[m].shape)
        mx = fx.max()
        near = xs[fx > mx - 1e-9]
        if near.max() - near.min() > 1e-3:   # 최댓값 갖는 점이 여러 개(또는 구간)
            continue
        sols.append((ch, fl))
    ok_unique = len(sols) == 1
    ch, fl = sols[0]
    fmd = sp.lambdify(x, sp.sympify(md_answer(2, "1"), locals={"x": x}), "numpy")
    fnum = lambda v: float(fl[min(int(v / (np.pi / 4)), 7)](v))
    test = np.linspace(0, 2 * np.pi, 997)
    ok1 = ok_unique and all(abs(fnum(v) - float(fmd(v))) < 1e-12 for v in test)
    # [2-2] g(t) = max{x : f(x) >= f(t)} 를 격자로 독립 계산
    xs = np.linspace(0, 2 * np.pi, 400001)
    fvec = np.vectorize(fnum)
    fx = fvec(xs)
    gmd_expr = sp.sympify(md_answer(2, "2"), locals={"t": t})
    gmd = lambda v: float(gmd_expr.subs(t, v))
    err, worst = 0.0, None
    for v in list(np.linspace(0.013, 2 * np.pi, 211)) + [sp.Integer(0), sp.pi, 3 * sp.pi / 2, sp.pi / 2, 2 * sp.pi]:
        ft = fnum(float(v))
        gnum = xs[fx >= ft - 1e-7].max()   # 경계점은 기호값으로 넣어 부동소수 비교 오류 방지
        e = abs(gnum - gmd(v))
        if e > err:
            err, worst = e, v
    ok2 = err < 1e-3
    # [2-3] 검증된 g 로 ∫ g sin t 를 기호적분 + 수치적분 교차
    I_sym = sp.integrate(gmd_expr * sp.sin(t), (t, 0, 2 * sp.pi))
    import mpmath as mp
    I_num = mp.quad(lambda v: gmd(float(v)) * mp.sin(v), [0, mp.pi / 2, mp.pi, 3 * mp.pi / 2, 2 * mp.pi])
    ans3 = sp.sympify(md_answer(2, "3"))
    ok3 = sp.simplify(I_sym - ans3) == 0 and abs(float(I_num) - float(ans3)) < 1e-9
    report(2, ok1 and ok2 and ok3, f"choice={ch} unique={ok_unique} g_maxerr={err:.2e}@t={worst} I={sp.simplify(I_sym)}")


def p03():
    A, B, C, D = map(np.array, ([1, 0], [1, -1], [3, 0], [2, 1]))
    # [3-1] R 영역 = 선분쌍 Minkowski 합(평행사변형 4개)의 합집합. 격자로 넓이 추정
    segsP, segsQ = [(A, B), (B, C)], [(A, D), (D, C)]
    h = 0.002
    X, Y = np.meshgrid(np.arange(1.5, 6.5, h) + h / 2, np.arange(-1.5, 1.5, h) + h / 2)
    inside = np.zeros_like(X, dtype=bool)
    for p0, p1 in segsP:
        for q0, q1 in segsQ:
            o, u, w = p0 + q0, p1 - p0, q1 - q0
            M = np.array([u, w]).T
            Mi = np.linalg.inv(M)
            s = Mi[0, 0] * (X - o[0]) + Mi[0, 1] * (Y - o[1])
            r = Mi[1, 0] * (X - o[0]) + Mi[1, 1] * (Y - o[1])
            inside |= (s >= 0) & (s <= 1) & (r >= 0) & (r <= 1)
    area_grid = inside.sum() * h * h
    # 같은 영역의 경계 다각형(교점 E, F 는 sympy 로 직접 계산) 신발끈 넓이 — 격자와 교차 확인
    xx = sp.symbols("xx")
    Ex = sp.solve(sp.Eq(-(xx - 4), (xx - 3) / 2), xx)[0]
    Fx = sp.solve(sp.Eq((xx - 2) / 2 - 1, -(xx - 3)), xx)[0]
    poly = [(2, 0), (3, 1), (Ex, -(Ex - 4)), (5, 1), (6, 0), (4, -1), (Fx, -(Fx - 3)), (2, -1)]
    shoe = sp.Rational(1, 2) * abs(sum(poly[i][0] * poly[(i + 1) % 8][1] - poly[(i + 1) % 8][0] * poly[i][1] for i in range(8)))
    ans1 = sp.sympify(md_answer(3, "1"))
    ok1 = shoe == ans1 and abs(area_grid - float(ans1)) < 0.02
    # [3-2] P: 꺾은선 ABC, S: 위쪽 반원. 격자 전수 + 기호 최솟값
    ss = np.linspace(0, 1, 2001)
    Ps = np.vstack([A + np.outer(ss, B - A), B + np.outer(ss, C - B)])
    ph = np.linspace(0, np.pi, 2001)
    Ss = np.stack([2 + np.cos(ph), np.sin(ph)], axis=1)
    dots = Ps @ Ss.T
    a2 = kv(md_answer(3, "2"))
    mx_md, mn_md = sp.sympify(a2["max"]), sp.sympify(a2["min"])
    phi = sp.symbols("phi", real=True)
    expr_B = 1 * (2 + sp.cos(phi)) + (-1) * sp.sin(phi)               # P=B 일 때 내적
    crit = list(sp.solveset(sp.diff(expr_B, phi), phi, sp.Interval(0, sp.pi))) + [0, sp.pi]
    mn_sym = sp.nsimplify(min(sp.simplify(expr_B.subs(phi, c)) for c in crit), [sp.sqrt(2)])
    ok2 = abs(dots.max() - float(mx_md)) < 1e-9 and abs(dots.min() - float(mn_md)) < 1e-5 \
        and sp.simplify(mn_sym - mn_md) == 0
    report(3, ok1 and ok2, f"area_shoelace={shoe} area_grid={area_grid:.4f} max={dots.max():.6f} min={dots.min():.6f} min_sym={mn_sym}")


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
