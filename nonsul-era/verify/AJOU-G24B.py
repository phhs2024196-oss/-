#!/usr/bin/env python3
"""AJOU-G24B (아주대 2024학년도 논술 자연계열 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "AJOU-G24B"
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


import mpmath as mp
mp.mp.dps = 40


def p01():
    T, X = sp.symbols("T X", positive=True)
    # [1-1](1) 법선(접선 기울기 2t 에 수직)과 포물선 사이 넓이(0<=x<=t)
    C = X**2 - 1
    slope = -1 / sp.diff(C, X).subs(X, T)
    LP = slope * (X - T) + C.subs(X, T)
    S = sp.integrate(sp.expand(LP - C), (X, 0, T))
    ok1 = sp.simplify(S - sp.sympify(md_answer(1, "1-1"), locals={"t": T})) == 0
    # [1-1](2) theta: 법선이 x축과 이루는 예각, tan(theta) = |slope| = 1/(2t) → t = 1/(2 tan θ)
    th = sp.symbols("theta", positive=True)
    S_th = S.subs(T, 1 / (2 * sp.tan(th)))
    th1 = sp.atan(sp.Rational(1, 2))
    rate = sp.simplify(sp.diff(S_th, th).subs(th, th1))
    ok2 = sp.simplify(rate - sp.sympify(md_answer(1, "1-2"))) == 0
    # [1-1](3) PQ 길이를 t 로 직접 계산해 최소화
    qx = sp.solve(sp.Eq(LP.subs(X, x), C.subs(X, x)), x)
    qx = [r for r in qx if sp.simplify(r - T) != 0][0]
    PQ = sp.sqrt((T - qx)**2 + (C.subs(X, T) - C.subs(X, qx))**2)
    f = sp.lambdify(T, PQ, "mpmath")
    t0 = min((mp.mpf(i) / 1000 for i in range(1, 5000)), key=f)
    tmin = mp.findroot(lambda v: mp.diff(f, v), t0)
    ok3 = abs(f(tmin) - mp.mpf(sp.N(sp.sympify(md_answer(1, "1-3")), 40))) < mp.mpf(10)**-20
    # [1-2](1) R = P + (법선 방향 단위벡터, x 감소 방향)
    d = sp.Matrix([-2 * T, 1]) / sp.sqrt(4 * T**2 + 1)
    assert sp.simplify(d[1] / d[0] - slope) == 0
    Rx, Ry = T + d[0], C.subs(X, T) + d[1]
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "2-1").replace(" ", "").split(";"))
    ok4 = sp.simplify(Rx - sp.sympify(parts["x"], locals={"t": T})) == 0 and \
        sp.simplify(Ry - sp.sympify(parts["y"], locals={"t": T})) == 0
    # (점 R 이 선분 PQ 위: |PQ|>=1 확인, t>=1)
    assert all(f(mp.mpf(v)) > 1 for v in (1, 1.5, 2, 3))
    # [1-2](2) 속력 → l(s) 의 정의로 극한을 수치 계산(s=2±h) + 정확값 speed(2)/4
    speed = sp.sqrt(sp.diff(Rx, T)**2 + sp.diff(Ry, T)**2)
    sp_f = sp.lambdify(T, speed, "mpmath")
    l = lambda s: mp.quad(sp_f, [1, s])
    h = mp.mpf(10)**-12
    lim_num = (l(2 + h) - l(2 - h)) / ((2 + h)**2 - (2 - h)**2)
    exact = sp.nsimplify(sp.simplify(speed.subs(T, 2) / 4))
    md = sp.sympify(md_answer(1, "2-2"))
    ok5 = abs(lim_num - mp.mpf(sp.N(md, 40))) < mp.mpf(10)**-15 and sp.simplify(sp.sqrtdenest(exact) - md) == 0
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"S={sp.factor(S)} dS/dθ={rate} minPQ={mp.nstr(f(tmin), 12)} lim≈{mp.nstr(lim_num, 15)}")


CARDS = ("가위", "바위", "보")
BEATER = {"보": "가위", "바위": "보", "가위": "바위"}   # BEATER[c] = c 를 이기는 카드


def is_f_arr(arr, fv):
    pos = {c: i + 1 for i, c in enumerate(arr)}
    return all(pos[BEATER[arr[i]]] <= fv[i] for i in range(3))


def p02():
    import ast
    # [2-1] 전수 세기
    c1 = sum(1 for _ in itertools.permutations(range(1, 8), 3))
    c2 = sum(1 for g, b, p in itertools.permutations(range(1, 10), 3) if g < b < p)
    c3 = sum(1 for g, b, p in itertools.permutations(range(1, 12), 3) if abs(g - p) == 1)
    ok1 = (c1, c2, c3) == (int(md_answer(2, "1-1")), int(md_answer(2, "1-2")), int(md_answer(2, "1-3")))
    arrs = list(itertools.permutations(CARDS))
    # [2-2](1)
    s1 = {"/".join(a) for a in arrs if is_f_arr(a, (2, 3, 3))}
    md1 = {seg.strip() for seg in md_answer(2, "2-1").split(";")}
    # [2-2](2)
    funcs = list(itertools.product((1, 2, 3), repeat=3))
    s2 = {g for g in funcs if g[0] == 2 and any(is_f_arr(a, g) for a in arrs)}
    md2 = {ast.literal_eval(seg.strip()) for seg in md_answer(2, "2-2").split(";")}
    # [2-2](3)
    s3 = {h for h in funcs if all(is_f_arr(a, h) for a in arrs)}
    md3 = {ast.literal_eval(seg.strip()) for seg in md_answer(2, "2-3").split(";")}
    # 제시문 예시 확인: f=(2,3,2) 에서 (바위/보/가위) O, (가위/보/바위) X
    ex = is_f_arr(("바위", "보", "가위"), (2, 3, 2)) and not is_f_arr(("가위", "보", "바위"), (2, 3, 2))
    ok2 = s1 == md1 and s2 == md2 and s3 == md3 and ex
    report(2, ok1 and ok2, f"{c1},{c2},{c3} / {sorted(s1)} / {sorted(s2)} / {sorted(s3)}")


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
