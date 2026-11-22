#!/usr/bin/env python3
"""HYU-G24B (한양대 2024학년도 논술 자연계열 오후1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys
from functools import lru_cache

import mpmath as mp
import sympy as sp

PREFIX = "HYU-G24B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", positive=True)
mp.mp.dps = 40


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {k.strip(): sp.sympify(v) for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) 5x5x1 직육면체 격자: (i,j,k), 0<=i,j<=5, 0<=k<=1. 색칠된 정육면체 = 맨 위 줄 왼쪽 → i∈{0,1}, j∈{4,5}
    bad = {(i, j, k) for i in (0, 1) for j in (4, 5) for k in (0, 1)}

    @lru_cache(None)
    def cnt(i, j, k):
        if (i, j, k) in bad:
            return 0
        if (i, j, k) == (0, 0, 0):
            return 1
        return sum(cnt(*p) for p in ((i - 1, j, k), (i, j - 1, k), (i, j, k - 1)) if min(p) >= 0)
    n1 = cnt(5, 5, 1)
    ok1 = n1 == int(sp.sympify(md_answer(1, "1")))
    # (2) 전수: a∈A(4,4,4,6,6,8), b∈B(0×6,1×4)
    A = [4] * 3 + [6] * 2 + [8]
    B = [0] * 6 + [1] * 4
    E = 0
    for a, b in itertools.product(A, B):
        X = sp.symbols("X")
        roots = sorted(sp.solve((X + 7) * (X**2 - a * b * X + sp.Rational(a, 2) - 6), X), key=lambda r: float(r))
        assert len(roots) == 3
        gap = min(roots[1] - roots[0], roots[2] - roots[1], key=lambda r: float(r))
        E += sp.Rational(1, len(A) * len(B)) * gap
    ok2 = sp.simplify(E - sp.sympify(md_answer(1, "2"))) == 0
    # (3) g,h 를 적분으로 구성(도함수 검증) 후 극한
    f = 1 / sp.sin(2 * x)**2
    G = sp.integrate(f, x)
    g = G - G.subs(x, sp.pi / 6)
    ok_g = sp.simplify(sp.diff(g, x) - f) == 0
    H = sp.integrate(g, x)
    h = H - H.subs(x, sp.pi / 6)
    ok_h = sp.simplify(sp.diff(h, x) - g) == 0
    L = sp.limit((h + g**2) / f, x, 0, "+")
    ok3 = ok_g and ok_h and sp.simplify(L - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"paths={n1} E(X)={sp.nsimplify(sp.expand(E))} lim={L}")


def p02():
    th = sp.symbols("theta", real=True)
    # (1) Q=(cosθ,sinθ), PQ=1 ⇒ P 는 θ+π/3, 작은 반원이 AB에 접 ⇒ 중점 높이 = 1/2
    sols = [s for s in sp.solve(sp.Eq((sp.sin(th + sp.pi / 3) + sp.sin(th)) / 2, sp.Rational(1, 2)), th)
            if 0 < float(s) < float(sp.pi / 2)]
    assert len(sols) == 1
    t0 = sols[0]
    ans1 = kv(md_answer(2, "1"))
    ok1 = abs(sp.N(sp.cos(t0) - ans1["cos"], 30)) < 1e-25 and abs(sp.N(sp.sin(t0) - ans1["sin"], 30)) < 1e-25
    # 조건 QB<PB 확인
    tf = mp.mpf(str(sp.N(t0, 40)))
    okc = mp.sqrt((mp.cos(tf) - 1)**2 + mp.sin(tf)**2) < mp.sqrt((mp.cos(tf + mp.pi / 3) - 1)**2 + mp.sin(tf + mp.pi / 3)**2)
    # (2) T 를 선분 PQ 위에서 움직이며 ST = 호 높이 - 선분 높이 최대(수치)
    Px, Py = mp.cos(tf + mp.pi / 3), mp.sin(tf + mp.pi / 3)
    Qx, Qy = mp.cos(tf), mp.sin(tf)
    def st(u):
        X = Px + u * (Qx - Px)
        Y = Py + u * (Qy - Py)
        return mp.sqrt(1 - X**2) - Y
    grid = [i / 2000 for i in range(2001)]
    u0 = max(grid, key=lambda u: st(mp.mpf(u)))
    um = mp.findroot(lambda u: mp.diff(st, u), u0)
    smax = st(um)
    ok2 = abs(smax - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "2")), 40)))) < 1e-25
    # (3) 3D: β 의 법선 n=(sinφ,0,cosφ), β 와 O 거리 d, 원 C 반지름 sqrt(1-d^2)=1/2.
    #     C 를 밑면으로 하는 반구(O 쪽, -n 방향)의 최저점 높이 = d cosφ - 1/2 = 0 (α 에 한 점에서 접)
    d = sp.sqrt(1 - sp.Rational(1, 4))
    c = sp.symbols("c", positive=True)
    cphi = sp.solve(sp.Eq(d * c - sp.Rational(1, 2), 0), c)[0]
    # 최저점이 실제로 -n 쪽 반구 위에 있음(아래 방향 -z 와 -n 의 내적 >= 0) 확인 후 정사영 넓이
    area = sp.pi * sp.Rational(1, 4) * cphi
    ok3 = cphi > 0 and sp.simplify(area - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and okc and ok2 and ok3,
           f"cos={sp.N(sp.cos(t0), 12)} sin={sp.N(sp.sin(t0), 12)} STmax={mp.nstr(smax, 15)} area={area}")


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
