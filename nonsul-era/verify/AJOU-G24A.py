#!/usr/bin/env python3
"""AJOU-G24A (아주대 2024학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "AJOU-G24A"
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
    # [1-1](1) h = max(x^2/2-2, log2 x) on [2,3]: 격자로 최대·최소 위치를 찾고 정확값 계산
    f = x**2 / 2 - 2
    g = sp.log(x, 2)
    h = lambda v: max(float(f.subs(x, v)), float(g.subs(x, v)))
    grid = [2 + sp.Rational(i, 2000) for i in range(2001)]
    hv = [h(v) for v in grid]
    xmax, xmin = grid[hv.index(max(hv))], grid[hv.index(min(hv))]
    hmax = sp.Max(f.subs(x, xmax), g.subs(x, xmax))
    hmin = sp.Max(f.subs(x, xmin), g.subs(x, xmin))
    parts = dict(seg.split("=") for seg in md_answer(1, "1-1").replace(" ", "").split(";"))
    ok1 = sp.simplify(hmax - sp.sympify(parts["max"])) == 0 and sp.simplify(hmin - sp.sympify(parts["min"])) == 0
    # [1-1](2) h = max(sin2x, 3/4 cos x) on (-pi,pi): 극소점(이웃보다 작음) 중 최대 x
    F = lambda v: max(mp.sin(2 * v), mp.mpf(3) / 4 * mp.cos(v))
    N = 20000
    xs = [-mp.pi + mp.pi * 2 * (i + 1) / (N + 2) for i in range(N + 1)]
    ys = [F(v) for v in xs]
    mins = [xs[i] for i in range(1, N) if ys[i] <= ys[i - 1] and ys[i] <= ys[i + 1]]
    a0 = max(mins)
    alpha = mp.findroot(lambda v: mp.sin(2 * v) - mp.mpf(3) / 4 * mp.cos(v), a0)
    ok2 = abs(mp.tan(alpha) - mp.mpf(sp.N(sp.sympify(md_answer(1, "1-2")), 40))) < mp.mpf(10)**-25
    # [1-2](1) t=1/3: cos(angle POQ) = 2t^2-1
    tt = sp.Rational(1, 3)
    P = sp.Matrix([-sp.sqrt(1 - tt**2), tt]); Q = sp.Matrix([sp.sqrt(1 - tt**2), tt])
    cth = P.dot(Q)
    sth = sp.sqrt(1 - cth**2)
    ok3 = sp.simplify(sth - sp.sympify(md_answer(1, "2-1"))) == 0
    # [1-2](2) A(theta) = 활꼴 넓이를 직접 적분: t = cos(theta/2), A = ∫(sqrt(1-x^2)-t)dx
    def A(th):
        tv = mp.cos(th / 2); w = mp.sqrt(1 - tv**2)
        return mp.quad(lambda u: mp.sqrt(1 - u**2) - tv, [-w, w])
    I = mp.quad(A, [0, mp.pi])
    ok4 = abs(I - mp.mpf(sp.N(sp.sympify(md_answer(1, "2-2")), 40))) < mp.mpf(10)**-15
    # [1-2](3) B = 호(θ) + 현(두 교점 거리), 극한은 sympy
    th = sp.symbols("theta", positive=True)
    tv = sp.cos(th / 2)
    B = th + 2 * sp.sqrt(1 - tv**2)
    L = sp.limit(B / sp.sin(th), th, 0, "+")
    ok5 = sp.simplify(L - sp.sympify(md_answer(1, "2-3"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"max={hmax} min={hmin} tanα={mp.nstr(mp.tan(alpha), 12)} sinθ={sth} ∫A={mp.nstr(I, 12)} lim={L}")


def p02():
    import ast
    import random
    random.seed(1)
    # [2-1](1) f∘f ≡ 3 인 함수를 무작위로 여러 개 만들어 합 계산(정의역은 1..3000 으로 절단, 합은 1..2024)
    sums = set()
    for _ in range(5):
        S = {3} | set(random.sample(range(1, 3001), 50))
        Sl = sorted(S)
        fmap = {v: (3 if v in S else random.choice(Sl)) for v in range(1, 3001)}
        assert all(fmap[fmap[v]] == 3 for v in range(1, 3001))
        sums.add(sum(fmap[fmap[fmap[v]]] for v in range(1, 2025)))
    ok1 = sums == {int(md_answer(2, "1-1"))}
    # [2-1](2) k>1 에서 2-상수 아님 & 3-상수
    def comp_const(k, m):
        fk = lambda v: v + 3 if v < k else k + 3
        vals = set()
        for v in range(1, k + 20):
            w = v
            for _ in range(m):
                w = fk(w)
            vals.add(w)
        return len(vals) == 1
    ks = [k for k in range(2, 200) if not comp_const(k, 2) and comp_const(k, 3)]
    ok2 = ks == sorted(ast.literal_eval(md_answer(2, "1-2")))
    # [2-2] X={1,2,3,4} 전수: 4-상수, f^4(1)=4, 단조
    sol = []
    for fv in itertools.product(range(1, 5), repeat=4):
        f = lambda v: fv[v - 1]
        f4 = [f(f(f(f(v)))) for v in range(1, 5)]
        if len(set(f4)) == 1 and f4[0] == 4 and list(fv) == sorted(fv):
            sol.append(fv)
    ok3 = sorted(sol) == sorted(ast.literal_eval(md_answer(2, "2")))
    # [2-3] X={1,2,3} 전수: 어떤 n 에 대해 n-상수(n<=10 이면 충분: 3원소 함수는 3번 안에 안정)
    cnt = 0
    for gv in itertools.product(range(1, 4), repeat=3):
        for m in range(1, 11):
            vals = set()
            for v in range(1, 4):
                w = v
                for _ in range(m):
                    w = gv[w - 1]
                vals.add(w)
            if len(vals) == 1:
                cnt += 1
                break
    ok4 = cnt == int(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3 and ok4, f"Σa_n={sums} k={ks} 2-2 함수 {len(sol)}개 2-3 개수={cnt}")


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
