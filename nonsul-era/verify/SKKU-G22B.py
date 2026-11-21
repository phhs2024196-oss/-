#!/usr/bin/env python3
"""SKKU-G22B (성균관대 2022학년도 수시 논술우수전형 자연계 2교시 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SKKU-G22B"
WIDTH = 2
RESULTS = []
x, m, alpha = sp.symbols("x m alpha", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def rows(seq, nrows, rev_even):
    """수열을 삼각형으로 배열(k번째 줄 k개). rev_even 이면 짝수 줄 역순."""
    out, i = [], 0
    for k in range(1, nrows + 1):
        r = seq[i:i + k]
        i += k
        if rev_even and k % 2 == 0:
            r = r[::-1]
        out.append(r)
    return out


def p01():
    import random
    loc = {"a": sp.Symbol("a"), "b": sp.Symbol("b"), "sqrt": sp.sqrt}
    exprs = {k: sp.sympify(md_answer(1, k), locals=loc) for k in ("1", "2", "3", "4")}
    ok = True
    random.seed(1)
    for _ in range(6):
        av = sp.Rational(random.randint(1, 30), random.randint(1, 7))
        bv = av + sp.Rational(random.randint(1, 30), random.randint(1, 7))
        M, A3, A2, A1, A0 = sp.symbols("M A3 A2 A1 A0")
        f = -x**4 + A3 * x**3 + A2 * x**2 + A1 * x + A0
        fp = sp.diff(f, x)
        # 원점을 지나고 기울기 M>0 인 직선이 a, b 에서 접함
        sol = sp.solve([f.subs(x, av) - M * av, fp.subs(x, av) - M, f.subs(x, bv) - M * bv, fp.subs(x, bv) - M],
                       [A3, A2, A1, A0], dict=True)[0]
        F = f.subs(sol)
        cs = [r for r in sp.solve(sp.diff(F, x) - M, x) if av < r < bv]
        c = cs[0]
        fpp = sp.diff(F, x, 2)
        ds = [r for r in sp.solve(fpp, x) if c < r < bv and sp.diff(F, x, 3).subs(x, r) < 0]
        d = ds[0]
        L = (F.subs(x, bv) - F.subs(x, c)) / (bv - c) * (x - bv) + F.subs(x, bv)
        es = [r for r in sp.solve(sp.expand(F - L), x) if r.is_real and c < r < bv]
        e = es[0]
        s = {loc["a"]: av, loc["b"]: bv}
        ok &= len(cs) == 1 and len(ds) == 1 and len(es) == 1
        ok &= sp.simplify(c - exprs["1"].subs(s)) == 0
        ok &= sp.simplify(d - exprs["2"].subs(s)) == 0
        ok &= sp.simplify(e - exprs["3"].subs(s)) == 0
        ok &= sp.simplify((e - d) / (bv - av) - exprs["4"]) == 0
    report(1, ok, f"무작위 (a,b) 6쌍에서 c,d,e,(e-d)/(b-a) 재계산 일치; ratio={float(exprs['4']):.10f}")


def p02():
    import numpy as np
    # (i) 교점이 2개인 c<0 탐색: y1-y2 의 최솟값=0 이 되는 c (접할 때)
    X = sp.symbols("X", real=True)
    y1 = lambda t: 2 * t * (t - 2) * abs(t - 1)
    # x>=1 쪽: y1=2x(x-2)(x-1), y2=-(x-1)+c; 접함: y1'=-1
    tt = [r for r in sp.solve(sp.diff(2 * X * (X - 2) * (X - 1), X) + 1, X) if r > 1]
    t0 = tt[0]
    cval = sp.simplify(2 * t0 * (t0 - 2) * (t0 - 1) + (t0 - 1))
    area = 2 * sp.integrate(2 * X * (X - 2) * (X - 1) - (-(X - 1) + cval), (X, 1, t0))
    p1 = kv(md_answer(2, "1"))
    grid = np.linspace(-3, 5, 400001)
    d = 2 * grid * (grid - 2) * np.abs(grid - 1) - (-np.abs(grid - 1) + float(cval))
    ok1 = (sp.simplify(cval - sp.sympify(p1["c"])) == 0 and sp.simplify(area - sp.sympify(p1["area"])) == 0
           and d.min() > -1e-9 and cval < 0)
    # (ii) S(a): 곡선, 역함수, y=-x 로 둘러싸인 영역 — 그린 정리(경계 적분)로 정확 계산
    #   곡선 (u, c(u)) 와 역함수 (c(u), u) 의 기여가 같고, y=-x 위 선분의 기여는 0
    p2 = kv(md_answer(2, "2"))
    k = sp.Symbol("k")
    Sk = sp.sympify(p2["S(k)"], locals={"k": k})
    ok2 = True
    U = sp.Symbol("U", real=True)
    for av in (1, 2, 3, 5, sp.Rational(7, 3)):
        t1, t2 = sorted(sp.solve((X - av)**2 - X, X), key=float)
        left = -(U - av)**2                          # u < a
        right = (U - av)**2                          # u >= a
        I = sp.integrate(U * sp.diff(left, U) - left, (U, t1, av)) + sp.integrate(U * sp.diff(right, U) - right, (U, av, t2))
        A = sp.Abs(sp.simplify(I))
        ok2 &= sp.simplify(A - Sk.subs(k, av)) == 0
        # 영역이 실제로 닫혀 있는지: 곡선과 y=-x, y=x 교점 확인
        ok2 &= sp.simplify(-(t1 - av)**2 + t1) == 0 and sp.simplify((t2 - av)**2 - t2) == 0 and t1 < av < t2
    tot = sp.summation(Sk, (k, 1, 12))
    ok2 &= tot == sp.sympify(p2["sum"])
    # (iii) b 의 해를 수치로 모두 찾고 곱을 비교
    ok3 = True
    A_ = sp.Symbol("a")
    prodmd = sp.sympify(md_answer(2, "3"), locals={"a": A_})
    import mpmath as mp
    for av in (1.0, 2.5, 0.7, -1.3):
        def F(bv):
            l = mp.quad(lambda t: (t - av)**2 * abs(t - av), [0, av, bv] if (0 - av) * (bv - av) < 0 else [0, bv])
            r = mp.quad(lambda t: (t - bv)**2 * abs(t - bv), [0, bv, av] if (0 - bv) * (av - bv) < 0 else [0, av])
            return l - r
        grid = [i / 50 * av for i in range(-50, 451)]
        vals = [F(g) for g in grid]
        roots = []
        for i in range(len(grid) - 1):
            if vals[i] == 0:
                roots.append(grid[i])
            elif vals[i] * vals[i + 1] < 0:
                roots.append(float(mp.findroot(F, (grid[i], grid[i + 1]), solver="bisect")))
        roots = sorted(set(round(r, 9) for r in roots))
        pr = 1.0
        for r in roots:
            pr *= r
        ok3 &= len(roots) == 3 and abs(pr - float(prodmd.subs(A_, av))) < 1e-6
    report(2, ok1 and ok2 and ok3, f"c={cval} area={sp.nsimplify(sp.simplify(area))} (i)={ok1} S(k) OK={ok2} sum={tot} b-곱 OK={ok3}")


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
