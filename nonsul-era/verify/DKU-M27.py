#!/usr/bin/env python3
"""DKU-M27 (단국대 2027 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "DKU-M27"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    f = lambda z: -z**3 - 3 * z**2
    # 1. 두 극한이 모두 유한한 0 아닌 값 γ 로 같은 (α,β,γ) 를 정수 후보에서 탐색(β 는 두 식 모두에 나타남)
    found = []
    cands = [v for v in range(-8, 9) if v != 0]
    for a_ in cands:
        for b_ in cands:
            L1 = sp.limit(f(b_ * x) / (2 * x)**a_, x, sp.oo)
            L2 = sp.limit(f(-x) * f(a_ * x) / x**b_, x, sp.oo)
            if L1.is_finite and L2.is_finite and L1 != 0 and L1 == L2:
                found.append((a_, b_, L1))
    md = sp.sympify(md_answer(1, "1").replace("[", "(").replace("]", ")"))
    ok1 = len(found) == 1 and tuple(found[0]) == tuple(md)
    # 2. 접선 y=l(x) at x=-3 ; S=∫|f-l|dx 를 구한 뒤 ∫_{-3}^{3}(f-p)dx=S 를 p 에 대해 푼다
    p = sp.symbols("p", real=True)
    l = f(-3) + sp.diff(f(x), x).subs(x, -3) * (x + 3)
    roots = sp.solve(sp.Eq(f(x), l), x)                          # 접점 -3(중근)과 다른 교점
    lo, hi = min(roots), max(roots)
    diff = sp.expand(f(x) - l)
    # 구간 [lo,hi] 에서 f-l 의 부호 확인
    sample = [lo + (hi - lo) * sp.Rational(i, 10) for i in range(1, 10)]
    sgn = {sp.sign(diff.subs(x, v)) for v in sample}
    S = sp.integrate(sp.Abs(diff) if len(sgn) > 1 else sgn.pop() * diff, (x, lo, hi))
    ps = sp.solve(sp.Eq(sp.integrate(f(x) - p, (x, -3, 3)), S), p)
    ok2 = (lo, hi) == (-3, 3) and ps == [sp.sympify(md_answer(1, "2"))]
    # 3. A(a-f(a)/m,0), B(0,f(a)-am). g(m)²=h(m) 의 m=2 에서 h'(2)=0 인 양수 a, 그 a 에서 m=2 가 전역 최솟값인지 확인
    a, m = sp.symbols("a m", real=True)
    fa = f(a)
    Ax, By = a - fa / m, fa - a * m
    h = sp.simplify(Ax**2 + By**2)
    eqs = sp.solve(sp.Eq(sp.diff(h, m).subs(m, 2), 0), a)
    pos = [r for r in eqs if r.is_real and r > 0]
    ok3 = len(pos) == 1 and sp.simplify(pos[0] - sp.sympify(md_answer(1, "3"))) == 0
    if ok3:
        hv = sp.lambdify(m, h.subs(a, pos[0]), "mpmath")
        grid = [mp.mpf(i) / 1000 for i in range(5, 100000, 7)]
        vals = [(hv(g_), g_) for g_ in grid]
        best = min(vals, key=lambda z: z[0])
        ok3 = abs(best[1] - 2) < mp.mpf("0.01") and hv(2) <= best[0] + mp.mpf("1e-9")
    report(1, ok1 and ok2 and ok3, f"(α,β,γ)={found} p={ps} a={pos}")


def p02():
    ks = [-3.0, -2.0, -1.5, -0.9, -0.5, -0.2]                    # 두 구간 k<-1, -1<k<0 에서 표본
    rs = [1.2, 2.0, 5.0]
    out = {}
    for kv in ks:
        for rv in rs:
            b = [rv**i for i in range(6)]                       # 첫째항 양수, 공비 r>1
            best, arg = None, []
            for perm in itertools.permutations(range(1, 7)):
                v = sum((kv ** perm[i]) * b[i] for i in range(6))
                if best is None or v > best + 1e-9:
                    best, arg = v, [perm]
                elif abs(v - best) <= 1e-9:
                    arg.append(perm)
            out.setdefault("lt" if kv < -1 else "mid", set()).add(tuple(arg))
    md = [tuple(int(z) for z in grp.split(",")) for grp in re.findall(r"\(([\d,]+)\)", md_answer(2, "1"))]
    ok1 = all(len(v) == 1 and len(next(iter(v))) == 1 for v in out.values()) and \
        {"lt": next(iter(out["lt"]))[0], "mid": next(iter(out["mid"]))[0]} == {"lt": md[0], "mid": md[1]}
    # 2. 6! 전수: 점 G_n 중 한 점을 제외한 모든 점과 직선 3x+4y=0 사이 거리가 자연수
    from fractions import Fraction
    good = []
    for perm in itertools.permutations(range(1, 7)):
        ds = []
        for nn in range(1, 7):
            gx = Fraction(2 * nn + 1 + 2**perm[nn - 1], 3)
            gy = Fraction(nn + 1)
            ds.append(abs(3 * gx + 4 * gy) / 5)
        nonint = [i for i, d in enumerate(ds) if d.denominator != 1]
        if len(nonint) <= 1 and len(nonint) == 1:             # 정확히 한 점만 자연수 아님
            good.append(perm)
    md2 = sorted(tuple(int(z) for z in grp.split(",")) for grp in re.findall(r"\(([\d,]+)\)", md_answer(2, "2")))
    ok2 = sorted(good) == md2
    report(2, ok1 and ok2, f"최대 f(k<-1)={md[0]}, f(-1<k<0)={md[1]}; 거리 조건 f={sorted(good)}")


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
