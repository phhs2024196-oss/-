#!/usr/bin/env python3
"""SKKU-G22A (성균관대 2022학년도 수시 논술우수전형 자연계 1교시 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SKKU-G22A"
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
    N = 50 * 51 // 2
    S1 = lambda n: n * n + n + 1
    a = [S1(1)] + [S1(n) - S1(n - 1) for n in range(2, N + 1)]       # S_n 로부터 a_n 직접 생성
    v1 = sum(r[-1] for r in rows(a, 50, False))
    v2 = sum(r[-1] for r in rows(a, 50, True))
    S3 = lambda n: 2**n
    b = [S3(1)] + [S3(n) - S3(n - 1) for n in range(2, N + 1)]
    prod = 1
    for r in rows(b, 50, True):
        prod *= r[-1]
    ok1 = v1 == sp.sympify(md_answer(1, "1"))
    ok2 = v2 == sp.sympify(md_answer(1, "2"))
    ok3 = sp.Integer(prod) == sp.sympify(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"(i)={v1} (ii)={v2} (iii)=2^{sp.log(prod, 2)}")


def p02():
    f = sp.Piecewise((-x**2 - x, x <= 0), (x**2 - x, True))
    b = sp.symbols("b", real=True)
    # (i) g(b): b>0, b<0 각각 접선과 다른 가지의 교점
    ok1 = True
    for bv in (sp.Rational(3, 7), sp.Integer(2), -sp.Rational(5, 3), -sp.Integer(1)):
        fb = (bv**2 - bv) if bv > 0 else (-bv**2 - bv)
        sl = (2 * bv - 1) if bv > 0 else (-2 * bv - 1)
        T = sl * (x - bv) + fb
        other = [r for r in sp.solve(T - (-x**2 - x), x) if r <= 0] + [r for r in sp.solve(T - (x**2 - x), x) if r > 0]
        other = [r for r in other if sp.simplify(r - bv) != 0]
        p = kv(md_answer(2, "1"))
        g = sp.sympify(p["g(x)"], locals={"x": x}); h = sp.sympify(p["h(x)"], locals={"x": x})
        ok1 &= len(other) == 1 and sp.simplify(other[0] - g.subs(x, bv)) == 0
        ok1 &= sp.simplify(h.subs(x, g.subs(x, bv)) - bv) == 0
    al = 1 - sp.sqrt(2)
    hfun = sp.sympify(kv(md_answer(2, "1"))["h(x)"], locals={"x": x})
    p2 = kv(md_answer(2, "2")); p3 = kv(md_answer(2, "3"))
    ratio_md = sp.sympify(md_answer(2, "4"), locals={"alpha": alpha}).subs(alpha, al)
    ok2 = ok3 = ok4 = True
    for mv in range(4):
        xs = [sp.Integer(1)]
        for _ in range(2 * mv + 2):
            xs.append(sp.expand(hfun.subs(x, xs[-1])))
        P = [(t, f.subs(x, t)) for t in xs]
        areas = []
        for j, (key, akey) in enumerate((("L2m", "A2m"), ("L2m1", "A2m1"))):
            n = 2 * mv + j
            (x1, y1), (x2, y2) = P[n], P[n + 1]
            sl = sp.simplify((y2 - y1) / (x2 - x1)); cst = sp.simplify(y1 - sl * x1)
            sub = {alpha: al, m: mv}
            ok2 &= sp.simplify(sl - sp.sympify(p2[key + "_slope"], locals={"alpha": alpha, "m": m}).subs(sub)) == 0
            ok2 &= sp.simplify(cst - sp.sympify(p2[key + "_const"], locals={"alpha": alpha, "m": m}).subs(sub)) == 0
            L = sl * x + cst
            lo, hi = sorted([x1, x2], key=lambda v: float(v))
            A = sp.Abs(sp.integrate(L - (-x**2 - x), (x, lo, 0)) + sp.integrate(L - (x**2 - x), (x, 0, hi)))
            # 둘러싸인 영역: 구간 안에서 L-f 부호 일정 확인(접선이므로 부호 불변)
            import numpy as np
            grid = np.linspace(float(lo), float(hi), 2001)[1:-1]
            vals = [float((L - f).subs(x, g)) for g in grid[::100]]
            ok3 &= all(v > 0 for v in vals) or all(v < 0 for v in vals)
            ok3 &= abs(float(A - sp.sympify(p3[akey], locals={"alpha": alpha, "m": m}).subs(sub))) < 1e-12 * max(1, float(A))
            areas.append(A)
        ok4 &= abs(float(areas[1] / areas[0] - ratio_md)) < 1e-9
    report(2, ok1 and ok2 and ok3 and ok4, f"g,h OK={ok1} L계수={ok2} A={ok3} ratio={float(ratio_md):.10f} ({ok4})")


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
