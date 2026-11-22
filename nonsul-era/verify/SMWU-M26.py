#!/usr/bin/env python3
"""SMWU-M26 (숙명여대 2026학년도 모의논술 자연계열) — sympy 검산. 정답 하드코딩 금지(YAML answer.value 를 읽어 독립 재계산과 비교)."""
import re
import sys

import sympy as sp

PREFIX = "SMWU-M26"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
n = sp.symbols("n", positive=True, integer=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    """'b_n = ...; c_n = ...' 형식을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1-1) 원 a^2+b^2=4 위에서 a+b 의 최솟값·최댓값 (매개변수 θ 로 독립 계산)
    th = sp.symbols("theta", real=True)
    s = 2 * sp.cos(th) + 2 * sp.sin(th)
    dom = sp.Interval(0, 2 * sp.pi)
    lo, hi = sp.minimum(s, th, dom), sp.maximum(s, th, dom)
    ans = sp.sympify(md_answer(1, "1"))          # 리스트 [lo, hi] 로 해석
    ok1 = sp.simplify(ans[0] - lo) == 0 and sp.simplify(ans[1] - hi) == 0
    # (1-2) α²+β²=γ² ⇔ 2γ²=Σ근² ; 수치로 t 를 훑어 조건을 직접 확인
    import numpy as np
    r2 = np.sqrt(2)
    def ok_t(tv):
        rts = np.roots([1, -(2 + r2) * tv, (1 + 2 * r2) * tv**2, -r2])
        if np.max(np.abs(rts.imag)) > 1e-6:
            return False
        rts = rts.real
        from itertools import permutations
        return any(g > 0 and abs(a * a + b * b - g * g) < 1e-6 for a, b, g in permutations(rts))
    hits = [tv for tv in np.linspace(0.01, 5, 49901) if ok_t(tv)]
    tt = sp.symbols("tt", positive=True)
    # 기호 확인: x=√2 t 가 근이 되는 양수 t
    sol = [v for v in sp.solve(sp.expand((sp.sqrt(2) * tt)**3 - (2 + sp.sqrt(2)) * tt * (sp.sqrt(2) * tt)**2
                                         + (1 + 2 * sp.sqrt(2)) * tt**2 * sp.sqrt(2) * tt - sp.sqrt(2)), tt) if v.is_real]
    ansv = sp.sympify(md_answer(1, "2"))
    ok2 = len(sol) == 1 and sp.simplify(sol[0] - ansv) == 0 and hits and all(abs(h - float(ansv)) < 1e-3 for h in hits)
    report(1, ok1 and ok2, f"a+b in [{lo}, {hi}]  t={sol} grid_hits~{hits[:1]}")


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    # (2-1) [증명-수치확인] 무작위 연속함수·구간에서 ∫f(x)dx = ∫f(a+b-x)dx
    import random
    random.seed(1)
    ok1 = True
    for _ in range(20):
        a0 = random.uniform(-3, 1); b0 = a0 + random.uniform(0.5, 4)
        c = [random.uniform(-2, 2) for _ in range(4)]
        f = lambda u: c[0] + c[1] * mp.sin(u) + c[2] * u**3 + c[3] * mp.exp(u / 3)
        ok1 &= abs(mp.quad(f, [a0, b0]) - mp.quad(lambda u: f(a0 + b0 - u), [a0, b0])) < 1e-20
    # (2-2) 수치 적분(독립) vs YAML 값
    g = lambda u: mp.sqrt(mp.log(u + 25)) / (mp.sqrt(mp.log(u + 25)) + mp.sqrt(mp.log(27 - u)))
    I = mp.quad(g, [-24, -23, 0, 25, 26])
    ok2 = abs(I - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "2")), 30)))) < 1e-15
    # 피적분함수 연속성: 분모 > 0 (양 끝 포함)
    ok2 &= min(mp.sqrt(mp.log(u + 25)) + mp.sqrt(mp.log(27 - u)) for u in mp.linspace(-24, 26, 2001)) > 0
    report(2, ok1 and ok2, f"[증명-수치확인] 20 cases; I={mp.nstr(I, 20)}")


def p03():
    a = sp.symbols("a", real=True)
    f = x**3 - 5 * x**2 + x + 8
    fp = sp.diff(f, x)
    tan = f.subs(x, a) + fp.subs(x, a) * (x - a)
    roots = sp.solve(sp.Eq(tan.subs(x, -3), 5), a)
    lines = {sp.expand(tan.subs(a, r)) for r in roots}
    md = {sp.expand(sp.sympify(seg.split("=", 1)[1], locals={"x": x})) for seg in md_answer(3, "1").split(";")}
    ok1 = lines == md and len(roots) == 3
    # (3-2) 기울기 순 정렬 → L1, L2 의 접점 B, C
    rs = sorted(roots, key=lambda r: fp.subs(x, r))
    L1, L2 = sp.expand(tan.subs(a, rs[0])), sp.expand(tan.subs(a, rs[1]))
    xB, xC = rs[0], rs[1]
    # 영역: x∈[-3,xC] 는 L1~L2 사이, x∈[xC,xB] 는 L1~곡선 사이 (xC=0<xB=3)
    okshape = xC < xB and all((f - L1).subs(x, v) >= 0 and (L2 - f).subs(x, v) >= 0 for v in [sp.Rational(k, 10) for k in range(0, 31)])
    S = sp.integrate(L2 - L1, (x, -3, xC)) + sp.integrate(f - L1, (x, xC, xB))
    ok2 = okshape and sp.simplify(S - sp.sympify(md_answer(3, "2"))) == 0
    report(3, ok1 and ok2, f"a={roots} lines={lines} S={S}")


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
