#!/usr/bin/env python3
"""DKU-G25A (단국대 2025학년도 논술 자연계열 오전) — 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "DKU-G25A"
WIDTH = 2
RESULTS = []
x, t, k, a = sp.symbols("x t k a")
NS = {"pi": sp.pi, "E": sp.E, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"DKU-G25A-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def ans_num(no, sub):
    return float(sp.N(sp.sympify(md_answer(no, sub), locals=NS)))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"DKU-G25A-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def close(u, v, tol=1e-9):
    return abs(float(u) - float(v)) < tol



def p01():
    s = sp.Symbol("s")
    al = sp.Symbol("al", positive=True)
    # g'=f 가 x=-2,0,2 에서 부호가 바뀌는 최고차 계수 양수 삼차함수: f=al*x(x+2)(x-2)
    f = al * x * (x + 2) * (x - 2)
    g = lambda u: sp.integrate(f.subs(x, s), (s, 1, u))
    # 1. ∫_0^2 f = -2 - g(0) 에서 al, 그 다음 g(-2)
    sol_a = sp.solve(sp.Eq(sp.integrate(f, (x, 0, 2)), -2 - g(0)), al)
    g_m2 = sp.simplify(g(-2).subs(al, sol_a[0]))
    ok1 = len(sol_a) == 1 and g_m2 == sp.sympify(md_answer(1, "1"))
    # 2. 접선 l, 그에 수직인 m, m 의 x절편 13
    f1p = sp.diff(f, x).subs(x, 1)
    mslope = -1 / f1p
    xint = sp.solve(sp.Eq(mslope * (x - 1) + f.subs(x, 1), 0), x)[0]
    sol2 = sp.solve(sp.Eq(xint, 13), al)
    ok2 = [v for v in sol2 if v > 0] == [sp.sympify(md_answer(1, "2"))] if False else [v for v in sol2 if v > 0] == [sp.sympify(md_answer(2, "2"), locals=NS)] if False else [v for v in sol2 if v > 0] == [sp.sympify(md_answer(1, "2"))]
    # 3. P(p,0): 접선이 정확히 2개 ⇔ 접점 방정식(삼차)이 이중근을 가짐(판별식=0)
    u, p = sp.symbols("u p", real=True)
    tang = sp.expand(sp.diff(f, x).subs(x, u) * (p - u) + f.subs(x, u))            # 접점 u 의 접선이 (p,0) 지남
    disc = sp.factor(sp.discriminant(sp.Poly(tang, u)))
    ps = [v for v in sp.solve(disc, p) if v.is_real and v > 0]
    P0 = ps[0]
    rts = sp.roots(sp.Poly(tang.subs(p, P0), u))
    simple = [r for r, m in rts.items() if m == 1]
    dbl = [r for r, m in rts.items() if m == 2]
    slopes = [sp.diff(f, x).subs(x, r) for r in (simple + dbl)]
    sol3 = [v for v in sp.solve(sp.Eq(slopes[0] * slopes[1], -1), al) if v > 0]
    ok3 = len(ps) == 1 and P0 == 2 and len(simple) == 1 and len(dbl) == 1 and sol3 == [sp.sympify(md_answer(3, "3"), locals=NS)] if False else (len(ps) == 1 and P0 == 2 and len(simple) == 1 and len(dbl) == 1 and len(sol3) == 1 and sp.simplify(sol3[0] - sp.sympify(md_answer(1, "3"), locals=NS)) == 0)
    report(1, ok1 and ok2 and ok3, f"a={sol_a} g(-2)={g_m2}; a(논제2)={sol2}; P={P0}, 접점 단순근 {simple} 이중근 {dbl}, a={sol3}")


def cnt(c, lo, hi, fx, brk, fvals):
    """[lo,hi] 에서 f(x)=c 의 근의 개수 (f 는 brk 로 나뉜 구간마다 단조)."""
    n = 0
    pts = [lo] + brk + [hi]
    for i in range(len(pts) - 1):
        va, vb = fvals[i], fvals[i + 1]
        a_, b_ = min(va, vb), max(va, vb)
        left_closed = True
        if a_ <= c <= b_:
            if i > 0 and abs(c - va) < 1e-12:    # 구간 경계(극값점)는 앞 구간에서 이미 셈
                continue
            n += 1
    return n


def p02():
    mp.mp.dps = 25
    # 1. 치환적분: 피적분함수 = -f'/f 임을 수치로 확인하고 a 를 구해 f(a) 계산
    f = lambda v: mp.cos(v) / (2 + mp.sin(v))
    integrand = lambda v: (2 * mp.sin(v) + 1) / (2 * mp.cos(v) + mp.sin(v) * mp.cos(v))
    ok_form = all(abs(integrand(v) + mp.diff(f, v) / f(v)) < 1e-15 for v in (0.3, 0.9, 1.3))
    a_sol = mp.findroot(lambda aa: mp.quad(integrand, [0, aa]) - 1, 1.3)
    ok1 = ok_form and 0 < a_sol < mp.pi / 2 and close(f(a_sol), ans_num(2, "1"), 1e-12)
    # 2. |3f-k|=1 ⇔ f=(k±1)/3. [π/2,2π] 에서 f 는 단조 세 구간. 구간 단조점은 f'=0 의 해(solveset)로 구한다.
    fs = sp.cos(x) / (2 + sp.sin(x))
    crit = sorted(v for v in sp.solveset(sp.diff(fs, x), x, sp.Interval(sp.pi / 2, 2 * sp.pi)) if True)
    lo, hi = sp.pi / 2, 2 * sp.pi
    pts = [lo] + crit + [hi]
    fv = [float(fs.subs(x, q)) for q in pts]
    brk = [float(q) for q in crit]

    def N(kv):
        return cnt((kv - 1) / 3, float(lo), float(hi), None, brk, fv) + cnt((kv + 1) / 3, float(lo), float(hi), None, brk, fv)
    # 검증용 독립 수치: 격자 부호변화(접하지 않는 일반 k)
    xs = np.linspace(float(lo), float(hi), 400001)
    fx = np.cos(xs) / (2 + np.sin(xs))

    def Ngrid(kv):
        n = 0
        for c in ((kv - 1) / 3, (kv + 1) / 3):
            s_ = np.sign(fx - c); s_ = s_[s_ != 0]
            n += int(np.sum(s_[:-1] * s_[1:] < 0))
        return n
    r3 = np.sqrt(3)
    bps = sorted({e for c in (0, 0.5, r3 / 3, -r3 / 3) for e in (3 * c + 1, 3 * c - 1)})
    tests = [bps[0] - 1] + [(bps[i] + bps[i + 1]) / 2 for i in range(len(bps) - 1)] + [bps[-1] + 1]
    agree = all(N(v) == Ngrid(v) for v in tests)
    expr = md_answer(2, "2")
    ev = lambda kv: bool(eval(expr, {"sqrt": np.sqrt, "k": kv}))
    ok2 = agree and all((N(v) == 2) == ev(v) for v in tests + bps)
    report(2, ok1 and ok2, f"f(a)={mp.nstr(f(a_sol), 10)} (a={mp.nstr(a_sol, 8)}); k 경계점 {len(bps)}개·구간 {len(tests)}개에서 개수=2 ⇔ 공식 범위")


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
