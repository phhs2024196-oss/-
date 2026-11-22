#!/usr/bin/env python3
"""DKU-G26B (단국대 2026학년도 논술 자연계열 오후) — 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "DKU-G26B"
WIDTH = 2
RESULTS = []
x, t, k, a = sp.symbols("x t k a")
NS = {"pi": sp.pi, "E": sp.E, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"DKU-G26B-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def ans_num(no, sub):
    return float(sp.N(sp.sympify(md_answer(no, sub), locals=NS)))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"DKU-G26B-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def close(u, v, tol=1e-9):
    return abs(float(u) - float(v)) < tol



def p01():
    mp.mp.dps = 30
    f = lambda v: mp.cos(v) - mp.sin(v)
    g = lambda v: f(v) * mp.e**(-v)
    # 1. 극한을 작은 x 로 수치 계산, 양수 a 의 최소 영점
    def L(av):
        xx = mp.mpf("1e-9")
        return mp.quad(lambda s: mp.e**(-(av * s)**2) * f(av * s), [1, 1 + xx]) / xx
    grid = [mp.mpf(i) / 200 for i in range(1, 400)]
    vals = [L(v) for v in grid]
    i0 = next(i for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
    r1 = mp.findroot(L, grid[i0], tol=1e-12, verify=False)
    ok1 = abs(r1 - ans_num(1, "1")) < 1e-6
    # 2. 두 곡선이 만나는 모든 점에서 접선 일치. 교점 ⇔ f(p)=b, 접선 일치 ⇔ 추가로 g'(p)=h'(p)
    def check(bv):
        roots = []
        xs = np.linspace(0, 10 * np.pi, 2000001)
        fv = np.cos(xs) - np.sin(xs) - bv
        for i in np.where(fv[:-1] * fv[1:] < 0)[0]:
            roots.append(float(mp.findroot(lambda z: f(z) - bv, xs[i], tol=1e-20, verify=False)))
        return roots
    gp = lambda v: mp.diff(g, v)
    hp = lambda v, bv: mp.diff(lambda z: bv * mp.e**(-z), v)
    bad = []
    for bv in np.linspace(0.05, 1.41, 137):
        rs = check(bv)
        worst = max((abs(gp(r) - hp(r, bv)) for r in rs), default=1)
        bad.append(float(worst))
    ok_scan = min(bad) > 1e-5
    bs = mp.sqrt(2)
    cand = [mp.mpf(7) * mp.pi / 4 + 2 * j * mp.pi for j in range(5)]
    tang = all(abs(g(c) - bs * mp.e**(-c)) < 1e-20 and abs(gp(c) - hp(c, bs)) < 1e-18 for c in cand)
    xs = np.linspace(0, 10 * np.pi, 2000001)
    # b=√2 이면 f ≤ √2 이고 등호는 후보점에서만 → 교점은 후보점뿐
    on_only = np.all((np.cos(xs) - np.sin(xs) < np.sqrt(2) - 1e-9) | (np.abs(((xs - 7 * np.pi / 4 + np.pi) % (2 * np.pi)) - np.pi) < 1e-3))
    pts = [c for c in cand if c <= 10 * mp.pi]
    ok2 = ok_scan and tang and on_only and close(bs + pts[1], ans_num(1, "2"), 1e-12)
    # 3. 넓이 A_k
    zs = [mp.pi / 4 + j * mp.pi for j in range(0, 7)]
    zchk = all(abs(g(z)) < 1e-25 for z in zs)
    A = lambda kk: abs(mp.quad(g, [zs[kk - 1], zs[kk]]))
    val = A(1) + A(6) / A(3)
    ok3 = zchk and close(val, ans_num(1, "3"), 1e-20)
    report(1, ok1 and ok2 and ok3, f"a={mp.nstr(r1, 8)} b+p2={mp.nstr(bs + pts[1], 10)} (b≠√2 탐색 최소 접선오차 {min(bad):.1e}) A1+A6/A3={mp.nstr(val, 12)}")


def p02():
    # 1. f(x)=(x+2)(x-3)(x-a), f(1)=6 → a. g 를 가지(branch)로 직접 구성해 조건(1)(2)·극한을 확인
    aa = sp.Symbol("aa")
    fx = (x + 2) * (x - 3) * (x - aa)
    asol = sp.solve(sp.Eq(fx.subs(x, 1), 6), aa)[0]
    fpoly = sp.Poly(sp.expand(fx.subs(aa, asol)), x)
    co = [float(c) for c in fpoly.all_coeffs()]

    def roots_at(tv):
        r = np.roots(co[:3] + [co[3] - tv])
        return sorted(z.real for z in r if abs(z.imag) < 1e-7)

    def gbranch(tv):
        r = roots_at(tv)
        return r[0] if tv < 6 else (r[-1] if tv > 6 else 1.0)
    ts = np.linspace(-20, 40, 60001)
    gv = np.array([gbranch(v) for v in ts])
    jumps = [round(ts[i + 1], 3) for i in range(len(ts) - 1) if abs(gv[i + 1] - gv[i]) > 0.3]
    ok_g = abs(gbranch(0) + 2) < 1e-9 and abs(float(fpoly.eval(3))) < 1e-9 and len(jumps) > 0 and all(abs(float(q) - 6) < 0.01 for q in jumps)
    lim_l, lim_r = gbranch(6 - 1e-9), gbranch(6 + 1e-9)
    ya, yb = [float(sp.N(sp.sympify(q, locals=NS))) for q in md_answer(2, "1")[1:-1].split(",")]
    ok1 = asol == 2 and ok_g and abs(lim_l - ya) < 1e-6 and abs(lim_r - yb) < 1e-6
    # 2. f=x^3-3x^2+c (극값 0,2). 정수 n 에 대해 c=2n+2, 연속 h 존재 ⇔ m≤n, 연속 k 비존재 ⇔ m>n-1
    cands = []
    for n in range(1, 11):
        c = 2 * n + 2                              # f(h(n))=2n=f(1)=c-2
        m = c - 4                                   # 극솟값 f(2)
        if m <= n and m > n - 1:
            cands.append((n, c))
    n0, c0 = cands[0]
    cc = [1, -3, 0, c0]

    def hbranch(tv):                                # t+n 수준에서 가장 큰 근 (x≥2 가지: 연속)
        r = np.roots(cc[:3] + [cc[3] - (tv + n0)])
        return max(z.real for z in r if abs(z.imag) < 1e-7)
    d = 1e-6
    hp = (hbranch(n0 + d) - hbranch(n0 - d)) / (2 * d)
    ok2 = len(cands) == 1 and n0 == 2 and c0 == 6 and abs(hp - ans_num(2, "2")) < 1e-6
    report(2, ok1 and ok2, f"a={asol}, 극한=({lim_l:.6f},{lim_r:.6f}) 불연속점 {sorted(set(float(q) for q in jumps))}; (n,c)={cands}, h'(2)={hp:.8f}")


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
