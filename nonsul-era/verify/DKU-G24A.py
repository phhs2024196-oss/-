#!/usr/bin/env python3
"""DKU-G24A (단국대 2024학년도 논술 자연계열 오전) — 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "DKU-G24A"
WIDTH = 2
RESULTS = []
x, t, k, a = sp.symbols("x t k a")
NS = {"pi": sp.pi, "E": sp.E, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"DKU-G24A-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def ans_num(no, sub):
    return float(sp.N(sp.sympify(md_answer(no, sub), locals=NS)))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"DKU-G24A-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def close(u, v, tol=1e-9):
    return abs(float(u) - float(v)) < tol



def p01():
    mp.mp.dps = 25
    f = lambda v: np.abs(np.abs(np.tan(v)) - 1)
    xs = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 2000001)
    zs = sorted(set(np.round(xs[np.abs(f(xs)) < 1e-5], 3)))
    area = mp.quad(lambda v: abs(abs(mp.tan(v)) - 1), [-mp.pi / 4, 0, mp.pi / 4])
    ok1 = zs == [round(-np.pi / 4, 3), round(np.pi / 4, 3)] and close(area, ans_num(1, "1"), 1e-12)
    # 2. f 의 꺾임점 좌우 미분계수가 다르므로 g=fp 가 미분가능하려면 p(c)=0. 삼차 p 를 일반형으로 두고 푼다.
    pieces = {sp.pi / 4: (-sp.tan(x) + 1, sp.tan(x) - 1), 0: (sp.tan(x) + 1, -sp.tan(x) + 1), -sp.pi / 4: (-sp.tan(x) - 1, sp.tan(x) + 1)}
    jumps = {c: sp.simplify(sp.diff(r, x).subs(x, c) - sp.diff(l, x).subs(x, c)) for c, (l, r) in pieces.items()}
    c0, c1, c2, c3 = sp.symbols("c0:4")
    P = c0 + c1 * x + c2 * x**2 + c3 * x**3
    eqs = [P.subs(x, c) for c in pieces]                       # (f'(c+)-f'(c-)) p(c) = 0, 도약 ≠ 0
    eqs.append(1 + sp.diff(P, x).subs(x, 0))                    # h=|u|, u(0)=0 이 미분가능 ⇔ u'(0)=0
    sol = sp.solve(eqs, [c0, c1, c2, c3], dict=True)[0]
    Psol = P.subs(sol)
    ok2 = all(v != 0 for v in jumps.values()) and sp.simplify(Psol - sp.sympify(md_answer(1, "2"), locals=NS)) == 0
    # 3. h=|tan x+p| 의 극값 개수 (격자)
    pf = sp.lambdify(x, sp.sympify(md_answer(1, "2"), locals=NS), "numpy")
    xs = np.linspace(-1.55, 1.55, 2000001)
    u = np.tan(xs) + pf(xs)
    hv = np.abs(u)
    d = np.diff(hv)
    idx = np.where(d[:-1] * d[1:] < 0)[0]
    mono = np.all(np.diff(u) > 0)
    ok3 = len(idx) == 1 and abs(xs[idx[0] + 1]) < 1e-3 and mono and len(idx) == int(ans_num(1, "3"))
    report(1, ok1 and ok2 and ok3, f"zeros={zs} S={mp.nstr(area, 8)} p={sp.simplify(Psol)} 극값 {len(idx)}개 at {xs[idx[0]+1] if len(idx) else None:.4f}")


def p02():
    mp.mp.dps = 25
    f = lambda v: np.exp(v) / (np.exp(v) + 1)

    def N(kk):
        n_bc = 1 if 0 < f(kk) <= 1 else 0                          # 변 BC: x=k, 0≤y≤1
        xs = np.linspace(-kk, kk, 400001)
        s = np.sign(f(xs) - (xs + kk) / (2 * kk))                  # 변 AC. 변 AB(y=0) 와는 f>0 이라 만나지 않음
        s = s[s != 0]
        return n_bc + int(np.sum(s[:-1] * s[1:] < 0))
    lo, hi = 1.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if N(mid) == 4: hi = mid
        else: lo = mid
    thr = hi
    ok1 = abs(thr - 2) < 1e-3 and N(1.9) != 4 and N(2.1) == 4 and N(5) == 4 and N(0.5) != 4 \
        and abs(thr - float(re.findall(r"[-\d.]+", md_answer(2, "1"))[0])) < 1e-3
    # 2. (β^{-1}∘α)(ln2) 를 수치로: β(y)=α(ln2) 를 푼다
    alpha = mp.quad(lambda s: mp.sqrt(mp.e**s / (mp.e**s + 1)), [0, mp.log(2)])
    c = mp.sqrt(mp.mpf(1) / 2)
    beta = lambda y: mp.quad(lambda s: 2 / (1 - s**2), [c, y])
    y = mp.findroot(lambda yy: beta(yy) - alpha, 0.8)
    ok2 = close(y, ans_num(2, "2"), 1e-12)
    report(2, ok1 and ok2, f"k 임계≈{thr:.5f}; (β^-1∘α)(ln2)={mp.nstr(y, 10)}")


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
