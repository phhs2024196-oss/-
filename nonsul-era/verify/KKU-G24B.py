#!/usr/bin/env python3
"""KKU-G24B (건국대 2024학년도 KU논술우수자 자연계 B) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "KKU-G24B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


import mpmath as mp
from sympy.utilities.iterables import multiset_permutations


def p01():
    # 영역 R: t<=x<=1 에서 작은 원 위 ~ 큰 원 아래, t=1/2 에서 기호 적분
    h = sp.Rational(1, 2)
    fh = sp.simplify(sp.integrate(sp.sqrt(1 - x**2), (x, h, 1)) - sp.integrate(sp.sqrt(h**2 - x**2), (x, 0, 1 - h)))  # 작은 원: x-t 로 평행이동
    g = lambda tv: mp.quad(lambda u: mp.sqrt(1 - u**2) - mp.sqrt(max((1 - tv)**2 - (u - tv)**2, 0)), [tv, 1])
    grid = [k / 200 for k in range(1, 200)]
    gv = [g(v) for v in grid]
    k = max(range(len(gv)), key=lambda i: gv[i])
    tm = mp.findroot(lambda v: mp.diff(g, v), grid[k])
    pr = parts(1, "1")
    ok = sp.simplify(fh - pr["area_half"]) == 0 and abs(tm - mp.mpf(str(sp.N(pr["t_max"], 30)))) < 1e-10 \
        and abs(g(0.5) - float(pr["area_half"])) < 1e-12
    report(1, ok, f"f(1/2)={fh} t_max(수치)={mp.nstr(tm, 14)}")


def p02():
    c1 = c2 = 0
    for p in multiset_permutations(list("AAABBBBBCCC")):
        s = "".join(p)
        if all(i + 1 < len(s) and s[i + 1] == "B" for i, ch in enumerate(s) if ch == "C"):
            c1 += 1
        if "AB" not in s:
            c2 += 1
    ok = c1 == int(md_answer(2, "1")) and c2 == int(md_answer(2, "2"))
    report(2, ok, f"전수 (1)={c1} (2)={c2}")


def p03():
    mp.mp.dps = 30
    def S(tv):  # 두 원 내부 공통 부분: 교점 x = 1/(2t)
        xc = 1 / (2 * tv)
        return 2 * (mp.quad(lambda u: mp.sqrt(max(tv**2 - (u - tv)**2, 0)), [0, xc])
                    + mp.quad(lambda u: mp.sqrt(max(1 - u**2, 0)), [xc, 1]))
    d = mp.diff(S, 1)
    md = sp.sympify(md_answer(3, "1"))
    report(3, abs(d - mp.mpf(str(sp.N(md, 30)))) < 1e-15, f"S'(1) 수치미분={mp.nstr(d, 18)}")


def p04():
    mp.mp.dps = 30
    T, PH = sp.symbols("T PH", real=True)
    a, b = 1 - sp.sqrt(1 + T), 1 + sp.sqrt(1 + T)          # x^2 = 2x + T 의 두 근 (a<0<b)
    px, py = 1 + sp.cos(PH), 8 + sp.sin(PH)                 # 원 위의 점 P
    Fs = (px - a)**2 + (py - a**2)**2 + (px - b)**2 + (py - b**2)**2
    Ff = sp.lambdify((T, PH), Fs, "math")
    best = min((Ff(i / 100, 2 * math.pi * j / 720), i / 100, 2 * math.pi * j / 720)
               for i in range(1, 1500) for j in range(720))
    gr = sp.lambdify((T, PH), [sp.diff(Fs, T), sp.diff(Fs, PH)], "mpmath")
    sol = mp.findroot(lambda u, w: gr(u, w), (best[1], best[2]))
    tv = sol[0]
    ax = 1 - mp.sqrt(1 + tv)
    pr = parts(4, "1")
    ok = abs(ax - mp.mpf(str(sp.N(pr["Ax"], 30)))) < 1e-20 and abs(ax**2 - mp.mpf(str(sp.N(pr["Ay"], 30)))) < 1e-20 \
        and abs(2 * ax + tv - ax**2) < 1e-20 and ax < 0
    report(4, ok, f"전역격자+뉴턴 min t={mp.nstr(tv, 15)} 값={mp.nstr(Ff(float(tv), float(sol[1])), 12)} "
           f"A=({mp.nstr(ax, 12)},{mp.nstr(ax**2, 12)})")

def parts(no, sub):
    return {k.strip(): sp.sympify(v) for k, v in (s.split("=", 1) for s in md_answer(no, sub).split(";"))}


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
