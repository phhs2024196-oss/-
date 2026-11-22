#!/usr/bin/env python3
"""AJOU-G25M (아주대 2025학년도 논술 자연계열 의학과, 수학 문항) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G25M"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def same(a, b, tol=1e-12):
    return abs(complex(sp.N(a - b, 30))) < tol


def p01():
    oks = []
    a = sp.symbols("a", positive=True)

    def fper(av):
        return [(av * (sp.log(2 - 2 * t, 2) - 1), 0, sp.Rational(1, 2)),
                ((av + 2) * (t - sp.Rational(3, 2)) + 2, sp.Rational(1, 2), sp.Rational(3, 2)),
                (-4 * t + 8, sp.Rational(3, 2), 2)]
    # (1-1) 한 주기 적분 = 0 (주기 적분이 0 이 아니면 |∫_0^γ| → ∞)
    I = sum(sp.integrate(e, (t, lo, hi)) for e, lo, hi in fper(a))
    asol = sp.solve(sp.Eq(sp.simplify(I), 0), a)
    oks.append(len(asol) == 1 and sp.simplify(asol[0] - sp.sympify(md_answer(1, "1"))) == 0)
    # 유계성 직접 확인: a=asol 이면 ∫_0^γ 의 최대 |.| 가 γ 에 무관하게 작다
    # 이하 v = f(a=2)
    pieces = fper(2)
    Iv = sp.nsimplify(sp.simplify(sum(sp.integrate(e, (t, lo, hi)) for e, lo, hi in pieces)))
    Iv = sp.simplify(sp.expand_log(sp.re(Iv), force=True))
    s = sp.symbols("s", nonnegative=True)
    Gs = [sp.integrate(e, (t, lo, s)) for e, lo, hi in pieces]
    G0 = [0, Gs[0].subs(s, sp.Rational(1, 2)), Gs[0].subs(s, sp.Rational(1, 2)) + Gs[1].subs(s, sp.Rational(3, 2))]
    import mpmath as mp
    mp.mp.dps = 30
    Gf = [(lambda ex: (lambda z: mp.re(sp.lambdify(s, ex, "mpmath")(z))))(G0[i] + Gs[i]) for i in range(3)]
    Ivf = mp.mpf(str(sp.N(sp.re(Iv), 40)))

    def g(tt, inv_ln2=None):
        tt = mp.mpf(tt)
        nper = int(mp.floor(tt / 2)); r = tt - 2 * nper
        i = 0 if r < 0.5 else (1 if r < 1.5 else 2)
        return mp.mpf(1) / 2 + nper * Ivf + Gf[i](r)
    # (1-2)(1) g(4)
    g4 = sp.Rational(1, 2) + 2 * Iv
    oks.append(sp.simplify(g4 - sp.sympify(md_answer(1, "2-1"))) == 0)
    # (1-2)(2) g(t)=1/2 의 가장 작은 양의 근: 격자로 부호 변화 → findroot
    grid = [mp.mpf(i) / 4000 for i in range(1, 8001)]
    t1 = None
    for u, w in zip(grid, grid[1:]):
        if (g(u) - mp.mpf(1) / 2) * (g(w) - mp.mpf(1) / 2) <= 0:
            t1 = mp.findroot(lambda z: g(z) - mp.mpf(1) / 2, (u, w), solver="bisect"); break
    val = t1**2 - 4 * t1
    oks.append(abs(val - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "2-2")), 40)))) < mp.mpf("1e-20"))
    # (1-3)(1) g(t)=0 의 양의 근 개수 (0<t≤60, 격자 부호변화)
    grid = [mp.mpf(i) / 200 for i in range(1, 12001)]
    gv = [g(u) for u in grid]
    nroot = sum(1 for u, w in zip(gv, gv[1:]) if u * w < 0)
    oks.append(nroot == int(sp.sympify(md_answer(1, "3-1"))) and min(abs(z) for z in gv) > mp.mpf("1e-6"))
    # (1-3)(2) 접점 α∈(7/2,4): v(α)=m, g(α)=mα+1/2 를 연립해 풀고, 교점이 정확히 3개인지 확인
    v = lambda z: -4 * (z - 2) + 8
    al = mp.findroot(lambda z: g(z) - (v(z) * z + mp.mpf(1) / 2), mp.mpf("3.9"))
    m = v(al)
    d = [g(u) - (m * u + mp.mpf(1) / 2) for u in grid]
    sc = [grid[i] for i in range(len(d) - 1) if d[i] * d[i + 1] < 0]
    # 숨은 접점 탐지: d 의 극대점은 v(t)=m 인 t = α+2j (주기성). 극댓값 부호를 직접 확인
    dmax = [g(al + 2 * j) - (m * (al + 2 * j) + mp.mpf(1) / 2) for j in range(-1, 28)]
    near = not (dmax[0] > 0 and abs(dmax[1]) < mp.mpf("1e-25") and all(z < 0 for z in dmax[2:]))
    cnt = len(sc) + 1
    oks.append(cnt == 3 and max(sc) < al and m > 0 and not near
               and abs(al**2 - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "3-2")), 40)))) < mp.mpf("1e-15"))
    report(1, all(oks), f"a={asol} g(4)={sp.N(g4, 8)} t1={mp.nstr(t1, 10)} roots={nroot} α={mp.nstr(al, 12)} m={mp.nstr(m, 6)} 교점={cnt} {oks}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
