#!/usr/bin/env python3
"""EWHA-G26B (이화여대 2026학년도 수시 논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "EWHA-G26B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 조건을 만족하는 구체적 f 족: f = α cos(πx/4) + β sin(πx/4) + γ φ(x), φ=(2-x)^3 (x<2), 0 (x≥2)
    # (나)는 모든 x≥2 에서 성립, (가)로 α 결정(β, γ 는 자유 → t 가 자유 상수임을 반영)
    al, be, ga = sp.symbols("alpha beta gamma", real=True)
    C, S = sp.cos(sp.pi * x / 4), sp.sin(sp.pi * x / 4)
    f_hi = al * C + be * S                                   # x ≥ 2 에서의 f
    ok_nah = sp.simplify(f_hi.subs(x, x + 4) + f_hi) == 0
    condA = sp.integrate(sp.diff(f_hi, x, 2) * C, (x, 2, 6)) + sp.pi**2 / 16
    alv = sp.solve(condA, al)[0]
    f_hi = f_hi.subs(al, alv)
    v1 = sp.simplify(sp.integrate(f_hi * C, (x, 2, 6)))
    ok1 = ok_nah and sp.simplify(v1 - sp.sympify(md_answer(1, "1"))) == 0
    # a_n 을 실제 적분으로 계산(β, γ 수치 몇 쌍)해 a_n = n + t 와 극한 확인
    ok2 = True
    tvals = []
    for bv, gv in ((0, 0), (sp.Rational(3, 7), 1), (-2, sp.Rational(-5, 3))):
        f_lo = f_hi.subs(be, bv) + gv * (2 - x)**3
        t = sp.integrate(f_lo * C, (x, 0, 2))
        tvals.append(t)
        for n in range(1, 5):
            an = t + sp.integrate(f_hi.subs(be, bv) * C, (x, 2, 4 * n + 2))
            ok2 &= sp.simplify(an - (n + t)) == 0
    m, n, T = sp.symbols("m n T", positive=True, integer=True), sp.symbols("n", integer=True), sp.symbols("T", real=True)
    m = sp.symbols("m", positive=True, integer=True)
    lim = sp.limit(sp.summation(n + T, (n, 1, m)) / m**2, m, sp.oo)
    ok2 &= sp.simplify(lim - sp.sympify(md_answer(1, "2"))) == 0
    # (3) Σ(n+T)^2 의 최솟값 = 2025·2026·2027/12 (부등식 우변) — 수치 반례 없음
    Ssum = sp.expand(sp.summation((n + T)**2, (n, 1, 2026)))
    Tmin = sp.solve(sp.diff(Ssum, T), T)[0]
    ok3 = sp.simplify(Ssum.subs(T, Tmin) - sp.Rational(2025 * 2026 * 2027, 12)) == 0 and sp.diff(Ssum, T, 2) > 0
    ok3 &= all(Ssum.subs(T, tv) >= sp.Rational(2025 * 2026 * 2027, 12) for tv in tvals)
    # (4) 등호일 때 a_n = n + Tmin 으로 합을 정확히 계산
    a = [None] + [k + Tmin for k in range(1, 2027)]
    tot = sum((a[2 * j - 1] + a[2 * j]) / (a[2 * j - 1]**2 * a[2 * j]**2) for j in range(1, 1014))
    ok4 = sp.simplify(tot - sp.sympify(md_answer(1, "4"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"(1)={v1} lim={lim} Tmin={Tmin} (4)={tot} [증명-수치확인](3)")


def p02():
    f = (x / (x + 6))**x
    l0 = sp.limit(f, x, 0, "+")
    li = sp.limit(f, x, sp.oo)
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "1").replace(" ", "").split(";"))
    ok1 = sp.simplify(l0 - sp.sympify(parts["lim0"])) == 0 and sp.simplify(li - sp.sympify(parts["liminf"])) == 0
    # (2) [증명-수치확인] f' < 0 on (0,∞): 로그 격자
    fp = sp.lambdify(x, sp.diff(sp.log(f), x), "mpmath")
    ok2 = all(fp(mp.mpf(10) ** e) < 0 for e in mp.linspace(-6, 6, 2001))
    # (3) f(c)=1/15 의 해가 하나: 부호 변화 1회
    fn = sp.lambdify(x, f - sp.Rational(1, 15), "mpmath")
    grid = [fn(mp.mpf(10) ** e) for e in mp.linspace(-6, 4, 4001)]
    ch = sum(1 for u, v in zip(grid, grid[1:]) if u * v < 0)
    c = mp.findroot(fn, 2)
    ok3 = ch == 1 and 1 < c < 3
    report(2, ok1 and ok2 and ok3, f"lim0={l0} liminf={li} c≈{mp.nstr(c, 10)} [증명-수치확인](2)(3)")


def p03():
    a, b, al = sp.symbols("a b alpha", positive=True)
    k = sp.symbols("k", positive=True, integer=True)
    X = sp.symbols("X", positive=True)
    ok = True
    hk_md = sp.sympify(md_answer(3, "2"), locals={"k": k})
    for kv in range(1, 7):
        # 접함 조건: 직선 y=b(1-X/a) 와 y=3/X^k 가 X=α 에서 함숫값·기울기 일치
        sol = sp.solve([b * (1 - al / a) - 3 / al**kv, -b / a + 3 * kv / al**(kv + 1)], [a, b], dict=True)[0]
        area = 2 * sol[a] * sol[b]
        rect = 4 * al * 3 / al**kv
        hk = sp.simplify(area / rect)
        ok &= sp.simplify(hk - hk_md.subs(k, kv)) == 0 and not hk.has(al)
        if kv == 1:
            area1 = sp.simplify(area)
            per = 4 * sp.sqrt(sol[a]**2 + sol[b]**2)
            crit = [r for r in sp.solve(sp.diff(per**2, al), al) if r.is_positive]
            pmin = sp.nsimplify(sp.simplify(per.subs(al, crit[0])))
        # n(A∩B)=4 ⇔ 접함 [수치확인]: 접선보다 b 를 키우면 제1사분면 교점 2개(총 8), 줄이면 0개
        av = float(sol[a].subs(al, 1.3)); bv = float(sol[b].subs(al, 1.3))
        for scale, want in ((1.0, 1), (1.05, 2), (0.95, 0)):
            g = lambda s: scale * bv * (1 - s / av) - 3 / s**kv
            xs = [av * i / 20000 for i in range(1, 20000)]
            vals = [g(s) for s in xs]
            cnt = sum(1 for u, v in zip(vals, vals[1:]) if u * v < 0)
            if want == 1:
                ok &= cnt in (0, 2) and max(vals) > -1e-6 and max(vals) < 1e-6
            else:
                ok &= cnt == want
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "1").replace(" ", "").split(";"))
    ok1 = sp.simplify(area1 - sp.sympify(parts["area"])) == 0 and sp.simplify(pmin - sp.sympify(parts["min_perimeter"])) == 0
    ser = sp.summation((2 * k + 1) / (k**3 * hk_md), (k, 1, sp.oo))
    ok3 = sp.simplify(ser - sp.sympify(md_answer(3, "3"))) == 0
    report(3, ok and ok1 and ok3, f"area(k=1)={area1} minP={pmin} h_k=(k+1)^2/(2k) k=1..6 sum={ser}")


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
