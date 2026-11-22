#!/usr/bin/env python3
"""SGU-G24A (서강대 2024학년도 수시 논술 자연계열 1차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-G24A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """전수 열거(n=1..8): 증가·연속·미분가능·적분 조건을 정의대로 판정."""
    from fractions import Fraction as Fr
    X = sp.symbols("X")
    xs_neg = [Fr(-k, 40) for k in range(40, 0, -1)]          # [-1,0) 표본
    xs_pos = [Fr(k, 40) for k in range(1, 41)]           # (0,1] 표본

    def f(a, b, c, d, v):
        if v < 0:
            return v
        if v == 0:
            return Fr(d)
        return Fr(a, 2) * v * v + b * v + c

    def increasing(a, b, c, d):
        # x<0 쪽 상한 0(열린) ≤ ... 정의: x1<x2 ⇒ f(x1)<f(x2). 극한 비교 + 이차부분 엄격 증가를 직접 판정
        if not (d >= 0 and c >= d):           # sup_{x<0} f = 0 (미달성) 이므로 d>=0, inf_{x>0} f = c (미달성)
            return False
        # (0,1] 에서 q(x)=a/2 x^2+bx+c 의 엄격 증가: 미분 q'(x)=ax+b 가 [0,1]에서 ≥0 이고 항등 0 아님
        if not (b >= 0 and a + b >= 0 and (a, b) != (0, 0)):
            return False
        # 표본점 검사(보조)
        pts = xs_neg + [Fr(0)] + xs_pos
        vals = [f(a, b, c, d, v) for v in pts]
        return all(u < w for u, w in zip(vals, vals[1:]))

    ok = True
    msgs = []
    a1 = parts(md_answer(1, "1-1"))
    a2 = parts(md_answer(1, "1-2"))
    e3 = sp.sympify(md_answer(1, "1-3"), locals={"n": n})
    e4 = sp.sympify(md_answer(1, "1-4"), locals={"n": n})
    for N in range(1, 9):
        R = range(-N, N + 1)
        inc_ab = {(a, b) for a in R for b in R if increasing(a, b, N, 0)}         # (c,d)=(N,0)은 c≥d 만족
        inc_cd = {(c, d) for c in R for d in R if increasing(0, 1, c, d)}
        inc = [(a, b, c, d) for (a, b) in inc_ab for (c, d) in inc_cd]
        # 연속: x=0 에서 좌극한 0 = d = 우극한 c
        cont = lambda a, b, c, d: c == 0 and d == 0
        diff = lambda a, b, c, d: cont(a, b, c, d) and b == 1               # 우미분계수 b = 좌미분계수 1
        integ = lambda a, b, c, d: sp.integrate(X, (X, -1, 0)) + sp.integrate(sp.Rational(a, 2) * X**2 + b * X + c, (X, 0, 1)) <= 1
        sel = [q for q in inc if cont(*q) and diff(*q) and integ(*q)]
        ab2 = {(q[0], q[1]) for q in sel}
        cd2 = {(q[2], q[3]) for q in sel}
        n_cont = sum(1 for a in R for b in R for c in R for d in R if cont(a, b, c, d))
        n_cont_inc = sum(1 for q in inc if cont(*q))
        n_diff_inc = sum(1 for q in inc if diff(*q))
        P3 = sp.Rational(n_cont_inc, n_cont)
        P4 = sp.Rational(n_diff_inc, len(inc))
        sub = {n: N}
        ok &= len(inc_ab) == sp.sympify(a1["ab"], locals={"n": n}).subs(sub)
        ok &= len(inc_cd) == sp.sympify(a1["cd"], locals={"n": n}).subs(sub)
        ok &= len(ab2) == sp.sympify(a2["ab"], locals={"n": n}).subs(sub)
        ok &= len(cd2) == sp.sympify(a2["cd"], locals={"n": n}).subs(sub)
        ok &= sp.simplify(P3 - e3.subs(sub)) == 0
        ok &= sp.simplify(P4 - e4.subs(sub)) == 0
        msgs.append(f"n={N}:{len(inc_ab)},{len(inc_cd)},{len(ab2)},{P3},{P4}")
    report(1, ok, " ".join(msgs[:3]) + " ... (n=1..8 전수)")


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    # (2-1) 속력 수치 적분 vs md
    dx = lambda s: mp.e**(3 * s) + (s + 1) * mp.e**s
    dy = lambda s: mp.sqrt((2 * s - s**2) * mp.e**(2 * s) + 2 * s)
    L = mp.quad(lambda s: mp.sqrt(dx(s)**2 + dy(s)**2), [mp.mpf(1) / 2, mp.log(2)])
    v1 = sp.N(sp.sympify(md_answer(2, "2-1")), 30)
    ok1 = abs(L - mp.mpf(str(v1))) < mp.mpf("1e-20")
    # 인수분해 항등식 확인
    T = sp.symbols("T", real=True)
    lhs = (sp.exp(3 * T) + (T + 1) * sp.exp(T))**2 + (2 * T - T**2) * sp.exp(2 * T) + 2 * T
    ok1 &= sp.simplify(sp.expand(lhs - (2 * T + sp.exp(2 * T)) * (sp.exp(2 * T) + 1)**2)) == 0
    # (2-2) 부등식 수치 확인 + 극한·극대·변곡점
    g = 2 * sp.exp(x) - x**2
    okg = all(g.subs(x, sp.Rational(k, 10)) > 0 for k in range(0, 301))
    f = x * sp.exp(-x)
    lim = sp.limit(f, x, sp.oo)
    cr = sp.solve(sp.diff(f, x), x)
    ip = sp.solve(sp.diff(f, x, 2), x)
    a2 = parts(md_answer(2, "2-2"))
    mx = sp.sympify(a2["max"]); inf = sp.sympify(a2["inflection"])
    ok2 = okg and lim == sp.sympify(a2["limit"]) and cr == [mx[0]] and sp.simplify(f.subs(x, cr[0]) - mx[1]) == 0 \
        and ip == [inf[0]] and sp.simplify(f.subs(x, ip[0]) - inf[1]) == 0 and sp.limit(f, x, -sp.oo) == -sp.oo
    # (2-3) 접선 x절편 → 넓이 합 최대 = 1/e 인 c
    c, tt, X = sp.symbols("c tt X", real=True)
    tan = -sp.exp(-tt) * (X - tt) + sp.exp(-tt)
    t1 = sp.solve(tan, X)[0]
    S = (tt - c) * sp.exp(-tt) + (t1 - tt) * sp.exp(-t1)
    tc = sp.solve(sp.diff(S, tt), tt)[0]
    cval = sp.solve(sp.Eq(sp.simplify(S.subs(tt, tc)), sp.exp(-1)), c)
    ok3 = len(cval) == 1 and sp.simplify(cval[0] - sp.sympify(md_answer(2, "2-3"))) == 0 \
        and sp.simplify(sp.diff(S, tt, 2).subs(tt, tc)) .subs(c, cval[0]).evalf() < 0
    # (2-4) n=1..5 에서 반복 구성으로 a_n 을 직접 구하고, 일반식 1+1/e-Σe^{-k} 와 대조 후 극한
    ok4 = True
    for N in range(1, 6):
        pts = [tt]
        for _ in range(N):
            p = pts[-1]
            tn = sp.solve(-sp.exp(-p) * (X - p) + sp.exp(-p), X)[0]
            pts.append(tn)
        A = (tt - sp.exp(-1)) * sp.exp(-tt) + sum((pts[k] - pts[k - 1]) * sp.exp(-pts[k]) for k in range(1, N + 1))
        aN = sp.solve(sp.diff(A, tt), tt)[0]
        k = sp.symbols("k", integer=True, positive=True)
        ok4 &= sp.simplify(aN - (1 + sp.exp(-1) - sp.summation(sp.exp(-k), (k, 1, N)))) == 0
    k = sp.symbols("k", integer=True, positive=True)
    lim4 = 1 + sp.exp(-1) - sp.summation(sp.exp(-k), (k, 1, sp.oo))
    ok4 &= sp.simplify(lim4 - sp.sympify(md_answer(2, "2-4"))) == 0
    report(2, ok1 and ok2 and ok3 and ok4,
           f"L={mp.nstr(L, 12)} lim={lim} max_x={cr} infl_x={ip} c={cval} lim a_n={sp.N(lim4, 10)}")


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
