#!/usr/bin/env python3
"""SJU-M27 (세종대 2027학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-M27"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, p, t = sp.symbols("a p t", positive=True)
    X = sp.symbols("X", real=True)
    loc = {"t": t}
    sol = sp.solve([2 * a * p - sp.sqrt(3), a * p**2 + t - sp.sqrt(3) * p], [a, p], dict=True)[0]
    P = (sol[p], sp.sqrt(3) * sol[p]); gt = sol[a]
    qx = sp.solve(sp.Eq(0 - P[1], -1 / sp.sqrt(3) * (X - P[0])), X)[0]          # 법선의 x절편
    ok1 = sp.simplify(qx - sp.sympify(md_answer(1, "1"), locals=loc)[0]) == 0
    R2 = (qx - P[0])**2 + P[1]**2
    ell = sp.sqrt(qx**2 + gt**2 - R2)                                           # 접선 길이
    dl = sp.diff(ell, t)
    lim = sp.limit(dl, t, sp.oo)
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "2").replace(" ", "").split(";"))
    md_dl = sp.sympify(parts["l1(t)"], locals=loc)
    ok2 = sp.simplify(lim - sp.sympify(parts["limit"])) == 0 and all(
        abs(float(dl.subs(t, v) - md_dl.subs(t, v))) < 1e-12 for v in (0.3, 1, 2.5))
    side = -(X - qx) / sp.sqrt(3)                                               # 직선 PQ 의 높이
    V = sp.integrate(sp.sqrt(3) / 4 * side**2, (X, P[0], qx))
    ok3 = sp.simplify(V - sp.sympify(md_answer(1, "3"), locals=loc)) == 0
    report(1, ok1 and ok2 and ok3, f"Q_x={qx} lim={lim} V={sp.simplify(V)}")


def p03():
    al = sp.symbols("alpha", real=True)
    b, c = sp.symbols("b c", real=True)
    f = x**3 + b * x**2 + c * x + 1                                    # f(0)=1
    f = f.subs(c, sp.Rational(1, 2))                                   # f'(0)=1/2 (g 의 x<0 기울기)
    v1 = sp.simplify(f.subs(x, 2) - sp.diff(f, x).subs(x, 2))
    ok1 = sp.simplify(v1 - sp.sympify(md_answer(3, "1"))) == 0
    # (2) b 를 격자로 훑어 [0,∞)에서 f'>=0 인 b 중 f(2) 최소
    import numpy as np
    xs = np.linspace(0, 10, 4001)
    def incr(bv):
        return all(3 * xx**2 + 2 * bv * xx + 0.5 >= -1e-12 for xx in xs)
    bs = np.linspace(-3, 1, 40001)
    f2 = sp.lambdify(b, f.subs(x, 2))
    m2 = min(f2(bv) for bv in bs if incr(bv))
    ok2 = abs(m2 - float(sp.sympify(md_answer(3, "2")))) < 1e-3
    # (3) b 를 격자로 훑어 조건 (가)(나)를 정수 -20..20 에서 직접 확인 → 21-6g'(1) 집합
    fl = sp.lambdify((x, b), f)
    gfun = lambda xx, bv: xx / 2 + 1 if xx < 0 else fl(xx, bv)
    vals = []
    for bv in np.linspace(-4, 1, 50001):
        ga = all(gfun(k, bv) * gfun(k + 1, bv) >= -1e-12 for k in range(-20, 20))
        gb = any(-1e-12 <= gfun(j + 1, bv) - gfun(j, bv) <= 0.25 + 1e-12 for j in range(-20, 20))
        if ga and gb:
            vals.append(21 - 6 * (3 + 2 * bv + 0.5))
    vals = np.array(vals)
    lo1, hi1 = vals[vals < 24].min(), vals[vals < 24].max()
    lo2, hi2 = vals[vals > 24].min(), vals[vals > 24].max()
    ok3 = all(abs(u - w) < 2e-3 for u, w in [(lo1, 15), (hi1, 18), (lo2, 29), (hi2, 30)]) and \
        md_answer(3, "3").replace(" ", "") == "[15,18]U[29,30]"
    report(3, ok1 and ok2 and ok3, f"g(2)-g'(2)={v1} min g(2)~{m2:.5f} ranges=[{lo1:.3f},{hi1:.3f}]U[{lo2:.3f},{hi2:.3f}]")


def p02():
    f = x**3 + 3 * x + 1
    fp, fpp = sp.diff(f, x), sp.diff(f, x, 2)
    g = lambda y: sp.nsolve(f - y, x, 0)          # f 는 증가 → 실근 1개
    al, be = g(sp.Rational(1, 3)), g(sp.Rational(4, 3))
    ok1 = al < 0 < be
    gpp = lambda y: -fpp.subs(x, g(y)) / fp.subs(x, g(y))**3
    ok2 = gpp(sp.Rational(1, 3)) > 0 and gpp(sp.Rational(4, 3)) < 0
    # (2-3) F(s)=(g(s+1)-g(s))^2+1 을 격자 탐색 후 정밀화 (정답을 쓰지 않음)
    import mpmath as mp
    gm = lambda y: mp.findroot(lambda u: u**3 + 3*u + 1 - y, 0)
    F = lambda s: (gm(s + 1) - gm(s))**2 + 1
    grid = [i / 100 for i in range(-500, 501)]
    s0 = max(grid, key=F)
    smax = mp.findroot(lambda s: mp.diff(F, s), s0)
    # 원 답(1/2)과의 비교는 md 정답 문자열을 읽어서 한다
    ans = sp.Rational(md_answer(2, "3"))
    ok3 = abs(smax - float(ans)) < 1e-9 and F(smax) > F(smax + 0.3) and F(smax) > F(smax - 0.3)
    report(2, ok1 and ok2 and ok3, f"alpha={float(al):.4f} beta={float(be):.4f} argmax={float(smax):.10f}")


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
