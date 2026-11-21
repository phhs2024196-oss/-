#!/usr/bin/env python3
"""SJU-M24 (세종대 2024학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-M24"
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
    res = []
    for a1 in range(1, 1500):
        for a2 in range(1, 200):                       # (가) a2<200
            s = [a1, a2]
            for _ in range(7):
                s.append(2 * s[-2] if s[-1] <= 80 else s[-1] - 80)
            if s[4] == 4 and all(v >= 1 for v in s):
                res.append(s)
    v1 = max(s[1] for s in res)
    v2 = sum(set(s[0] for s in res))
    v3 = sum(set(s[8] for s in res if s[7] <= 90))
    ok = v1 == int(md_answer(1, "1")) and v2 == int(md_answer(1, "2")) and v3 == int(md_answer(1, "3"))
    # 대표 오답: a1=82 를 빼먹으면 820, a4=84 갈래를 빼면 24
    report(1, ok, f"max a2={v1} sum a1={v2} sum a9={v3} a9 set={sorted(set(s[8] for s in res if s[7] <= 90))}")


def p02():
    # 관계식을 u=x+y 로 매개화: e^u=x-y ⇒ x=(u+e^u)/2, y=(u-e^u)/2 (x'(u)>0 이므로 함수 y=f(x) 전체)
    u = sp.symbols("u", real=True)
    X, Y = (u + sp.exp(u)) / 2, (u - sp.exp(u)) / 2
    ok_rel = sp.simplify(sp.exp(X + Y) + Y - X) == 0
    u0 = sp.solve(sp.Eq(X, sp.Rational(1, 2)), u)[0]
    slope = sp.diff(Y, u) / sp.diff(X, u)
    ok1 = ok_rel and sp.simplify(Y.subs(u, u0) + sp.Rational(1, 2)) == 0 and \
        sp.simplify(slope.subs(u, u0) - sp.sympify(md_answer(2, "1"))) == 0
    y2 = sp.simplify(sp.diff(slope, u) / sp.diff(X, u))
    import numpy as np
    y2f = sp.lambdify(u, y2, "numpy")
    us = np.linspace(-30, 30, 1000)
    ok2 = bool(np.all(y2f(us) < 0)) and md_answer(2, "2") is None      # [증명-수치확인]
    # (3) 직선 x-2y-1=0 까지의 부호 있는 거리 d(u) 최소화
    d = (X - 2 * Y - 1) / sp.sqrt(5)
    crit = sp.solve(sp.diff(d, u), u)
    uc = crit[0]
    Q = sp.sympify(md_answer(2, "3"))
    df = sp.lambdify(u, d, "numpy")
    ok3 = len(crit) == 1 and sp.simplify(X.subs(u, uc) - Q[0]) == 0 and sp.simplify(Y.subs(u, uc) - Q[1]) == 0 and \
        bool(np.all(df(us) > 0)) and bool(np.all(df(us) >= float(d.subs(u, uc)) - 1e-12))
    report(2, ok1 and ok2 and ok3, f"slope={slope.subs(u, u0)} y''={y2} [증명-수치확인] Q=({sp.N(X.subs(u, uc), 8)},{sp.N(Y.subs(u, uc), 8)})")


def p03():
    import mpmath as mp
    mp.mp.dps = 20
    tt = sp.symbols("t", positive=True)
    md1 = sp.sympify(md_answer(3, "1"), locals={"t": tt})
    # 서로 다른 두 f 로 검산: (A) 1-|x|, (B) c(1-x^2)^p - (나)를 만족하도록 p 를 수치로 결정(c 무관)
    pB = mp.findroot(lambda p: mp.quad(lambda x: (1 - x**2)**p, [-1, 1]) - 1, 2.4)
    fs = [lambda x: 1 - abs(x), lambda x: (1 - x**2)**pB]
    Qd = lambda f, a, b: mp.quad(f, [a, 0, b] if a < 0 < b else [a, b])
    def hmin(f, t):
        g = lambda x: Qd(lambda y: (x - y) * f(y), -1, x) + t * Qd(lambda y: (y - x) * f(y), x, 1)
        grid = [mp.mpf(-1) + 2 * mp.mpf(i) / 400 for i in range(1, 400)]
        x0 = min(grid, key=g)
        gp = lambda x: Qd(f, -1, x) - t * Qd(f, x, 1)
        return mp.findroot(gp, x0), g
    ok1 = True
    for f in fs:
        assert abs(f(0) - mp.quad(f, [-1, 0, 1])) < 1e-12           # 조건 (나)
        for t in (0.3, 1, 2.5):
            h, g = hmin(f, t)
            ok1 &= g(h) <= g(h + 0.01) and g(h) <= g(h - 0.01)
            ok1 &= abs(Qd(f, -1, h) / f(0) - float(md1.subs(tt, t))) < 1e-12
    # (2) h'(1): 두 f 모두 대칭이라 h(1)=0. 중심차분
    ok2 = True
    for f in fs:
        hd = (hmin(f, 1 + 1e-5)[0] - hmin(f, 1 - 1e-5)[0]) / 2e-5
        ok2 &= abs(hmin(f, 1)[0]) < 1e-12 and abs(hd - float(sp.sympify(md_answer(3, "2")))) < 1e-6
    # (3) f=1-|x| 에서 md 식 두 조각을 읽어 수치 최소점과 비교
    s = md_answer(3, "3").split("=", 1)[1]
    e1, e2 = [sp.sympify(seg.split("(")[0] if False else seg[:seg.rindex("(")], locals={"t": tt}) for seg in s.split(";")]
    ok3 = True
    for t in (0.1, 0.5, 1, 1.7, 6):
        h = hmin(fs[0], t)[0]
        e = e1 if t <= 1 else e2
        ok3 &= abs(h - float(e.subs(tt, t))) < 1e-12
    report(3, ok1 and ok2 and ok3, f"p_B={mp.nstr(pB,8)} h(1)={mp.nstr(hmin(fs[0],1)[0],5)} checks ok")


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
