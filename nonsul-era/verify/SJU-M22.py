#!/usr/bin/env python3
"""SJU-M22 (세종대 2022학년도 논술 모의평가) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-M22"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.strip().split("=", 1) for seg in s.split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a = sp.symbols("a", positive=True)
    m = sp.symbols("m", real=True)
    # 중심 (0,√2) 에서 직선 mx - y - ma = 0 까지 거리 = 1
    roots = sp.solve(sp.Eq((sp.sqrt(2) + m * a)**2, m**2 + 1), m)
    P = parts(md_answer(1, "1"))
    md1 = sp.sympify(P["m1"], locals={"a": a}); md2 = sp.sympify(P["m2"], locals={"a": a})
    ok1 = True
    for av in [sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(9, 10)]:
        rv = sorted(float(r.subs(a, av)) for r in roots)
        ok1 &= abs(rv[0] - float(md1.subs(a, av))) < 1e-12 and abs(rv[1] - float(md2.subs(a, av))) < 1e-12 and rv[0] < 0 < rv[1]
    # (2) 각을 기하적으로 직접: θ = 2·arcsin(1/|PQ|)
    theta = 2 * sp.asin(1 / sp.sqrt(a**2 + 2))
    mdtan = sp.sympify(md_answer(1, "2"), locals={"a": a})
    ok2 = all(abs(sp.N(sp.tan(theta).subs(a, av) / mdtan.subs(a, av), 30) - 1) < 1e-20 for av in [sp.Rational(1, 10), sp.Rational(2, 5), sp.Rational(4, 5)])
    lim = sp.limit(sp.cos(theta) / a**2, a, 0, "+")
    ok3 = sp.simplify(lim - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"tanθ(0.5)={float(sp.tan(theta).subs(a,0.5)):.10f} cosθ={sp.simplify(sp.cos(theta))} lim={lim}")


def p02():
    # 후보 f 를 구간별로 재구성해 (가)(나) 직접 확인 + 각 소문항 값 비교
    k = sp.sympify(md_answer(2, "1"))
    fr = sp.sympify(md_answer(2, "2"), locals={"x": x})
    # (2-2) 독립 유도: u=f-x-2, u'=sqrt(u) → 2 sqrt(u) = x + C, f(4)=7
    C = sp.symbols("C")
    Cv = sp.solve(sp.Eq(2 * sp.sqrt(7 - 4 - 2), 4 + C), C)[0]
    f_right = ((x + Cv) / 2)**2 + x + 2
    ok2 = sp.expand(f_right - fr) == 0
    f = sp.Piecewise((x + k, x < 0), (x + 2, x <= 2), (fr, True))
    import mpmath as mp
    fl = sp.lambdify(x, f, "mpmath")
    mp.mp.dps = 30

    def rhs(v):
        pts = [0] + [b for b in (2,) if 0 < b < v] + [v]
        return mp.quad(lambda s: mp.sqrt(max(fl(s) - s - 2, 0)), pts) + v + 2
    ok_ga = all(abs(fl(v) - rhs(v)) < 1e-10 for v in [-3, -0.5, 0.5, 1.7, 2.5, 4, 6])
    ok_na = all(fl(v) > v + 2 for v in [2.01, 3, 10]) and abs(fl(4) - 7) < 1e-15
    ok_dom = all(fl(v) - v - 2 >= -1e-15 for v in [i / 10 for i in range(-50, 80)])
    ok1 = k == 2 and ok_ga
    # (2-3) [증명-수치확인] (0,2)에서 f>x+2 인 다른 해가 없는지: 연속·미분가능·f'>=1, f(0)=2,f(2)=4 이면 f=x+2 (평균값) — 값 비교
    ok3 = abs(fl(1) - float(sp.sympify(md_answer(2, "3")))) < 1e-15
    report(2, ok1 and ok2 and ok3 and ok_na and ok_dom, f"k={k} f(x>2)={sp.expand(f_right)} (가)확인={ok_ga} f(1)={fl(1)}")


def p03():
    import mpmath as mp
    mp.mp.dps = 30
    def f(v):
        return 1 + mp.pi / 2 * mp.cos(mp.pi / 2 * v) if abs(v) <= 1 else 2 * v**4 / (1 + v**2)
    def g(tt):
        F = lambda s: mp.quad(f, [tt, s] if (tt > 1 or s < -1 or (tt >= -1 and s <= 1)) else sorted(set([tt, s] + [p for p in (-1, 1) if min(tt, s) < p < max(tt, s)]))) - 2
        lo, hi = mp.mpf(tt), mp.mpf(tt) + 3
        for _ in range(120):
            mid = (lo + hi) / 2
            if F(mid) > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    ok1 = abs(g(-1)) < 1e-25
    # (3-2) [증명-수치확인] t 격자에서 t<=g(t)<=t+2, 그리고 f>=1
    ok2 = all(tt <= g(tt) <= tt + 2 for tt in [-5, -2.5, -1, -0.3, 0, 0.7, 1, 2, 5]) and all(f(v) >= 1 - 1e-25 for v in [i / 50 for i in range(-300, 301)])
    # (3-3) g'(t) 수치미분, 큰 t
    gp = lambda tt: mp.diff(g, tt)
    vals = [gp(tt) for tt in (50, 200, 1000)]
    ok3 = abs(vals[-1] - float(sp.sympify(md_answer(3, "3")))) < 1e-2 and abs(vals[-1] - 1) <= abs(vals[0] - 1)
    report(3, ok1 and ok2 and ok3, f"g(-1)={float(g(-1)):.2e} g'(50,200,1000)={[round(float(v),6) for v in vals]}")


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
