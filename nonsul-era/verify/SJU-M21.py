#!/usr/bin/env python3
"""SJU-M21 (세종대 2021학년도 모의논술고사 자연계열) — sympy 검산. 정답 하드코딩 금지, 설정으로부터 독립 재계산."""
import re
import sys
from fractions import Fraction as Fr
from itertools import product, combinations

import sympy as sp

PREFIX = "SJU-M21"
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


def parts(no: int, sub: str) -> dict:
    """'k1 = v1; k2 = v2' 형식의 정답을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def S(s: str):
    return sp.sympify(s, locals={"E": sp.E, "e": sp.E, "x": x, "t": t, "M": sp.Symbol("M", positive=True)})


def same(a, b, tol=1e-12) -> bool:
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 or abs(complex(sp.N(a - b, 30))) < tol
    except Exception:  # noqa: BLE001
        return abs(complex(sp.N(a - b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    th, r = sp.symbols("theta r", positive=True)
    rr = sp.solve(sp.Eq(r * sp.cos(th) + 2 * r * sp.sin(th), 2), r)[0]       # x+2y=2
    S_ = sp.Rational(1, 2) * 2 * rr * sp.sin(th)
    ok1 = sp.simplify(S_ - sp.sympify(md_answer(1, "1"), locals={"theta": th})) == 0
    dS = sp.simplify(sp.diff(S_, th))
    ok2 = all(float(dS.subs(th, v)) > 0 for v in [i / 1000 * float(sp.pi / 2) for i in range(1, 1001)])
    c = sp.Symbol("c", positive=True)
    w = c / dS                                                                # dθ/dt
    import mpmath as mp
    wf = sp.lambdify(th, w.subs(c, 1), "mpmath")
    tmax = mp.findroot(lambda u: mp.diff(wf, u), 1.1)
    gridbest = max((i / 10000 * mp.pi / 2 for i in range(1, 10001)), key=wf)
    P = (float(rr.subs(th, tmax) * sp.cos(tmax)), float(rr.subs(th, tmax) * sp.sin(tmax)))
    md = S(md_answer(1, "3"))
    ok3 = abs(P[0] - float(md[0])) < 1e-9 and abs(P[1] - float(md[1])) < 1e-9 and abs(gridbest - tmax) < 1e-3
    report(1, ok1 and ok2 and ok3, f"[증명-수치확인] dS/dθ={dS}>0, argmax θ={float(tmax):.10f} P=({P[0]:.10f},{P[1]:.10f})")


def p02():
    from functools import lru_cache

    @lru_cache(None)
    def win(a, b):        # A 가 앞면 a 번, B 가 뒷면 b 번 더 필요할 때 A 승리 확률
        if a == 0:
            return Fr(1)
        if b == 0:
            return Fr(0)
        return Fr(1, 2) * win(a - 1, b) + Fr(1, 2) * win(a, b - 1)
    # (1) 전수: 7번 던지기 2^7 열거 → 종료 시점
    tot = Fr(0)
    for seq in product((0, 1), repeat=7):
        h = tl = 0
        for i, s in enumerate(seq, 1):
            h += s; tl += 1 - s
            if h == 4 or tl == 4:
                tot += Fr(i, 2**7); break
    ok1 = same(sp.Rational(tot.numerator, tot.denominator), S(md_answer(2, "1")))
    p2 = win(3, 2) * 10000
    ok2 = same(sp.Rational(p2.numerator, p2.denominator), S(md_answer(2, "2")))
    best = max(range(3, 40), key=lambda k: win(k, 3) * 2**k * 10000 - 5000 * (k - 3)**2)
    vals = {k: float(win(k, 3) * 2**k * 10000 - 5000 * (k - 3)**2) for k in range(3, 8)}
    ok3 = -best == int(S(md_answer(2, "3")))
    report(2, ok1 and ok2 and ok3, f"E(X)={tot} A상금={p2} best n={best} 수익={vals}")


def p03():
    import random
    M = sp.Symbol("M", positive=True)
    # (3-1)(3-2) [증명-수치확인]: f(0)=f(1)=0 인 무작위 다항식에서 |f| 최대점 a 의 f'(a)=0, |f(a)|<=max|f''|/8
    random.seed(1)
    ok12 = True
    for _ in range(200):
        deg = random.randint(2, 6)
        cs = [random.uniform(-3, 3) for _ in range(deg - 1)]
        q = sum(cc * x**i for i, cc in enumerate(cs))
        f = x * (1 - x) * q
        if sp.Poly(f, x).degree() < 1:
            continue
        fl = sp.lambdify(x, f); f1 = sp.lambdify(x, sp.diff(f, x)); f2 = sp.lambdify(x, sp.diff(f, x, 2))
        xs = [i / 4000 for i in range(4001)]
        a = max(xs, key=lambda u: abs(fl(u)))
        Mv = max(abs(f2(u)) for u in xs)
        ok12 = ok12 and abs(f1(a)) < 5e-3 * (1 + Mv) and abs(fl(a)) <= Mv / 8 + 1e-9
    # (3-3) 등호: f''=±M 인 이차식 결정 후 md 답과 비교
    b, c0 = sp.symbols("b c0")
    sols = []
    for sgn in (1, -1):
        f = sgn * M * x**2 / 2 + b * x + c0
        s = sp.solve([f.subs(x, 0), f.subs(x, 1)], [b, c0], dict=True)[0]
        ff = sp.expand(f.subs(s))
        if sp.simplify(sp.Abs(ff.subs(x, sp.Rational(1, 2))) - M / 8) == 0:
            sols.append(ff)
    mdf = [sp.expand(S(seg.split("=", 1)[1])) for seg in md_answer(3, "3").split(" or ")]
    ok3 = len(sols) == 2 and all(any(sp.simplify(u - v) == 0 for v in mdf) for u in sols) and len(mdf) == 2
    report(3, ok12 and ok3, f"[증명-수치확인] 무작위 200개 다항식 f'(a)=0, |f(a)|<=M/8 / 등호 f={sols}")


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
