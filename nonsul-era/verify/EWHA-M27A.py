#!/usr/bin/env python3
"""EWHA-M27A (이화여대 2027학년도 모의논술 자연계열Ⅰ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import random
import re
import sys

import sympy as sp

PREFIX = "EWHA-M27A"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[증명-수치확인] (1)~(4) 무작위 표본 반례 탐색, (2) 근 개수 격자 조사."""
    random.seed(1)
    f = lambda t: t * math.log(t)
    ok1 = ok3 = ok4 = True
    for _ in range(20000):
        x, y = sorted(random.uniform(1.0001, 50) for _ in range(2))
        if y - x > 1e-9 and not f(x) < f(y):
            ok1 = False
        x, y = sorted(random.uniform(1e-6, 50) for _ in range(2))
        if y - x > 1e-6:
            if not math.log(x) <= (f(y) - f(x)) / (y - x):
                ok3 = False
            if not f((x + y) / 2) <= (f(x) + f(y)) / 2 + 1e-12:
                ok4 = False
    # (2) x ln x - pi 의 부호 변화 개수 (0,∞) 격자 + 증가 구간 확인
    X = sp.symbols("x", positive=True)
    g = X * sp.log(X) - sp.pi
    grid = [10 ** (k / 1000) for k in range(-6000, 3000)]
    vals = [float(g.subs(X, v)) if False else v * math.log(v) - math.pi for v in grid]
    changes = sum(1 for u, w in zip(vals, vals[1:]) if u * w < 0)
    root = sp.nsolve(g, X, 2)
    crit = sp.solve(sp.diff(g, X), X)
    ok2 = changes == 1 and crit == [sp.exp(-1)] and float(root) > float(sp.exp(-1))
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] 반례 없음(2만 표본), 근 개수={changes}, 근≈{float(root):.6f}")


def p02():
    cnt = {}
    for s in itertools.combinations(range(1, 13), 4):
        m = min(b - a for a, b in zip(s, s[1:]))
        cnt[m] = cnt.get(m, 0) + 1
    tot = sum(cnt.values())
    ge2 = sum(v for k, v in cnt.items() if k >= 2)
    eq2 = cnt.get(2, 0)
    ex = sp.Rational(sum(k * v for k, v in cnt.items()), tot)
    ok = (ge2 == int(md_answer(2, "1")) and eq2 == int(md_answer(2, "2"))
          and ex == sp.Rational(md_answer(2, "3")) and max(cnt) == 3
          and ge2 == math.comb(9, 4))
    report(2, ok, f"전수: 분포={dict(sorted(cnt.items()))} E={ex}")


def p03():
    al = sp.symbols("alpha", positive=True)
    X, Y = sp.symbols("X Y", real=True)
    tang = sp.expand((Y - 1 / al) + (X - al) / al**2)          # = 0
    c0 = -tang.subs({X: 0, Y: 0})                               # 상수항을 우변으로
    l = sp.diff(tang, X) / c0
    m = sp.diff(tang, Y) / c0
    ok1 = sp.simplify(l + m - sp.sympify(md_answer(3, "1"), locals={"alpha": al})) == 0
    # (2) 절편 a=1/l, b=1/m, 넓이 2ab 상수 + ab 값에 따른 교점 개수(제1사분면 × 4)
    a, b = 1 / l, 1 / m
    area = sp.simplify(2 * a * b)
    ok2 = area.free_symbols == set() and sp.simplify(area - sp.sympify(md_answer(3, "2"))) == 0

    def n_inter(av, bv):  # x/a + (1/x)/b = 1, x>0 의 해 개수 × 4
        x = sp.symbols("x", positive=True)
        sols = sp.solve(sp.Eq(x / av + 1 / (x * bv), 1), x)
        return 4 * len(set(sols))
    ok2 = ok2 and n_inter(sp.Rational(1), sp.Rational(4)) == 4 and n_inter(1, 3) == 0 and n_inter(2, 3) == 8
    # (3) ab=4 에서 둘레 4*sqrt(a^2+b^2) 최소
    t = sp.symbols("t", positive=True)
    P = 4 * sp.sqrt(t**2 + 16 / t**2)
    cr = [c for c in sp.solve(sp.diff(P, t), t)]
    pmin = sp.simplify(min(P.subs(t, c) for c in cr))
    ok3 = sp.simplify(pmin - sp.sympify(md_answer(3, "3"))) == 0
    report(3, ok1 and ok2 and ok3, f"l+m={sp.simplify(l + m)} area={area} minP={pmin}")


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
