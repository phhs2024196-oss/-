#!/usr/bin/env python3
"""YSU-G23 (연세대 2023학년도 수시 논술 자연계열 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import random
import re
import sys

import sympy as sp

PREFIX = "YSU-G23"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/YSU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def simulate(students, N=2023):
    """전등 N개를 직접 시뮬레이션: (누른 횟수 리스트, 켜짐 리스트)."""
    cnt = [0] * (N + 1)
    for k in students:
        for n in range(k, N + 1, k):
            cnt[n] += 1
    return cnt, [c % 2 == 1 for c in cnt]


def p01():
    cnt, on = simulate(range(1, 2024))
    ok1 = cnt[2023] == sp.sympify(md_answer(1, "1"))
    lit = [n for n in range(1, 2024) if on[n]]
    three = [n for n in lit if cnt[n] == 3]
    pr2 = sp.Rational(len(three), len(lit))
    ok2 = pr2 == sp.sympify(md_answer(1, "2"))
    _, on3 = simulate(range(4, 2022))
    nlit = sum(on3[1:])
    pr3 = sp.Rational(nlit, 2023)
    ok3 = pr3 == sp.sympify(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"presses(2023)={cnt[2023]} lit={len(lit)} three={len(three)} p2={pr2} lit3={nlit} p3={pr3}")


def measure(fun, t, N=200000):
    """{x in [0,1] : f(x) > t} 의 길이(중점 격자 수치)."""
    return sum(1 for i in range(N) if fun((i + 0.5) / N) > t) / N


def p02():
    x, t = sp.symbols("x t", real=True)
    # (2-1) 해집합을 sympy 로
    f1 = 36 * x**3 - 27 * x**2 - 4 * x + 9
    S = sp.solveset(f1 > 6, x, sp.Interval(0, 1))
    g6 = S.measure
    ok1 = sp.simplify(g6 - sp.sympify(md_answer(2, "1"))) == 0
    # (2-2) g(t)=1 <=> t <= min f (최솟값과 같을 때는 한 점만 빠짐)
    f2 = 8 * x**3 - 11 * x**2 + 3 * x + 5
    cands = [sp.Integer(0), sp.Integer(1)] + [r for r in sp.solve(sp.diff(f2, x), x) if 0 <= r <= 1]
    mn = min(f2.subs(x, r) for r in cands)
    S_t = sp.Interval(-sp.oo, mn)
    md2 = sp.sympify(md_answer(2, "2"))
    # 경계 확인: t=mn 에서 길이 1, t=mn+eps 에서 길이<1
    lam = sp.lambdify(x, f2)
    edge = measure(lam, float(mn), 20000) == 1.0 and measure(lam, float(mn) + 1e-3, 20000) < 1.0
    ok2 = md2 == S_t and edge
    # (2-3) 네 경우의 공식을 무작위 (a,b,c) 와 t 에서 수치 길이와 비교
    a, b, c = sp.symbols("a b c", real=True)
    cases = []
    for seg in md_answer(2, "3").split("; "):
        cond, expr = seg.split(": ", 1)
        cases.append((sp.sympify(cond, locals={"a": a, "b": b, "c": c}),
                      sp.sympify(expr, locals={"a": a, "b": b, "c": c, "t": t})))
    rng = random.Random(23)
    ok3, tested, hit = True, 0, set()
    while tested < 60:
        av, bv = rng.uniform(0.2, 4), -rng.uniform(0.2, 4)
        cv = rng.uniform(-3, av - bv + 2)
        sub = {a: av, b: bv, c: cv}
        holds = [i for i, (cd, _) in enumerate(cases) if bool(cd.subs(sub))]
        if len(holds) != 1:
            ok3 = False
            break
        hit.add(holds[0])
        fv = lambda xx: min(av * xx, bv * xx + cv)
        expr = cases[holds[0]][1].subs(sub)
        lo = min(0, bv + cv, cv) - 1
        hi = max(av, cv, av * cv / (av - bv)) + 1
        for _ in range(5):
            tv = rng.uniform(lo, hi)
            if abs(float(expr.subs(t, tv)) - measure(fv, tv, 20000)) > 2e-4:
                ok3 = False
        tested += 1
    ok3 = ok3 and hit == {0, 1, 2, 3}
    # (2-4) 적분값: ∫0^1 f dx 를 sympy 로, 그리고 ∫0^2023 g(t)dt 를 수치로 직접
    X = sp.symbols("X", real=True)
    inner = a * (X - b) * (X - c)
    I = sp.integrate(inner, (X, 0, c)) + sp.integrate(-inner, (X, c, 1))
    md4 = sp.sympify(md_answer(2, "4"), locals={"a": a, "b": b, "c": c})
    ok4 = sp.simplify(I - md4) == 0
    import numpy as np
    for _ in range(4):
        av, bv, cv = -rng.uniform(0.1, 9.9), -rng.uniform(0.1, 9.9), rng.uniform(0.05, 0.95)
        xs = (np.arange(4000) + 0.5) / 4000
        fx = np.abs(av * (xs - bv) * (xs - cv))
        ok4 = ok4 and fx.max() < 2023
        ts = np.linspace(0, 2023, 400001)
        ts_mid = (ts[:-1] + ts[1:]) / 2
        srt = np.sort(fx)
        gt = 1 - np.searchsorted(srt, ts_mid, side="right") / len(xs)   # 길이 = f>t 비율
        num = float(np.sum(gt) * (ts[1] - ts[0]))
        ok4 = ok4 and abs(num - float(md4.subs({a: av, b: bv, c: cv}))) < 5e-3
    report(2, ok1 and ok2 and ok3 and ok4, f"g(6)={g6} range=(-oo,{mn}] cases_hit={sorted(hit)} int=∫0^1 f dx")


def p03():
    m, x, y = sp.symbols("m x y", positive=True)
    rng = random.Random(3)
    # (3-1) 무작위 P,Q 의 중점이 YAML 방정식·범위를 만족, 역으로 곡선 위 점에서 P,Q 복원
    parts = [p.strip() for p in md_answer(3, "1").split(";")]
    eq = sp.sympify(parts[0].split("=")[0], locals={"m": m, "x": x, "y": y})
    ineqs = [sp.sympify(p, locals={"m": m, "x": x, "y": y}) for p in parts[1:]]
    ok1 = True
    for _ in range(50):
        mv = rng.uniform(1, 4)
        x2 = rng.uniform(0, 1 / (2 * mv) * 1.9)
        # x1>=0 이 되도록 (x1-x2)^2 + m^2 (x1+x2)^2 = 1 에서 x1 풀기
        x1s = [r for r in sp.solve((sp.Symbol("u") - x2)**2 + mv**2 * (sp.Symbol("u") + x2)**2 - 1, sp.Symbol("u")) if r.is_real and r >= 0]
        for x1 in x1s:
            mx_, my_ = (x1 + x2) / 2, (mv * x1 - mv * x2) / 2
            sub = {m: mv, x: mx_, y: my_}
            ok1 = ok1 and abs(float(eq.subs(sub))) < 1e-9 and all(bool(q.subs(sub)) for q in ineqs)
        # 역방향: 곡선 위 점 → x1=x+y/m, x2=x-y/m >= 0, PQ=1
        xv = rng.uniform(1 / (2 * (mv**2 + 1) ** 0.5), 1 / (2 * mv))
        yv = rng.choice([1, -1]) * mv / 2 * (1 - 4 * mv**2 * xv**2) ** 0.5
        X1, X2 = xv + yv / mv, xv - yv / mv
        ok1 = ok1 and X1 >= -1e-12 and X2 >= -1e-12 and abs((X1 - X2)**2 + mv**2 * (X1 + X2)**2 - 1) < 1e-9
    # (3-2) s(t) 를 이차방정식에서 독립적으로 풀고 미분
    t, s = sp.symbols("t s", real=True)
    roots = sp.solve((s - t)**2 + m**2 * (s + t)**2 - 1, s)
    md2 = sp.sympify(md_answer(3, "2"), locals={"m": m})
    ok2 = True
    for mv in (1, sp.Rational(3, 2), 2, sp.sqrt(15) / 3, 5):
        t0 = 1 / (2 * sp.sqrt(1 + mv**2))
        rs = [r.subs(m, mv) for r in roots]
        good = [r for r in rs if float(r.subs(t, t0)) >= 0]
        if len(good) != 1:
            ok2 = False
            continue
        v = sp.diff(good[0], t).subs(t, t0)
        vel = (v, mv * v)
        ok2 = ok2 and all(abs(float(vel[i] - md2[i].subs(m, mv))) < 1e-12 for i in range(2))
    # (3-3) 좌표로 직접 내심 계산
    mv = sp.sqrt(15) / 3
    t0 = sp.Rational(1, 2) / sp.sqrt(1 + mv**2)
    Q = sp.Matrix([t0, -mv * t0])
    sv = [r.subs({m: mv, t: t0}) for r in roots]
    sv = [r for r in sv if r >= 0][0]
    P = sp.Matrix([sv, mv * sv])
    O = sp.Matrix([0, 0])

    def incircle(A, B, C):
        a_, b_, c_ = (B - C).norm(), (C - A).norm(), (A - B).norm()
        I = (a_ * A + b_ * B + c_ * C) / (a_ + b_ + c_)
        area = sp.Abs((B - A)[0] * (C - A)[1] - (B - A)[1] * (C - A)[0]) / 2
        return I.applyfunc(sp.simplify), sp.simplify(2 * area / (a_ + b_ + c_))

    I1, R = incircle(O, P, Q)
    x3 = sp.simplify(I1[0] - R)                       # C1 과 x축의 원점 쪽 교점 (중심이 x축 위)
    A2, B2 = sp.Matrix([x3, mv * x3]), sp.Matrix([x3, -mv * x3])
    I2, r = incircle(O, A2, B2)
    dist = sp.sqrt((I1[0] - I2[0])**2 + (I1[1] - I2[1])**2)
    md3 = sp.sympify(md_answer(3, "3"))
    ok3 = abs(sp.N(dist - md3, 30)) < 1e-25 and sp.simplify(I1[1]) == 0
    report(3, ok1 and ok2 and ok3, f"OP={sp.nsimplify(sp.simplify(P.norm()))} R={sp.nsimplify(sp.N(R, 40), [sp.sqrt(15)])} dist={sp.N(dist, 15)}")


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
