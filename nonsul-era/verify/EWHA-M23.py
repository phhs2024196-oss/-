#!/usr/bin/env python3
"""EWHA-M23 (이화여대 2023학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import random
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "EWHA-M23"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str):
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
    # [증명-수치확인] 접선 교점으로 f 를 독립 유도 후 (1)~(4) 수치 반례 탐색
    a, X = sp.symbols("a X", positive=True)
    tangent = a**2 + 2 * a * (X - a)
    f = sp.lambdify(a, sp.solve(sp.Eq(tangent, 2), X)[0])
    rnd = random.Random(2023)
    ok = True
    # (1) a1>0 이면 a2>=sqrt2
    for v in [10 ** rnd.uniform(-4, 4) for _ in range(2000)]:
        ok &= f(v) >= math.sqrt(2) - 1e-12
    # (2)(3) a1>=sqrt2 이면 a_n>=sqrt2, 감소 / (4) 임의 a1>0 에서 b_n 감소, b_n=a_{n+1}
    for _ in range(300):
        a1 = 10 ** rnd.uniform(-3, 3)
        seq = [a1]
        for _n in range(200):
            seq.append(f(seq[-1]))
        b = [0.5 * (s + 2 / s) for s in seq[:-1]]
        ok &= all(abs(b[i] - seq[i + 1]) < 1e-9 * max(1, seq[i + 1]) for i in range(len(b)))
        ok &= all(b[i + 1] <= b[i] + 1e-12 for i in range(len(b) - 1))
        if a1 >= math.sqrt(2):
            ok &= all(s >= math.sqrt(2) - 1e-12 for s in seq)
            ok &= all(seq[i + 1] <= seq[i] + 1e-12 for i in range(len(seq) - 1))
    kinds = [md_answer(1, s) for s in ("1", "2", "3", "4")]
    report(1, ok and all(k is None for k in kinds), f"[증명-수치확인] f(a)={sp.simplify(sp.solve(sp.Eq(tangent, 2), X)[0])}, a1 무작위 300개×n≤200 반례 없음")


def p02():
    t, x = sp.symbols("t x", positive=True)
    f = t * x**3 + 3 * x**2 + 3 * x + 2023
    fp = sp.diff(f, x)
    r = sp.solve(fp, x)                       # 두 근(0<t<1)
    tv = sp.Rational(3, 10)
    rs = sorted(r, key=lambda e: float(e.subs(t, tv)))
    a, b = rs
    v1 = sp.simplify(sp.integrate(fp, (x, a, b)) / (t * (b - a) ** 3))
    ok1 = sp.simplify(v1 - sp.sympify(md_answer(2, "1"))) == 0
    BC = b - a
    AC = f.subs(x, a) - f.subs(x, b)
    g = BC / sp.sqrt(BC**2 + AC**2)
    t0s = [s for s in sp.solve(sp.Eq(sp.simplify(g**2), sp.Rational(4, 5)), t) if s.is_positive and s < 1]
    ok2 = len(t0s) == 1 and t0s[0] == sp.sympify(md_answer(2, "2"))
    t0 = t0s[0]
    A0, B0 = [sp.nsimplify(sp.simplify(e.subs(t, t0))) for e in (a, b)]
    f0 = f.subs(t, t0)
    area = sp.simplify(sp.integrate(f0 - f0.subs(x, B0), (x, A0, B0)))
    # 영역 확인: [a,b] 에서 f >= f(b), x>b 에서는 f>f(b) (다시 만나지 않음)
    okreg = all(float((f0 - f0.subs(x, B0)).subs(x, xv)) > 0 for xv in np.linspace(float(B0) + 0.01, float(B0) + 50, 50))
    ok3 = sp.simplify(area - sp.sympify(md_answer(2, "3"))) == 0 and okreg
    lim = sp.limit(sp.simplify(g / t), t, 0, "+")
    ok4 = sp.simplify(lim - sp.sympify(md_answer(2, "4"))) == 0
    report(2, ok1 and ok2 and ok3 and ok4, f"(1)={v1} (2)t0={t0} (3)넓이={area} (4)lim={lim}")


def boundary(x):
    return 3 / 8 * (np.abs(x) - np.abs(x - 2) + 4)


def contact_r(a):
    """원 (x-a)^2+(y-r)^2=r^2 가 경계 아래에 머무는 최대 r (A 와 닿기만 하는 원) — 이분법."""
    th = np.linspace(0, 2 * np.pi, 4001)

    def inside_ok(r):
        xs, ys = a + r * np.cos(th), r + r * np.sin(th)
        return np.all(ys <= boundary(xs) + 1e-12)

    lo, hi = 1e-6, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if inside_ok(mid) else (lo, mid)
    return lo


def contacts(a, r):
    th = np.linspace(0, 2 * np.pi, 20001)
    xs, ys = a + r * np.cos(th), r + r * np.sin(th)
    gap = boundary(xs) - ys
    idx = np.where(gap < 1e-6)[0]
    if len(idx) == 0:
        return 0
    # 연속 구간 개수(접촉점 개수)
    groups = 1 + int(np.sum(np.diff(idx) > 50))
    if idx[0] == 0 and idx[-1] == len(th) - 1 and groups > 1:
        groups -= 1
    return groups


def p03():
    a, X, Y = sp.symbols("a X Y", real=True)
    # (1) 수선의 발
    sol = sp.solve([sp.Eq(Y, sp.Rational(3, 4) * (X + 1)), sp.Eq((Y - (a + 1) / 3) * sp.Rational(3, 4), -(X - a))], [X, Y])
    md1 = sp.sympify(md_answer(3, "1"), locals={"a": a})
    ok1 = sp.simplify(sol[X] - md1[0]) == 0 and sp.simplify(sol[Y] - md1[1]) == 0
    # (3) 수치: 각 a 에서 닿는 원의 r 을 이분법으로 → md 의 구간별 식과 비교
    fmd = sp.lambdify(a, sp.sympify(md_answer(3, "3"), locals={"a": a}))
    errs = []
    for av in list(np.linspace(-0.99, 4, 120)) + [0.0, 0.25, 19 / 8, 0.1, 0.2]:
        errs.append(abs(contact_r(av) - float(fmd(av))))
    ok3 = max(errs) < 1e-4
    # (2) 접촉점 x좌표가 불연속으로 뛰는 a(두 부분에 동시에 닿는 a)를 찾아 이분법으로 정밀화
    th = np.linspace(0, 2 * np.pi, 20001)

    def cx(av):
        r = contact_r(av)
        xs, ys = av + r * np.cos(th), r + r * np.sin(th)
        return xs[np.argmin(boundary(xs) - ys)]

    grid = np.linspace(-3, 6, 451)
    cxs = [cx(v) for v in grid]
    jumps = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if abs(cxs[i + 1] - cxs[i]) > 0.3]
    twos = []
    for lo, hi in jumps:
        clo = cx(lo)
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if abs(cx(mid) - clo) < 0.3 else (lo, mid)
        twos.append((lo + hi) / 2)
    md2 = [float(sp.sympify(v)) for v in str(md_answer(3, "2")).strip("[]").split(",")]
    ok2 = len(twos) == len(md2) and all(abs(u - v) < 1e-6 for u, v in zip(sorted(twos), sorted(md2)))
    report(3, ok1 and ok2 and ok3, f"(1) 발={sol[X]},{sol[Y]} (2) 두 점 접촉 a≈{[round(v, 8) for v in twos]} (3) 최대오차 {max(errs):.2e}")


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
