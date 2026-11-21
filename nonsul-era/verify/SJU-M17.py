#!/usr/bin/env python3
"""SJU-M17 (세종대 2017학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "SJU-M17"
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


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, b, c, d, s = sp.symbols("a b c d s", real=True)
    line = sp.Matrix([0, 0, d]) + s * sp.Matrix([a, b, c - d])
    s0 = sp.solve(line[2], s)[0]
    Q = [sp.simplify(line[0].subs(s, s0)), sp.simplify(line[1].subs(s, s0))]
    loc = {"a": a, "b": b, "c": c, "d": d}
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "1").replace(" ", "").split(";"))
    ok1 = sp.simplify(Q[0] - sp.sympify(parts["x"], locals=loc)) == 0 and sp.simplify(Q[1] - sp.sympify(parts["y"], locals=loc)) == 0
    # (2) 반원 a=1+cos θ, c=sin θ (0≤θ≤π) 의 그림자 점이 md 곡선 식을 만족하는지 + y 범위
    X, Y = sp.symbols("x y", real=True)
    eqs = re.sub(r"\(y>=1\)", "", md_answer(1, "2").replace(" ", ""))
    lhs, rhs = eqs.split("=")
    F = sp.sympify(lhs, locals={"x": X, "y": Y}) - sp.sympify(rhs, locals={"x": X, "y": Y})
    import math
    ok2 = True; ys = []
    for i in range(0, 1001):
        th = math.pi * i / 1000
        av, cv = 1 + math.cos(th), math.sin(th)
        xv, yv = 3 * av / (3 - cv), 3 / (3 - cv)
        ys.append(yv)
        ok2 &= abs(float(F.subs({X: xv, Y: yv}))) < 1e-9
    ok2 &= min(ys) >= 1 - 1e-12 and "y>=1" in md_answer(1, "2").replace(" ", "")
    # (3) y 최대점 (격자 최대 + 정확: c=1)
    i = max(range(1001), key=lambda k: ys[k])
    th = math.pi * i / 1000
    R = (3 * (1 + math.cos(th)) / (3 - math.sin(th)), ys[i])
    Rm = sp.sympify(md_answer(1, "3"))
    ok3 = abs(R[0] - float(Rm[0])) < 1e-6 and abs(R[1] - float(Rm[1])) < 1e-6
    report(1, ok1 and ok2 and ok3, f"Q=({Q[0]},{Q[1]}) curve ok={ok2} Rmax~({R[0]:.6f},{R[1]:.6f})")


def p02():
    tt, u = sp.symbols("tt u", real=True)
    # f^2 = sin^2 가 조건식을 만족하는지 독립 확인 (f=sin, 그리고 구간별 부호는 f^2 불변)
    f2 = lambda z: sp.sin(z)**2
    lhs = sp.integrate((x - tt) * f2(tt), (tt, 0, x)) + sp.integrate(tt * f2(x - tt), (tt, 0, x))
    ok0 = sp.simplify(lhs - (x**2 / 2 - sp.sin(x)**2 / 2)) == 0
    # 역으로 조건식에서 f^2 를 유도: 두 번 미분
    G = (x**2 / 2 - sp.sin(x)**2 / 2) / 2
    ok0 &= sp.simplify(sp.diff(G, x, 2) - sp.sin(x)**2) == 0
    gpi = sp.integrate(tt * f2(sp.pi - tt), (tt, 0, sp.pi))
    ok1 = sp.simplify(gpi - sp.sympify(md_answer(2, "1"))) == 0
    # (2) 구간별 ±∫sin = ±2 의 100개 합: 가능한 값 집합을 DP 로
    piece = sp.integrate(sp.sin(x), (x, 0, sp.pi))
    S = {0}
    for _ in range(100):
        S = {v + piece for v in S} | {v - piece for v in S}
    nums = [int(v) for v in re.findall(r"-?\d+", md_answer(2, "2").split("}")[0])]
    md_set = set(range(nums[0], nums[-1] + 1, nums[1] - nums[0]))
    ok2 = S == md_set
    # (3) p=|sin|/200 : 적분 1, 평균
    k = 1 / sum(sp.integrate((-1)**n * sp.sin(x), (x, n * sp.pi, (n + 1) * sp.pi)) for n in range(100))  # ∫|sin| 구간별
    EX = sum(sp.integrate(x * (-1)**n * sp.sin(x), (x, n * sp.pi, (n + 1) * sp.pi)) for n in range(100)) * k
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "3").replace(" ", "").split(";"))
    pm = sp.sympify(parts["p(x)"], locals={"x": x})
    ok3 = all(abs(float(pm.subs(x, v) - k * abs(sp.sin(v)))) < 1e-14 for v in [0.3, 2.0, 4.5, 100.0]) and \
        sp.simplify(EX - sp.sympify(parts["E(X)"])) == 0
    report(2, ok0 and ok1 and ok2 and ok3, f"g(pi)={gpi} |set|={len(S)} min={min(S)} max={max(S)} k={k} E={sp.simplify(EX)}")


def p03():
    a, c, X = sp.symbols("a c X", real=True)
    y = sp.exp(a * x)
    tl = sp.diff(y, x).subs(x, c) * (X - c) + y.subs(x, c)
    md = md_answer(3, "1").replace(" ", "").split("=", 1)[1]
    ok1 = sp.simplify(tl - sp.sympify(md, locals={"x": X, "a": a, "c": c})) == 0
    al, ap = sp.symbols("alpha ap", positive=True)
    import mpmath as mp
    mp.mp.dps = 40
    ns = mp.findroot([lambda A_, P_: A_**2 - mp.e**(P_ * A_), lambda A_, P_: 2 * A_ - P_ * mp.e**(P_ * A_)], (2.5, 0.8))
    sol = [{al: ns[0], ap: ns[1]}]
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "2").replace(" ", "").split(";"))
    Qm = sp.sympify(parts["Q"])
    ok2 = any(abs(s[ap] - sp.sympify(parts["a"]).evalf(40)) < 1e-25 and abs(s[al] - Qm[0].evalf(40)) < 1e-25
              and abs(s[al]**2 - Qm[1].evalf(40)) < 1e-25 for s in sol)
    import mpmath as mp
    mp.mp.dps = 40
    import numpy as np
    def corners(aval, k):
        """e^{a x}-x^k 의 부호가 바뀌는 근(미분불가능점) 위치 목록: 부동소수 격자 + 이분법 정밀화"""
        u = lambda z: mp.e**(aval * z) - mp.mpf(z)**k
        grid = np.linspace(-60, 60, 240001)
        vals = np.exp(float(aval) * grid) - grid**k
        idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        pts = []
        for i in idx:
            lo, hi = mp.mpf(grid[i]), mp.mpf(grid[i + 1])
            for _ in range(150):
                mid = (lo + hi) / 2
                if u(lo) * u(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            r = (lo + hi) / 2
            if abs(mp.diff(u, r)) > 1e-12:
                pts.append(r)
        return pts
    n3 = len(corners(mp.mpf(1) / 3, 2))
    ok3 = n3 == int(md_answer(3, "3"))
    aval = mp.mpf(11) / (2 * mp.e)
    allpts, cum, nfound = [], 0, None
    for k in range(1, 16):
        ps = corners(aval, k)
        allpts += ps
        cum += len(ps)
        if cum == 17 and nfound is None:
            nfound = k
    distinct = len(allpts) == len(set(round(float(p), 9) for p in allpts))
    ok4 = nfound == int(md_answer(3, "4")) and distinct
    report(3, ok1 and ok2 and ok3 and ok4, f"a,Q={sol} (3-3)={n3} (3-4) n={nfound} distinct={distinct}")


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
