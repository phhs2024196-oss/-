#!/usr/bin/env python3
"""AJOU-G25B (아주대 2025학년도 논술 자연계열 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G25B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def same(a, b, tol=1e-12):
    return abs(complex(sp.N(a - b, 30))) < tol


def p01():
    oks = []
    # (1-1)(1): (F(x))^2 = (x-2j)^2 on [2j-1, 2j+1] 과 y=x-c 의 교점 수를 c=i/400 격자에서 정확히 셈
    X = sp.symbols("X", real=True)

    def count(cv):
        # (x-2j)^2 = x-c  ⇔  x^2-(4j+1)x+4j^2+c=0, 판별식 8j+1-4c (정확한 유리수 계산)
        pts = set()
        for j in range(-3, 4):
            D = 8 * j + 1 - 4 * cv
            if D < 0:
                continue
            for s in {((4 * j + 1) + sp.sqrt(D)) / 2, ((4 * j + 1) - sp.sqrt(D)) / 2}:
                if 2 * j - 1 <= s <= 2 * j + 1:
                    pts.add(sp.nsimplify(s))
        return len(pts)
    cs = [sp.Rational(i, 400) for i in range(0, 800) if count(sp.Rational(i, 400)) == 2]
    oks.append(sp.FiniteSet(*cs) == sp.sympify(md_answer(1, "1-1")))
    # (1-1)(2): F 와 y=bx (0<b<1/3) 의 r1, r2 를 구간별 적분으로
    b = sp.symbols("b", positive=True)
    al = sp.solve(sp.Eq(2 - X, b * X), X)[0]
    be = sp.solve(sp.Eq(X - 2, b * X), X)[0]
    r1 = sp.integrate((1 - b) * X, (X, 0, 1)) + sp.integrate(2 - X - b * X, (X, 1, al))
    r2 = sp.integrate(b * X - (2 - X), (X, al, 2)) + sp.integrate(b * X - (X - 2), (X, 2, be))
    found = [(m, sp.limit(b**m * r1 / r2, b, 0, "+")) for m in range(1, 6)]
    good = [(m, L) for m, L in found if L.is_finite and L != 0]
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "1-2").replace(" ", "").split(";"))
    oks.append(good == [(int(parts["m"]), sp.sympify(parts["limit"]))])
    # (1-1)(3): f=x(x-1/2)^2, g=p^2 x, r1=r2 를 넓이 그대로 세워 풀기
    p = sp.symbols("p", positive=True)
    f, g = X * (X - sp.Rational(1, 2))**2, p**2 * X
    roots = sorted([s for s in sp.solve(sp.Eq(f, g), X) if s != 0], key=lambda e: e.subs(p, sp.Rational(1, 10)))
    R1 = sp.integrate(f - g, (X, 0, roots[0])); R2 = sp.integrate(g - f, (X, roots[0], roots[1]))
    ps = [s for s in sp.solve(sp.Eq(R1, R2), p) if 0 < s < sp.Rational(1, 4)]
    oks.append(ps == [sp.sympify(md_answer(1, "1-3"))])
    # (1-2)(1): S = ∫_0^a (2h-h)
    S1 = sp.integrate(sp.log(2 * X + 1), (X, 0, sp.Rational(1, 2)))
    oks.append(sp.simplify(S1 - sp.sympify(md_answer(1, "2-1"))) == 0)
    # (1-2)(2): T 를 y 방향 적분으로 직접: ∫_0^{2a^3} ((y/2)^{1/3} - k y^{1/3}) dy, a=1 과 a=2 에서 확인
    k, y = sp.symbols("k y", positive=True)
    ks = []
    for a in (1, 2):
        S = sp.integrate(X**3, (X, 0, a))
        T = sp.integrate((y / 2)**sp.Rational(1, 3) - k * y**sp.Rational(1, 3), (y, 0, 2 * a**3))
        ks.append(sp.solve(sp.Eq(S, T), k)[0])
    kmd = sp.sympify(md_answer(1, "2-2"))
    oks.append(all(sp.simplify(kk - kmd) == 0 for kk in ks) and kmd < sp.cbrt(4) / 2)
    report(1, all(oks), f"c={cs} m,lim={good} p={ps} S={S1} k={ks[0]} {oks}")


def p02():
    oks = []
    th = sp.symbols("theta", positive=True)
    # (2-1): 중심 O, 반지름 10. 빗변 EF 는 기울기 -tanθ, 원에 접하고 원은 왼쪽. F=(b,-10), E=(c,20)
    b = (10 + 10 / sp.cos(th)) / sp.tan(th)          # 거리 조건 |tanθ·b-10|/secθ = 10
    c = b - 30 / sp.tan(th)
    L = (b + 10) + (c + 10) + 20 * sp.pi              # BF + AE(=GC, 점대칭) + FG
    oks.append(sp.simplify(L.subs(th, sp.pi / 4) - sp.sympify(md_answer(2, "1-1"))) == 0)
    import numpy as np
    Lf = sp.lambdify(th, L, "numpy")
    grid = np.linspace(1e-3, np.pi / 2, 200001)
    vals = Lf(grid)
    i = int(np.argmin(vals))
    crit = [s for s in sp.solve(sp.diff(L, th), th) if 0 < s <= sp.pi / 2]
    Lmin = sp.simplify(L.subs(th, crit[0]))
    oks.append(abs(vals[i] - float(Lmin)) < 1e-6 and sp.simplify(Lmin - sp.sympify(md_answer(2, "1-2"))) == 0
               and c.subs(th, crit[0]) >= -10)
    # (2-2): 정육각형 배치를 직접 만들어 셈 (m=n=50)
    r3 = sp.sqrt(3)
    hexes = []
    for j in range(1, 100):
        y0, y1 = (j - 1) * r3, (j - 1) * r3 + 4 * r3 / 3
        off = 0 if j % 2 == 1 else 1
        for i in range(50):
            hexes.append((j, 2 * i + off, 2 * i + off + 2, float(y0), float(y1)))

    def XM(kk, ll):
        inside = [h for h in hexes if h[4] <= ll + 1e-12 and h[2] <= kk]
        return len(inside), len({h[0] for h in inside})
    x21 = XM(10, 6)[0]
    oks.append(x21 == int(sp.sympify(md_answer(2, "2-1"))))
    ls = [ll for ll in range(2, 120, 2) if all(XM(kk, ll)[1] == ll // 2 + 1 for kk in (3, 4, 5, 50, 100))]  # k≥3 이면 층 판정은 k 와 무관(대표값 확인)
    oks.append(min(ls) == int(sp.sympify(md_answer(2, "2-2"))))
    l0 = min(ls)
    ks = [kk for kk in range(2, 101, 2) if 4 * XM(kk, l0)[0] > kk * l0]
    oks.append(min(ks) == int(sp.sympify(md_answer(2, "2-3"))))
    report(2, all(oks), f"L(π/4)={sp.nsimplify(L.subs(th, sp.pi/4))} θ*={crit} Lmin={Lmin} X(10,6)={x21} ℓ={l0} k={min(ks)} {oks}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
