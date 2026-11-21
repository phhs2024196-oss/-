#!/usr/bin/env python3
"""KKU-G23A (건국대 2023학년도 KU논술우수자 자연계 A 문제 1~5) — sympy 검산.
공식 해설 미확보 세트: 자체 풀이 답을 문제 설정으로부터 독립 재계산해 비교한다.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지. p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "KKU-G23A"
WIDTH = 2
RESULTS = []
LOC = {"e": sp.E, "E": sp.E}


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """두 원의 교점을 solve 로 구하고, 각 교점 A 에 대해 방향각 격자 + 기호 최대로 XY 최댓값."""
    import mpmath as mp
    x, y, s, th = sp.symbols("x y s theta", real=True)
    c1 = x**2 + 6*x + y**2 - 8*y
    c2 = x**2 - 2*x + y**2
    pts = sp.solve([c1, c2], [x, y], dict=True)
    best = {}
    for P in pts:
        ax, ay = P[x], P[y]
        line = {x: ax + s*sp.cos(th), y: ay + s*sp.sin(th)}
        sx = [r for r in sp.solve(sp.expand(c1.subs(line)), s) if r != 0][0]
        sy = [r for r in sp.solve(sp.expand(c2.subs(line)), s) if r != 0][0]
        L = sp.simplify(sx - sy)
        crit = sp.solve(sp.diff(L, th), th)
        cand = [sp.simplify(sp.Abs(L.subs(th, c))) for c in crit] + [sp.Abs(L.subs(th, 0))]
        Lmax = max(cand, key=lambda v: float(v))
        # 격자 확인 + X, Y ≠ A
        fL = sp.lambdify(th, L, "mpmath")
        gmax = max(abs(fL(mp.pi * i / 20000)) for i in range(20000))
        best[(ax, ay)] = (sp.nsimplify(Lmax), float(gmax))
    ans = sp.sympify(md_answer(1, "1"))
    ok = all(sp.simplify(v[0] - ans) == 0 and abs(v[1] - float(ans)) < 1e-6 for v in best.values())
    report(1, ok, f"교점별 최대 XY={ {k: v[0] for k, v in best.items()} } (격자 {[round(v[1], 6) for v in best.values()]})")


def p02():
    """렌즈 넓이를 원의 내부 공통부분 적분(세로 단면)으로 직접 구한 뒤 x 에 대해 수치 적분."""
    import mpmath as mp
    mp.mp.dps = 30

    def A(d):
        # 중심 (0,0),(d,0) 반지름 1: 공통부분 = 2 * ∫_{d/2}^{1} 2√(1-u²) du
        if d >= 2:
            return mp.mpf(0)
        return 4 * mp.quad(lambda u: mp.sqrt(1 - u**2), [d / 2, 1])

    val = mp.quad(A, [0, 2])
    u = sp.symbols("u", real=True)
    d = sp.symbols("d", positive=True)
    Asym = 4 * sp.integrate(sp.sqrt(1 - u**2), (u, d / 2, 1))
    sym = sp.simplify(sp.integrate(sp.diff(Asym, d) * (-d), (d, 0, 2)))   # 부분적분: ∫A = -∫ d A'(d)
    ans = sp.sympify(md_answer(2, "1"))
    ok = abs(val - mp.mpf(str(sp.N(ans, 30)))) < mp.mpf("1e-20") and sp.simplify(sym - ans) == 0
    report(2, ok, f"∫A dx 수치={mp.nstr(val, 20)} 기호(부분적분)={sym}")


def p03():
    """sin∠AXB = |XA×XB|/(|XA||XB|) 를 a 의 함수로 두고 임계점(기호)·격자 최댓값 비교."""
    a = sp.symbols("a", real=True)
    X = (a, sp.sqrt(2) / 4 * a**2)
    u = (2 - X[0], -X[1]); v = (4 - X[0], -X[1])
    S2 = (u[0]*v[1] - u[1]*v[0])**2 / ((u[0]**2 + u[1]**2) * (v[0]**2 + v[1]**2))   # sin² (a≠0)
    num = sp.factor(sp.numer(sp.together(sp.diff(S2, a))))
    crit = [r for r in sp.real_roots(sp.Poly(num, a)) if r != 0]
    vals = [sp.sqrt(sp.nsimplify(sp.simplify(S2.subs(a, r)))) for r in crit]
    smax = max(vals, key=lambda t: float(t))
    f = sp.lambdify(a, S2)
    grid = max(f(i / 1000) for i in range(-30000, 30001) if i != 0) ** 0.5
    ans = sp.sympify(md_answer(3, "1"))
    ok = sp.simplify(smax - ans) == 0 and abs(grid - float(ans)) < 1e-6
    report(3, ok, f"임계점 a={[sp.N(r, 8) for r in crit]} max sin={sp.radsimp(smax)} 격자={grid:.9f}")


def p04():
    """S=(4cos t,4sin t), P=(0,1): 반직선 PS 의 방향각 θ(t)=atan2 를 수치미분. θ=0, π/2 인 t 에서 dθ/dt."""
    import mpmath as mp
    mp.mp.dps = 30
    theta = lambda t: mp.atan2(4 * mp.sin(t) - 1, 4 * mp.cos(t))       # 그림의 각(반직선 기준)
    t0 = mp.findroot(lambda t: theta(t), 0.25)                          # θ=0
    t1 = mp.findroot(lambda t: theta(t) - mp.pi / 2, 1.5)               # θ=π/2
    d0, d1 = mp.diff(theta, t0), mp.diff(theta, t1)
    # 기울기각 해석(직선, mod π)일 때 추가로 생기는 t=3π/2 의 값(보고용)
    d_low = mp.diff(lambda t: mp.acot((4 * mp.cos(t)) / (4 * mp.sin(t) - 1)), 3 * mp.pi / 2)
    ans = sp.sympify(md_answer(4, "1"))
    ok = abs(d0 - mp.mpf(str(sp.N(ans[0], 30)))) < 1e-20 and abs(d1 - mp.mpf(str(sp.N(ans[1], 30)))) < 1e-20 \
        and abs(mp.sin(t0) - mp.mpf(1) / 4) < 1e-25 and abs(t1 - mp.pi / 2) < 1e-25
    report(4, ok, f"θ=0: t={mp.nstr(t0, 10)} dθ/dt={mp.nstr(d0, 15)}; θ=π/2: t={mp.nstr(t1, 10)} "
                  f"dθ/dt={mp.nstr(d1, 15)} [기울기각 해석 t=3π/2: {mp.nstr(d_low, 15)}]")


def p05():
    """f, g 를 정의대로 계산해 1..6200 전수 조사."""
    def f(n):
        s, p = 0, 10
        while p <= n:
            s += n // p
            p *= 10
        return s
    g = lambda n: f(10 * n) - 10 * f(n)
    cnt = sum(1 for n in range(1, 6201) if g(n) == 8)
    digit_ok = all(g(n) == sum(map(int, str(n))) for n in range(1, 6201))
    ok = cnt == int(sp.sympify(md_answer(5, "1"))) and f(123) == 13
    report(5, ok, f"count={cnt} (f(123)={f(123)}, g=자릿수합 전수 확인={digit_ok}, 중복조합 C(11,3)-C(5,3)+6={sp.binomial(11,3)-sp.binomial(5,3)+6})")


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
