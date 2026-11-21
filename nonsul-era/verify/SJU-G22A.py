#!/usr/bin/env python3
"""SJU-G22A (세종대 2022학년도 수시 논술고사 A) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-G22A"
WIDTH = 2
RESULTS = []
mp.mp.dps = 30


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.strip().split(" = ", 1) for seg in s.split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    x, t = sp.symbols("x t", positive=True)
    f = t * sp.exp(-x**2)
    # (1) f'' 부호 변화 점
    f2 = sp.lambdify((x, t), sp.diff(f, x, 2))
    xs = np.linspace(1e-3, 4, 40001)
    q = [xs[i] for i in range(len(xs) - 1) if f2(xs[i], 1.7) * f2(xs[i + 1], 1.7) < 0]
    ok1 = len(q) == 1 and abs(q[0] - float(sp.sympify(md_answer(1, "1")))) < 1e-4
    # (2) 여러 t 에서 원점까지 최소 거리를 직접 수치로
    dist = sp.sympify(md_answer(1, "2"), locals={"t": t, "log": sp.log})
    ok2 = True
    for tv in (0.75, 1.0, 2.0, 5.0):
        d = lambda X: mp.sqrt(X**2 + tv**2 * mp.e**(-2 * X**2))
        g = [mp.mpf(i) / 1000 for i in range(1, 5000)]
        x0 = min(g, key=d)
        xm = mp.findroot(lambda X: mp.diff(d, X), x0)
        ok2 &= abs(d(xm) - mp.mpf(sp.N(dist.subs(t, tv), 30))) < 1e-15
    # (3) t 를 움직이며: P(α) 에서 OP 에 수직인 직선의 x 절편 r 을 직접 계산해 최소화
    def P_of(tv):
        d2 = lambda X: X**2 + tv**2 * mp.e**(-2 * X**2)
        g = [mp.mpf(i) / 200 for i in range(1, 1000)]
        return mp.findroot(lambda X: mp.diff(d2, X), min(g, key=d2))
    def r_of(tv):
        a = P_of(tv); b = tv * mp.e**(-a**2)
        return a + b * b / a                 # S: (r,0), (S-P)·(P-O)=0 ⇒ r·a = a²+b²
    ts = [0.72 + 0.01 * i for i in range(300)]
    t0 = min(ts, key=r_of)
    tbest = mp.findroot(lambda T: mp.diff(r_of, T), t0)
    ok3 = abs(P_of(tbest) - mp.mpf(sp.N(sp.sympify(md_answer(1, "3")), 30))) < 1e-8
    report(1, ok1 and ok2 and ok3, f"Q_x~{q[0]:.5f}; dist ok; argmin_t r ~ {float(tbest):.8f} (sqrt(e/2)={float(mp.sqrt(mp.e/2)):.8f}), P_x={float(P_of(tbest)):.10f}")


def p02():
    # [증명-수치확인] (2-1)(2-2): 조건을 만족하는 예 f=x^2, g=2x 및 볼록 함수들에서 부등식·f'=g 확인,
    # 조건 불만족 예(g 가 f' 와 다름)에서 부등식이 깨짐을 확인
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, 2000); Y = rng.uniform(-5, 5, 2000)
    cases = [(lambda u: u**2, lambda u: 2 * u), (lambda u: np.exp(u), lambda u: np.exp(u)), (lambda u: 0 * u + 2, lambda u: 0 * u)]
    ok1 = all(np.all(f(X) - f(Y) <= (X - Y) * g(X) + 1e-9) and np.all((X - Y) * g(Y) <= f(X) - f(Y) + 1e-9) for f, g in cases)
    bad = lambda u: 2 * u + 0.1                     # g ≠ f'
    ok1 &= not np.all((lambda u: u**2)(X) - (lambda u: u**2)(Y) <= (X - Y) * bad(X) + 1e-12)
    # (2-3) 8f(x)+f(-2x)=18: 다항식 해 공간을 기호로 → 상수 2 뿐 (볼록·미분가능 조건과 무관한 독립 확인)
    x = sp.symbols("x")
    cs = sp.symbols("c0:6")
    fpoly = sum(c * x**i for i, c in enumerate(cs))
    sol = sp.solve(sp.Poly(sp.expand(8 * fpoly + fpoly.subs(x, -2 * x) - 18), x).coeffs(), cs, dict=True)
    fsol = fpoly.subs(sol[0])
    # 단조 f' 논증의 수치 확인: f' 가 비감소이고 4φ(x)=φ(-2x) 이면 φ≡0 — 반례 탐색(무작위 비감소 계단함수)
    # 다항식 해 공간 {c3*x^3+2} 중 (2-2)에서 얻은 "f'=g 는 비감소" 를 만족하는 것만 남김
    free = [c for c in cs if fsol.has(c)]
    fp = sp.diff(fsol, x)
    keep = []
    for cv in np.linspace(-3, 3, 61):
        fpv = sp.lambdify(x, fp.subs({c: cv for c in free}))
        xs_ = np.linspace(-5, 5, 1001)
        if np.all(np.diff(np.broadcast_to(fpv(xs_), xs_.shape)) >= -1e-12):
            keep.append(sp.nsimplify(fsol.subs({c: cv for c in free})))
    ok3 = len(set(keep)) == 1 and sp.simplify(keep[0] - sp.sympify(md_answer(2, "3"))) == 0
    for _ in range(200):
        knots = np.sort(rng.uniform(-3, 3, 6)); vals = np.sort(rng.normal(size=7))
        phi = lambda u: vals[np.searchsorted(knots, u)]
        xs = np.linspace(-1.5, 1.5, 301)
        if np.allclose(4 * phi(xs), phi(-2 * xs)) and np.any(np.abs(phi(xs)) > 1e-12):
            ok3 = False
    report(2, ok1 and ok3, f"[증명-수치확인] 부등식·f'=g 예시 확인, 다항식 해 {fsol} 중 f\' 비감소 → {set(keep)}")


def p03():
    x = sp.symbols("x", real=True)
    a, b, c = sp.symbols("a b c", real=True)
    f = x**3 + a * x**2 + b * x + c
    g = f * sp.exp(x)
    # (1)(2): (나) g(3)=0, g'(3)=0 + (가) 를 수치로 확인하며 계수 결정
    s = sp.solve([g.subs(x, 3), sp.diff(g, x).subs(x, 3)], [b, c], dict=True)[0]
    fa = f.subs(s)                                    # a 한 개 남음
    # (가): G(x)=∫_1^x g ≥ 0 ⇒ G'(1)=g(1)=0 이 필요 → a 결정 후 G≥0 수치 확인
    aval = sp.solve(fa.subs(x, 1), a)
    cand = []
    for av in aval:
        fv = sp.expand(fa.subs(a, av)); gv = fv * sp.exp(x)
        G = sp.lambdify(x, sp.integrate(gv, (x, 1, x)))
        if all(G(v) >= -1e-9 for v in np.linspace(-30, 8, 3801)):
            # 극솟값 확인
            gl = sp.lambdify(x, gv)
            if gl(2.99) > 0 and gl(3.01) > 0:
                cand.append(fv)
    ok1 = len(cand) == 1 and cand[0].subs(x, 1) == 0
    gv = cand[0] * sp.exp(x)
    ans = parts(md_answer(3, "2"))
    ok2 = sp.simplify(gv - sp.sympify(ans["g"], locals={"x": x})) == 0
    d1 = sp.lambdify(x, sp.diff(gv, x)); d2 = sp.lambdify(x, sp.diff(gv, x, 2))
    xs = np.linspace(-12.0001, 8, 400001)
    s1, s2 = np.sign(d1(xs)), np.sign(d2(xs))
    mx = [xs[i] for i in range(len(xs) - 1) if s1[i] > 0 > s1[i + 1]]
    mn = [xs[i] for i in range(len(xs) - 1) if s1[i] < 0 < s1[i + 1]]
    ip = [xs[i] for i in range(len(xs) - 1) if s2[i] * s2[i + 1] < 0]
    near = lambda A, B: len(A) == len(B) and all(abs(u - float(v)) < 1e-3 for u, v in zip(sorted(A), sorted(B)))
    ok2 &= near(mx, sp.sympify(ans["max"])) and near(mn, sp.sympify(ans["min"])) and near(ip, sp.sympify(ans["inflection"]))
    # (3) h(t): F(x)=g(x+1)-g(t)-g'(t)(x-t), x≠t 의 서로 다른 근의 개수
    gm = sp.lambdify(x, gv, "mpmath"); gpm = sp.lambdify(x, sp.diff(gv, x), "mpmath")
    gn = sp.lambdify(x, gv); gpn = sp.lambdify(x, sp.diff(gv, x))
    X = np.linspace(-60, 6, 660001)
    def h(tv):
        F = gn(X + 1) - gn(tv) - gpn(tv) * (X - tv)
        roots = []
        sg = np.sign(F)
        for i in np.where(sg[:-1] * sg[1:] < 0)[0]:
            roots.append(X[i])
        # 접하는 근(부호 불변 극값 ≈0) 탐지
        Fa = np.abs(F)
        loc = np.where((Fa[1:-1] < Fa[:-2]) & (Fa[1:-1] < Fa[2:]) & (Fa[1:-1] < 1e-6))[0] + 1
        for i in loc:
            if all(abs(X[i] - r) > 1e-3 for r in roots):
                roots.append(X[i])
        return len([r for r in roots if abs(r - tv) > 1e-6])
    # 공통접선 α: 접선이 y=g(x+1) 에도 접함 → 미지수 (t, u): g'(t)=g'(u+1), g(u+1)=g(t)+g'(t)(u-t)
    alpha, u0 = mp.findroot([lambda T, U: gpm(T) - gpm(U + 1),
                             lambda T, U: gm(U + 1) - gm(T) - gpm(T) * (U - T)], (2.8, 1.5))
    alpha = float(alpha)
    hs = {tv: h(tv) for tv in np.linspace(2.0005, 2.9995, 400)}
    below = [v for tv, v in hs.items() if tv < alpha - 2e-3]
    above = [v for tv, v in hs.items() if tv > alpha + 2e-3]
    special = [h(2.0), h(alpha), h(3.0)]
    cnt = special.count(2) + sum(1 for v in below + above if v == 2)   # 격자(α 근방 제외)에서 2 가 나오면 개수 증가
    ok3 = max(below) <= 1 and set(above) == {3} and special == [2, 2, 2] and cnt == int(md_answer(3, "3"))
    report(3, ok1 and ok2 and ok3, f"f={cand[0]}; max={[round(v, 3) for v in mx]} min={[round(v, 3) for v in mn]} infl={[round(v, 3) for v in ip]}; alpha={alpha:.6f}, h<α∈{sorted(set(below))}, h>α∈{sorted(set(above))}, h(2,α,3)={special}")


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
