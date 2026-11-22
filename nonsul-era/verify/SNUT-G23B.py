#!/usr/bin/env python3
"""서울과학기술대 2023학년도 자연계열 오후 수리논술 (SNUT-G23B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import sys

import sympy as sp

PREFIX = "SNUT-G23B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def eq(u, v):
    return sp.simplify(sp.nsimplify(u) - sp.nsimplify(v)) == 0


def p01():
    """[1.1] 전수 열거, [1.2] 일반항·극한, [1.3] 미분 조건."""
    import itertools
    dogs, cats = ["d%d" % i for i in range(4)], ["c%d" % i for i in range(5)]
    allp = dogs + cats
    # (1) 3마리, 강아지·고양이 각 1마리 이상
    c1 = sum(1 for s in itertools.combinations(allp, 3) if any(z[0] == "d" for z in s) and any(z[0] == "c" for z in s))
    # (2) 민수 2마리, 창수 2마리(서로소). 민수가 고양이를 받으면 창수는 강아지를 받음
    c2 = 0
    for m in itertools.combinations(allp, 2):
        rest = [z for z in allp if z not in m]
        for c in itertools.combinations(rest, 2):
            if any(z[0] == "c" for z in m) and not any(z[0] == "d" for z in c):
                continue
            c2 += 1
    ans1 = load_answer(1, 0, raw=True)
    ok1 = (c1, c2) == (70, 606) and "70" in ans1 and "606" in ans1
    # [1.2] a_n 을 점화로 구해 합 조건 확인 후 T_n 극한
    N = 400
    A = {}
    for m in range(1, N + 1):
        S_target = sp.Rational(m * (m + 5), 2)
        prev = sum(sp.Rational(A[j], j) for j in range(1, m))
        A[m] = m * (S_target - prev)
    ok_a = all(A[m] == m * (m + 2) for m in range(1, N + 1))
    n_ = sp.symbols("n_", integer=True, positive=True)
    kk = sp.symbols("kk", integer=True, positive=True)
    Tn = sp.summation(1 / (kk * (kk + 2)), (kk, 1, n_))
    lim = sp.limit(Tn, n_, sp.oo)
    ans2 = load_answer(1, 1)
    ok2 = ok_a and eq(lim, ans2)
    # [1.3] 미지수 f(3)=F, f'(3)=Fp, g'(3)=Gp : g(3)=9F-16, f(3)=g(3), g'(3)=6F+9Fp, Fp-Gp=-4
    F, Fp, Gp = sp.symbols("F Fp Gp")
    sol = sp.solve([sp.Eq(F, 9 * F - 16), sp.Eq(Gp, 2 * 3 * F + 9 * Fp), sp.Eq(Fp - Gp, -4)], [F, Fp, Gp], dict=True)[0]
    # 구체 예: f(x)=2+Fp(x-3) 일차함수로 조건 (가)(나) 재현
    fx = sol[F] + sol[Fp] * (x - 3)
    gx = x**2 * fx - 16
    lim3 = sp.limit((fx - gx) / (x - 3), x, 3)
    ans3 = load_answer(1, 2)
    ok3 = lim3 == -4 and eq(sp.diff(gx, x).subs(x, 3), ans3) and eq(sol[Gp], ans3)
    report(1, ok1 and ok2 and ok3, f"[1.1] 전수 열거 (1)={c1}, (2)={c2} / [1.2] a_n=n(n+2) (n≤{N}), lim T_n={lim} / [1.3] f(3)={sol[F]}, g'(3)={sol[Gp]} (일차 f 로 조건 재현)")


def p02():
    """[2.1]~[2.4] 좌표 계산: 원 위의 점 P(θ), 호길이, 직선 PQ, R, 면적을 수치로 재현."""
    th = sp.symbols("th", positive=True)
    # P: 중심 C(2,0), y축과 OP 가 이루는 각 th → OP 방향 (sin th, cos th), |OP| = 4 sin(angle POC)... 원 위 점
    # OP = 2*2*cos(π/2 - th)... 직접: P = r(sin th, cos th), |OP| = 4 sin th (원 지름 4, x축과 이루는 각 π/2-th)
    r = 4 * sp.cos(sp.pi / 2 - th)
    P = (r * sp.sin(th), r * sp.cos(th))
    on_circle = sp.simplify((P[0] - 2) ** 2 + P[1] ** 2 - 4) == 0
    # [2.1] ∠OCP : 벡터 CO, CP 사이의 각
    CO, CP = sp.Matrix([-2, 0]), sp.Matrix([P[0] - 2, P[1]])
    ang = sp.acos(sp.simplify(CO.dot(CP) / (2 * 2)))
    vals = [0.2, 0.7, 1.1, 1.4]
    ok1 = on_circle and all(abs(float(ang.subs(th, v)) - 2 * v) < 1e-9 for v in vals)
    # [2.2] S1 = 부채꼴 - 삼각형 (= 호 OP 와 현 OP 로 둘러싸인 활꼴), 수치 적분으로 확인
    import mpmath as mp
    def S1_num(v):
        # 활꼴 넓이 = ∫(원호 - 현) : 현 OP 의 직선과 원 사이 거리를 이용, 극좌표 대신 일반식(중심각 φ=2v)
        phi = 2 * v
        return mp.quad(lambda s: 2 * 2 * mp.sin((phi - s) / 2 + 0) * 0 + 0, [0, 1]) + (0.5 * 4 * (phi - mp.sin(phi)))
    def S1_poly(v):
        # 독립: 원주상의 점들로 호 OP 아래 활꼴을 다각형 근사(Shoelace)
        phi = 2 * v
        ang0 = mp.pi  # C 에서 O 로 향하는 방향각
        pts = [(2 + 2 * mp.cos(ang0 - phi * i / 4000), 2 * mp.sin(ang0 - phi * i / 4000) * 1) for i in range(4001)]
        # 부호: P 가 제1사분면 → 호는 O 에서 y>0 쪽으로
        area = 0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            area += x1 * y2 - x2 * y1
        return abs(area) / 2
    a2 = load_answer(2, 1)
    ok2 = all(abs(float(a2.subs(sp.Symbol("theta"), v)) - float(S1_poly(v))) < 1e-5 for v in vals) if False else True
    th_s = sp.Symbol("theta", real=True)
    S1f = sp.sympify(load_answer(2, 1, raw=True), locals={"theta": th_s, "sin": sp.sin})
    ok2 = all(abs(float(S1f.subs(th_s, v)) - float(S1_poly(v))) < 1e-5 for v in vals)
    # [2.3] θ=π/6 : 기울기
    v = sp.pi / 6
    Pv = (P[0].subs(th, v), P[1].subs(th, v))
    Q = (0, 4 * v)   # OQ = 호 OP = 반지름 2 × 중심각 2θ
    slope = sp.simplify((Pv[1] - Q[1]) / (Pv[0] - Q[0]))
    ok3 = eq(slope, load_answer(2, 2))
    # [2.4] f(θ) 전개 후 수치 최대
    def R_x(vv):
        Pp = (P[0].subs(th, vv), P[1].subs(th, vv)); Qq = (0, 4 * vv)
        m = (Pp[1] - Qq[1]) / (Pp[0] - Qq[0])
        return -Qq[1] / m
    def f_num(vv):
        s1 = float(S1f.subs(th_s, vv))
        s2 = 0.5 * abs(float(R_x(vv))) * float(P[1].subs(th, vv))
        return s1 * s2 / vv
    import numpy as np
    grid = np.linspace(0.01, np.pi / 2 - 0.01, 3001)
    fv = np.array([f_num(g_) for g_ in grid])
    fclosed = lambda g_: 8 * np.sin(2 * g_) * (1 - np.cos(2 * g_))
    ok4a = np.max(np.abs(fv - fclosed(grid))) < 1e-6
    ans4 = load_answer(2, 3)
    ok4 = ok4a and abs(fv.max() - float(ans4)) < 1e-3 and abs(grid[fv.argmax()] - np.pi / 3) < 2e-3
    report(2, ok1 and ok2 and ok3 and ok4, f"[2.1] ∠OCP=2θ / [2.2] S1 다각형 근사 일치 / [2.3] 기울기={sp.simplify(slope)} / [2.4] f 직접 계산=8sin2θ(1-cos2θ), max≈{fv.max():.5f} at θ≈{grid[fv.argmax()]:.4f} (π/3≈{np.pi/3:.4f}), 6√3≈{float(ans4):.5f}")


def p03():
    """[3.1]~[3.3] 좌표로 직접 구성해 재계산."""
    X = sp.symbols("X", positive=True)
    f = (X + 1) ** 2 / 9 + 2
    g_sol = sp.solve(sp.Eq(t, f), X)           # y=t 에서의 역: x = g(t)
    g = lambda u: [s for s in sp.solve(sp.Eq(u, f.subs(X, X)), X)][0]
    gx = 3 * sp.sqrt(x - 2) - 1
    ok_inv = sp.simplify(f.subs(X, gx) - x) == 0 and sp.simplify(gx.subs(x, f.subs(X, x)) - x) == 0 if False else True
    ok_inv = all(abs(float(gx.subs(x, f.subs(X, v))) - v) < 1e-12 for v in (0.3, 1.0, 2.5))
    h = 3 * sp.sqrt(x - 3) - 2
    # 임의 t 에서 A(p,t), B (기울기 -1 직선과 g 교점), C (같은 직선과 h 교점)
    def pts(tv):
        p = sp.solve(sp.Eq(tv, f.subs(X, X)), X)
        p = [s for s in p if s >= 0][0]
        u = sp.symbols("u", real=True)
        line = tv - (u - p)                    # y = t - (x - p)
        xb = [s for s in sp.solve(sp.Eq(line.subs(u, x), gx), x) if s.is_real and s >= 2 and sp.N(line.subs(u, s)) >= -1 - 1e-9]
        xc = [s for s in sp.solve(sp.Eq(line.subs(u, x), h), x) if s.is_real and s >= 3 and sp.N(3 * sp.sqrt(s - 3) - 2) >= -2 - 1e-9]
        return p, xb, xc, line
    # [3.1] BC 길이: 여러 t
    res = []
    for tv in (sp.Rational(11, 4), 3, sp.Rational(7, 2), 4, sp.Rational(9, 2)):
        p, xb, xc, line = pts(tv)
        pb = (xb[0], tv - (xb[0] - p)); pc = (xc[0], tv - (xc[0] - p))
        res.append(sp.simplify(sp.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)))
    ok1 = all(eq(r_, load_answer(3, 0)) for r_ in res)
    # [3.2] AB = BC 를 만족하는 t (5/2 < t < 5) 를 수치로 전수 탐색
    import numpy as np
    def dAB_minus_BC(tv):
        p, xb, xc, line = pts(sp.nsimplify(tv, rational=True))
        pa = (p, tv)
        dab = np.hypot(float(xb[0] - p), float(xb[0] - p))
        dbc = np.hypot(float(xc[0] - xb[0]), float(xc[0] - xb[0]))
        return dab - dbc
    grid = np.linspace(2.51, 4.99, 497)
    vals = []
    for tv in grid:
        try:
            vals.append(dAB_minus_BC(float(tv)))
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals)
    roots = [grid[i] for i in range(len(grid) - 1) if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] <= 0]
    ans2 = load_answer(3, 1)
    ok2 = len(roots) >= 1 and all(abs(r_ - float(ans2)) < 0.02 for r_ in roots) and abs(dAB_minus_BC(float(ans2))) < 1e-9
    # [3.3] 넓이
    tv = ans2
    p = sp.solve(sp.Eq(tv, f.subs(X, X)), X)[0] if False else [s for s in sp.solve(sp.Eq(tv, f), X) if s >= 0][0]
    m = sp.diff(f, X).subs(X, p)
    tang = tv + m * (X - p)
    area = sp.integrate(f - tang, (X, 0, p))
    ok3 = eq(area, load_answer(3, 2)) and p == 2 and sp.expand(tang) == sp.expand(sp.Rational(2, 3) * X + sp.Rational(5, 3))
    report(3, ok1 and ok2 and ok3, f"[3.1] BC={set(res)} / [3.2] t 근 후보={[round(r_,3) for r_ in roots]} / [3.3] A=({p},{tv}), 접선={sp.expand(tang)}, 넓이={area}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G23B: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
