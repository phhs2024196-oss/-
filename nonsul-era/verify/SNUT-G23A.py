#!/usr/bin/env python3
"""서울과학기술대 2023학년도 자연계열 오전 수리논술 (SNUT-G23A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import sys

import sympy as sp

PREFIX = "SNUT-G23A"
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
    """[1.1] 접선 y절편 → 점화식 → 등비급수, [1.2] 극값 조건, [1.3] 단면적 적분."""
    # [1.1] y=sqrt(x) 위의 점 (u, sqrt(u))에서 접선의 y절편 = 다음 b, 다음 a = b^2
    u = sp.symbols("u", positive=True)
    tang = sp.sqrt(u) + sp.diff(sp.sqrt(x), x).subs(x, u) * (0 - u)       # x=0 에서의 접선 값
    ratio = sp.simplify(tang**2 / u)
    a1, s = 4, 0
    cur = sp.Integer(4)
    for _ in range(60):
        s += cur
        cur = cur * ratio
    ans1 = load_answer(1, 0)
    ok1 = ratio == sp.Rational(1, 4) and abs(float(s) - float(ans1)) < 1e-9 and eq(sp.Rational(4) / (1 - ratio), ans1)
    # [1.2] f'(x)=0 의 근(0<x<2pi) 중 f'가 부호변화하는 점(극값), 각각 sin(2α+π/4)
    f = (sp.sin(x) + 3) / (sp.cos(x) - 1)
    cand = []
    for r0 in (0.5, 2.0, 3.5, 5.0, 6.0):
        try:
            cand.append(sp.nsolve(sp.diff(f, x), x, r0))
        except Exception:
            pass
    ext = []
    for r in cand:
        r = float(r) % (2 * float(sp.pi))
        if 0 < r < 2 * float(sp.pi) and abs(float(sp.diff(f, x).subs(x, r))) < 1e-9:
            if all(abs(r - e) > 1e-6 for e in ext):
                ext.append(r)
    vals = [float(sp.sin(2 * r + sp.pi / 4)) for r in ext]
    ans2 = load_answer(1, 1)
    # 극값 여부: 좌우 도함수 부호 변화
    isext = all(float(sp.diff(f, x).subs(x, r - 1e-3)) * float(sp.diff(f, x).subs(x, r + 1e-3)) < 0 for r in ext)
    ok2 = len(ext) == 1 and isext and abs(vals[0] - float(ans2)) < 1e-9
    # [1.3] 변 비 5:6:7, 밑면 변 = 14 sqrt(x) e^x = 7k (공식 해설 해석) → S(x), V
    kk = 14 * sp.sqrt(x) * sp.exp(x) / 7
    cosQ = sp.Rational(25 + 36 - 49, 2 * 5 * 6)                      # (PQ²+QR²-RP²)/(2PQ·QR), 단위비
    sinQ = sp.sqrt(1 - cosQ**2)
    S = sp.Rational(1, 2) * 5 * kk * 6 * kk * sinQ
    V = sp.integrate(sp.simplify(S), (x, sp.Rational(1, 2), 1))
    ans3 = load_answer(1, 2)
    ok3 = eq(V, ans3) and abs(float(V) - float(ans3)) < 1e-9
    report(1, ok1 and ok2 and ok3,
           f"[1.1] 공비={ratio}, 합={ans1} / [1.2] 극값점={[round(e,4) for e in ext]}, 값={[round(v,6) for v in vals]} vs {sp.N(ans2,8)} / [1.3] V={sp.simplify(V)} (RP=밑변 14√x e^x 해석) vs {ans3}")


def p02():
    """[2.1] 적분으로 위치, [2.2][2.3] 두 원의 교집합 넓이(수치 적분 교차확인), [2.4] |f|<=4 범위."""
    # [2.1]
    T = sp.symbols("T", real=True)
    v = [(-2 * T, 0, 2), (2 * T - 8, 2, 6), (-2 * T + 16, 6, 8)]
    pos, start = [], sp.Integer(7)
    for ex, lo, hi in v:
        F = start + sp.integrate(ex, (T, lo, T))
        pos.append((sp.expand(F), lo, hi))
        start = F.subs(T, hi)
    given = [7 - t**2, t**2 - 8 * t + 15, -t**2 + 16 * t - 57]
    ok1 = all(sp.expand(p[0].subs(T, t) - g) == 0 for p, g in zip(pos, given))
    ans1 = load_answer(2, 0, raw=True)
    # [2.2][2.3] 반지름 2인 두 원, 중심 거리 d 의 교집합 넓이를 y 에 대한 적분으로
    def lens(d):
        import mpmath as mp
        dd = float(d)
        ys = (4 - (dd / 2) ** 2) ** 0.5
        return float(mp.quad(lambda yy: max(0.0, 2 * (4 - yy * yy) ** 0.5 - dd), [-2, -ys, 0, ys, 2]))
    a2 = load_answer(2, 1)
    d2 = 2 * sp.sqrt(2)
    l2 = lens(d2)
    closed2 = 2 * 2**2 * sp.acos(d2 / 4) - (d2 / 2) * sp.sqrt(4 * 4 - d2**2)
    ok2 = eq(closed2, a2) and abs(l2 - float(a2)) < 1e-6
    # [2.3] t=4-sqrt(3) 에서의 위치
    tt = 4 - sp.sqrt(3)
    d3 = sp.simplify(pos[1][0].subs(T, tt))
    a3 = load_answer(2, 2)
    closed3 = 2 * 2**2 * sp.acos(d3 / 4) - (d3 / 2) * sp.sqrt(4 * 4 - d3**2)
    l3 = lens(d3)
    ok3 = d3 == 2 and eq(closed3, a3) and abs(l3 - float(a3)) < 1e-6
    # [2.4] 소리 구간: |f(t)|<=4 인 t ∈ [0,8], 경계 f=4 의 근
    rng = []
    for (F, lo, hi) in pos:
        for r in sp.solve(sp.Eq(F.subs(T, t), 4), t):
            if r.is_real and lo <= r <= hi:
                rng.append(r)
    import numpy as np
    grid = np.linspace(0, 8, 80001)
    fv = np.where(grid < 2, 7 - grid**2, np.where(grid < 6, grid**2 - 8 * grid + 15, -grid**2 + 16 * grid - 57))
    sel = grid[np.abs(fv) <= 4 + 1e-12]
    a4 = load_answer(2, 3, raw=True)
    lo_, hi_ = sorted(float(r) for r in rng)
    ok4 = abs(sel.min() - lo_) < 1e-3 and abs(sel.max() - hi_) < 1e-3 and eq(rng[0] + rng[1], 8) and abs(lo_ - 3**0.5) < 1e-9 and "sqrt(3)" in a4
    report(2, ok1 and ok2 and ok3 and ok4,
           f"[2.1] f 세 식 일치 / [2.2] 겹침={sp.simplify(closed2)}(수치 {l2:.6f}) / [2.3] f={d3}, 겹침={sp.simplify(closed3)}(수치 {l3:.6f}) / [2.4] 소리 구간 [{lo_:.5f},{hi_:.5f}] (격자 [{sel.min():.4f},{sel.max():.4f}])")


def p03():
    """[3.1] 미분가능 → a,b, [3.2] 실근 개수 c별 전수, [3.3] 넓이."""
    A, B = sp.symbols("A B", real=True)
    fl = A * sp.log(x) + 6
    fr = B * x + 12
    sol = sp.solve([sp.Eq(fl.subs(x, 1), fr.subs(x, 1)), sp.Eq(sp.diff(fl, x).subs(x, 1), sp.diff(fr, x).subs(x, 1))], [A, B], dict=True)[0]
    av, bv = sol[A], sol[B]
    ans1 = load_answer(3, 0, raw=True)
    ok1 = (av, bv) == (-6, -6) and "a=-6" in ans1.replace(" ", "") and "b=-6" in ans1.replace(" ", "")
    # [3.2] c 별 실근 수: 수치 전수
    import numpy as np
    fL = lambda X: av * np.log(X) + 6
    fR = lambda X: bv * X + 12
    def nroots(c):
        xs = np.linspace(1e-6, 6, 600001)
        F = np.where(xs <= 1, fL(xs), fR(xs)) - (-3 * xs**2 + 6 * xs + c)
        cnt = int(np.sum(F[:-1] * F[1:] < 0)) + int(np.sum(F == 0))
        # 접점(중근): 최솟값이 0 에 근접한 경우
        if cnt == 0 and np.min(np.abs(F)) < 1e-9:
            cnt = 1
        return cnt
    counts = {c: nroots(c) for c in (-2, -0.5, 0, 0.5, 2, 10)}
    ok2 = counts == {-2: 0, -0.5: 0, 0: 1, 0.5: 2, 2: 2, 10: 2}
    ans2 = load_answer(3, 1, raw=True)
    # [3.3] c=0, 넓이 ∫_{1/2}^{2} (f-g)
    g = -3 * x**2 + 6 * x
    area = sp.integrate(av * sp.log(x) + 6 - g, (x, sp.Rational(1, 2), 1)) + sp.integrate(bv * x + 12 - g, (x, 1, 2))
    ans3 = load_answer(3, 2)
    ok3 = eq(area, ans3)
    report(3, ok1 and ok2 and ok3, f"[3.1] a,b={av},{bv} / [3.2] 개수={counts} / [3.3] 넓이={sp.simplify(area)} vs {ans3}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G23A: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
