#!/usr/bin/env python3
"""HIU-G21A (홍익대 2021학년도 수시 논술 자연계열 서울 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G21A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n")


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    """'키=값; 키=값' 형식의 답을 사전으로."""
    out = {}
    for seg in s.split(";"):
        k, _, v = seg.partition("=")
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def walk(N: int, start: int):
    """시작 좌표 start 에서 N 회 시행: (성공, 보류, 실패) 확률을 정확히 계산(상태 전이)."""
    half = sp.Rational(1, 2)
    dist = {start: sp.Integer(1)}
    S = F = sp.Integer(0)
    for _ in range(N):
        nd = {}
        for pos, p in dist.items():
            for q in (pos + 1, pos - 1):
                if q == 3:
                    S += p * half
                elif q == 0:
                    F += p * half
                else:
                    nd[q] = nd.get(q, 0) + p * half
        dist = nd
    return S, sum(dist.values(), sp.Integer(0)), F


def p01():
    Ns = range(1, 16)
    tab = {(N, s): walk(N, s) for N in Ns for s in (1, 2)}
    # (1)
    a1 = parts(md_answer(1, "1"))
    ok1 = (tab[(3, 1)][1] == sp.sympify(a1["D(3,1)"]) and tab[(4, 2)][1] == sp.sympify(a1["D(4,2)"])
           and all(tab[(N, 1)][1] == sp.sympify(a1["D(n,1)"]).subs(n, N)
                   and tab[(N, 2)][1] == sp.sympify(a1["D(n,2)"]).subs(n, N) for N in Ns))
    # (2)(3) 점화식: 'S(n-1,2)' 등을 기호 Y 로 바꿔 대입
    Y = sp.Symbol("Y")
    r2 = sp.sympify(parts(md_answer(1, "2"))["S(n,1)"].replace("S(n-1,2)", "Y"), locals={"Y": Y})
    r3 = sp.sympify(parts(md_answer(1, "3"))["S(n,2)"].replace("S(n-1,1)", "Y"), locals={"Y": Y})
    ok2 = all(tab[(N, 1)][0] == r2.subs(Y, tab[(N - 1, 2)][0]) for N in Ns if N >= 2)
    ok3 = all(tab[(N, 2)][0] == r3.subs(Y, tab[(N - 1, 1)][0]) for N in Ns if N >= 2)
    # (4) S(n,2)=F(n,1), a_n+b_n
    a4 = parts(md_answer(1, "4"))
    sym = all(tab[(N, 2)][0] == tab[(N, 1)][2] for N in Ns)
    ab = sp.sympify(a4["a_n+b_n"])
    ok4 = sym and all(tab[(N, 1)][0] + tab[(N, 2)][0] == ab.subs(n, N) for N in Ns) \
        and sp.limit(ab, n, sp.oo) == sp.sympify(a4["lim(a_n+b_n)"])
    # (5) 점화식을 n=2000 까지 정확히 돌려 극한 추정 + 고정점 연립
    a, b = sp.Integer(0), sp.Rational(1, 2)          # S(1,1)=0, S(1,2)=1/2
    for _ in range(200):
        a, b = b / 2, a / 2 + sp.Rational(1, 2)
    A, B = sp.symbols("A B")
    fix = sp.solve([A - r2.subs(Y, B), B - r3.subs(Y, A)], [A, B])
    a5 = parts(md_answer(1, "5"))
    la, lb = sp.sympify(a5["lim a_n"]), sp.sympify(a5["lim b_n"])
    ok5 = fix[A] == la and fix[B] == lb and abs(float(a - la)) < 1e-30 and abs(float(b - lb)) < 1e-30
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"D ok={ok1} rec2={ok2} rec3={ok3} (4)={ok4} lim=({fix[A]},{fix[B]})")


def p02():
    h = sp.symbols("h", nonnegative=True)
    V1 = sp.integrate(sp.pi * (1 - sp.sqrt(h))**2, (h, 0, 1))
    ok1 = sp.simplify(V1 - sp.sympify(md_answer(2, "1"))) == 0
    X, Yt = sp.exp(-t) * sp.cos(t), sp.exp(-t) * sp.sin(t)
    speed = sp.sqrt(sp.diff(X, t)**2 + sp.diff(Yt, t)**2)
    L = sp.integrate(sp.simplify(speed), (t, 0, sp.pi))
    Lnum = sp.Integral(speed, (t, 0, sp.pi)).evalf(30)
    ans2 = sp.sympify(md_answer(2, "2"))
    ok2 = sp.simplify(L - ans2) == 0 and abs(Lnum - ans2.evalf(30)) < 1e-20
    # (3) 독립 탐색: 높이 H, 밑면 반지름 R 인 원뿔이 빈 공간(높이 y 에서 반지름 1-sqrt(y), 즉 y <= (1-r)^2)에
    #     들어갈 조건 H(1-r/R) <= (1-r)^2 (0<=r<=R<=1). R 마다 허용 최대 H 를 구해 부피 최대화.
    import numpy as np
    rs = np.linspace(0, 1, 4001)

    def best(R):
        r = rs[rs < R]
        return np.min((1 - r)**2 / (1 - r / R))

    Rg = np.linspace(0.3, 1.0, 1401)
    vols = [np.pi / 3 * R**2 * best(R) for R in Rg]
    i = int(np.argmax(vols))
    ans3 = float(sp.sympify(md_answer(2, "3")))
    ok3 = abs(vols[i] - ans3) < 1e-4 and abs(Rg[i] - 0.75) < 5e-3
    # 원 해설의 V(a) 식도 기호로 확인
    a = sp.symbols("a", real=True)
    Va = sp.pi / 12 * (a + 1)**2 * (1 - a**2)
    crit = [c for c in sp.solve(sp.diff(Va, a), a) if 0 <= c <= 1]
    ok3 = ok3 and crit == [sp.Rational(1, 2)] and sp.simplify(Va.subs(a, crit[0]) - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and ok2 and ok3, f"V1={V1} L={sp.simplify(L)} Vmax~{vols[i]:.6f} at R={Rg[i]:.4f}")


def swept_area(N: int, grid: int = 700, steps: int = 160) -> float:
    """회전 도형이 훑는 넓이를 격자로 직접 계산(정답 식을 쓰지 않음)."""
    import numpy as np
    ang = np.pi / 2 + 2 * np.pi * np.arange(N) / N
    fixed = np.c_[np.cos(ang), np.sin(ang)]
    # 첫 위치: 고정 도형을 한 변(꼭짓점 0-1)에 대해 대칭이동
    p, q = fixed[0], fixed[1]
    d = (q - p) / np.linalg.norm(q - p)

    def refl(v):
        w = v - p
        return p + 2 * (w @ d) * d - w
    poly = np.array([refl(v) for v in fixed])
    R = 4.2
    g = np.linspace(-R, R, grid)
    GX, GY = np.meshgrid(g, g)
    P = np.c_[GX.ravel(), GY.ravel()]
    covered = np.zeros(len(P), bool)

    def inside(pg):
        c = pg.mean(0)
        ok = np.ones(len(P), bool)
        for k in range(len(pg)):
            a_, b_ = pg[k], pg[(k + 1) % len(pg)]
            nrm = np.array([b_[1] - a_[1], a_[0] - b_[0]])
            s = (c - a_) @ nrm
            ok &= ((P - a_) @ nrm) * s >= 0
        return ok
    pivot_idx = 0           # 고정 도형의 꼭짓점 번호(회전축)
    theta_tot = 2 * np.pi - 2 * np.pi * (N - 2) / N
    for _ in range(N):
        piv = fixed[pivot_idx]
        # 회전 방향: 고정 도형 둘레를 꼭짓점 번호 감소 방향으로 돈다(대칭이동 방향과 일관되게 부호 선택)
        best_sign = None
        for sgn in (1, -1):
            th = sgn * theta_tot
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            new = (poly - piv) @ Rm.T + piv
            # 회전 후 고정 도형과 다른 변을 공유해야 함(겹치지 않음): 중심 거리 = 2*apothem
            if abs(np.linalg.norm(new.mean(0)) - 2 * np.cos(np.pi / N)) < 1e-9 and \
               np.linalg.norm(new.mean(0) - poly.mean(0)) > 1e-6:
                best_sign = sgn
        for s_ in np.linspace(0, 1, steps):
            th = best_sign * theta_tot * s_
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            covered |= inside((poly - piv) @ Rm.T + piv)
        th = best_sign * theta_tot
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        poly = (poly - piv) @ Rm.T + piv
        # 다음 회전축: 새 위치에서 고정 도형과 공유하는 꼭짓점 중 현재 축이 아닌 것
        shared = [k for k in range(N) if k != pivot_idx and np.min(np.linalg.norm(poly - fixed[k], axis=1)) < 1e-9]
        pivot_idx = shared[0]
    covered &= ~inside(fixed)
    cell = (2 * R / (grid - 1))**2
    return covered.sum() * cell


def p03():
    ans1 = sp.sympify(md_answer(3, "1"))
    A3 = swept_area(3)
    ok1 = abs(A3 - float(ans1)) / float(ans1) < 0.01
    # (2) 좌표로 가장 먼 두 꼭짓점 거리 계산 후 식과 비교
    ok2 = True
    for N in (3, 5, 7, 9, 11):
        pts = [sp.Matrix([sp.cos(2 * sp.pi * k / N), sp.sin(2 * sp.pi * k / N)]) for k in range(N)]
        dmax = max(float((pts[0] - pts[k]).norm()) for k in range(1, N))
        ok2 &= abs(dmax - float(sp.sympify(md_answer(3, "2").split("=")[1]).subs(n, N))) < 1e-12
    # (3) S_n 식: 격자 넓이와 비교(n=5,7), n=3 값 일치, 극한 = 고리 넓이 pi(3^2-1^2)
    a3 = parts(md_answer(3, "3"))
    Sn = sp.sympify(a3["S_n"])
    errs = {}
    for N in (5, 7):
        A = swept_area(N)
        errs[N] = abs(A - float(Sn.subs(n, N))) / float(Sn.subs(n, N))
    ok3 = all(e < 0.01 for e in errs.values()) and sp.simplify(Sn.subs(n, 3) - ans1) == 0
    lim = sp.limit(Sn, n, sp.oo)
    ok3 = ok3 and sp.simplify(lim - sp.sympify(a3["limit"])) == 0 and sp.simplify(lim - sp.pi * (3**2 - 1**2)) == 0
    report(3, ok1 and ok2 and ok3, f"grid A3={A3:.4f} vs {float(ans1):.4f}; relerr n=5,7 {errs}; lim={lim}")


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
