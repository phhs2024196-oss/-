#!/usr/bin/env python3
"""SJU-M16 (세종대 2016학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-M16"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    x, y, z, k = sp.symbols("x y z k", real=True)
    c = sp.sqrt(3)
    sph = x**2 + y**2 + (z - c)**2 - 4
    # (1) 윗면 z=2√3 에서 x^2+y^2 = r^2, 옆면 x^2+y^2=3 과 만나는 z 의 최댓값
    r = sp.sqrt(sp.solve(sph.subs({z: 2 * c, y: 0}), x)[-1]**2)
    zs = sp.solve(sph.subs({x: 0, y: c}), z)
    a = max(zs, key=lambda v: float(v))
    P = parts(md_answer(1, "1"))
    ok1 = sp.simplify(r - sp.sympify(P["r"])) == 0 and sp.simplify(a - sp.sympify(P["a"])) == 0
    # (2) 단면 원 넓이
    S = sp.pi * sp.solve(sph.subs({y: 0, z: k}), x)[-1]**2
    ok2 = sp.simplify(S - sp.sympify(md_answer(1, "2"), locals={"k": k})) == 0
    # (3) 높이별 단면 = min(구 단면, 원기둥 단면 3π) 를 0..2√3 에서 적분
    Sz = sp.pi * (4 - (z - c)**2)
    V = sp.integrate(Sz, (z, 0, c - 1)) + sp.integrate(3 * sp.pi, (z, c - 1, c + 1)) + sp.integrate(Sz, (z, c + 1, 2 * c))
    # 수치 교차 확인(몬테카를로 대신 격자 적분)
    import numpy as np
    zz = np.linspace(0, 2 * float(c), 200001)
    area = np.minimum(np.pi * np.maximum(4 - (zz - float(c))**2, 0), 3 * np.pi)
    Vnum = np.trapezoid(area, zz)
    ok3 = sp.simplify(V - sp.sympify(md_answer(1, "3"))) == 0 and abs(Vnum - float(V)) < 1e-6
    wrong = sp.nsimplify(3 * sp.pi * 2 + sp.integrate(Sz, (z, c + 1, 2 * c)))   # 2배 누락 오답 재현
    report(1, ok1 and ok2 and ok3, f"r={r} a={a} V={sp.simplify(V)}≈{float(V):.6f} (2배 누락 오답 {sp.simplify(wrong)})")


def p02():
    z, x, y = sp.symbols("z x y", real=True)
    f = z / 2 + sp.sqrt((80 - z)**2 + 300)
    crit = [s for s in sp.solve(sp.diff(f, z), z) if 0 <= s <= 80]
    cand = crit + [0, 80]
    tmin = min(sp.simplify(f.subs(z, v)) for v in cand)
    ok1 = sp.simplify(tmin - sp.sympify(md_answer(2, "1"))) == 0
    # (2) 일반 x,y 에서 최적 z = 50 일 때 관계: 수치로 여러 (x,y) 를 md 관계식에서 뽑아 실제 최적 z 확인
    import numpy as np
    rel = md_answer(2, "2").replace(" ", "").split("=", 1)[1]
    yf = sp.lambdify(x, sp.sympify(rel))
    ok2 = True
    for xv in (60.0, 80.0, 130.0):
        yv = yf(xv)
        zs = np.linspace(0, xv, 200001)
        T = zs / 2 + np.sqrt((xv - zs)**2 + yv**2)
        ok2 &= abs(zs[T.argmin()] - 50) < 1e-2
    # (3) 격자 위 각 점의 최소 도달 시간 ≤ 60 인 넓이 (z 는 -120..120 탐색, 닫힌꼴 최소화)
    # 점 (X,Y) 에 대해 최소 시간 = min_z |z|/2 + sqrt((X-z)^2+Y^2)
    def tmin_pt(X, Y):
        # 최적 z 후보: 직접 수영(z=0) 또는 z = X - sign(X)*Y/√3 (|z| 가 X 와 같은 부호일 때)
        cands = [0.0]
        zc = X - np.sign(X) * Y / np.sqrt(3)
        if zc * X > 0:
            cands.append(zc)
        return min(abs(zz) / 2 + np.hypot(X - zz, Y) for zz in cands)
    h = 0.5
    Xs = np.arange(-130 + h / 2, 130, h)
    Ys = np.arange(h / 2, 70, h)
    cnt = 0
    for Yv in Ys:
        for Xv in Xs:
            if tmin_pt(Xv, Yv) <= 60:
                cnt += 1
    area = cnt * h * h
    ok3 = abs(area - float(sp.sympify(md_answer(2, "3")))) / area < 2e-3
    report(2, ok1 and ok2 and ok3, f"tmin={tmin} relation y={rel} 격자넓이≈{area:.1f}")


def p03():
    B = sp.Matrix([1, 1, 1])
    ang = lambda u, v: sp.acos(sp.simplify(u.dot(v) / (u.norm() * v.norm())))
    P = sp.Matrix([1, 2, 1]); Q = sp.Matrix([sp.Rational(3, 2), sp.Rational(3, 2), 1 + sp.sqrt(2) / 2])
    d1 = ang(P - B, Q - B)
    ok1 = sp.simplify(d1 - sp.sympify(md_answer(3, "1"))) == 0
    A = sp.Matrix([2, 2, 1 - sp.sqrt(2)])
    t = sp.symbols("t", real=True)
    L = B + t * (B - A)
    ts = sp.solve(sp.expand((L - B).dot(L - B) - 1), t)
    C = max((L.subs(t, v) for v in ts), key=lambda p: float((p - A).norm()))
    ok2 = sp.simplify(sp.Matrix(sp.sympify(md_answer(3, "2"))) - C) == sp.zeros(3, 1)
    # 구 밖 점 A 에서 구면 위 점 X 까지 최단 = 보이면 |AX|, 아니면 √(AB²-1) + (∠ABX - arccos(1/AB))
    AB = (A - B).norm()
    tang = sp.sqrt(AB**2 - 1); a0 = sp.acos(1 / AB)
    def dist(X):
        th = ang(A - B, X - B)
        return (X - A).norm() if float(th) <= float(a0) + 1e-12 else tang + th - a0
    d3 = sp.simplify(dist(C))
    ok3 = sp.simplify(d3 - sp.sympify(md_answer(3, "3"))) == 0
    # (4) 원 z=3/2 위를 수치로 훑어 최솟값
    import mpmath as mp
    r = mp.sqrt(mp.mpf(3) / 4)
    Af = [mp.mpf(2), mp.mpf(2), 1 - mp.sqrt(2)]
    def dnum(ph):
        X = [1 + r * mp.cos(ph), 1 + r * mp.sin(ph), mp.mpf(3) / 2]
        u = [Af[i] - 1 for i in range(3)]; v = [X[i] - 1 for i in range(3)]
        th = mp.acos(sum(u[i] * v[i] for i in range(3)) / (mp.sqrt(sum(w * w for w in u)) * mp.sqrt(sum(w * w for w in v))))
        a0n = mp.acos(1 / mp.mpf(2))
        if th <= a0n:
            return mp.sqrt(sum((X[i] - Af[i])**2 for i in range(3)))
        return mp.sqrt(3) + th - a0n
    grid = [2 * mp.pi * i / 3600 for i in range(3600)]
    p0 = min(grid, key=dnum)
    pm = mp.findroot(lambda s: mp.diff(dnum, s), p0)
    dmin = dnum(pm)
    ok4 = abs(dmin - mp.mpf(float(sp.N(sp.sympify(md_answer(3, "4")), 30)))) < 1e-9
    report(3, ok1 and ok2 and ok3 and ok4, f"d(P,Q)={d1} C={list(C)} d(A,C)={d3} min d(A,D)≈{mp.nstr(dmin, 12)}")


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
