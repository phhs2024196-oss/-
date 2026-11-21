#!/usr/bin/env python3
"""KKU-G25 (건국대 2025학년도 KU논술우수자 자연계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "KKU-G25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


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
    # 기호: 각 OAB 예각 조건 AB^2+9 > OB^2 을 정리해 풀기
    A = sp.Matrix([3 * sp.cos(t), 3 * sp.sin(t)])
    B = sp.Matrix([7 + sp.cos(3 * t), sp.sin(3 * t)])
    expr = sp.simplify(sp.expand_trig(((A - B).dot(A - B) + 9 - B.dot(B))))
    sol = sp.solveset(expr > 0, t, sp.Interval(0, 2 * sp.pi))
    md = sp.sympify(md_answer(1, "1"), locals={"Interval": sp.Interval})
    ok_sym = sp.simplify(sol.symmetric_difference(md)) == sp.S.EmptySet or sol == md
    # 수치: 선분 AB 와 원 x^2+y^2=9 의 교점 개수를 직접 센다(격자)
    def count(tv):
        ax, ay = 3 * math.cos(tv), 3 * math.sin(tv)
        bx, by = 7 + math.cos(3 * tv), math.sin(3 * tv)
        dx, dy = bx - ax, by - ay
        # |A + s d|^2 = 9 → s (s |d|^2 + 2 A·d) = 0 ; s=0 은 A
        s2 = -2 * (ax * dx + ay * dy) / (dx * dx + dy * dy)
        return 2 if 1e-9 < s2 <= 1 else 1
    grid = [2 * math.pi * k / 4000 for k in range(4001)]
    ok_num = all((count(tv) == 2) == bool(md.contains(sp.Float(tv))) for tv in grid
                 if min(abs(tv - math.pi / 3), abs(tv - 5 * math.pi / 3)) > 1e-6)
    report(1, ok_sym and ok_num, f"solveset={sol} grid4001 ok={ok_num}")


def p02():
    # 정사면체 좌표 (한 모서리 6)
    A = sp.Matrix([0, 0, 0]); B = sp.Matrix([6, 0, 0]); C = sp.Matrix([3, 3 * sp.sqrt(3), 0])
    Hc = (A + B + C) / 3
    D = Hc + sp.Matrix([0, 0, sp.sqrt(36 - Hc.dot(Hc))])
    E = D + (B - D) * sp.Rational(2, 6)          # DE = 2
    import numpy as np
    An, Dn, En = (np.array([float(v) for v in M]) for M in (A, D, E))
    def vol(phi, s, L):
        u = np.array([math.cos(phi), math.sin(phi), 0.0])
        P, Q = An + s * u, An - (L - s) * u
        return abs(np.linalg.det(np.array([P - Dn, Q - Dn, En - Dn]))) / 6
    best = {}
    for L in (1.0, 2.5):
        cand = max(((vol(ph, s, L), ph) for ph in np.linspace(0, math.pi, 3601) for s in (0.0, L / 2, L)))
        best[L] = cand[1]
    ok_dir = all(abs(math.cos(ph)) < 2e-3 for ph in best.values())   # PQ ⊥ AB (y 방향)
    # 최대일 때 PQ 는 A 를 지나는 y 방향 직선: 평면 EPQ 의 법선과 평면 ABC 법선의 각
    u = sp.Matrix([0, 1, 0])
    nrm = u.cross(E - A)
    cos_th = sp.nsimplify(sp.simplify(abs(nrm[2]) / sp.sqrt(nrm.dot(nrm))))
    ok = ok_dir and sp.simplify(cos_th - sp.sympify(md_answer(2, "1"))) == 0
    report(2, ok, f"max dir phi={[round(v, 4) for v in best.values()]} cos={cos_th}")


def p03():
    th = sp.symbols("theta", real=True)
    A = sp.Matrix([0, 0]); B = sp.Matrix([sp.sqrt(3), 0]); D = sp.Matrix([sp.cos(th), sp.sin(th)])
    # C: B, D 에서 각각 거리 2 인 점 중 A 의 반대쪽(볼록 사각형)
    M = (B + D) / 2
    BD = D - B
    h = sp.sqrt(4 - BD.dot(BD) / 4)
    perp = sp.Matrix([-BD[1], BD[0]]) / sp.sqrt(BD.dot(BD))
    Cs = [M + h * perp, M - h * perp]
    th0 = sp.pi / 2
    # A 에서 더 먼 쪽 선택
    Cc = max(Cs, key=lambda Cv: float(Cv.subs(th, th0).dot(Cv.subs(th, th0))))
    f = sp.sqrt(Cc.dot(Cc))
    val = sp.nsimplify(sp.simplify(sp.diff(f, th).subs(th, th0)))
    ok_len = sp.simplify((Cc - D).dot(Cc - D).subs(th, th0) - 4) == 0 and sp.simplify(f.subs(th, th0) - sp.sqrt(7)) == 0
    # 수치 미분 교차 확인
    fn = sp.lambdify(th, f, "mpmath")
    import mpmath as mp
    mp.mp.dps = 30
    num = mp.diff(fn, mp.pi / 2)
    md = sp.sympify(md_answer(3, "1"))
    ok = ok_len and sp.simplify(val - md) == 0 and abs(num - float(md)) < 1e-12
    report(3, ok, f"f(pi/2)={sp.simplify(f.subs(th, th0))} f'(pi/2)={val} numeric={mp.nstr(num, 15)}")


def p04():
    tt = sp.symbols("t", positive=True)
    # (1) 넓이 f = 사다리꼴 - ∫x^2, θ = ∠OPA (벡터 PO=(0,-10), PA=(t,t^2-10))
    F = tt * (10 + tt**2) / 2 - sp.integrate(x**2, (x, 0, tt))
    theta = sp.acos((10 - tt**2) / sp.sqrt(tt**2 + (tt**2 - 10)**2))
    dth = sp.diff(theta, tt)
    fprime = sp.diff(F, tt) / dth
    md1 = sp.sympify(md_answer(4, "1"), locals={"t": tt})
    import mpmath as mp
    ok1 = all(abs(complex((fprime - md1).subs(tt, v).evalf(30))) < 1e-20 for v in (sp.Rational(1, 2), sp.Rational(17, 10), sp.Rational(31, 10), sp.Rational(22, 5)))
    # (2) g(θ)=f(θ+π/2)-f(θ): 수치로 g'(θ)=0 을 찾는다
    mp.mp.dps = 40
    def t_of(thv):                      # ∠OPA = thv 인 곡선 위 점의 x좌표
        return mp.findroot(lambda s: mp.acos((10 - s**2) / mp.sqrt(s**2 + (s**2 - 10)**2)) - thv,
                           mp.sqrt(10) * thv / (mp.pi / 2) + mp.mpf("0.01"))
    def Fn(s):
        return s**3 / 6 + 5 * s
    def g(thv):
        return Fn(t_of(thv + mp.pi / 2)) - Fn(t_of(thv))
    # 격자에서 g' 부호 변화
    grid = [mp.mpf(k) / 200 * (mp.pi / 2) for k in range(2, 199)]
    dg = [mp.diff(g, v) for v in grid]
    roots = [mp.findroot(lambda v: mp.diff(g, v), grid[i]) for i in range(len(grid) - 1) if dg[i] * dg[i + 1] < 0]
    ok_unique = len(roots) == 1
    th0 = roots[0]
    a, b = t_of(th0), t_of(th0 + mp.pi / 2)
    perp = abs((a**2 - 10) / a * (b**2 - 10) / b + 1) < 1e-12
    parts = dict(seg.split("=", 1) for seg in md_answer(4, "2").replace(" ", "").split(";"))
    want = {"a**2+b**2": a**2 + b**2, "a*b": a * b, "g": g(th0)}
    ok2 = ok_unique and perp and all(abs(want[k] - mp.mpf(sp.N(sp.sympify(parts[k]), 40))) < 1e-12 for k in want)
    report(4, ok1 and ok2, f"(1) f'(θ) ok={ok1}; (2) roots={len(roots)} a={mp.nstr(a, 12)} b={mp.nstr(b, 12)} "
           f"a2+b2={mp.nstr(a**2 + b**2, 15)} ab={mp.nstr(a * b, 15)} g={mp.nstr(g(th0), 15)}")


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
