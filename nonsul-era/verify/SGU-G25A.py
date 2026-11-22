#!/usr/bin/env python3
"""SGU-G25A (서강대 2025학년도 수시 논술 1차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
PREFIX = "SGU-G25A"
import re
import sys

import sympy as sp

WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")
x, y = sp.symbols("x y", real=True)


def p01():
    A, B = sp.Matrix([5, 0]), sp.Matrix([-5, 0])
    s = sp.symbols("s", positive=True)
    # (1) 원 |AC|=6 위의 여러 C 에서 조건 (ii)(iii)을 만족하는 P 를 직접 구해 md 의 곡선 위에 있는지 확인
    lhs, rhs = md_answer(1, "1").split("=")
    curve = sp.sympify(lhs, locals={"x": x, "y": y}) - sp.sympify(rhs)
    ok1 = True
    for ang in (2.25, 2.5, 2.8, 3.0, -2.4, -2.9):          # |BC|<=8 이어야 P 존재 (cos ang <= -3/5)
        C = A + 6 * sp.Matrix([sp.cos(ang), sp.sin(ang)])
        P = A + s * (C - A) / 6
        sols = [v for v in sp.solve(sp.Eq((P - B).dot(P - B), (8 - (6 - s))**2), s) if 0 <= v <= 6]
        for v in sols:
            Pv = P.subs(s, v)
            ok1 &= abs(float(curve.subs({x: Pv[0], y: Pv[1]}))) < 1e-9
        ok1 &= len(sols) >= 1
    # (2) 초점 ±5, 거리의 차 2 → a=1, b^2=24 (제시문 [나]). Q 제2사분면, 접선 [다] 의 y절편 R
    a2, b2 = 1, 5**2 - 1
    yq = sp.symbols("yq", positive=True)
    xq2 = a2 * (1 + yq**2 / b2)                      # x_q^2
    Ry = -b2 / yq                                     # x_q x/a2 - yq y/b2 = 1 에서 x=0
    QR2 = xq2 + (yq - Ry)**2
    crit = [c for c in sp.solve(sp.diff(QR2, yq), yq) if c.is_positive]
    best = min(crit, key=lambda c: float(QR2.subs(yq, c)))
    OQ2 = sp.nsimplify(sp.simplify(xq2.subs(yq, best) + best**2))
    ok2 = sp.simplify(OQ2 - sp.sympify(md_answer(1, "2"))) == 0 and float(QR2.subs(yq, best)) < float(QR2.subs(yq, best * 1.01))
    # (3) C=P: |AC|=6, |BC|=8, D = 선분 BC ∩ y축, cos(angle COD)
    cx, cy = sp.symbols("cx cy", real=True)
    Csol = [c for c in sp.solve([(cx - 5)**2 + cy**2 - 36, (cx + 5)**2 + cy**2 - 64], [cx, cy], dict=True) if c[cy] > 0][0]
    Cp = sp.Matrix([Csol[cx], Csol[cy]])
    u = sp.symbols("u")
    uD = sp.solve(B[0] + u * (Cp[0] - B[0]), u)[0]
    D = B + uD * (Cp - B)
    cosa = sp.simplify(Cp.dot(D) / (Cp.norm() * D.norm()))
    ok3 = sp.simplify(cosa - sp.sympify(md_answer(1, "3"))) == 0
    # (4) angle A = th, C = A + 6(-cos th, sin th), P on AC with |BP|-|AP|=2, f = area BCP
    th = sp.symbols("th", positive=True)
    C4 = A + 6 * sp.Matrix([-sp.cos(th), sp.sin(th)])
    P4 = A + s * (C4 - A) / 6
    sv = sp.solve(sp.expand((P4 - B).dot(P4 - B) - (s + 2)**2).subs(sp.sin(th)**2, 1 - sp.cos(th)**2), s)[0]
    P4 = P4.subs(s, sv)
    cross = ((C4 - B)[0] * (P4 - B)[1] - (C4 - B)[1] * (P4 - B)[0]) / 2
    be = sp.acos(sp.Rational(4, 5))
    sgn = sp.sign(cross.subs(th, be).evalf())
    fp = sp.nsimplify(sp.N((sgn * sp.diff(cross, th)).subs(th, be), 50), rational=True, tolerance=1e-30)
    ok4 = sp.simplify(fp - sp.sympify(md_answer(1, "4"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"P on curve(6 C); OQ^2={OQ2}; cos a={cosa}; f'(b)={fp}")


def fmin(tv):
    """g(θ)^2 = h(cosθ) 의 [-1,1] 최솟값의 제곱근 (끝점·임계점 비교)."""
    t = sp.nsimplify(tv)
    th = sp.symbols("th")
    g2 = sp.expand((4 * sp.cos(th) - 3 * t)**2 + 4 * t**2 * sp.sin(th)**2)
    X = sp.symbols("X", real=True)
    h = sp.expand(g2.subs(sp.sin(th)**2, 1 - sp.cos(th)**2).subs(sp.cos(th), X))
    cands = [sp.Integer(-1), sp.Integer(1)] + [c for c in sp.solve(sp.diff(h, X), X) if -1 <= c <= 1]
    return sp.sqrt(min((h.subs(X, c) for c in cands), key=lambda v: float(v)))


def p02():
    t = sp.symbols("t", positive=True)
    import numpy as np
    # 무작위 격자 대조: θ 전수 탐색과 fmin 일치
    ths = np.linspace(0, 2 * np.pi, 200001)
    okg = all(abs(np.min(np.sqrt((4 * np.cos(ths) - 3 * tv)**2 + 4 * tv**2 * np.sin(ths)**2)) - float(fmin(tv))) < 1e-6
              for tv in (0.3, 1, 1.5, 2, 3.7))
    ok1 = sp.simplify(fmin(1) - sp.sympify(md_answer(2, "1"))) == 0
    e2 = sp.sympify(md_answer(2, "2"), locals={"t": t})
    ok2 = all(abs(float(fmin(tv) - e2.subs(t, sp.nsimplify(tv)))) < 1e-12 for tv in (0.1, 0.37, 0.5, 0.8, 0.99))
    e3 = sp.sympify(md_answer(2, "3"), locals={"t": t})
    ok3 = all(abs(float(fmin(tv) - e3.subs(t, sp.nsimplify(tv)))) < 1e-12 for tv in (1.01, 4 / 3, 1.7, 2, 2.5, 6))
    # (4) t=1 에서 연속·미분가능: 수치 좌·우 차분몫
    import mpmath as mp
    mp.mp.dps = 40
    F = lambda v: mp.mpf(str(sp.N(fmin(sp.Rational(v)), 50)))
    f1 = F(1)
    hs = ["1/1000000", "1/10000000"]
    left = [(F(1 - sp.Rational(h)) - f1) / (-mp.mpf(sp.Rational(h).p) / sp.Rational(h).q) for h in hs]
    right = [(F(1 + sp.Rational(h)) - f1) / (mp.mpf(sp.Rational(h).p) / sp.Rational(h).q) for h in hs]
    p = parts(md_answer(2, "4").split(";", 2)[2])
    d1 = float(sp.sympify(p["f1(1)"]))
    s4 = md_answer(2, "4")
    ok4 = "연속" in s4 and "미분가능" in s4 and "불가능" not in s4 and \
        all(abs(float(v) - d1) < 1e-4 for v in left + right) and abs(F(1 + sp.Rational(1, 10**9)) - f1) < 1e-8
    report(2, okg and ok1 and ok2 and ok3 and ok4,
           f"f(1)={fmin(1)}; (2)(3) 격자 일치; left={float(left[-1]):.6f} right={float(right[-1]):.6f}")


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
