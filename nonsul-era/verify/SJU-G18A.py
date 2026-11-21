#!/usr/bin/env python3
"""SJU-G18A (세종대 2018학년도 수시 논술고사[A]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "SJU-G18A"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)
E = sp.E


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str) -> dict:
    """'키=식; 키=식' 형태의 정답을 dict 로."""
    return {a.strip(): b.strip() for a, b in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def S(s, **loc):
    return sp.sympify(s, locals={"e": sp.E, "x": x, "t": t, **loc})


def same(a, b, tol=1e-12):
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 if not (isinstance(a, float) or isinstance(b, float)) else abs(float(a) - float(b)) < tol
    except Exception:  # noqa: BLE001
        return abs(float(a) - float(b)) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    y = sp.Symbol("y", nonnegative=True)
    pG = sp.Rational(8, 10)
    PO_G = lambda w: sp.integrate(sp.Rational(1, 30), (x, 0, sp.Min(w, 30)))
    PO_N = lambda w: sp.integrate(sp.Rational(1, 60), (x, 0, sp.Min(w, 60)))
    PO = pG * PO_G(15) + (1 - pG) * PO_N(15)
    ok1 = PO == S(md_answer(1, "1"))
    ok2 = pG * PO_G(15) / PO == S(md_answer(1, "2"))
    wmin = next(w for w in range(0, 61) if pG * PO_G(w) + (1 - pG) * PO_N(w) >= sp.Rational(6, 10))  # 분 단위 전수
    tmin = 8 * 60 + 15 - wmin
    ok3 = md_answer(1, "3") == f"{tmin // 60}:{tmin % 60:02d}" and pG * PO_G(wmin - sp.Rational(1, 100)) + (1 - pG) * PO_N(wmin - sp.Rational(1, 100)) < sp.Rational(6, 10)
    report(1, ok1 and ok2 and ok3, f"P(O)={PO} P(G|O)={pG * PO_G(15) / PO} 대기 {wmin}분 → {tmin // 60}:{tmin % 60:02d}")


def p02():
    import numpy as np
    O = np.zeros(3); A = np.array([1., -1, 2]); H = np.array([2., 0, 4]); d = np.array([1., 1, 2])
    ok_line = np.allclose(np.cross(H - A, d), 0)
    u = np.array([1., -1, 0]) / np.sqrt(2); w = np.cross(d / np.linalg.norm(d), u)
    # 평면 위 점 P = H + s u + r w 를 격자로 훑음
    g = np.linspace(-4, 4, 1601)
    S1, S2 = np.meshgrid(g, g)
    P = H[None, None, :] + S1[..., None] * u + S2[..., None] * w
    ok = (np.linalg.norm(P, axis=2) <= 5 + 1e-12) & (np.linalg.norm(P - H, axis=2) >= 1 - 1e-12)
    AP = np.linalg.norm(P - A, axis=2)
    mn = AP[ok].min()
    ok1 = abs(mn - float(S(md_answer(2, "1")))) < 2e-3 and mn >= float(S(md_answer(2, "1"))) - 1e-9
    OPn = np.linalg.norm(P, axis=2); i = np.unravel_index(np.where(ok, OPn, 1e9).argmin(), OPn.shape)
    Pm = np.array([float(v) for v in S(md_answer(2, "2"))])
    ok2 = np.linalg.norm(P[i] - Pm) < 1e-2 and abs(Pm @ d - 10) < 1e-12 and np.linalg.norm(Pm) <= np.linalg.norm(P[i]) + 1e-12 and np.linalg.norm(Pm - H) >= 1
    # (3) 정확한 최댓값 위치: 원(반지름 √(25-|OP2|²)) 위에서 |AP| 최대 — 원 둘레를 세밀히
    P2 = Pm; rr = np.sqrt(25 - P2 @ P2)
    th = np.linspace(0, 2 * np.pi, 4000001)
    Q = P2[:, None] + rr * (np.outer(u, np.cos(th)) + np.outer(w, np.sin(th)))
    j = np.linalg.norm(Q - A[:, None], axis=0).argmax()
    ok3 = abs(Q[0, j] - float(S(md_answer(2, "3")))) < 1e-5 and AP[ok].max() <= np.linalg.norm(Q[:, j] - A) + 1e-9
    report(2, ok_line and ok1 and ok2 and ok3, f"min AP~{mn:.5f}, argmin OP~{np.round(P[i], 3)}, argmax AP x={Q[0, j]:.8f}")


def p03():
    a, b2, g2 = sp.symbols("a beta gamma")
    F = a + b2 * (x - 1) + g2 * (x - 1)**2
    f = (x - 1) * F + x * sp.exp(x**2)
    f1 = sp.limit(f, x, 1)
    ok1 = sp.simplify(f1 - S(md_answer(3, "1"))) == 0
    fp1 = sp.limit((f - f1) / (x - 1), x, 1)
    ok2 = sp.simplify(fp1 - S(md_answer(3, "2"), a=a)) == 0
    F3 = F.subs(b2, 3)
    f3 = (x - 1) * F3 + x * sp.exp(x**2)
    fp = sp.diff(f3, x)
    gp1 = sp.limit((fp - fp.subs(x, 1)) / (x - 1), x, 1)
    ok3 = sp.simplify(gp1 - S(md_answer(3, "3"))) == 0
    ok4 = True
    for cc in (sp.Integer(1), sp.Rational(-7, 3), sp.Integer(5)):
        F4 = cc * x * (2 * x - 1)
        ok4 &= sp.simplify(x * F4.subs(x, 1 - x) + (1 - x) * F4) == 0
        val = sp.integrate((x - 1) * F4 + x * sp.exp(x**2), (x, 0, 1))
        ok4 &= sp.simplify(val - S(md_answer(3, "4"))) == 0
    report(3, ok1 and ok2 and ok3 and ok4, f"f(1)={f1} f'(1)={sp.simplify(fp1)} g'(1)={sp.simplify(gp1)} ∫f(F=c·x(2x-1))={sp.simplify(val)}")


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
