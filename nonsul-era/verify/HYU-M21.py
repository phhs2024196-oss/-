#!/usr/bin/env python3
"""HYU-M21 (한양대 2021학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산 후 YAML answer.value 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "HYU-M21"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 1. [0,pi] 격자에서 sin(x-cos x)-sin x 의 부호: 등호 지점만 0, 나머지 음수
    d = sp.lambdify(x, sp.sin(x - sp.cos(x)) - sp.sin(x))
    import numpy as np
    eq = float(sp.sympify(kv(md_answer(1, "1"))["equal_at"]))
    xs = np.linspace(0, np.pi, 200001)
    v = d(xs)
    zeros = xs[np.abs(v) < 1e-9]
    ok1 = abs(d(eq)) < 1e-12 and len(zeros) > 0 and np.all(np.abs(zeros - eq) < 1e-3) and np.all(v[np.abs(xs - eq) > 1e-3] < 0)
    h = sp.pi / 2
    I2 = sp.integrate(x * sp.cos(x)**2 + x**3 * sp.cos(x) + sp.sin(x), (x, -h, h))
    ok2 = sp.simplify(I2 - sp.sympify(md_answer(1, "2"))) == 0
    f = 2 * x * sp.cos(2 * x) - sp.sin(x)
    g = 2 * x * f - sp.cos(2 * x)
    I3 = sp.integrate(sp.expand(g), (x, -h, h)) + sp.integrate(sp.expand(2 * x * g - f), (x, -h, h))
    ok3 = sp.simplify(I3 - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"equal_at={eq:.6f} I2={I2} I3={sp.simplify(I3)}")


def p02():
    # 1. 항등식: 여러 t 에서 수치 대입
    a = kv(md_answer(2, "1"))
    A, B, C, D = (sp.sympify(a[k]) for k in "abcd")
    lhs = sp.sin(t) * sp.sin(sp.pi / 3 - t)
    ok1 = all(abs(float((lhs - (A * sp.cos(B * t + C) + D)).subs(t, v))) < 1e-12 for v in (0, 0.3, 1.1, 2.7, -4.2))
    # 2. 좌표로 회전 선분의 교점을 직접 구해 S(t) 계산(반시계 회전)
    T = sp.symbols("T", real=True)
    Ap, Bp, Cp = sp.Matrix([0, 0]), sp.Matrix([1, 0]), sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 2])
    rot = lambda v, th: sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]]) * v
    def meet(P1, d1, P2, d2):
        u, w = sp.symbols("u w", real=True)
        s = sp.solve(list(P1 + u * d1 - P2 - w * d2), [u, w], dict=True)[0]
        return P1 + s[u] * d1, s[u], s[w]
    dl, dm, dn = rot(Bp - Ap, T), rot(Cp - Bp, T), rot(Ap - Cp, T)
    P, *_ = meet(Ap, dl, Bp, dm)
    Q, *_ = meet(Bp, dm, Cp, dn)
    R, *_ = meet(Cp, dn, Ap, dl)
    S = sp.Abs(((Q - P)[0] * (R - P)[1] - (Q - P)[1] * (R - P)[0])) / 2
    Sf = sp.lambdify(T, S)
    from scipy.integrate import quad
    val, _ = quad(lambda v: float(Sf(v)), 0, float(sp.pi / 3), epsabs=1e-13, epsrel=1e-13)
    ok2 = abs(val - float(sp.sympify(md_answer(2, "2")))) < 1e-9
    # 3. 넓이 함수와 극한
    tt = sp.symbols("tt", positive=True)
    def tangent(F):
        return sp.diff(F, x).subs(x, 1) * (x - 1) + F.subs(x, 1)
    F1, F2 = x * sp.log(x), sp.log(x)
    Pt = sp.integrate(F1 - tangent(F1), (x, 1, tt))
    Qt = sp.integrate(tangent(F2) - F2, (x, 1, tt))
    lim = sp.limit(Pt / (tt * Qt), tt, sp.oo)
    ok3 = sp.simplify(lim - sp.sympify(md_answer(2, "3"))) == 0 and float(Pt.subs(tt, 3)) > 0 and float(Qt.subs(tt, 3)) > 0
    report(2, ok1 and ok2 and ok3, f"intS={val:.12f} lim={lim}")


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
