#!/usr/bin/env python3
"""SGU-G25B (서강대 2025학년도 수시 논술 2차 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
PREFIX = "SGU-G25B"
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
b = sp.symbols("b", positive=True)


def p01():
    X, Y, s = sp.symbols("X Y s", real=True)
    F, Fp = sp.Matrix([sp.sqrt(3) * b, 0]), sp.Matrix([-sp.sqrt(3) * b, 0])
    ell = X**2 / (4 * b**2) + Y**2 / b**2 - 1
    loc = {"b": b}
    # (1)(2) ∠F'PF = 90°: (P-F')·(P-F)=0, x>0, y>0
    sol = [d for d in sp.solve([ell, (X - Fp[0]) * (X - F[0]) + Y**2], [X, Y], dict=True)
           if d[X].subs(b, 1) > 0 and d[Y].subs(b, 1) > 0][0]
    P = sp.Matrix([sol[X], sol[Y]])
    l1, l2 = sp.sqrt(sp.simplify((P - Fp).dot(P - Fp))), sp.sqrt(sp.simplify((P - F).dot(P - F)))
    p1 = parts(md_answer(1, "1"))
    ok1 = all(abs(float((u - sp.sympify(p1[k], locals=loc)).subs(b, bv))) < 1e-12
              for k, u in (("l1*l2", l1 * l2), ("l1", l1), ("l2", l2)) for bv in (1, 2.7))
    mdP = sp.sympify(md_answer(1, "2"), locals=loc)
    ok2 = all(sp.simplify(P[i] - mdP[i]) == 0 for i in range(2))
    # (3) ∠F'FQ = 135°: Q = F + s(cos45°, sin45°), s>0
    Qs = F + s * sp.Matrix([1, 1]) / sp.sqrt(2)
    sv = [v for v in sp.solve(ell.subs({X: Qs[0], Y: Qs[1]}), s) if v.subs(b, 1) > 0][0]
    Q = sp.simplify(Qs.subs(s, sv))
    ang = sp.simplify((Fp - F).dot(Q - F) / ((Fp - F).norm() * (Q - F).norm()))
    l3, l4 = sp.sqrt(sp.simplify((Q - Fp).dot(Q - Fp))), sp.sqrt(sp.simplify((Q - F).dot(Q - F)))
    p3 = parts(md_answer(1, "3"))
    ok3 = sp.simplify(ang + sp.sqrt(2) / 2) == 0 and all(
        abs(float((u - sp.sympify(p3[k], locals=loc)).subs(b, bv))) < 1e-12 for k, u in (("l3", l3), ("l4", l4)) for bv in (1, 3.1))
    # (4) Q 에서의 접선과 x축의 교점 R, θ = ∠FQR
    Rx = sp.solve((Q[0] * X / (4 * b**2)).subs(Y, 0) - 1, X)[0]
    R = sp.Matrix([Rx, 0])
    v1, v2 = F - Q, R - Q
    tanth = sp.nsimplify(sp.N((abs(v1[0] * v2[1] - v1[1] * v2[0]) / v1.dot(v2)).subs(b, 1), 50), [sp.sqrt(6)])
    ok4 = sp.simplify(tanth - sp.sympify(md_answer(1, "4"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"l1l2={sp.simplify(l1*l2)} P={list(P)} l4={sp.radsimp(l4)} tan={tanth}")


def p02():
    import numpy as np
    xx = sp.symbols("x", real=True)
    # g = f + |cos f| (조건 (i)(ii) 에서 g-f = sqrt(1 - sin^2 f) >= 0) — 조건 성립 수치 확인
    gfun = lambda f: f + sp.Abs(sp.cos(f))
    okc = all(abs(float((gfun(sp.Float(v)) - v)**2 + sp.sin(v)**2 - 1)) < 1e-12 for v in (0.3, 1.9, -4.2))
    # (1) ∫_0^π 10x + |cos 10x| : cos10x 의 영점에서 구간을 나눠 부호별 적분
    cuts = [0] + [sp.pi / 20 + k * sp.pi / 10 for k in range(10)] + [sp.pi]
    I = sp.integrate(10 * xx, (xx, 0, sp.pi)) + sum(
        sp.Abs(sp.integrate(sp.cos(10 * xx), (xx, cuts[i], cuts[i + 1]))) for i in range(len(cuts) - 1))
    ok1 = sp.simplify(I - sp.sympify(md_answer(2, "1"))) == 0
    # (2) g(x)=x+|cos x|: x=π/2 좌·우 미분계수, (0,π)\{π/2} 에서 g'>0
    h = sp.symbols("h", real=True)
    g = gfun(xx)
    dq = (g.subs(xx, sp.pi / 2 + h) - g.subs(xx, sp.pi / 2)) / h
    dl, dr = sp.limit(dq, h, 0, "-"), sp.limit(dq, h, 0, "+")
    xs = np.linspace(1e-4, np.pi - 1e-4, 200001)
    gv = xs + np.abs(np.cos(xs))
    s2 = md_answer(2, "2")
    ok2 = dl != dr and np.all(np.diff(gv) > 0) and "미분가능하지 않" in s2 and "증가" in s2
    # (3) 방정식 g(x)-kx+π/2(k-1)=0 의 [0,π] 실근 개수(=1)인 k 를 격자로 판정해 md 집합과 비교
    S = sp.sympify(md_answer(2, "3"))
    xl = np.linspace(0, np.pi / 2 - 1e-6, 40001)
    xr = np.linspace(np.pi / 2 + 1e-6, np.pi, 40001)
    def has_root(xa, k):
        v = xa + np.abs(np.cos(xa)) - k * xa + np.pi / 2 * (k - 1)
        return bool(np.any(v <= 0) and np.any(v > 0))
    bad = []
    for k in np.linspace(0.005, 4, 1600):
        if min(abs(k - c) for c in (1 - 2 / np.pi, 1 + 2 / np.pi, 2, 1)) < 3e-3:
            continue
        one = not has_root(xl, k) and not has_root(xr, k)     # x=π/2 는 항상 근
        if one != bool(S.contains(sp.Float(k))):
            bad.append(k)
    ok3 = not bad and abs(np.pi / 2 + abs(np.cos(np.pi / 2)) - np.pi / 2) < 1e-12
    # (4) g(x)=x 이면 x = φ(f(x)), φ(u)=u+|cos u|. f'(a)=1/φ'(f(a)) = 2/3 ⇔ φ'(u)=3/2
    u = sp.symbols("u", real=True)
    cands = []
    for k in range(-2, 3):
        for expr, cond in ((1 - sp.sin(u), lambda w: sp.cos(w) > 0), (1 + sp.sin(u), lambda w: sp.cos(w) < 0)):
            for w in sp.solveset(sp.Eq(expr, sp.Rational(3, 2)), u, sp.Interval(2 * k * sp.pi, 2 * (k + 1) * sp.pi, False, True)):
                if cond(w):
                    a = w + sp.Abs(sp.cos(w))
                    if sp.pi / 2 < a < sp.pi + 1:
                        cands.append(sp.simplify(a))
    ok4 = len(cands) == 1 and sp.simplify(cands[0] - sp.sympify(md_answer(2, "4"))) == 0
    report(2, okc and ok1 and ok2 and ok3 and ok4,
           f"int={I}; g'(pi/2-)={dl} g'(pi/2+)={dr}; k-grid mismatch={len(bad)}; a={cands}")


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
