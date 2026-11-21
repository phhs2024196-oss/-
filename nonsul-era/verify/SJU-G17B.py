#!/usr/bin/env python3
"""SJU-G17B (세종대 2017학년도 수시 논술고사[B] 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G17B"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


import math


def p01():
    T = sp.symbols("T", positive=True)
    r = sp.Function("r")
    # 관계식 미분: 2 r r' = (t-1/3) r, r>0 → r' = t/2 - 1/6 ; r = t^2/4 - t/6 + k
    k = sp.symbols("k", positive=True)
    rp = sp.solve(sp.Eq(2 * r(t) * sp.Symbol("D"), (t - sp.Rational(1, 3)) * r(t)), sp.Symbol("D"))[0]
    r1 = sp.integrate(rp, (t, 0, t)) + k
    kk = [s for s in sp.solve(sp.Eq(r1.subs(t, 0)**2, sp.integrate(r1.subs(t, x), (x, 0, 1))), k) if s > 0][0]
    r1 = r1.subs(k, kk)
    # 원래 관계식이 모든 t 에서 성립하는지 확인
    rel = sp.simplify(r1**2 - (sp.integrate((x - sp.Rational(1, 3)) * r1.subs(t, x), (x, 0, t)) + sp.integrate(r1.subs(t, x), (x, 0, 1))))
    ok0 = rel == 0 and all(r1.subs(t, v) > 0 for v in [0, 0.33, 1, 5])
    v1 = abs(sp.diff(r1, t).subs(t, 2))
    ok1 = v1 == sp.Rational(md_answer(1, "1"))
    ok2 = r1.subs(t, 2) == sp.Rational(md_answer(1, "2"))
    # (3) 기하: O1=(0,r1), O2=(X,r2), |O1O2|=r1+r2, X>0 ; P 내분점 ; 접선 ⟂ O1O2, x절편
    r2 = 1 / (t**2 + 1)
    X = sp.sqrt((r1 + r2)**2 - (r1 - r2)**2)
    O1 = sp.Matrix([0, r1]); O2 = sp.Matrix([X, r2])
    P = O1 + r1 / (r1 + r2) * (O2 - O1)
    d = O2 - O1
    # 접선: d·(Y-P)=0, y=0 → x
    qx = sp.simplify((d[0] * P[0] + d[1] * P[1]) / d[0])
    speed = abs(sp.diff(qx, t).subs(t, 2))
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "3").replace(" ", "").split(";"))
    Pm, Qm = sp.sympify(parts["P"]), sp.sympify(parts["Q"])
    ok3 = (sp.simplify(P[0].subs(t, 2) - Pm[0]) == 0 and sp.simplify(P[1].subs(t, 2) - Pm[1]) == 0 and
           sp.simplify(qx.subs(t, 2) - Qm[0]) == 0 and sp.simplify(sp.nsimplify(speed) - sp.sympify(parts["speed"])) == 0
           and abs(float(speed) - float(sp.sympify(parts["speed"]))) < 1e-12)
    report(1, ok0 and ok1 and ok2 and ok3, f"r1={r1} |r1'(2)|={v1} P={[sp.nsimplify(c.subs(t,2)) for c in P]} q(2)={sp.nsimplify(qx.subs(t,2))} speed={float(speed):.12f}")


def p02():
    A = sp.Matrix([4, 4, 3]); n = sp.Matrix([1, 2, 2]); B = sp.Matrix([2, sp.Rational(3, 2), 2])
    s = sp.symbols("s")
    Hs = A + s * n
    H = Hs.subs(s, sp.solve(Hs.dot(n) - 9, s)[0])
    ok1 = list(H) == list(sp.sympify(md_answer(2, "1")))
    # (2) 평면 위 Q 의 매개화: B + u e1 + v e2 (정규직교), |Q-B|>=2 에서 |AQ| 최소 — 극좌표 격자 + 해석
    e1 = sp.Matrix([2, -1, 0]) / sp.sqrt(5); e2 = n.cross(e1) / n.norm()
    import numpy as np
    Af = np.array([float(v) for v in A]); Bf = np.array([float(v) for v in B])
    E1 = np.array([float(v) for v in e1]); E2 = np.array([float(v) for v in e2])
    th = np.linspace(0, 2 * np.pi, 7201)[:, None]; rr = np.linspace(2, 2.5, 201)[None, :]
    Q = Bf[None, None, :] + rr[..., None] * (np.cos(th)[..., None] * E1 + np.sin(th)[..., None] * E2)
    best = float(np.sqrt(((Q - Af)**2).sum(-1)).min())
    ok2 = abs(best - float(sp.sympify(md_answer(2, "2")))) < 1e-5
    # (3) [증명-수치확인] ℓ' 위 점과 B 의 거리 최소 > 2
    tt = sp.symbols("tt", real=True)
    L = sp.Matrix([2 + 2 * tt, -1 + 2 * tt, 6 - 3 * tt])
    Lp = L - ((L.dot(n) - 9) / n.dot(n)) * n
    d2 = sp.expand((Lp - B).dot(Lp - B))
    m = sp.minimum(d2, tt)
    ok3 = sp.simplify(sp.diff(L, tt).dot(n)) == 0 and m > 4 and md_answer(2, "3") is None
    report(2, ok1 and ok2 and ok3, f"H={list(H)} min|AQ|~{best:.6f} [증명-수치확인] min d(B,l')^2={m}")


def p03():
    a, b = sp.symbols("a b", nonzero=True)
    sol = sp.solve([sp.Eq(a, 2), sp.Eq(a * b, 5 * a)], [a, b], dict=True)[0]
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "1").replace(" ", "").split(";"))
    ok1 = sol[a] == sp.sympify(parts["a"]) and sol[b] == sp.sympify(parts["b"])
    import random
    import mpmath as mp
    random.seed(1)
    # (2) 상계 2 : 무작위 p(x)=2cos(φ(x)), q=2sin(φ), φ∈[0,π/2]
    vals = []
    for _ in range(200):
        c0, c1, c2 = random.random(), random.random(), random.random()
        phi = lambda X: (math.pi / 2) * (c0 + c1 * X + c2 * X * X) / (c0 + c1 + c2 + 1e-9) % (math.pi / 2)
        vals.append(mp.quad(lambda X: 2 * mp.cos(phi(X)) * 2 * mp.sin(phi(X)), [0, 1]))
    att2 = sp.integrate(sp.sqrt(2) * sp.sqrt(2), (x, 0, 1))
    ok2 = max(vals) <= 2 + 1e-12 and att2 == sp.sympify(md_answer(3, "2"))
    # (3) 코시–슈바르츠 상계 sqrt(6*∫(x^3+1)) 와 등호 함수
    W = sp.integrate(x**3 + 1, (x, 0, 2))
    g = sp.sqrt(6 / W) * sp.sqrt(x**3 + 1)
    att3 = sp.integrate(g * sp.sqrt(x**3 + 1), (x, 0, 2))
    ok3 = sp.simplify(sp.integrate(g**2, (x, 0, 2)) - 6) == 0 and sp.simplify(att3 - sp.sqrt(6 * W)) == 0 \
        and sp.simplify(att3 - sp.sympify(md_answer(3, "3"))) == 0
    # (4) h=2e^{cx}(c>0) 족에서 (다)=6 인 k 를 풀고 h(k) 최대화 + 다른 꼴(2+cx^2 …) 비교
    def hk_exp(c):
        k = mp.findroot(lambda K: mp.quad(lambda X: 9 * (2 * mp.e**(c * X))**2 + 4 * (2 * c * mp.e**(c * X))**2, [0, K]) - 6, 0.05)
        return 2 * mp.e**(c * k)
    best = max(hk_exp(c) for c in [0.2 * i for i in range(1, 30)])
    def hk_poly(c):
        h = lambda X: 2 + c * X**2
        hp = lambda X: 2 * c * X
        k = mp.findroot(lambda K: mp.quad(lambda X: 9 * h(X)**2 + 4 * hp(X)**2, [0, K]) - 6, 0.05)
        return h(k)
    bp = max(hk_poly(c) for c in [0.5 * i for i in range(1, 30)])
    target = float(sp.sympify(md_answer(3, "4")))
    eq = hk_exp(1.5)
    ok4 = abs(eq - target) < 1e-12 and best <= target + 1e-12 and bp <= target + 1e-12
    report(3, ok1 and ok2 and ok3 and ok4, f"a,b={sol} max∫pq~{float(max(vals)):.6f} (3)={att3} (4) h(k) at c=3/2: {float(eq):.12f}, family max {float(best):.6f}, poly max {float(bp):.6f}")


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
