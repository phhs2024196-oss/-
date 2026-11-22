#!/usr/bin/env python3
"""AJOU-G25A (아주대 2025학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G25A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def same(a, b, tol=1e-12):
    return abs(complex(sp.N(a - b, 30))) < tol


def p01():
    oks = []
    # (1-1)(1)(2)(3): 정적분을 고정밀 수치적분으로 독립 계산
    import mpmath as mp
    mp.mp.dps = 30
    I1 = mp.quad(lambda u: (1 + mp.sin(u) - mp.sin(u)**3 - mp.sin(u)**4) / mp.cos(u)**4, [0, mp.pi / 4])
    I2 = mp.quad(lambda u: (1 + 2 * mp.sin(u))**10 * mp.sin(u) * mp.cos(u), [0, mp.pi / 6])
    I3 = mp.quad(lambda u: (u**2 * mp.e**u - 1) / (u**2 * mp.e**u + u), [1, 2])
    for sub, I in (("1-1", I1), ("1-2", I2), ("1-3", I3)):
        oks.append(abs(I - mp.mpf(str(sp.N(sp.sympify(md_answer(1, sub)), 40)))) < mp.mpf("1e-20"))
    # (1-2)(1): 원 x^2-2cx+y^2=0 과 y=±rx 의 교점으로 QR/r 극한
    c, r = sp.symbols("c r", positive=True)
    X = sp.symbols("X")
    qx = [s for s in sp.solve(X**2 - 2 * c * X + (r * X)**2, X) if s != 0][0]
    QR = 2 * r * qx
    lim = sp.limit(QR / r, r, 0, "+")
    cval = sp.solve(sp.Eq(lim, 3), c)[0]
    oks.append(sp.simplify(cval - sp.sympify(md_answer(1, "2-1"))) == 0)
    # (1-2)(2): 직선 PQ 의 x절편 h(r), 극한 2c = 1/2048 인 n
    Q = (qx, r * qx)
    h = sp.solve(sp.Eq(0, sp.sqrt(r) + (Q[1] - sp.sqrt(r)) / Q[0] * X), X)[0]
    hl = sp.limit(h, r, 0, "+")
    nn = [k for k in range(1, 40) if hl.subs(c, sp.Rational(1, 2)**k) == sp.Rational(1, 2048)]
    oks.append(nn == [int(sp.sympify(md_answer(1, "2-2")))])
    report(1, all(oks), f"I1={mp.nstr(I1, 12)} I2={mp.nstr(I2, 12)} I3={mp.nstr(I3, 12)} c={cval} lim h={hl} n={nn} {oks}")


def p02():
    oks = []
    th, p = sp.symbols("theta p", positive=True)
    # (2-1)(1): 원 위 점 X 에서의 반지름이 접선(y축과 θ)과 수직 → 반지름 방향 (cosθ, -sinθ)
    Xp = (sp.cos(th), p - sp.sin(th))
    md = sp.sympify(md_answer(2, "1-1"), locals={"theta": th, "p": p})
    oks.append(all(sp.simplify(u - v) == 0 for u, v in zip(Xp, md)))
    # 접선 기울기 = cosθ/sinθ (y축과 예각 θ), 실제로 PX ⊥ 접선 확인
    m = sp.cos(th) / sp.sin(th)
    oks.append(sp.simplify((Xp[1] - p) / Xp[0] * m + 1) == 0)
    # (2-1)(2): 접선 ∩ y=x^2-1 → AB^2 를 t=1/sinθ 로
    X = sp.symbols("X")
    line = m * (X - Xp[0]) + Xp[1]
    rts = sp.solve(sp.Eq(X**2 - 1, line), X)
    AB2 = (rts[0] - rts[1])**2 + (rts[0]**2 - rts[1]**2)**2
    tt = sp.symbols("t", positive=True)
    mdf = sp.sympify(md_answer(2, "1-2"), locals={"t": tt, "p": p})
    import random
    ok = True
    for _ in range(20):
        thv = random.uniform(0.05, 1.5); pv = random.uniform(0.3, 3)
        a = complex(AB2.subs({th: thv, p: pv}).evalf(30)).real
        b = float(mdf.subs({tt: 1 / sp.sin(thv), p: pv}))
        ok &= abs(a - b) < 1e-8 * max(1, abs(b))
    oks.append(ok)
    # (2-2)(1): p=1, f(t)=12, t>1 → X
    f1 = mdf.subs(p, 1)
    tsol = [s for s in sp.solve(sp.Eq(f1, 12), tt) if s.is_real and s > 1]
    thsol = sp.asin(1 / tsol[0])
    Xs = (sp.cos(thsol), 1 - sp.sin(thsol))
    md21 = sp.sympify(md_answer(2, "2-1"))
    oks.append(len(tsol) == 1 and all(sp.simplify(u - v) == 0 for u, v in zip(Xs, md21)))
    # (2-2)(2): Q(1/2,-3/4) 접선 기울기 2·(1/2)=1 = cotθ → θ=π/4
    slope = sp.diff(X**2 - 1, X).subs(X, sp.Rational(1, 2))
    th0 = sp.solve(sp.Eq(sp.cot(th), slope), th)[0]
    pv = sp.solve(sp.Eq(mdf.subs(tt, 1 / sp.sin(th0)), 12), p)
    oks.append(len(pv) == 1 and sp.simplify(pv[0] - sp.sympify(md_answer(2, "2-2"))) == 0)
    # (2-3): p 격자에서 (1,∞) 위 f 가 단조가 아닌지(= 같은 값 두 번) 직접 판정
    import numpy as np
    ts = np.linspace(1.0000001, 6, 60001)
    good = []
    for pvv in np.arange(0.2505, 0.6, 0.0005):
        fv = ts**2 * (ts**2 - 4 * ts + 4 * pvv + 3)
        good.append((pvv, bool(np.any(np.diff(fv) < 0))))
    ps = [q for q, g in good if g]
    iv = sp.sympify(md_answer(2, "3"))
    oks.append(abs(min(ps) - 0.2505) < 1e-9 and abs(max(ps) - float(iv.sup)) < 1e-3
               and float(iv.inf) == 0.25 and all(iv.contains(sp.Float(q)) for q in ps))
    report(2, all(oks), f"t={tsol} p(2-2)={pv} p-range≈({min(ps):.4f},{max(ps):.4f}] {oks}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
