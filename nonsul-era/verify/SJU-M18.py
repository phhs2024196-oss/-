#!/usr/bin/env python3
"""SJU-M18 (세종대 2018학년도 논술 모의평가(자연계열)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "SJU-M18"
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
    a, b, c, fv = sp.symbols("a b c fv")
    f = a * sp.exp(-x) + b * x + c
    cdf = lambda v: 2 * v / (1 + v)
    fm1 = sp.solve(sp.Eq(cdf(fv), 0), fv); fp1 = sp.solve(sp.Eq(cdf(fv), 1), fv)
    d = md_parts(1, "1")
    ok1 = fm1 == [S(d["f(-1)"])] and fp1 == [S(d["f(1)"])]
    s2 = sp.solve([f.subs(x, -1) - fm1[0], f.subs(x, 1) - fp1[0], c - sp.Rational(1, 2)], [a, b, c], dict=True)[0]
    P2 = sp.simplify(cdf(f.subs(s2).subs(x, sp.Rational(1, 2))))
    ok2 = P2 == S(md_answer(1, "2"))
    import numpy as np
    sab = sp.solve([f.subs(x, -1) - fm1[0], f.subs(x, 1) - fp1[0]], [a, b], dict=True)[0]
    F = sp.lambdify((x, c), cdf(f.subs(sab)))
    xs = np.linspace(-1, 1, 4001)
    valid = []
    for cv in np.linspace(-1, 3, 40001):
        fv_ = sp.lambdify((x, c), f.subs(sab))(xs, cv)
        if np.any(np.abs(1 + fv_) < 1e-9):
            continue
        Fx = F(xs, cv)
        if np.all(np.diff(Fx) >= -1e-13):
            valid.append(cv)
    d3 = md_parts(1, "3")
    ok3 = abs(min(valid) - float(S(d3["min"]))) < 2e-4 and abs(max(valid) - float(S(d3["max"]))) < 2e-4
    report(1, ok1 and ok2 and ok3, f"f(-1)={fm1} f(1)={fp1} P={P2} c∈[{min(valid):.4f},{max(valid):.4f}]")


def p02():
    import numpy as np
    r_, xa = sp.symbols("r xa", positive=True)
    sol = sp.solve([xa**2 + r_**2 - 5, (5 - xa)**2 + r_**2 - 10], [xa, r_], dict=True)
    ok1 = len(sol) == 1 and sol[0][r_] == S(md_answer(2, "1"))
    xo, R = float(sol[0][xa]), float(sol[0][r_])
    A = np.array([0., 0, 0]); B = np.array([5., 0, 0]); O = np.array([xo, 0, 0])
    V = [O + R * np.array([0, np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)]) for k in range(6)]
    def perim(m):
        pts = []
        for i in range(6):
            s = np.linspace(0, 1, m, endpoint=False)
            pts.append(V[i][None, :] * (1 - s[:, None]) + V[(i + 1) % 6][None, :] * s[:, None])
        return np.vstack(pts)
    Pp = perim(600)
    best = max(np.linalg.norm((Pp - A) + (q - B), axis=1).max() for q in Pp[::3])
    ok2 = abs(best - float(S(md_answer(2, "2")))) < 1e-6
    P = (V[0] + 2 * V[1]) / 3            # 한 변을 2:1 로 내분
    best3 = np.linalg.norm((P - A)[None, :] + (perim(20000) - B), axis=1).max()
    P_ = (2 * V[0] + V[1]) / 3           # 반대 방향 내분도 대칭으로 같음
    best3b = np.linalg.norm((P_ - A)[None, :] + (perim(20000) - B), axis=1).max()
    ok3 = abs(best3 - float(S(md_answer(2, "3")))) < 1e-9 and abs(best3b - best3) < 1e-9
    report(2, ok1 and ok2 and ok3, f"r={sol[0][r_]} max={best:.8f} max(2:1)={best3:.10f}")


def p03():
    import numpy as np
    h = x * sp.exp(-x)
    h1, h2 = sp.diff(h, x), sp.diff(h, x, 2)
    cx = sp.solve(h1, x); ix = sp.solve(h2, x)
    d = md_answer(3, "1")
    ok1 = cx == [1] and ix == [2] and sp.simplify(S(d.split(";")[0].split("=")[1].split(" (")[0]) - h.subs(x, 1)) == 0 \
        and sp.Matrix(S(d.split("inflection=")[1])) == sp.Matrix([2, h.subs(x, 2)]) and h2.subs(x, 1) < 0
    # (2) t e^{-t}=1/e 의 실근 개수: 수치
    ts = np.linspace(-5, 30, 3500001)
    hv = ts * np.exp(-ts)
    ok2 = abs(hv.max() - np.exp(-1)) < 1e-12 and abs(ts[hv.argmax()] - float(S(md_answer(3, "2")))) < 1e-5
    # (3) f=x^4+px^3+qx^2+rx+1 (f(0)=1) 격자 전수: g''=f'(1-f) 의 실근이 1개 ⇔ 답의 꼴
    fam = md_answer(3, "3").split(";")
    bb, cc = sp.symbols("b c")
    f2 = S(fam[1].split(" (")[0], b=bb, c=cc)
    def nreal(coef):
        fp = np.polyder(coef); one_minus = -np.array(coef); one_minus[-1] += 1
        rts = np.concatenate([np.roots(fp), np.roots(np.trim_zeros(one_minus, "f"))])
        re_ = np.unique(np.round(rts[np.abs(rts.imag) < 1e-6].real, 6))
        return len(re_)
    vals = [Fraction(v, 2) for v in range(-6, 7)]
    ok3 = True; hits = 0
    for p_, q_, r_ in itertools.product(vals, repeat=3):
        coef = [1, float(p_), float(q_), float(r_), 1]
        good = nreal(coef) == 1
        # 답의 꼴: r=0 이고 p=4b/3, q=2c 로 b^2<4c 또는 b=c=0
        bv, cv = Fraction(3, 4) * p_, q_ / 2
        inform = r_ == 0 and (bv * bv < 4 * cv or (bv == 0 and cv == 0))
        ok3 &= good == inform
        hits += good
    ok3 &= S(fam[0]) == x**4 + 1
    # (4)
    integ = sp.integrate(f2, (x, -1, 1))
    best = min(float(integ.subs({bb: B_, cc: C_})) for B_ in np.linspace(-2, 2, 81) for C_ in np.linspace(1e-6, 2, 81) if B_**2 < 4 * C_)
    v0 = sp.integrate(x**4 + 1, (x, -1, 1))
    ok4 = v0 == S(md_answer(3, "4")) and best > float(v0) and sp.simplify(integ - v0 - sp.Rational(4, 3) * cc) == 0
    report(3, ok1 and ok2 and ok3 and ok4, f"극대 x={cx} 변곡 x={ix} f(0)=1, 격자 {len(vals)**3}개 중 조건 만족 {hits}개 모두 답의 꼴, min∫={v0}")


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
