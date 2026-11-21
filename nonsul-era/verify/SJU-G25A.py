#!/usr/bin/env python3
"""SJU-G25A (세종대 2025학년도 수시 논술고사 A형) — sympy/수치 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-G25A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {k.strip(): sp.sympify(v.strip()) for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, b, t, y = sp.symbols("a b t y", positive=True)
    f = x * (x - a) * (x - b)                                            # (가) f(0)=f(a)=0
    bs = sp.solve(sp.integrate(f, (x, 0, a)), b)[0]                      # (나)
    f = sp.expand(f.subs(b, bs))
    cps = sp.solve(sp.diff(f, x), x)
    mx = [c for c in cps if sp.simplify(sp.diff(f, x, 2).subs(x, c)) .subs(a, 1) < 0][0]
    v1 = sp.simplify(f.subs(x, mx))
    ok1 = sp.simplify(v1 - sp.sympify(md_answer(1, "1"), locals={"a": a})) == 0
    X = sp.symbols("X", real=True)
    g = sp.integrate(t * f.subs(x, X - t), (t, 0, X))                     # 정의 그대로 적분
    v2 = sp.simplify(sp.diff(g, X, 2).subs(X, -a))
    ok2 = sp.simplify(v2 - sp.sympify(md_answer(1, "2"), locals={"a": a})) == 0
    # (3) 여러 a 에서 p(x)=e^{h(x)}+e^{h(a-x)} 를 격자 최소화
    ok3 = True
    md3 = sp.sympify(md_answer(1, "3"), locals={"a": a})
    for av in (0.4, 1.0, 2.0, 3.0, 4.5):
        h = sp.lambdify(x, (f + x).subs(a, av), "numpy")
        xs = np.linspace(-av - 1, 2 * av + 1, 400001)
        pv = np.exp(h(xs)) + np.exp(h(av - xs))
        ok3 &= abs(pv.min() - float(md3.subs(a, av))) < 1e-6 * float(md3.subs(a, av))
    report(1, ok1 and ok2 and ok3, f"b={bs} max={v1} g''(-a)={v2} p_min=2e^(a/2) 수치 일치")


def p02():
    def area(xa, xb):                                                    # 영역을 x 로 직접 적분
        ya, yb = 1 / xa, 1 / xb
        return mp.quad(lambda u: ya / xa * u - yb / xb * u, [0, xa]) + mp.quad(lambda u: 1 / u - yb / xb * u, [xa, xb])
    F = sp.Function("f")
    T = sp.symbols("t", real=True)
    md1 = dict(seg.split("=", 1) for seg in md_answer(2, "1").replace(" ", "").split(";"))
    Sexpr = sp.sympify(md1["S(t)"], locals={"f": F, "t": T})
    ok1 = True
    for xa, xb in [(0.3, 0.9), (1.3, 4.1), (2.0, 2.5)]:
        val = Sexpr.subs({F(T + 1): xb, F(T): xa})
        ok1 &= abs(area(mp.mpf(xa), mp.mpf(xb)) - float(val)) < 1e-12
    S2 = area(mp.e**8, mp.e**27)
    ok1 &= abs(S2 - float(sp.sympify(md1["S(2)"]))) < 1e-9
    # (2) S=2, f(0)=2 → f(n+1)/f(n)=e^2 를 넓이 공식 없이: area(f(n),r f(n))=2 를 r 에 대해 풀기
    r = mp.findroot(lambda r: area(mp.mpf(2), 2 * r) - 2, 5)
    ssum = mp.nsum(lambda n: 1 / (2 * r**n), [1, mp.inf])
    ok2 = abs(ssum - float(sp.sympify(md_answer(2, "2")))) < 1e-12
    # (3) f'/f 상수 k: f(0)=2, f'(0)=3 → f=2e^{kt} (dsolve), S 상수·부등식 확인
    k = sp.Rational(3, 2)
    sol = sp.dsolve(sp.Eq(F(T).diff(T), sp.Rational(3, 2) * F(T)), F(T), ics={F(0): 2}).rhs
    ok3 = sp.simplify(sol.diff(T).subs(T, 0) - 3) == 0 and sp.simplify(sp.log(sol.subs(T, T + 1) / sol)).is_constant()
    ok3 &= sp.simplify(sol.diff(T, 2) * sol - sol.diff(T)**2) >= 0
    ok3 &= sp.simplify(sol.subs(T, sp.Rational(1, 2)) - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and ok2 and ok3, f"S(2)={float(S2):.10f} ratio={float(r):.6f} sum={float(ssum):.10f} f={sol}")


def clip_below(poly, yl=1.0):
    """다각형을 y<=yl 로 자른 다각형(Sutherland–Hodgman)."""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        pin, qin = P[1] <= yl, Q[1] <= yl
        if pin:
            out.append(P)
        if pin != qin:
            s = (yl - P[1]) / (Q[1] - P[1])
            out.append((P[0] + s * (Q[0] - P[0]), yl))
    return out


def shoelace(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2


def p03():
    th = sp.symbols("theta", real=True)
    md1 = {k.strip(): sp.sympify(v, locals={"theta": th}) for k, v in (s.split("=", 1) for s in md_answer(3, "1").split(";"))}
    ok1 = True
    for tv in np.linspace(0, 1.5, 16):
        A = np.array([float(md1["Ax"].subs(th, tv)), float(md1["Ay"].subs(th, tv))])
        B = np.array([float(md1["Bx"].subs(th, tv)), float(md1["By"].subs(th, tv))])
        ok1 &= abs(np.linalg.norm(A) - 2 * np.sqrt(2)) < 1e-12 and abs(np.linalg.norm(B) - 2) < 1e-12 and abs(np.linalg.norm(A - B) - 2) < 1e-12
        ok1 &= abs(np.arctan2(B[1], B[0]) - tv) < 1e-12 and (B[0] * A[1] - B[1] * A[0]) > 0          # A 는 OB 의 반시계 쪽
    # D 가 선분 OB 위: 삼각형과 y=1 교점 중 오른쪽 점이 OB 위에 있는 θ 를 격자로
    def CD(tv):
        A = (2 * np.sqrt(2) * np.cos(tv + np.pi / 4), 2 * np.sqrt(2) * np.sin(tv + np.pi / 4))
        B = (2 * np.cos(tv), 2 * np.sin(tv))
        pts = []
        for P, Q in [((0, 0), A), (A, B), (B, (0, 0))]:
            if (P[1] - 1) * (Q[1] - 1) <= 0 and P[1] != Q[1]:
                s = (1 - P[1]) / (Q[1] - P[1]); pts.append((P[0] + s * (Q[0] - P[0]), (P, Q)))
        pts.sort(key=lambda z: z[0])
        return pts[0][0], pts[-1][0], pts[-1][1], A, B
    ths = np.sort(np.append(np.linspace(0, np.pi / 2 - 1e-6, 200001), np.pi / 6 + 1e-13))
    onOB = [tv for tv in ths if CD(tv)[2][1] == (0, 0) or CD(tv)[2][0] == (0, 0)]
    ok1 &= abs(min(onOB) - float(md1["theta_min"])) < 1e-4 and abs(max(onOB) - float(md1["theta_sup"])) < 1e-4
    # (2) md f(t) 가 실제 CD 와 같은지 + 최소·최대
    md2 = dict((k.strip(), v) for k, v in (s.split("=", 1) for s in md_answer(3, "2").split(";")))
    T = sp.symbols("t", positive=True)
    ft = sp.lambdify(T, sp.sympify(md2["f(t)"], locals={"t": T}))
    lens = np.array([CD(tv)[1] - CD(tv)[0] for tv in onOB])
    ok2 = all(abs(ft(np.tan(tv)) - (CD(tv)[1] - CD(tv)[0])) < 1e-9 for tv in onOB[::5000])
    ok2 &= abs(lens.min() - float(sp.sympify(md2["min"]))) < 1e-8 and abs(lens.max() - float(sp.sympify(md2["max"]))) < 1e-8
    # (3) 다각형 자르기로 a, b 직접 계산
    prods = []
    for tv in ths[::20]:
        _, _, _, A, B = CD(tv)
        tri = [(0.0, 0.0), B, A]
        bb = shoelace(clip_below(tri)); aa = shoelace(tri) - bb
        prods.append(aa * bb)
    prods = np.array(prods)
    md3 = parts(md_answer(3, "3"))
    ok3 = abs(prods.min() - float(md3["min"])) < 1e-6 and abs(prods.max() - float(md3["max"])) < 1e-6
    report(3, ok1 and ok2 and ok3, f"theta∈[{min(onOB):.5f},{max(onOB):.5f}) CD∈[{lens.min():.8f},{lens.max():.8f}] ab∈[{prods.min():.7f},{prods.max():.7f}]")


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
