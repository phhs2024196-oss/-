#!/usr/bin/env python3
"""SJU-G16 (세종대 2016학년도 수시 논술고사 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G16"
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


def p01():
    a, b = sp.symbols("a b", real=True)
    V = sp.Function("V")
    sol = sp.dsolve(sp.Eq(V(t).diff(t), a * V(t) + b), V(t), ics={V(0): 0}).rhs     # V=(b/a)(e^{at}-1)
    A = sp.symbols("A", positive=True)   # e^a
    Vs = (b / a) * (A**t - 1)
    r = sp.solve([sp.Eq(Vs.subs(t, 1), 1), sp.Eq(Vs.subs(t, 2), 3)], [A, b], dict=True)
    ok_ode = sp.simplify(sol - (b / a) * (sp.exp(a * t) - 1)) == 0
    Aval = [s for s in r if s[A] > 0][0][A]
    aval = sp.log(Aval)
    bval = sp.simplify([s for s in r if s[A] > 0][0][b].subs(a, aval))
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "2").replace(" ", "").split(";"))
    ok2 = sp.simplify(aval - sp.sympify(parts["a"])) == 0 and sp.simplify(bval - sp.sympify(parts["b"])) == 0
    # (3) md 의 f 로 부피를 직접 적분해 V(t)=2^t-1 재현 + f(0)=0, f'>0, f>0 [증명-수치확인]
    yv = sp.symbols("y", positive=True)
    fm = sp.sympify(md_answer(1, "3"), locals={"x": yv})
    Vt = sp.integrate(sp.pi * fm**2, (yv, 0, sp.sqrt(sp.Symbol("T", positive=True))))
    T = sp.Symbol("T", positive=True)
    ok3 = sp.simplify(Vt - (Aval**T - 1)) == 0
    import mpmath as mp
    fl = sp.lambdify(yv, fm, "mpmath")
    ok1 = all(fl(v) > 0 and mp.diff(fl, v) > 0 for v in [0.01, 0.3, 1, 2, 3])
    report(1, ok_ode and ok1 and ok2 and ok3, f"V={sol} e^a={Aval} a={aval} b={bval} V(T) from f = {sp.simplify(Vt)} [증명-수치확인] f>0,f'>0")


def p02():
    th, w = sp.symbols("theta w", positive=True)
    # β = xy평면, L = x축, α: (X, Y, Y tanθ) (Y≥0). 정사영 정삼각형 AB'C' (한 변 d), B' 각 w, C' 각 w+π/3
    def tri(d, wv, tv):
        Bp = sp.Matrix([d * sp.cos(wv), d * sp.sin(wv), 0]); Cp = sp.Matrix([d * sp.cos(wv + sp.pi / 3), d * sp.sin(wv + sp.pi / 3), 0])
        lift = lambda P: sp.Matrix([P[0], P[1], P[1] * sp.tan(tv)])
        return sp.Matrix([0, 0, 0]), lift(Bp), lift(Cp), Bp, Cp
    A, B, C, Bp, Cp = tri(2, w, th)
    # (1) 직선 B'C' 와 L 의 예각
    dvec = Cp - Bp
    sinu = sp.simplify(sp.Abs(dvec[1]) / dvec.norm()).subs(w, sp.asin(sp.Rational(1, 3)))
    ok1 = abs(float(sinu) - float(sp.sympify(md_answer(2, "1")))) < 1e-14
    AB2 = sp.simplify((B - A).dot(B - A))
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "2").replace(" ", "").split(";"))
    loc = {"theta": th, "w": w}
    ok2 = sp.simplify(AB2 - sp.sympify(parts["AB2"], locals=loc)) == 0 and \
        sp.simplify(sp.sqrt(AB2.subs({th: sp.pi / 3, w: sp.pi / 6})) - sp.sympify(parts["AB"])) == 0
    tot = sp.simplify(sum(((P - Q).dot(P - Q) for P, Q in [(B, A), (C, B), (A, C)])))
    p3 = dict(seg.split("=", 1) for seg in md_answer(2, "3").replace(" ", "").split(";"))
    ok3 = sp.simplify(tot - sp.sympify(p3["f"], locals=loc)) == 0 and sp.simplify(tot.subs(th, sp.pi / 4) - sp.sympify(p3["f(pi/4)"])) == 0
    # (4) 수치: d, θ 를 풀고(넓이 1, 합 8) w 무관 확인
    import mpmath as mp
    def eqs(d, tv, wv=0.37):
        A_, B_, C_, Bp_, Cp_ = tri(d, wv, tv)
        area = (B_ - A_).cross(C_ - A_).norm() / 2
        s = sum(((P - Q).dot(P - Q) for P, Q in [(B_, A_), (C_, B_), (A_, C_)]))
        return [float(area - 1), float(s - 8)]
    sols = set()
    for d0 in [0.8, 1.0, 1.2, 1.6, 2.0]:
        for t0 in [0.3, 0.7, 1.0, 1.3]:
            try:
                r = mp.findroot(lambda d, tv: eqs(d, tv), (d0, t0))
                if 0 < r[1] < mp.pi / 2 and r[0] > 0:
                    sols.add(round(float(r[0]), 10))
            except Exception:
                pass
    ok4 = len(sols) == 1 and abs(sols.pop() - float(sp.sympify(md_answer(2, "4")))) < 1e-9
    report(2, ok1 and ok2 and ok3 and ok4, f"sin u={float(sinu):.12f} AB^2={AB2} sum={tot} (4) ok={ok4}")


def p03():
    import mpmath as mp
    mp.mp.dps = 30
    def shadow(tv):
        icpts = [1 - 1 / mp.sin(tv), 5 - 3 / mp.sin(tv)]          # 중심-반지름/sin t (접선 x절편)
        return max(0, -min(icpts))
    ok1 = abs(shadow(mp.pi / 4) - mp.mpf(str(sp.N(sp.sympify(md_answer(3, "1")), 35)))) < 1e-25
    ans = md_answer(3, "2").replace(" ", "")
    pieces = re.findall(r"([^,;=]+?)\((0<t<=pi/6|pi/6<t<pi/2)\)", ans)
    g1 = sp.sympify(pieces[0][0], locals={"t": t}); g2 = sp.sympify(pieces[1][0], locals={"t": t})
    ok2 = all(abs(shadow(v) - (g1 if v <= mp.pi / 6 else g2).subs(t, v)) < 1e-20 for v in [mp.mpf(k) / 100 for k in range(1, 157)])
    # 미분불가능점: 좌우 수치미분 비교로 격자 탐색
    bad = []
    for k in list(range(1, 157)) + ["pi6"]:
        v = mp.pi / 6 if k == "pi6" else mp.mpf(k) / 100
        hh = mp.mpf("1e-14")
        L = (shadow(v) - shadow(v - hh)) / hh; R = (shadow(v + hh) - shadow(v)) / hh
        if abs(L - R) > 1e-6 * (1 + abs(L)):
            bad.append(v)
    tm = sp.sympify(ans.split(";")[1].split("=")[1])
    hh = mp.mpf("1e-12"); tv = mp.mpf(float(tm))
    jump = abs((shadow(tv) - shadow(tv - hh)) / hh - (shadow(tv + hh) - shadow(tv)) / hh)
    ok2 = ok2 and jump > 1 and all(abs(b - tv) < 1e-12 for b in bad) and len(bad) == 1
    # (3) h(4): shadow(t)=4 인 t, h'(4)=1/g'(h(4))
    t4 = mp.findroot(lambda z: shadow(z) - 4, 0.3)
    hp = 1 / mp.diff(shadow, t4)
    ok3 = abs(hp - float(sp.sympify(md_answer(3, "3")))) < 1e-15
    report(3, ok1 and ok2 and ok3, f"g(pi/4)={float(shadow(mp.pi/4)):.12f} nondiff~{[float(b) for b in bad]} h'(4)={float(hp):.15f}")


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
