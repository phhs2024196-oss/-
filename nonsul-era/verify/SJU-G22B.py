#!/usr/bin/env python3
"""SJU-G22B (세종대 2022학년도 수시 논술고사 [B]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G22B"
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


def parts(s: str) -> dict:
    return dict(seg.strip().split("=", 1) for seg in s.split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    l = sp.symbols("l", positive=True)
    # 좌표로 직접 구성: B=(0,0), C=(l,0), A 는 B 에서 π/4 방향 거리 2√2 l
    B = sp.Matrix([0, 0]); C = sp.Matrix([l, 0]); A = 2 * sp.sqrt(2) * l * sp.Matrix([sp.cos(sp.pi / 4), sp.sin(sp.pi / 4)])
    CA, CB = A - C, B - C
    cosC = sp.simplify(CA.dot(CB) / (CA.norm() * CB.norm()))
    sinC = sp.simplify(sp.sqrt(1 - cosC**2))
    ok1 = sp.simplify(sinC - sp.sympify(md_answer(1, "1"))) == 0 and cosC < 0
    area = sp.simplify(sp.Abs((A - B)[0] * (C - B)[1] - (A - B)[1] * (C - B)[0]) / 2)   # = l^2
    S = sp.integrate(sp.Rational(4, 3) * t + 6, (t, 0, t)) + area.subs(l, 1)          # S(0) = 넓이(ℓ=1)
    lt = sp.sqrt(S / area.subs(l, 1))                                                  # 넓이 ∝ ℓ^2
    lim = sp.limit(lt / t, t, sp.oo)
    ok2 = sp.simplify(lim - sp.sympify(md_answer(1, "2"))) == 0
    # 외심: 수직이등분선 교점
    X, Y = sp.symbols("X Y")
    sol = sp.solve([(X - A[0])**2 + (Y - A[1])**2 - X**2 - Y**2, (X - C[0])**2 + (Y - C[1])**2 - X**2 - Y**2], [X, Y], dict=True)[0]
    Z = sp.Matrix([sol[X], sol[Y]])
    poly = [Z, B, C, A]                                                                # 사각형 ZBCA (신발끈)
    T = sp.Abs(sum(poly[i][0] * poly[(i + 1) % 4][1] - poly[(i + 1) % 4][0] * poly[i][1] for i in range(4))) / 2
    T = sp.simplify(T)
    t0 = [r for r in sp.solve(sp.Eq(lt, 5), t) if r >= 0][0]
    Tt = sp.simplify(T.subs(l, lt))
    rate = sp.simplify(sp.diff(Tt, t).subs(t, t0))
    ok3 = sp.simplify(rate - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"sinC={sinC} cosC={cosC} lim={lim} t0={t0} T/l^2={sp.simplify(T/l**2)} T'={rate}")


def p02():
    a, b, c = sp.symbols("a b c")
    g = x**3 + a * x**2 + b * x + c
    eqs = [sp.Eq(g.subs(x, 2), 4 * g.subs(x, 1)), sp.Eq(sp.diff(g, x).subs(x, 2), 2 * sp.diff(g, x).subs(x, 1)),
           sp.Eq(sp.integrate(g, (x, 1, 2)), 6)]
    sol = sp.solve(eqs, [a, b, c], dict=True)[0]
    G = g.subs(sol)
    ok3 = sp.expand(G - sp.sympify(md_answer(2, "3"), locals={"x": x})) == 0

    # f 를 실제로 구성: [1,2) 에서 G, 그 밖은 f(x)=4^k G(x/2^k)
    def f(v):
        v = sp.nsimplify(v)
        if v == 0:
            return 0
        k = sp.floor(sp.log(v, 2))
        return 4**k * G.subs(x, v / 2**k)
    import mpmath as mp
    mp.mp.dps = 30
    fn = lambda v: float(f(v)) if v > 0 else 0.0
    # 경계 미분가능성(좌우 도함수) 수치 확인
    h = 1e-7
    okd = abs((fn(2 + h) - fn(2)) / h - (fn(2) - fn(2 - h)) / h) < 1e-4
    # (2-1) 직접 수치 적분
    I1 = mp.quad(lambda u: 2**u * fn(2**u), [1, 1.5, 2, 2.5, 3])
    ok1 = abs(I1 - float(sp.sympify(md_answer(2, "1")))) < 1e-8 * abs(I1)
    # (2-2) [2^-k, 2^-k+1] 구간별 정확 적분 합
    I2 = sum(sp.integrate(4**(-k) * G.subs(x, x * 2**k), (x, sp.Rational(1, 2**k), sp.Rational(2, 2**k))) for k in range(1, 60))
    ok2 = abs(float(I2) - float(sp.sympify(md_answer(2, "2")))) < 1e-12
    report(2, ok1 and ok2 and ok3 and okd, f"g={sp.expand(G)} I1={float(I1):.10f} I2~{float(I2):.12f} C1경계={okd}")


def p03():
    # (3-1) g'(x) = f'(cos x)(-sin x) = -e^x sin x, g(0)=f(1)=1/2
    gp = -sp.exp(x) * sp.sin(x)
    g1 = sp.integrate(gp, (x, 0, x)) + sp.Rational(1, 2)
    ok1 = sp.simplify(g1 - sp.sympify(md_answer(3, "1"), locals={"x": x})) == 0
    # (3-2) g(x)=g(2π-x)
    g2 = g1.subs(x, 2 * sp.pi - x)
    P = parts(md_answer(3, "2"))
    okg = sp.simplify(g2 - sp.sympify(P["g(x)"], locals={"x": x})) == 0
    import numpy as np
    gfull = sp.lambdify(x, sp.Piecewise((g1, x <= sp.pi), (g2, True)))
    xs = np.linspace(0, 2 * np.pi, 200001)
    ys = np.array([gfull(v) for v in xs])
    zer = xs[:-1][np.sign(ys[:-1]) != np.sign(ys[1:])]
    d2 = sp.lambdify(x, sp.Piecewise((sp.diff(g1, x, 2), x <= sp.pi), (sp.diff(g2, x, 2), True)))
    d2s = np.array([d2(v) for v in xs])
    infl = xs[:-1][np.sign(d2s[:-1]) * np.sign(d2s[1:]) < 0]
    mdz = [float(sp.sympify(s)) for s in P["zeros"].split(",")]
    mdi = [float(sp.sympify(s)) for s in P["inflections"].split(",")]
    ok2 = okg and len(zer) == 2 and all(abs(u - w) < 1e-4 for u, w in zip(zer, mdz)) \
        and len(infl) == 2 and all(abs(u - w) < 1e-4 for u, w in zip(infl, mdi)) \
        and abs(ys.min() - float(sp.sympify(P["m"]))) < 1e-6 and abs(ys.max() - float(sp.sympify(P["M"]))) < 1e-6
    # (3-3) h(t)=두 교점 거리 수치 계산 → h(k)=π 인 k, h'(k) 수치미분
    import mpmath as mp
    mp.mp.dps = 40
    g1f = sp.lambdify(x, g1, "mpmath")

    def bisect(F, lo, hi, n=200):
        flo = F(lo)
        for _ in range(n):
            mid = (lo + hi) / 2
            if (F(mid) > 0) == (flo > 0):
                lo, flo = mid, F(mid)
            else:
                hi = mid
        return (lo + hi) / 2
    x1 = lambda tt: bisect(lambda u: g1f(u) - tt, mp.mpf(0), mp.pi)               # [0,π] 감소 구간의 교점
    x2 = lambda tt: bisect(lambda u: g1f(2 * mp.pi - u) - tt, mp.pi, 2 * mp.pi)  # [π,2π] 교점 (g(2π-u))
    hfun = lambda tt: x2(tt) - x1(tt)
    k = bisect(lambda tt: hfun(tt) - mp.pi, mp.mpf(-10), mp.mpf("0.49"))
    hp = mp.diff(hfun, k)
    Q = parts(md_answer(3, "3"))
    ok3 = abs(k - float(sp.sympify(Q["k"]))) < 1e-9 and abs(hp - float(sp.sympify(Q["h1"]))) < 1e-7
    report(3, ok1 and ok2 and ok3, f"zeros={np.round(zer,5)} infl={np.round(infl,5)} m={ys.min():.6f} M={ys.max():.6f} k={float(k):.8f} h'(k)={float(hp):.8f}")


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
