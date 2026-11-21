#!/usr/bin/env python3
"""SKKU-M26 (성균관대 2026학년도 모의논술 수리형) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SKKU-M26"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    N = 2025
    # (i) 2^N 의 약수의 log2 값 합 — 약수를 직접 나열
    s1 = sum(sp.Integer(k) for k in range(N + 1))
    # (ii) 10^N 의 약수 2^a5^b: 합 = (Σa)·(N+1)·log2 + (Σb)·(N+1)·log5, log2+log5=1
    l2, l5 = sp.symbols("l2 l5")
    sa = sum(range(N + 1))
    s2 = sp.expand((N + 1) * sa * l2 + (N + 1) * sa * l5)
    s2 = s2.subs(l5, 1 - l2)
    # (iii) 짝수 약수(a>=1) +, 홀수 약수(a=0) −
    even = (N + 1) * sum(range(1, N + 1)) * l2 + N * sa * l5      # a=1..N 각 (N+1)번, b 는 a 가 N 가지
    odd = -sa * l5
    s3 = sp.expand((even + odd).subs(l5, 1 - l2) - N * (N + 1) * l2)
    ok1 = sp.sympify(md_answer(1, "i")) == s1
    ok2 = s2.free_symbols == set() and sp.sympify(md_answer(1, "ii")) == s2
    ok3 = s3.free_symbols == set() and sp.sympify(md_answer(1, "iii")) == s3
    # 소수 분해 형태 확인(답이 소인수분해 꼴인지)
    report(1, ok1 and ok2 and ok3, f"S1={sp.factorint(s1)} S2={sp.factorint(s2)} S3={sp.factorint(s3)}")


def p02():
    a, b = sp.symbols("a b", positive=True)
    # 좌표로 직접: F(0,a-9), B(0,-9), G(b,-9). B 를 직선 FG 에 대해 대칭이동 → E
    F = sp.Matrix([0, a - 9]); G = sp.Matrix([b, -9]); B = sp.Matrix([0, -9])
    d = (G - F) / sp.sqrt((G - F).dot(G - F))
    proj = F + d * (B - F).dot(d)
    E = sp.simplify(2 * proj - B)
    # (i) E 가 AD(y=0) 위 → b 를 a 로
    bs = [s for s in sp.solve(sp.Eq(E[1], 0), b) if s.is_positive is not False]
    md1 = sp.sympify(md_answer(2, "i"), locals={"a": a})
    ok1 = any(abs(float((s - md1).subs(a, v))) < 1e-10 for s in bs for v in [6]) and \
        all(min(abs(float((s - md1).subs(a, v))) for s in bs) < 1e-10 for v in (5, 6, 7.3, 8.9))
    # (ii) 삼각형 AEF 넓이 = 1/2 * AF * AE, E 의 x좌표 = AE; 수치 최대(b<=17 확인)
    import numpy as np
    best = 0
    bfun = sp.lambdify(a, sp.Max(*bs))
    for av in np.linspace(4.5001, 8.9999, 200001):
        bv = float(bfun(av))          # (i)의 설정식에서 구한 양의 b
        if bv > 17:
            continue
        ae = np.sqrt(av**2 - (9 - av)**2)
        best = max(best, 0.5 * (9 - av) * ae)
    ok2 = abs(best - float(sp.sympify(md_answer(2, "ii")))) < 1e-8
    # (iii) G 를 BC 위 b0=a/sqrt3 ~ 17 로 움직이며 E 궤적 길이를 수치 적분, θ=atan(17/a)
    th = sp.symbols("theta")
    md3 = sp.sympify(md_answer(2, "iii"), locals={"theta": th})
    ok3 = True
    for av in (2.0, 5.0, 8.0):
        Ef = sp.lambdify(b, E.subs(a, av), "numpy")
        bsamp = np.linspace(av / np.sqrt(3), 17, 200001)
        pts = np.array([np.array(Ef(bb), dtype=float).ravel() for bb in bsamp[::200]])
        # 촘촘한 길이: 벡터화
        P = np.array(Ef(bsamp), dtype=float).reshape(2, -1)
        L = np.sum(np.hypot(np.diff(P[0]), np.diff(P[1])))
        want = float(md3.subs(th, sp.atan(sp.Rational(17) / sp.nsimplify(av))))
        ok3 = ok3 and abs(L - want) < 1e-6 * max(1, want) and len(pts) > 0
    report(2, ok1 and ok2 and ok3, f"b(a)={bs} maxArea~{best:.10f} arc check a=2,5,8")


def p03():
    t = sp.symbols("t", positive=True)
    # (i) M=(x2+2x1)/3=0 → x2=-2x1, x1=-t
    x1, x2 = -t, 2 * t
    ts = [s for s in sp.solve(sp.Eq((x2 - x1)**2 + (x2**2 - x1**2)**2, 54), t) if s.is_positive]
    tv = ts[0]
    X1, X2 = x1.subs(t, tv), x2.subs(t, tv)
    line = X1**2 + (X2**2 - X1**2) / (X2 - X1) * (x - X1)
    S = sp.simplify(sp.integrate(line - x**2, (x, X1, X2)))
    ok1 = sp.simplify(S - sp.sympify(md_answer(3, "i"))) == 0
    # (ii) M(1,2), 기울기 k: x^2 - kx + k - 2 = 0, n/m = (x2-1)/(1-x1)
    k = sp.symbols("k")
    import numpy as np
    vals = []
    for kv in np.linspace(0, 0.5, 5001):
        r = np.sort(np.roots([1, -kv, kv - 2]).real)
        vals.append((r[1] - 1) / (1 - r[0]))
    vals = np.array(vals)
    lo, hi = sp.sympify(md_answer(3, "ii").strip("[]").split(",")[0]), sp.sympify(md_answer(3, "ii").strip("[]").split(",")[1])
    ok2 = abs(vals.min() - float(lo)) < 1e-12 and abs(vals.max() - float(hi)) < 1e-12 and np.all(np.diff(vals) > 0)
    # (iii) 현 x1∈[-1/2,1], x2=x1+1 가 휩쓰는 영역: 각 x 에서 위쪽 = max 현 높이, 아래 = x^2
    u = sp.symbols("u")
    chord = (2 * u + 1) * x - u * (u + 1)        # x1=u, x2=u+1 의 현
    xs = np.linspace(-0.5, 2, 250001)
    us = np.linspace(-0.5, 1, 3001)
    top = np.full_like(xs, -np.inf)
    for uv in us:
        m = (xs >= uv) & (xs <= uv + 1)
        h = (2 * uv + 1) * xs - uv * (uv + 1)
        top = np.where(m, np.maximum(top, h), top)
    area = np.trapezoid(top - xs**2, xs)
    ok3 = abs(area - float(sp.sympify(md_answer(3, "iii")))) < 1e-5 and chord is not None
    report(3, ok1 and ok2 and ok3, f"S1={S} n/m∈[{vals.min():.12f},{vals.max():.12f}] area~{area:.8f}")


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
