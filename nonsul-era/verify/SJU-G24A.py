#!/usr/bin/env python3
"""SJU-G24A (세종대 2024학년도 수시 논술 A형) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G24A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


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
    N = sp.symbols("n", positive=True, integer=True)
    T, X = sp.symbols("T X", real=True)
    neg = -sp.integrate((T + 3) * (T + 1) / ((N + 3) * (N + 1)), (T, 0, X))
    pos = sp.integrate(T**3 / N**3 - (T + 3) * (T + 1) / ((N + 3) * (N + 1)), (T, 0, X))
    # (1) x<0 임계점과 이계 부호로 극소 찾기
    d1 = sp.diff(neg, X)
    crit = sp.solve(d1, X)
    mins = [c for c in crit if c < 0 and sp.simplify(sp.diff(d1, X).subs(X, c)) > 0]
    v1 = sp.simplify(neg.subs(X, mins[0]))
    ok1 = len(mins) == 1 and sp.simplify(v1 - sp.sympify(md_answer(1, "1"))) == 0
    # (2) 여러 n 에 대해 격자 최소와 md 식 비교
    md2 = sp.sympify(md_answer(1, "2"), locals={"n": N})
    import numpy as np
    ok2 = True
    for nv in (1, 2, 5, 11):
        fn = sp.lambdify(X, sp.Piecewise((neg.subs(N, nv), X < 0), (pos.subs(N, nv), True)), "numpy")
        xs = np.linspace(-10, nv + 10, 400001)
        m = float(np.min(fn(xs)))
        ok2 &= abs(m - float(md2.subs(N, nv))) < 1e-6
    # (3) 세 점 P,Q,R 의 가장 큰 내각의 tan, n 큰 값 수치 + 기호 극한
    def tan_max_angle(nv):
        Fn = lambda x: float((neg if x < 0 else pos).subs({N: nv, X: x}))
        pts = [np.array([-3.0, Fn(-3)]), np.array([-1.0, Fn(-1)]), np.array([float(nv), Fn(nv)])]
        angs = []
        for i in range(3):
            u = pts[(i + 1) % 3] - pts[i]; v = pts[(i + 2) % 3] - pts[i]
            angs.append(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
        return np.tan(max(angs))
    tq = (neg.subs(X, -1)) / 2                       # PQ 기울기
    tb = (pos.subs(X, N) - neg.subs(X, -1)) / (N + 1)  # QR 기울기
    lim = sp.limit(sp.simplify((tb - tq) / (1 + tb * tq)), N, sp.oo)
    ans = sp.sympify(md_answer(1, "3"))
    ok3 = sp.simplify(lim - ans) == 0 and abs(tan_max_angle(2000) - float(ans)) < 1e-3 and \
        all(tan_max_angle(k) < 0 for k in (1, 2, 3, 10))
    report(1, ok1 and ok2 and ok3, f"min(x<0)={v1} lim tan={lim} tan(n=2000)={tan_max_angle(2000):.6f}")


def p02():
    a, b, c, X = sp.symbols("a b c X", real=True)
    f = 2 * X + 1 + (X - a)**2 * (X - b)**2           # 최고차 1, 두 점에서 y=2x+1 에 접함
    eqs = [sp.diff(f, X).subs(X, c) - 2,
           (b - a)**2 + (f.subs(X, b) - f.subs(X, a))**2 - 5,
           a + b + c - sp.Rational(3, 2)]
    sols = [s for s in sp.solve(eqs, [a, b, c], dict=True)
            if all(v.is_real for v in s.values()) and s[a] < s[c] < s[b]]
    s0 = sols[0]
    ok1 = len(sols) == 1 and sp.simplify(s0[c] - sp.sympify(md_answer(2, "1"))) == 0
    F = sp.expand(f.subs(s0)); F1 = sp.diff(F, X)
    import numpy as np
    xs = np.linspace(float(s0[a]), float(s0[c]), 200001)[1:-1]
    f1 = sp.lambdify(X, F1, "numpy")(xs)
    ok_inv = f1.min() > 0                              # 역함수 존재
    mn = 1 / f1.max()
    ok2 = ok_inv and abs(mn - float(sp.sympify(md_answer(2, "2")))) < 1e-9
    # (3) 변곡점, L, k, P, 곡선 max(f,L) 위 최단점 수치 탐색
    infl = sorted(sp.solve(sp.diff(F, X, 2), X), key=float)
    al, be = infl
    L = sp.expand((F.subs(X, be) - F.subs(X, al)) / (be - al) * (X - al) + F.subs(X, al))
    roots = [r for r in sp.solve(sp.Eq(F, L), X) if r.is_real]
    k = max(roots, key=float)
    Px = float(k + 2 * F.subs(X, k))
    fF = sp.lambdify(X, F, "numpy"); fL = sp.lambdify(X, L, "numpy")
    xs = np.linspace(-3, 4, 1400001)
    ys = np.maximum(fF(xs), fL(xs))
    xq = xs[np.argmin((xs - Px)**2 + ys**2)]
    ans = sp.sympify(md_answer(2, "3"))
    ok3 = abs(xq - float(ans)) < 1e-4 and sp.simplify(k - ans) == 0
    report(2, ok1 and ok2 and ok3, f"(a,b,c)={tuple(s0[v] for v in (a,b,c))} min(f^-1)'={mn:.10f} k={sp.nsimplify(k)} argmin~{xq:.6f}")


def p03():
    import mpmath as mp
    mp.mp.dps = 20
    f = lambda x: mp.quad(lambda t: t * mp.e**(-(t - x)**2), [0, x])
    A = mp.quad(lambda u: mp.e**(-u**2), [0, 2])
    Asym = sp.symbols("A")
    d2 = mp.diff(f, 2)
    ok1 = abs(d2 - float(sp.sympify(md_answer(3, "1"), locals={"A": Asym}).subs(Asym, sp.Float(str(A))))) < 1e-12
    grid = [mp.mpf(i) / 100 for i in range(1, 201)]
    vals = [f(x) / x for x in grid]
    mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    md2 = sp.sympify(md_answer(3, "2"), locals={"A": Asym}).subs(Asym, sp.Float(str(A), 30))
    ok2 = mono and abs(max(vals) - mp.mpf(str(sp.N(md2, 30)))) < 1e-12
    # (3) 기호: t=xs 치환 f=x^2*int_0^1 s*exp(-x^2(1-s)^2) ds 를 x 에 대해 급수 전개(정의에서 직접)
    x, s_ = sp.symbols("x s", positive=True)
    integrand = sp.series(s_ * sp.exp(-x**2 * (1 - s_)**2), x, 0, 8).removeO()
    fs = sp.expand(x**2 * sp.integrate(sp.expand(integrand), (s_, 0, 1)))
    expr = fs**5 * sp.diff(fs, x) / x**11
    lim = sp.limit(expr, x, 0)
    ok3 = sp.simplify(lim - sp.sympify(md_answer(3, "3"))) == 0
    report(3, ok1 and ok2 and ok3, f"f'(2)={mp.nstr(d2,12)} A={mp.nstr(A,12)} max f/x={mp.nstr(max(vals),12)} lim={lim}")


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
