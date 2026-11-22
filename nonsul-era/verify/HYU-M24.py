#!/usr/bin/env python3
"""HYU-M24 (한양대 2024학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산 후 YAML answer.value 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "HYU-M24"
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
    import math
    from itertools import combinations
    import numpy as np
    # 1. n=1,2,3 전수 세기
    nn = sp.symbols("n", positive=True, integer=True)
    expr = sp.sympify(md_answer(1, "1"), locals={"n": nn})
    ok1 = True
    for N in (1, 2, 3):
        ev = [k for k in range(1, 4 * N + 3) if k % 2 == 0]
        od = [k for k in range(1, 4 * N + 3) if k % 2 == 1]
        ce = sum(math.comb(len(ev), r) for r in range(N + 1, len(ev) + 1))
        co = sum(math.comb(len(od), r) for r in range(0, N + 1))
        if N == 1:   # 작은 경우는 부분집합을 직접 나열해 확인
            cnt = 0
            A = list(range(1, 4 * N + 3))
            for r in range(len(A) + 1):
                for S in combinations(A, r):
                    e = sum(1 for v in S if v % 2 == 0); o = len(S) - e
                    cnt += (e >= N + 1 and o <= N)
            ok1 &= cnt == ce * co
        ok1 &= ce * co == int(expr.subs(nn, N))
    # 2. 가법성 + 미분가능 → f'(x)=f'(0) 상수, f(0)=0, f(3)=6 → f(x)=2x
    c = sp.symbols("c")
    cval = sp.solve(sp.Eq(3 * c, 6), c)[0]
    f = cval * x
    k = sp.integrate(f**9 * f.subs(x, 2 - x)**10 * sp.diff(f, x), (x, 0, 2))
    ok2 = sp.simplify(sp.binomial(20, 10) * k - sp.sympify(md_answer(1, "2"))) == 0
    # 3. 좌표: A(0,0), D=5(3/5,4/5), C=D+(1,0), B=(b,0) 를 cos∠BCD=-1/√5 로 결정
    b = sp.symbols("b", positive=True)
    A = sp.Matrix([0, 0]); D = sp.Matrix([3, 4]); C = D + sp.Matrix([1, 0]); B = sp.Matrix([b, 0])
    u, w = B - C, D - C
    bs = [s_ for s_ in sp.solve(sp.Eq(u.dot(w)**2 * 5, u.dot(u) * w.dot(w)), b) if (u.dot(w)).subs(b, s_) < 0]
    bv = bs[0]
    X, Y = sp.symbols("X Y", real=True)
    Pm = sp.Matrix([X, Y])
    eq = ((Pm - A) + (Pm - C)).dot(2 * (Pm - B.subs(b, bv)) + (Pm - D)) - sp.Rational(67, 8)
    # 원 중심·반지름을 이차식 계수로 직접 추출
    poly = sp.Poly(sp.expand(eq), X, Y)
    a2 = poly.coeff_monomial(X**2)
    cx = -poly.coeff_monomial(X) / (2 * a2); cy = -poly.coeff_monomial(Y) / (2 * a2)
    r2 = cx**2 + cy**2 - poly.coeff_monomial(1) / a2
    ths = np.linspace(0, 2 * np.pi, 400001)
    px = float(cx) + math.sqrt(float(r2)) * np.cos(ths); py = float(cy) + math.sqrt(float(r2)) * np.sin(ths)
    resid = max(abs(float(eq.subs({X: px[i], Y: py[i]}))) for i in (0, 1000, 200000))
    mx = np.sqrt(px**2 + py**2).max()
    ok3 = bv == 6 and resid < 1e-9 and abs(mx - float(sp.sympify(md_answer(1, "3")))) < 1e-9
    report(1, ok1 and ok2 and ok3, f"k*C={sp.binomial(20, 10) * k} B=({bv},0) center=({cx},{cy}) r2={r2} max|AP|={mx:.12f}")


def p02():
    import math
    from fractions import Fraction
    from itertools import product
    import numpy as np
    from scipy.optimize import minimize
    # 1. A,B 를 단위원 위 AB=√3 로 두고 P(x,y,√(1-x²-y²)) 에 대해 부피 최대화
    A = np.array([-math.sqrt(3) / 2, -0.5]); B = np.array([math.sqrt(3) / 2, -0.5])
    def area(h):
        return abs((B - A)[0] * (h - A)[1] - (B - A)[1] * (h - A)[0]) / 2
    def negV(v):
        r2 = v[0]**2 + v[1]**2
        if r2 >= 1:
            return 1.0
        return -area(np.array(v)) * math.sqrt(1 - r2) / 3
    best = min((minimize(negV, s0, method="Nelder-Mead", options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 20000})
                for s0 in ([0, 0.3], [0.2, -0.2], [-0.4, 0.5], [0.1, 0.8])), key=lambda r: r.fun)
    ar = area(best.x)
    ok1 = abs(ar - float(sp.sympify(md_answer(2, "1")))) < 1e-7
    # 2. 여러 m 에서 (0,1] 의 교점 개수(부호 변화 + 끝점)를 세어 YAML 범위와 대조
    rng = sp.sympify(md_answer(2, "2"))
    xs = np.linspace(1e-6, 1, 2000001)
    def meets_extra(m):
        d = xs**m / m - xs**2 / (1 + np.sqrt(1 - xs**2))     # g 를 소거 오차 없는 꼴로
        return bool(np.any(np.sign(d[:-1]) * np.sign(d[1:]) < 0) or abs(d[-1]) < 1e-12)
    ms = [0.2, 0.5, 0.9, 0.999, 1.0, 1.1, 1.5, 1.9, 1.99, 2.0, 2.01, 3.0, 7.0]
    ok2 = all((not meets_extra(m)) == bool(rng.contains(sp.Float(m)) if m != 1.0 and m != 2.0 else rng.contains(sp.Integer(int(m)))) for m in ms)
    # 3. 세 번 시행 전수(4*6)^3 정확 기댓값
    def mvA(a):
        return 3 if a % 3 == 0 else (1 if a % 3 == 1 else -2)
    def mvB(b_):
        return 1 if b_ % 3 == 0 else -2
    tot = Fraction(0)
    for rolls in product(product(range(1, 5), range(1, 7)), repeat=3):
        tot += sum(mvA(a) + mvB(b_) for a, b_ in rolls)
    E = tot / 24**3
    ok3 = sp.Rational(E.numerator, E.denominator) == sp.sympify(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"area={ar:.10f} E={E}")


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
