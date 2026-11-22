#!/usr/bin/env python3
"""KMU-G26B (국민대 2026학년도 수시 논술 자연계 오후1, 수학 1~8번) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "KMU-G26B"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
X = sp.Symbol("x")  # YAML 식(sympify) 의 x 와 맞추기 위한 기호


def md_item(no: int):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KMU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8"))


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    it = md_item(no)
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def md_equivalents(no: int, sub: str):
    it = md_item(no)
    return [str(e) for e in next(s["answer"]["equivalents"] for s in it["subparts"] if str(s["id"]) == sub)]


def same(expr, no: int, sub: str) -> bool:
    return sp.simplify(sp.nsimplify(expr) - sp.sympify(md_answer(no, sub))) == 0


def same_expr(expr, no: int, sub: str) -> bool:
    """식 정답: answer.value 와 equivalents 모두 독립 계산 식과 같은지."""
    e = sp.sympify(expr).subs(x, X)
    forms = [md_answer(no, sub)] + md_equivalents(no, sub)
    return all(sp.simplify(sp.sympify(s, locals={"x": X}) - e) == 0 for s in forms)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, r = sp.symbols("a r")
    sols = [s for s in sp.solve([a * (1 + r + r**2) - 26, a / (1 - r) - 27], [a, r], dict=True)
            if s[r].is_real and abs(s[r]) < 1]
    s = sols[0]
    an = lambda k: s[a] * s[r]**(k - 1)  # noqa: E731
    ok1 = len(sols) == 1 and same(an(4), 1, "1")
    n = sp.Symbol("n", positive=True, integer=True)
    c = (3 * n**2 - n) / (9 * n**2 - 3 * n - 2)
    lim_b = sp.limit(c, n, sp.oo)  # a_n→0, a_n+b_n-c_n→0
    ok2 = same(lim_b, 1, "2")
    # (3) lim{a_n+b_n} + Σ(a+b-1/3) = lim_b + 28 + Σ(c_n-1/3)
    tail = sp.summation(sp.apart(c - sp.Rational(1, 3), n), (n, 1, sp.oo))
    # 수치 확인: 부분합
    num = sum(float((c - sp.Rational(1, 3)).subs(n, k)) for k in range(1, 200001))
    total = lim_b + 28 + tail
    A = sp.nsimplify(total - 29)
    ok3 = abs(num - float(tail)) < 1e-5 and same(A, 1, "3")
    report(1, ok1 and ok2 and ok3, f"r={s[r]} a1={s[a]} a4={an(4)} lim b={lim_b} tail={tail} A={A}")


def p02():
    th = sp.Symbol("theta", positive=True)
    X0, Y0 = sp.cos(th), sp.sin(th)
    m = -X0 / Y0
    xi = sp.solve(sp.Eq(0 - Y0, m * (x - X0)), x)[0]
    yi = sp.simplify(Y0 + m * (0 - X0))
    A = sp.simplify(xi * yi / 2)
    # A = 1/sin(alpha) → alpha = asin(1/A); 수치로 alpha/theta 확인 (여러 θ in (0, π/4])
    ratios = []
    for v in [0.1, 0.3, 0.5, 0.7]:
        Av = float(A.subs(th, v))
        ratios.append(math.asin(1 / Av) / v)
    ok1 = all(abs(q - 2) < 1e-9 for q in ratios) and same(sp.Integer(round(ratios[0])), 2, "1")
    # (2) R: 기울기 -1/m 인 직선과 x축 교점
    xr = sp.solve(sp.Eq(0 - Y0, (-1 / m) * (x - X0)), x)[0]
    area = sp.simplify(sp.Abs((xi - xr) * Y0) / 2)
    f = sp.simplify(2 * area)
    lim2 = sp.limit(f.subs(th, x) / x, x, 0, "+")
    ok2 = same(lim2, 2, "2")
    # (3) 1/A_k with θ_k = kπ/(2n); S_n/n 수치 + 적분
    def Sn_over_n(N):
        return sum(float(1 / A.subs(th, k * sp.pi / (2 * N))) for k in range(1, N)) / N
    val = Sn_over_n(800)
    integ = sp.integrate(sp.sin(x), (x, 0, sp.pi)) / sp.pi
    ok3 = abs(val - float(integ)) < 1e-4 and same(integ, 2, "3")
    report(2, ok1 and ok2 and ok3, f"alpha/theta={ratios[0]:.6f} xr={sp.simplify(xr)} f={f} lim={lim2} S_n/n(800)={val:.6f} 적분={integ}")


def p03():
    k = sp.Symbol("k", positive=True)
    f = lambda z: 1 / z**4  # noqa: E731
    A = sp.integrate(f(x) - f(k), (x, 1, k))
    B, C = sp.symbols("B C")
    # B, C 상수: k 의 여러 값에서 맞추기 (B 는 정수 후보 전수)
    cand = []
    for Bv in range(1, 10):
        Cv = sp.solve(sp.Eq(A.subs(k, 2), sp.Rational(1, Bv) + sp.Rational(1, 2**Bv) * (sp.Rational(1, 2) + C)), C)
        if Cv and all(sp.simplify(A.subs(k, kv) - (sp.Rational(1, Bv) + (1 / sp.Integer(kv)**Bv) * (1 / sp.Integer(kv) + Cv[0]))) == 0 for kv in [3, 5, 7]):
            cand.append((Bv, Cv[0]))
    ok1 = len(cand) == 1 and same(cand[0][0] * cand[0][1], 3, "1")
    lim2 = sp.limit(A, k, sp.oo)
    ok2 = same(lim2, 3, "2")
    def S(N):
        return sum(sp.Rational(1) * (f(1 + sp.Rational(j, N)) - f(1 + sp.Rational(j + 1, N))) * sp.Rational(j, N) for j in range(1, N + 1))
    vals = [float(S(N)) for N in (2000, 4000)]
    exact = sp.integrate(f(x), (x, 1, 2)) - f(sp.Integer(2))
    ok3 = abs(vals[1] - float(exact)) < 1e-3 and abs(vals[1] - float(exact)) < abs(vals[0] - float(exact)) and same(exact, 3, "3") and same(A.subs(k, 2), 3, "3")
    report(3, ok1 and ok2 and ok3, f"A(k)={sp.simplify(A)} (B,C)={cand} lim={lim2} S(4000)={vals[1]:.6f} 극한={exact}")


def p04():
    # 밑면: 위 y=sqrt(x), 아래 max(-x, -1), 0<=x<=2 (y=sqrt x 와 y=-x 는 x=0 에서 만남)
    lower = lambda v: max(-v, -1.0)  # noqa: E731
    side1 = sp.sqrt(x) - (-x)       # 0<=x<=1
    side2 = sp.sqrt(x) - (-1)       # 1<=x<=2
    # 구간 판정 수치 확인
    ok_reg = all(lower(v) == -v for v in [0.25, 0.3, 1 / 3]) and all(lower(v) == -1 for v in [1.2, 1.5, 11 / 6])
    ok1 = ok_reg and same_expr(side1**2, 4, "1")
    ok2 = ok_reg and same_expr(side2**2, 4, "2")
    V = sp.integrate(sp.expand(side1**2), (x, 0, 1)) + sp.integrate(sp.expand(side2**2), (x, 1, 2))
    # 수치 적분 교차 확인
    import numpy as np
    xs = np.linspace(0, 2, 400001)
    side = np.sqrt(xs) - np.maximum(-xs, -1)
    Vn = np.trapz(side**2, xs) if hasattr(np, "trapz") else np.trapezoid(side**2, xs)
    alts = [md_answer(4, "3")] + md_equivalents(4, "3")
    ok3 = abs(Vn - float(V)) < 1e-6 and all(sp.simplify(sp.sympify(s) - V) == 0 for s in alts)
    report(4, ok1 and ok2 and ok3, f"V={sp.nsimplify(V)}={float(V):.6f} 수치={Vn:.6f} 동치표기 {len(alts)}개 확인")


def p05():
    sols = []
    for a1 in range(-200, 201):
        for m in range(1, 200):
            seq = [a1 + 2 * i for i in range(m)]
            if sum(seq) == 120 and sum(abs(v) for v in seq) == 144:
                ks = [i + 1 for i, v in enumerate(seq) if v == 0 and i + 1 >= 2]
                for kk in ks:
                    sols.append((a1, m, kk))
    ok = len(sols) == 1
    a1, m, kk = sols[0]
    s1 = sum(a1 + 2 * i for i in range(kk - 1))
    # a1 = A k + 2 가 모든 k 에 대해 성립하는 일반식: a_k=0 → a1 = 2-2k
    kk_s, A = sp.symbols("k A")
    Aval = sp.solve(sp.Eq(sp.solve(sp.Eq(sp.Symbol("a1") + 2 * (kk_s - 1), 0), sp.Symbol("a1"))[0], A * kk_s + 2), A)[0]
    ok = ok and same(s1, 5, "1") and same(Aval, 5, "2") and Aval * kk + 2 == a1 and same(m, 5, "3")
    report(5, ok, f"전수 해={sols} 합(k-1)={s1} A={Aval} m={m}")


def p06():
    # (1) 최소 차수: 차수 d 다항식 일반형으로 4개 조건 연립이 해를 갖는 최소 d
    nmin = None
    for d in range(0, 6):
        cs = sp.symbols(f"c0:{d + 1}")
        f = sum(cs[i] * x**i for i in range(d + 1))
        eqs = [f.subs(x, 3) - 5, f.subs(x, -2) + 7, f.subs(x, -1), f.subs(x, 1)]
        if d >= 1:
            eqs_lead = cs[d]
        sol = sp.solve(eqs, cs, dict=True)
        if sol and (d == 0 or any(sp.simplify(s.get(cs[d], cs[d])) != 0 for s in sol)):
            nmin = d
            break
    ok1 = same(nmin, 6, "1")
    cs = sp.symbols("c0:4")
    f = sum(cs[i] * x**i for i in range(4))
    sol = sp.solve([f.subs(x, 3) - 5, f.subs(x, -2) + 7, f.subs(x, -1), f.subs(x, 1), cs[2] + sp.Rational(23, 20)], cs, dict=True)
    s = sol[0]
    a0, a1 = s[cs[0]], s[cs[1]]
    ok2 = len(sol) == 1 and same(a1 / a0, 6, "2")
    B0 = sp.fraction(sp.nsimplify(abs(a0)))[1]
    B1 = sp.fraction(sp.nsimplify(abs(a1)))[1]
    ok3 = same(math.gcd(int(B0), int(B1)), 6, "3")
    report(6, ok1 and ok2 and ok3, f"n={nmin} f={sp.expand(f.subs(s))} a1/a0={a1 / a0} B0={B0} B1={B1}")


def p07():
    A, B = sp.symbols("A B")
    v = t**3 - A * t**2 + B * t
    sol = sp.solve([sp.diff(v, t).subs(t, 1), sp.integrate(v, (t, 1, 3))], [A, B], dict=True)
    s = sol[0]
    vv = v.subs(s)
    # 거리: |v| 수치 적분 (부호 가정 없이)
    import numpy as np
    vf = sp.lambdify(t, vv)
    ts = np.linspace(0, 1, 200001)
    dn = np.trapz(np.abs(vf(ts)), ts) if hasattr(np, "trapz") else np.trapezoid(np.abs(vf(ts)), ts)
    roots = [r for r in sp.solve(vv, t) if r.is_real and 0 < r < 1]
    d = sp.integrate(vv, (t, 0, 1)) if not roots else None
    ok = len(sol) == 1 and same(s[A], 7, "1") and same(s[B], 7, "2") and d is not None and abs(dn - float(d)) < 1e-6 and same(d, 7, "3")
    report(7, ok, f"A={s[A]} B={s[B]} 거리={d} 수치={dn:.6f} (0,1) 안 v=0 근={roots}")


def p08():
    A, B, C = sp.symbols("A B C")
    y = sp.Function("y")(x)
    F1 = A * x * sp.exp(x * y) - 2
    F2 = x - x * sp.sin(y) - B * x**2 - C
    Aval = sp.solve(F1.subs(y, 0).subs(x, 1), A)[0]
    yp = sp.Symbol("yp")
    d1 = sp.diff(F1.subs(A, Aval), x).subs(sp.Derivative(y, x), yp).subs(y, 0).subs(x, 1)
    d2 = sp.diff(F2, x).subs(sp.Derivative(y, x), yp).subs(y, 0).subs(x, 1)
    s1 = sp.solve(d1, yp)[0]
    s2 = sp.solve(d2, yp)[0]
    sol = sp.solve([sp.Eq(s1, s2), F2.subs(y, 0).subs(x, 1)], [B, C], dict=True)[0]
    ok = same(Aval, 8, "1") and same(sol[B], 8, "2") and same(sol[C], 8, "3")
    report(8, ok, f"A={Aval} 기울기1={s1} 기울기2={s2} B={sol[B]} C={sol[C]}")


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
