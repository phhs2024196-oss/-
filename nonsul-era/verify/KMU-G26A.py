#!/usr/bin/env python3
"""KMU-G26A (국민대 2026학년도 수시 논술 자연계 오전, 수학 1~8번) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "KMU-G26A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KMU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def same(expr, no: int, sub: str) -> bool:
    return sp.simplify(sp.nsimplify(expr) - sp.sympify(md_answer(no, sub))) == 0


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 가) 두 부등식의 공통 자연수 — 전수 확인
    nats = [k for k in range(1, 100) if -k + 3 <= 4 * k - 1 and 2 * k - 3 <= 2 - k]
    a = nats[0]
    # |f| 가 x=a 에서 극댓값 3/4 → f 는 x=a 에서 극솟값 -3/4 (최고차항 1)
    f = (x - a)**2 - sp.Rational(3, 4)
    ok1 = nats == [1] and same(f.subs(x, 2), 1, "1")
    # 나) g(x)=x^2+bx+c 를 정수·유리 격자로 훑지 않고, 조건을 수치로 직접 확인:
    b, c = sp.symbols("b c", real=True)
    g = x**2 + b * x + c
    h = sp.integrate(g.subs(x, t), (t, 0, x / 2))
    # x>=2 에서 h 최소가 x=6 → h'(6)=0 (내부 최소점) ; |h(6)|>=|h(4)| 와 h(x)>=h(6) 동시 만족 후보를 풀이
    sols = sp.solve([sp.diff(h, x).subs(x, 6), h.subs(x, 4)], [b, c], dict=True)
    cand = []
    import numpy as np
    for s in sols:
        hs = sp.lambdify(x, h.subs(s))
        xs = np.linspace(2, 60, 58001)
        hv = hs(xs)
        if hv.min() >= hs(6) - 1e-12 and np.all(np.abs(hv) >= abs(hs(4)) - 1e-12):
            cand.append(s)
    # h(6)>=0 경우(h(4)≠0 허용)의 반례 탐색: h'(6)=0 만 두고 c 를 격자로 훑어 조건 만족하는 것이 없는지
    extra = 0
    for cv in np.linspace(-50, 50, 2001):
        bv = float(sp.solve(sp.diff(h, x).subs(x, 6).subs(c, cv), b)[0])
        hs = sp.lambdify(x, h.subs({b: bv, c: cv}))
        xs = np.linspace(2, 60, 5801); hv = hs(xs)
        if hv.min() >= hs(6) - 1e-9 and np.all(np.abs(hv) >= abs(hs(4)) - 1e-9) and abs(hs(4)) > 1e-6:
            extra += 1
    g2 = g.subs(cand[0]) if len(cand) == 1 else None
    ok2 = g2 is not None and same(g2.subs(x, 3), 1, "2")
    ok3 = g2 is not None and same(g2.subs(x, 6), 1, "3") and extra == 0
    report(1, ok1 and ok2 and ok3, f"nats={nats} f(2)={f.subs(x, 2)} g={sp.expand(g2) if g2 is not None else None} g(6)={g2.subs(x, 6) if g2 is not None else None} 반례={extra}")


def md_expr(no: int, sub: str):
    """'y=...' 꼴이면 우변, '(a, b)' 꼴이면 Tuple 로 sympify(문자 e 는 exp(1) 로 쓴다)."""
    s = md_answer(no, sub).strip()
    if s.startswith("y="):
        s = s[2:]
    return sp.sympify(s, locals={"x": x, "h": sp.Function("h")})


def p02():
    p = sp.symbols("p", positive=True)  # π 를 문자로 두고 계수 비교(π 는 초월수)
    F = sp.Function("f")
    k = sp.exp(2 * x**2 + 3 * x) + sp.cos((sp.pi * x - sp.pi) / 2)
    # 가) 극한 = (f∘k)'(x) : 정의대로 차분몫의 극한을 다항 f 로 점검
    hh = sp.symbols("hh")
    ftest = lambda u: u**3 - 2 * u
    lim = sp.limit((ftest(k.subs(x, x + hh)) - ftest(k)) / hh, hh, 0)
    okdef = sp.simplify((lim - sp.diff(ftest(k), x)).subs(x, sp.Rational(1, 3))) == 0
    g = sp.diff(F(k), x)
    d1, d2 = sp.symbols("d1 d2")  # f'(k(0)) , f''(k(0))
    k0, k1 = k.subs(x, 0), k.subs(x, 1)
    kp = sp.diff(k, x)
    g0 = (d1 * kp).subs(x, 0)
    fp1 = sp.solve(sp.Eq(g0, sp.Rational(1, 4)), d1)[0]
    # (1) (A+Bπ)/(72+Cπ²) = f'(1), A,B,C 정수 탐색
    found1 = []
    for B in range(-10, 11):
        for C in range(-10, 11):
            r = sp.cancel(fp1.subs(sp.pi, p) * (72 + C * p**2) - B * p)
            if r.free_symbols == set() and r.is_integer:
                found1.append((int(r), B, C))
    # (2) g(1)=7 → f'(k(1)) = 7/k'(1)
    v2 = sp.simplify(7 / kp.subs(x, 1))
    # (3) g'(x) = f''(k)k'^2 + f'(k)k'' ; g'(0)=0
    kpp = sp.diff(k, x, 2)
    fpp1 = sp.solve(sp.Eq(d2 * kp.subs(x, 0)**2 + fp1 * kpp.subs(x, 0), 0), d2)[0]
    found3 = []
    for B in range(-10, 11):
        for C in range(1, 8):
            r = sp.cancel(-fpp1.subs(sp.pi, p) * (6 - B * p)**C)
            if r.free_symbols == set() and r.is_integer:
                found3.append((int(r), B, C))
    ok1 = k0 == 1 and len(found1) == 1 and same(sp.prod(found1[0]), 2, "1")
    ok2 = sp.simplify(k1 - (sp.exp(5) + 1)) == 0 and same(v2, 2, "2")
    ok3 = len(found3) == 1 and same(sp.prod(found3[0]), 2, "3")
    report(2, okdef and ok1 and ok2 and ok3, f"f'(1)={sp.simplify(fp1)} (A,B,C)={found1} f'(e^5+1)={v2} f''(1)={sp.simplify(fpp1)} (A,B,C)={found3}")


def p03():
    H = sp.Function("h")
    F = sp.Function("f")
    # (1) d/dx ∫_x^1 h(t)dt
    d = sp.diff(sp.Integral(H(t), (t, x, 1)), x).doit()
    ok1 = sp.simplify(d - md_expr(3, "1")) == 0
    # (2) 일반적인 h = c1 + c2 t + c3 e^t (가) 만족) 에 대해 A 계산 → c 에 무관한지
    c1, c2, c3 = sp.symbols("c1 c2 c3")
    hgen = c1 + c2 * t + c3 * sp.exp(t)
    cons = sp.integrate(x**2 * hgen.subs(t, x), (x, 0, 1)) - 1
    c1v = sp.solve(cons, c1)[0]
    gx = x * sp.integrate(hgen.subs(c1, c1v), (t, x, 1))
    A = sp.simplify(sp.integrate(gx, (x, 0, 1)))
    ok2 = A.free_symbols == set() and same(A, 3, "2")
    # (3) 라) 미분: x>0 에서 sqrt(x^2)=x
    xp = sp.symbols("xp", positive=True)
    s, u = sp.symbols("s u", positive=True)
    lhs = sp.diff(sp.Integral(A * sp.exp(sp.sqrt(s)) * F(xp).diff(xp).subs(xp, sp.sqrt(s)), (s, 1, xp**2)), xp).doit() \
        + xp * F(xp).diff(xp)
    fp = sp.solve(sp.Eq(lhs, 2 * xp), F(xp).diff(xp))[0]
    fp1 = sp.simplify(fp.subs(xp, 1))
    ok3 = same(fp1, 3, "3")
    # (4) x=1 대입 → 두 적분 0 → f(1)=1
    Y = sp.Symbol("Y")
    at1 = sp.Integral(A * sp.exp(sp.sqrt(s)) * F(s), (s, 1, 1)).doit() + sp.Integral(t * F(t), (t, 1, 1)).doit()
    f1 = sp.solve(sp.Eq(at1 + Y, 1**2), Y)[0]  # x=1 대입: 두 적분 0
    tang = fp1 * (x - 1) + f1
    ok4 = sp.simplify(md_expr(3, "4") - tang) == 0
    report(3, ok1 and ok2 and ok3 and ok4, f"(1)={d} A={A} f'(x)={sp.simplify(fp)} f'(1)={fp1} 접선 y={sp.simplify(tang)}")


def p04():
    sols = [s for s in sp.solve(sp.sin(x) - sp.cos(x), x) if 0 <= s <= sp.pi / 2]
    pt = sp.Tuple(sols[0], sp.sin(sols[0])) if len(sols) == 1 else None
    ok1 = pt is not None and sp.simplify(sp.Matrix(list(pt)) - sp.Matrix(list(md_expr(4, "1")))) == sp.zeros(2, 1)
    # (2) 위: sin+cos, 아래: max(sin, cos) — 구간별 적분, 수치로 경계 확인
    top, a = sp.sin(x) + sp.cos(x), sols[0]
    area = sp.integrate(top - sp.cos(x), (x, 0, a)) + sp.integrate(top - sp.sin(x), (x, a, sp.pi / 2))
    import numpy as np
    xs = np.linspace(0, np.pi / 2, 20001)
    num = (np.trapezoid if hasattr(np,'trapezoid') else np.trapz)(np.sin(xs) + np.cos(xs) - np.maximum(np.sin(xs), np.cos(xs)), xs)
    ok2 = same(area, 4, "2") and abs(num - float(area)) < 1e-6
    # (3) √(1+y'^2) = √2·√(1-cos x sin x) 항등식 확인 후 주어진 값 6/5 사용
    ident = sp.simplify(1 + sp.diff(top, x)**2 - 2 * (1 - sp.cos(x) * sp.sin(x))) == 0
    L = sp.sqrt(2) * sp.Rational(6, 5)
    ok3 = ident and same(L, 4, "3")
    import mpmath
    true_I = mpmath.quad(lambda z: mpmath.sqrt(1 - mpmath.cos(z) * mpmath.sin(z)), [0, mpmath.pi / 2])
    report(4, ok1 and ok2 and ok3, f"교점={pt} 넓이={sp.simplify(area)}(수치 {num:.8f}) 길이=√2·(6/5) [주의: 실제 ∫√(1-cos sin)={float(true_I):.6f}≠6/5, 문제의 가정값 사용]")


def p05():
    # 적분을 계산해 f 를 명시적으로 푼다
    s = sp.symbols("s")
    I1 = sp.integrate(2 - sp.cos(t), (t, 0, 3 * x))
    I2 = sp.integrate(sp.sec(s / 4)**2, (s, 0, x + sp.pi))
    Y = sp.symbols("Y")
    f = sp.solve(sp.Eq(I1 + I2 + sp.log(Y - 3), 4), Y)[0]
    f0 = sp.simplify(f.subs(x, 0))
    fp0 = sp.simplify(sp.diff(f, x).subs(x, 0))
    line = -1 / fp0 * x + f0
    ok = same(f0, 5, "1") and same(fp0, 5, "2") and sp.simplify(md_expr(5, "3") - line) == 0
    report(5, ok, f"f(x)={f} f(0)={f0} f'(0)={fp0} 직선 y={line}")


def p06():
    import itertools
    U = range(4)
    subsets = [frozenset(c) for r in range(5) for c in itertools.combinations(U, r)]
    Uset = frozenset(U)
    # (1) P∪Q^c=U, P≠Q, Q≠∅ : q→p 항상 참? p→q 는?
    cases = [(P, Q) for P in subsets for Q in subsets if Q and (P | (Uset - Q)) == Uset and P != Q]
    nec = all(Q <= P for P, Q in cases)
    suf_never = all(not (P <= Q) for P, Q in cases)
    a1 = "필요조건" if nec and suf_never else ("충분조건" if not nec else "?")
    # (2) s: P⊆Q^c , t: P∩Q=∅ — 모든 (P,Q) 에서 동치?
    allpq = [(P, Q) for P in subsets for Q in subsets if Q]
    eq = all(((P <= Uset - Q) == (not (P & Q))) for P, Q in allpq)
    a2 = "필요충분조건" if eq else "?"
    # (3) P⊆Q, R⊆P 를 만족하는 정수 a, b
    Pset = sp.Union(sp.Interval(-3, 1), sp.Interval.open(5, sp.oo))
    As = [a for a in range(-50, 51) if Pset.is_subset(sp.Interval.open(a, sp.oo))]
    Bs = [b for b in range(-50, 51) if sp.Interval(b, sp.oo).is_subset(Pset)]
    v3 = max(As) * min(Bs)
    ok = a1 == md_answer(6, "1") and a2 == md_answer(6, "2") and same(v3, 6, "3")
    report(6, ok, f"(1)={a1}({len(cases)}경우) (2)={a2} a_max={max(As)} b_min={min(Bs)} ab={v3}")


def p07():
    n = sp.symbols("n", positive=True, integer=True)
    r = sorted(sp.solve(sp.Eq(x**2, -x**2 + 2 * n * x + n), x), key=lambda e: e.subs(n, 1))
    al, be = r
    fn = sp.expand((be - al)**2)
    ok1 = same(fn.subs(n, 5), 7, "1")
    S = sp.integrate((-x**2 + 2 * n * x + n) - x**2, (x, al, be))
    C = sp.simplify(S.subs(n, 2) / sp.sqrt(2))
    ok2 = C.is_rational and same(C, 7, "2")
    T = sp.Abs(al * be**2 - be * al**2) / 2
    T = sp.simplify(T)
    D = sp.limit(sp.simplify(S / (n * T)), n, sp.oo)
    ok3 = same(D, 7, "3")
    report(7, ok1 and ok2 and ok3, f"f(n)={fn} f(5)={fn.subs(n, 5)} S_2={sp.simplify(S.subs(n, 2))} C={C} T_n={T} D={D}")


def p08():
    # [그림] f: x절편 -8,-2, y절편 3 / g: x절편 -2,7, y절편 -3
    f = sp.Rational(3, 16) * (x + 8) * (x + 2)
    g = sp.Rational(3, 14) * (x + 2) * (x - 7)
    ok_fig = f.subs(x, 0) == 3 and g.subs(x, 0) == -3
    # 부등식: 진수 조건 f>0, -1/g(-x)>0, log_3 f ≤ log_{1/3}(-1/g(-x)) — 정수 전수 + 실수 해집합
    def holds(v):
        fv, gv = f.subs(x, v), g.subs(x, -v)
        if not (fv > 0 and gv < 0):
            return False
        return sp.log(fv, 3) - sp.log(-1 / gv, sp.Rational(1, 3)) <= 0
    ints = [v for v in range(-40, 41) if holds(v)]
    sol = sp.Intersection(sp.solveset(f > 0, x, sp.S.Reals), sp.solveset(-g.subs(x, -x) > 0, x, sp.S.Reals),
                                      sp.solveset(f + g.subs(x, -x) <= 0, x, sp.S.Reals))
    M, m = max(ints), min(ints)
    s = md_answer(8, "1").replace(" ", "")
    Mv, mv = (sp.sympify(z.split("=")[1]) for z in s.split(","))
    ok1 = ok_fig and (Mv, mv) == (M, m)
    v2 = sp.simplify(m * sp.log(f.subs(x, M), 3) * sp.log(-1 / g.subs(x, m + 1), sp.Rational(1, 3)))
    ok2 = same(v2, 8, "2")
    report(8, ok1 and ok2, f"해집합={sol} 정수={ints} M={M} m={m} (2)={v2}")


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
