#!/usr/bin/env python3
"""가천대 2023학년도 논술전형 자연E 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산하고 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G23E"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "GCU" / "PAST"
x, t, k, a, b, c, n, th = sp.symbols("x t k a b c n theta", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def yaml_answer(seq: int) -> str:
    it = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def tex(s: str):
    """간단한 LaTeX 수(분수·근호) → sympy."""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\ ", "")
    for _ in range(5):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"(\d)sqrt", r"\1*sqrt", s)
    s = re.sub(r"\)\(", ")*(", s)
    s = re.sub(r"\)sqrt", ")*sqrt", s)
    return sp.nsimplify(sp.sympify(s))


def same(v, s) -> bool:
    return sp.simplify(sp.nsimplify(v) - tex(s)) == 0


def nums(s: str):
    """YAML 집합형 답에서 수(분수·근호 포함)를 순서대로 추출."""
    parts = re.split(r"[①②③④]", s.replace("\\ ", " "))
    return [p.strip().strip(",").strip() for p in parts if p.strip()]


def cubic_level_k(f, lo, hi, minval):
    """닫힌구간 [lo,hi] 최솟값 = minval 로 상수 a 결정 후, y=k 와 교점 2개인 k 전체."""
    crit = [r for r in sp.solve(sp.diff(f, x), x) if r.is_real]
    cands = [f.subs(x, p) for p in crit if lo <= p <= hi] + [f.subs(x, lo), f.subs(x, hi)]
    at0 = [sp.simplify(e.subs(a, 0)) for e in cands]
    amin = cands[at0.index(min(at0))]
    aval = sp.solve(sp.Eq(amin, minval), a)[0]
    g = f.subs(a, aval)
    ks = []
    for kk in sorted({sp.nsimplify(g.subs(x, p)) for p in crit} | {sp.Integer(0)}):
        roots = sp.Poly(g - kk, x).real_roots()
        if len(set(roots)) == 2:
            ks.append(kk)
    return aval, ks


def p07():
    f = -x**3 + 3*x**2 + 9*x + a
    aval, ks = cubic_level_k(f, -1, 6, -40)
    s = sum(ks)
    report(7, same(s, yaml_answer(7)), f"a={aval}, 교점 2개인 k={ks}, 합={s}")


def p08():
    cs = [r for r in sp.solve(sp.Eq((1 - x**2) / x, sp.Rational(-8, 3)), x) if abs(r) <= 1]
    vals = set()
    for cv in cs:
        for tv in (sp.acos(cv), -sp.acos(cv)):
            assert sp.simplify(sp.sin(tv) * sp.tan(tv) + sp.Rational(8, 3)) == 0
            vals.add(sp.nsimplify(sp.simplify(sp.sin(3 * sp.pi / 2 - tv))))
    v = vals.pop()
    report(8, not vals and same(v, yaml_answer(8)), f"cos={cs}, sin(3π/2-θ)={v}")


def p09():
    A = sp.symbols("A", positive=True)
    seq = lambda kk: A + (kk - 1) * 2 * A
    s = sum(3 / (seq(kk) * seq(kk + 1)) for kk in range(1, 11))
    sol = sp.solve(sp.Eq(sp.simplify(s), sp.Rational(5, 7)), A)
    a3 = [sp.simplify(seq(3).subs(A, r)) for r in sol]
    report(10 - 1, len(a3) == 1 and same(a3[0], yaml_answer(9)), f"a={sol}, a3={a3}")


def p10():
    # f=(x-b)^2(x-c): x<=a 에서 f<=0 인 최대 a 를 수치 스캔으로 확인 → 그 값이 4 인 경우
    def sup_a(bv, cv):
        xs = [sp.Rational(i, 100) for i in range(-2000, 2000)]
        fx = lambda z: (z - bv)**2 * (z - cv)
        for z in xs:
            if fx(z) > 0:
                return z
        return None
    vals = []
    for bv in sp.solve(sp.Eq(-b**2 * 4, -16), b):        # f(0)=-b^2 c, 단근 c 후보 4
        sa = sup_a(bv, 4)
        assert abs(sa - 4) <= sp.Rational(1, 50), (bv, sa)
        vals.append(((x - bv)**2 * (x - 4)).subs(x, 3))
    # 단근이 4가 아니면 최댓값 4 조건 불가(중근 쪽은 부호가 안 바뀜) 확인: c=3 이면 sup≈3
    assert abs(sup_a(-sp.sqrt(sp.Rational(16, 3)), 3) - 3) <= sp.Rational(1, 50)
    m = min(vals)
    report(10, same(m, yaml_answer(10)), f"f(3) 후보={vals}, 최솟값={m}")


def p11():
    cnt = []
    for av in sp.solve(2*a**2 - 9*a + 4, a):
        for nv in range(1, 7):
            if av**(3*nv**2 - 17*nv) < av**(nv**2 - 30):
                cnt.append((av, nv))
    report(11, same(len(cnt), yaml_answer(11)), f"순서쌍={cnt}")


def p12():
    eqs = sp.Poly(a*x**2 - 2*b*x + c - a*(x - 3)**2, x).coeffs() + [a + c - 10]
    sol = [s for s in sp.solve(eqs, [a, b, c], dict=True) if all(v != 0 for v in s.values())]
    got = [int(v) for v in re.findall(r"=\s*(-?\d+)", yaml_answer(12))]
    ok = len(sol) == 1 and [sol[0][a], sol[0][b], sol[0][c]] == got
    report(12, ok, f"(a,b,c)={sol}")


def p13():
    u, A = sp.symbols("u A", real=True)
    p0, p1, p2 = sp.symbols("p0 p1 p2")
    f = u**2 * (p0 + p1*u + p2*u**2)            # 극한 존재 → (x-a)^2 인수 필요
    ex = sp.series(2/f - 5/u**2, u, 0, 1).removeO()
    eqs = [ex.coeff(u, -2), ex.coeff(u, -1), ex.coeff(u, 0) - 5]
    sol = sp.solve(eqs, [p0, p1, p2], dict=True)[0]
    F = f.subs(sol)
    Av = [r for r in sp.solve(sp.Eq(F.subs(u, 1 - A), sp.Rational(-24, 5)), A) if r.is_real and r > 0]
    val = [sp.simplify(F.subs(u, 8 - r)) for r in Av]
    report(13, len(val) == 1 and same(val[0], yaml_answer(13)), f"계수={sol}, a={Av}, f(8)={val}")


def p14():
    R = 6 / (2 * sp.sin(sp.pi / 3))
    P = lambda deg: sp.Matrix([R * sp.cos(sp.rad(deg)), R * sp.sin(sp.rad(deg))])
    # 호: DA=2∠ABD=90°, AB=2∠ADB=150°, BC=2∠BAC=60°, CD=2∠CAD=60°
    A_, B_, C_, D_ = P(0), P(150), P(210), P(270)
    ang = lambda U, V, W: float(sp.acos(((U - V).dot(W - V)) / ((U - V).norm() * (W - V).norm())))
    import math
    assert abs(ang(B_, A_, D_) - math.pi/3) < 1e-12 and abs(ang(A_, B_, D_) - math.pi/4) < 1e-12
    assert abs(ang(B_, A_, C_) - ang(C_, A_, D_)) < 1e-12 and abs(float((B_ - D_).norm()) - 6) < 1e-12
    s1, s2 = sp.symbols("s1 s2")
    ss = sp.solve(list(A_ + s1*(C_ - A_) - B_ - s2*(D_ - B_)), [s1, s2])
    E_ = A_ + ss[s1]*(C_ - A_)
    area = lambda U, V, W: sp.Abs(((V - U)[0]*(W - U)[1] - (V - U)[1]*(W - U)[0]) / 2)
    BC = sp.simplify((B_ - C_).norm()); x2 = sp.simplify((A_ - B_).norm()**2)
    ABC = sp.simplify(area(A_, B_, C_)); ST = sp.simplify(area(A_, B_, E_) - area(C_, D_, E_))
    want = nums(yaml_answer(14))
    got = [BC, x2, ABC, ST]
    ok = all(same(g, w) for g, w in zip(got, want)) and len(want) == 4
    report(14, ok, f"BC={BC}, x^2={x2}, △ABC={ABC}, S-T={ST}")


def p15():
    left = x**3 - 3*a*x**2 + 2*a*x + 1
    av = sp.solve(sp.Eq(left.subs(x, 1), -4 + 3), a)[0]
    fpw = sp.Piecewise((left.subs(a, av), x <= 1), (-4*x + 3, True))
    val = sp.integrate(fpw, (x, -1, 9)) - sp.integrate(fpw, (x, 2, 9))
    wrong = sp.integrate(left.subs(a, av), (x, -1, 2))
    report(15, same(val, yaml_answer(15)), f"a={av}, 값={val} (구간 미분할 오답 {wrong})")


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
