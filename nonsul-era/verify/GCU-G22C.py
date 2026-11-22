#!/usr/bin/env python3
"""가천대 2022학년도 논술고사 자연 C형 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G22C"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "GCU" / "PAST"
x, t, a, b, k, m = sp.symbols("x t a b k m", real=True)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def yans(seq):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return d["subparts"][0]["answer"]["value"]


def nums(s):
    """YAML answer 문자열에서 수(분수 포함)를 sympy 로 읽는다."""
    s = s.replace("\\log_2 3", "log(3,2)")
    return [sp.sympify(v) for v in re.findall(r"-?\d+/\d+|-?\d+|log\(3,2\)", s)]


def p07():
    y1 = sp.Rational(1, 2)**(x - 2) - 1
    yint = y1.subs(x, 0); xint = sp.solve(y1, x)[0]
    a_hi = sp.solve(sp.log(0 + a, 2) - yint, a)[0]
    a_lo = sp.solve(sp.log(xint + a, 2) - 0, a)[0]
    # 수치 확인: 범위 안/밖 a 에서 제1사분면 교점 존재 여부
    def meets(av):
        h = lambda xv: float(sp.log(xv + av, 2)) - float(y1.subs(x, xv))
        xs = [float(xint) * i / 400 for i in range(1, 400)]
        vals = [h(v) for v in xs if v + av > 0]
        return any(v1 * v2 < 0 for v1, v2 in zip(vals, vals[1:]))
    ya = yans(7)
    ok = (yint, xint, a_hi, a_lo) == (3, 2, 8, -1) and meets(3) and meets(-0.5) and not meets(9) and not meets(-1.5) \
        and "③ 8" in ya and "④ -1" in ya
    report(7, ok, f"절편 (0,{yint}),({xint},0), {a_lo}<a<{a_hi} (수치: a=3,-0.5 교점 있음, 9,-1.5 없음)")


def p08():
    d = (sp.log(8, 3) - sp.log(sp.Rational(1, 4), 3)) / 10
    terms = [3**(sp.log(sp.Rational(1, 4), 3) + kk*d) for kk in range(2, 10)]
    S = sp.nsimplify(sp.simplify(sum(terms)))
    target = sp.Rational(15, 2)*(sp.sqrt(2) + 1)
    ok = sp.simplify(d - sp.log(2, 3)/2) == 0 and sp.simplify(terms[0] - sp.Rational(1, 2)) == 0 \
        and sp.simplify(terms[1]/terms[0] - sp.sqrt(2)) == 0 and sp.simplify(S - target) == 0 and "15/2" in yans(8)
    report(8, ok, f"d={sp.simplify(d)}, 첫째항={sp.simplify(terms[0])}, 합={S}")


def p09():
    A = sp.symbols("A", positive=True)
    found = []
    for av in sp.solve(sp.Eq(sp.log(A + 2, A), 2), A):
        if av > 1:
            L = sp.log(av + 2, av)
            r = sp.simplify(L/(L + 1))
            f = 2*r**x
            vals = [f.subs(x, v) for v in (-2, 1)]
            if sp.simplify(max(vals) - sp.Rational(9, 2)) == 0:
                found.append((av, min(vals)))
    ok = found == [(2, sp.Rational(4, 3))] and "a=2" in yans(9).replace(" ", "") and "4/3" in yans(9)
    report(9, ok, f"(a, 최솟값)={found}")


def p10():
    roots = []
    for s in (sp.Rational(-1, 2), -1):
        for xv in sp.solveset(sp.sin(x) - s, x, sp.Interval.Ropen(0, 2*sp.pi)):
            if sp.cos(xv) != 0 and sp.simplify(3 + 3*sp.cos(xv)*sp.tan(xv) - 2*sp.cos(xv)**2) == 0:
                roots.append(xv)
    eqchk = sp.simplify((3 + 3*sp.sin(x) - 2*(1 - sp.sin(x)**2)) - (2*sp.sin(x)**2 + 3*sp.sin(x) + 1))
    tot = sum(roots)
    ok = eqchk == 0 and sp.simplify(tot - 3*sp.pi) == 0 and yans(10).strip() == "3π"
    report(10, ok, f"해={roots}, 합={tot}")


def p11():
    sols = sp.solve([sp.Eq(a*(1 + 2) + 1, a**2 + b), sp.Eq(-2*a, 2*a**2*(-1))], [a, b], dict=True)
    sols = [s for s in sols if s[a] != 0]
    s = sols[0]
    f = sp.Piecewise((s[a]*(1 - 2*x) + 1, x < -1), (s[a]**2*x**2 + s[b], True))
    cands = [-2, 2, -1, 0]
    vals = [f.subs(x, c) for c in cands]
    ya = yans(11).replace(" ", "")
    ok = len(sols) == 1 and max(vals) == 7 and min(vals) == 3 and "최댓값7" in ya and "최솟값3" in ya
    report(11, ok, f"(a,b)=({s[a]},{s[b]}), 후보값 {dict(zip(cands, vals))}")


def p12():
    seq = [sp.Integer(3)]
    def is_int_log(v):
        return v > 0 and sp.log(v, 2).is_integer
    while len(seq) < 1939:
        v = seq[-1]
        seq.append(sp.log(1/v, 2) if is_int_log(v) else v + 5)
    tot = sum(seq)
    ok = tot == int(yans(12))
    report(12, ok, f"앞 8항={seq[:8]}, 직접 합={tot}")


def p13():
    # [증명-수치확인] 무작위 다항식·a,b 에서 g(a)g(b)<0 이고 (a,b) 안에 근이 실제로 있음
    import random
    random.seed(1)
    ok = True
    for _ in range(20):
        coeffs = [random.randint(-5, 5) for _ in range(4)]
        f = sum(cf*x**i for i, cf in enumerate(coeffs))
        aa, bb = sorted(random.sample([sp.Rational(21) + sp.Rational(i, 10) for i in range(1, 10)], 2))
        if f.subs(x, aa) == f.subs(x, bb):
            continue
        g = 2*f - f.subs(x, aa) - f.subs(x, bb)
        sgn = g.subs(x, aa)*g.subs(x, bb) < 0
        ident = sp.simplify(g.subs(x, bb) + g.subs(x, aa)) == 0
        rts = [r for r in sp.Poly(g, x).nroots() if r.is_real and aa < r < bb]
        ok &= bool(sgn and ident and rts)
    ok &= yans(13) is None
    report(13, ok, "[증명-수치확인] 무작위 3차 다항식 20개: g(a)=-g(b), g(a)g(b)<0, (a,b) 안에 g(c)=0 인 c 존재")


def p14():
    f = x**3 + 2*x + 2
    g2 = [r for r in sp.solve(f - 2, x) if r.is_real][0]
    g5 = [r for r in sp.solve(f - 5, x) if r.is_real][0]
    val = g5*5 - g2*2 - sp.integrate(f, (x, g2, g5))
    # 수치: 역함수 적분 = ∫ x f'(x) dx (치환) 로 독립 확인
    val2 = sp.integrate(x*sp.diff(f, x), (x, g2, g5))
    ok = val == val2 == sp.sympify(yans(14))
    report(14, ok, f"g(2)={g2}, g(5)={g5}, ∫g={val} (치환 ∫x f'(x)dx={val2})")


def p15():
    f = (5*x**2 + 7*x + 2)/9
    A = (2, f.subs(x, 2)); m1 = sp.diff(f, x).subs(x, 2)
    av = sp.solve(sp.diff(f, x)*m1 + 1, x)[0]
    B = (av, f.subs(x, av)); m2 = sp.diff(f, x).subs(x, av)
    P = sp.solve([sp.Eq(b - A[1], m1*(a - A[0])), sp.Eq(b - B[1], m2*(a - B[0]))], [a, b])
    area = sp.Abs(sp.Matrix([[A[0], A[1], 1], [B[0], B[1], 1], [P[a], P[b], 1]]).det())/2
    ok = A[1] == 4 and area == sp.sympify(yans(15))
    report(15, ok, f"A={A}, B={B}, P=({P[a]},{P[b]}), 넓이={area}")


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
