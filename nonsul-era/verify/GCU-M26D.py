#!/usr/bin/env python3
"""가천대 2026학년도 의예 논술 모의고사 수학(문제 1~4) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-M26D"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)




def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, t = sp.symbols("a t", real=True)
    X = 2 * sp.cos(t) + 2 * sp.sqrt(3) * a * sp.sin(t)**2
    Y = sp.sqrt(3) / 32 * (sp.tan(t) + a * sp.cot(t))
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    # vx=0 (sin t≠0) → cos t = 1/(2√3 a), vy=0 → tan^2 t = a. t 를 소거해 a 결정
    sols = []
    for av in sp.solve(sp.Eq(sp.tan(sp.acos(1 / (2 * sp.sqrt(3) * a)))**2, a), a):
        if av.is_real and av > 0:
            tv = sp.acos(1 / (2 * sp.sqrt(3) * av))
            if 0 < tv < sp.pi / 2 and sp.simplify(vx.subs({a: av, t: tv})) == 0 and sp.simplify(vy.subs({a: av, t: tv})) == 0:
                acc = sp.sqrt(sp.diff(X, t, 2)**2 + sp.diff(Y, t, 2)**2).subs({a: av, t: tv})
                sols.append((av, tv, sp.nsimplify(sp.simplify(acc))))
    ok = len(sols) == 1 and sols[0][0] == sp.Rational(1, 3) and sols[0][2] == sp.Rational(2, 3)
    report(1, ok, f"(a, t, b) = {sols} (YAML a=1/3, b=2/3)")


def p02():
    m, a, p, q = sp.symbols("m a p q", positive=True)
    xA = 8
    xB = xA * (1 + m**2)                 # A 에서 l 에 수직인 직선의 x절편
    xD = xB * (1 + m**2)
    AB = sp.sqrt((xB - xA)**2 + (m * xA)**2)
    CD = sp.sqrt((xD - xB)**2 + (m * xB)**2)
    msol = [s for s in sp.solve(sp.Eq(3 * AB, 2 * CD), m) if s > 0][0]
    yA, xC = msol * xA, xB.subs(m, msol)
    yC = msol * xC
    asol = sp.solve(sp.Eq(sp.log(1 + xA) / sp.log(a), yA), a)[0]       # A 는 y=log_a(1+x) 위
    eqs = [sp.Eq(sp.log(1 + xA - p) / sp.log(asol) + q, yA), sp.Eq(sp.log(1 + xC - p) / sp.log(asol) + q, yC)]
    sol = sp.solve(eqs, [p, q], dict=True)[0]
    val = sp.nsimplify(sp.N(3**(sol[q] / (2 * sp.sqrt(2))), 40), [sp.Integer(1)])
    ok = (sp.simplify(asol - 3**(1 / (2 * sp.sqrt(2)))) == 0 and sol[p] == 7 and val == sp.Rational(9, 2) and xC == 12)
    report(2, ok, f"기울기={msol}, C=({xC},{yC}), a={asol}, p={sol[p]}, 3^(q/2√2)={val} (YAML a=3^(1/(2√2)), p=7, 9/2)")


def c_val(k, a):
    """c_k = Σ 1/(b_n b_{n+1}), b_n = max(a_n,10). 앞부분은 직접 합, 뒷부분(모두 a_n≥10)은 부분분수 망원합."""
    n, s = 1, sp.Integer(0)
    an = lambda j: a + (j - 1) * k
    b = lambda j: 10 if an(j) < 10 else an(j)
    while an(n) < 10:
        s += sp.Rational(1, 1) / (b(n) * b(n + 1))
        n += 1
    return s + sp.Rational(1, 1) / (k * an(n))           # Σ_{j≥n} 1/(a_j a_{j+1}) = 1/(k a_n)


def p03():
    # c_5 = 17/500 을 만족하는 정수가 아닌 a 찾기: p(=a_n≥10 첫 n)별로 c_5 는 a 의 단조식 → sympy 로 방정식 풀이
    A = sp.symbols("A", real=True)
    cands = []
    for p in range(1, 60):
        expr = sp.Rational(p - 1, 100) if p >= 2 else 0
        an_p = A + 5 * (p - 1)
        expr = (sp.Rational(p - 2, 100) + 1 / (10 * an_p) if p >= 2 else 0) + 1 / (5 * an_p)
        for s in sp.solve(sp.Eq(expr, sp.Rational(17, 500)), A):
            if s.is_real and not s.is_integer and s + 5 * (p - 1) >= 10 and (p == 1 or s + 5 * (p - 2) < 10):
                cands.append(s)
    cands = sorted(set(cands))
    ok_a = cands == [sp.Rational(5, 2)]
    a = cands[0]
    c2, c10 = c_val(2, a), c_val(10, a)
    kk = sp.symbols("kk", positive=True)
    ck_large = 1 / (10 * (a + kk)) + 1 / (kk * (a + kk))      # k≥8 이면 a_1<10≤a_2
    ok_form = all(c_val(j, a) == ck_large.subs(kk, j) for j in range(8, 30))
    lim = sp.limit(kk * ck_large, kk, sp.oo)
    ok = ok_a and ok_form and c2 == sp.Rational(61, 700) and c10 == sp.Rational(2, 125) and lim == sp.Rational(1, 10)
    report(3, ok and c_val(5, a) == sp.Rational(17, 500), f"a={cands}, c2={c2}, c10={c10}, lim k c_k={lim} (YAML 61/700, 2/125, 1/10)")


def p04():
    a = sp.symbols("a", negative=True)
    b = (1 / sp.E - 1) / 2                        # 2b+1 = 1/e
    # 역함수 평행이동 곡선 y=8a x^3+2b x 와 x=|y| (반사하면 넓이 보존)
    al2 = sp.solve(sp.Eq(8 * a * x**2 + 2 * b, -1), x**2)
    al = sp.sqrt(-(2 * b + 1) / (8 * a))
    A_expr = sp.integrate((8 * a * x**3 + 2 * b * x) - (-x), (x, 0, al))
    asol = sp.solve(sp.Eq(sp.simplify(A_expr), 1 / (160 * sp.E**2)), a)
    # 수치 독립 확인: 원래 좌표에서 f = (g 의 역함수 조건) → 평행이동된 f 곡선과 y=|x| 사이 넓이를 수치적분
    import mpmath as mpm
    av = float(asol[0])
    bv = float(b)
    # 원래 좌표: f 의 그래프 = {(f^{-1}(u), u)} 를 (-1,-1) 평행이동 → (8a v^3+2b v, v). y=|x| 와 만나는 v 를 수치로 찾고
    # 수평 단면(가로 폭)으로 넓이를 적분: 폭 = (곡선의 x) - (직선 y=-x 의 x=-v)
    Xc = lambda v: 8 * av * v**3 + 2 * bv * v
    vlow = mpm.findroot(lambda v: Xc(v) + v, -0.1)
    num = mpm.quad(lambda v: abs(Xc(v) - (-v)), [vlow, 0])
    ok = asol == [-5] and abs(num - 1 / (160 * mpm.e**2)) < 1e-12
    report(4, ok, f"A(a)={sp.simplify(A_expr)}, a={asol}, 수치넓이={num} (YAML -5)")


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
