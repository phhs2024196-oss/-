# -*- coding: utf-8 -*-
"""S02. 절댓값·구간별 함수의 연속성과 미분가능성 — sympy/수치 검산"""
import sys
import itertools
from sympy import (symbols, Rational, integrate, expand, Poly, solve, diff, Abs,
                   Piecewise, limit, oo, sqrt, Max, nsimplify, S, real_roots, simplify)
import mpmath as mp

mp.mp.dps = 40
x, h = symbols('x h', real=True)
n = symbols('n', positive=True, integer=True)
RESULTS = []


def report(pid, ok, msg):
    ok = bool(ok)
    RESULTS.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def one_sided(F, x0, eps=mp.mpf('1e-15')):
    """수치 한쪽 미분계수 (mpmath, 40자리)"""
    x0 = mp.mpf(x0)
    r = (F(x0 + eps) - F(x0)) / eps
    l = (F(x0) - F(x0 - eps)) / eps
    return l, r


def is_diff(F, x0, tol=mp.mpf('1e-8')):
    l, r = one_sided(F, x0)
    return abs(l - r) < tol


def count_nondiff(F, cands):
    return sum(0 if is_diff(F, c) else 1 for c in cands)


# S02-01  (x^2+ax+b)|x-1| 미분가능, g(3)=8 -> int_0^3 g = 17/4
def p01():
    a, b = symbols('a b', real=True)
    q = x**2 + a * x + b
    g = q * Abs(x - 1)
    # x=1 에서 좌우 미분계수 (h->0+) 를 기호적으로 계산
    R = limit((g.subs(x, 1 + h) - g.subs(x, 1)) / h, h, 0, '+')
    L = limit((g.subs(x, 1 - h) - g.subs(x, 1)) / (-h), h, 0, '+')
    sol = solve([R - L, g.subs(x, 3) - 8], [a, b], dict=True)
    G = g.subs(sol[0])
    ans = integrate(G, (x, 0, 3))
    ok = len(sol) == 1 and ans == Rational(17, 4)
    report('S02-01', ok, f'(a,b)={sol[0]}, int={ans}')


# S02-02  0 (x<0), p (0<=x<=2), 4 (x>2), p 삼차 이하, f 미분가능 -> int_{-1}^{3} f = 8
def p02():
    c = symbols('c0:4')
    p = sum(c[k] * x**k for k in range(4))
    eqs = [p.subs(x, 0), diff(p, x).subs(x, 0), p.subs(x, 2) - 4, diff(p, x).subs(x, 2)]
    sol = solve(eqs, c, dict=True)
    P = expand(p.subs(sol[0]))
    ans = integrate(P, (x, 0, 2)) + 4 * 1
    # 대표 오답: 연속만 고려해 일차식 p=2x -> 4+4 = 8 이 아니라 ... p=2x 이면 int_0^2 = 4 -> 8 (같음)
    #           미분가능만 쓰고 p(2)=4 를 빠뜨려 p 결정 실패. 대신 넓이를 int_0^2 p 만 답 -> 4
    ok = len(sol) == 1 and ans == 8 and expand(P - (3 * x**2 - x**3)) == 0
    report('S02-02', ok, f'p={P}, int={ans}')


# S02-03  g=|x^2-2|x|-k| 비미분점 개수 N(k);  N(-1+)+N(-1)+N(0) = 9
def p03():
    def N(k):
        k = mp.mpf(k)
        F = lambda t: abs(t**2 - 2 * abs(t) - k)
        cands = [mp.mpf(0)]
        for s in (1, -1):  # 양수부 근 |x| = 1 ± sqrt(1+k)
            if 1 + k >= 0:
                for r in (1 + mp.sqrt(1 + k), 1 - mp.sqrt(1 + k)):
                    if r > 0:
                        cands.append(s * r)
        cands = sorted(set([mp.nstr(c, 30) for c in cands]))
        return count_nondiff(F, [mp.mpf(c) for c in cands])
    table = {k: N(k) for k in ['-2', '-1', '-0.999', '-0.5', '-0.001', '0', '0.5', '3']}
    ans = N('-0.999') + N('-1') + N('0')
    ok = ans == 9 and table['-2'] == 1 and table['-0.5'] == 5 and table['3'] == 3
    report('S02-03', ok, f'N 표본 {table}, 값={ans}')


# S02-04  f=x (x<1), x-2 (x>=1); g=x^2+ax+b ; g∘f, f∘g 연속 -> min g(2) = 5
def p04():
    def f(t):
        return t if t < 1 else t - 2

    def cont_ok(a, b):
        g = lambda t: t * t + a * t + b
        # g∘f 연속: x=1 에서 좌극한 g(1) = g(f(1)) = g(-1)
        if abs(g(1) - g(-1)) > 1e-12:
            return False
        # f∘g 연속: g(x0)=1 인 점 근방에서 수치 점검
        disc = a * a - 4 * (b - 1)
        if disc < 0:
            return True
        r = mp.sqrt(disc)
        for x0 in ((-a + r) / 2, (-a - r) / 2):
            e = mp.mpf('1e-12')
            vals = [f(g(x0 + e)), f(g(x0 - e)), f(g(x0))]
            if max(vals) - min(vals) > 1e-6:
                return False
        return True
    feas = [(a, b) for a in [mp.mpf(i) / 4 for i in range(-12, 13)]
            for b in [mp.mpf(j) / 4 for j in range(-12, 21)] if cont_ok(a, b)]
    vals = [4 + 2 * a + b for a, b in feas]
    mn = min(vals)
    ok = mn == 5 and all(a == 0 and b >= 1 for a, b in feas)
    report('S02-04', ok, f'가능한 (a,b)는 a=0,b>=1 ({len(feas)}개 격자점), min g(2)={mn}')


# S02-05  g = lim (x^{2n} f + p)/(x^{2n}+1), f=x^4+cx, p=ax^2+b, 미분가능 -> int_{-2}^{2} g = 176/15
def p05():
    a, b, c = symbols('a b c', real=True)
    f = x**4 + c * x
    p = a * x**2 + b
    # 극한 식이 구간별로 무엇이 되는지 표본점에서 sympy limit 로 확인
    expr = (x**(2 * n) * f + p) / (x**(2 * n) + 1)
    chk = []
    for xv in [Rational(1, 2), Rational(-1, 3), 2, -3]:
        lv = limit(expr.subs(x, xv).subs({a: 2, b: -1, c: 0}), n, oo)
        tv = (p if abs(xv) < 1 else f).subs(x, xv).subs({a: 2, b: -1, c: 0})
        chk.append(simplify(lv - tv) == 0)
    g1 = limit(expr.subs(x, 1), n, oo)
    gm1 = limit(expr.subs(x, -1), n, oo)
    eqs = [p.subs(x, 1) - f.subs(x, 1), g1 - p.subs(x, 1),
           p.subs(x, -1) - f.subs(x, -1), gm1 - p.subs(x, -1),
           diff(p, x).subs(x, 1) - diff(f, x).subs(x, 1),
           diff(p, x).subs(x, -1) - diff(f, x).subs(x, -1)]
    sol = solve(eqs, [a, b, c], dict=True)
    s = sol[0]
    ans = integrate(p.subs(s), (x, -1, 1)) + 2 * integrate(f.subs(s), (x, 1, 2))
    ok = all(chk) and len(sol) == 1 and ans == Rational(176, 15)
    report('S02-05', ok, f'(a,b,c)={s}, int={ans}, 구간별 극한 확인')


# S02-06  f=ax^3+bx^2+cx on [0,2), f(x+2)=f(x)+2, 미분가능 -> int_0^6 f = 18 (a 무관)
def p06():
    a, b, c = symbols('a b c', real=True)
    f = a * x**3 + b * x**2 + c * x
    eqs = [f.subs(x, 2) - (f.subs(x, 0) + 2), diff(f, x).subs(x, 2) - diff(f, x).subs(x, 0)]
    sol = solve(eqs, [b, c], dict=True)[0]
    F = f.subs(sol)
    I = sum(integrate(F + 2 * m, (x, 0, 2)) for m in range(3))
    I = simplify(I)
    # 수치: a=0.7 일 때 실제 확장 함수로 적분 및 x=2,4 에서 미분가능
    av = mp.mpf('0.7')
    Fn = lambda t: (lambda u, m: float(F.subs({a: 0.7, x: u})) + 2 * m)(t - 2 * int(mp.floor(t / 2)), int(mp.floor(t / 2)))
    num = mp.quad(lambda t: Fn(t), [0, 2, 4, 6])
    d_ok = all(abs(one_sided(lambda t: mp.mpf(Fn(t)), x0, mp.mpf('1e-6'))[0]
                   - one_sided(lambda t: mp.mpf(Fn(t)), x0, mp.mpf('1e-6'))[1]) < 1e-4 for x0 in (2, 4))
    ok = I == 18 and not I.free_symbols and abs(num - 18) < 1e-9 and d_ok
    report('S02-06', ok, f'b,c={sol}, int_0^6={I} (a 무관), 수치 a=0.7 -> {mp.nstr(num, 12)}')


# S02-07  max{x^2+ax+b, |x|} 미분가능 -> min int_{-1}^{1} p = 7/6
def p07():
    best = None
    grid_a = [mp.mpf(i) / 20 for i in range(-40, 41)]
    grid_b = [mp.mpf(j) / 40 for j in range(-20, 81)]
    for av in grid_a:
        for bv in grid_b:
            p = lambda t: t * t + av * t + bv
            H = lambda t: max(p(t), abs(t))
            # 후보 비미분점: 0 과 p=|x| 의 교점
            cands = [mp.mpf(0)]
            for s in (1, -1):  # p(x) = s x
                A, B, C = 1, av - s, bv
                D = B * B - 4 * A * C
                if D >= 0:
                    for r in ((-B + mp.sqrt(D)) / 2, (-B - mp.sqrt(D)) / 2):
                        if s * r >= 0:
                            cands.append(r)
            if count_nondiff(H, cands) == 0:
                val = mp.mpf(2) / 3 + 2 * bv
                if best is None or val < best:
                    best = val
    # 기호: 조건 p>=|x| 에서 x=±1/2 대입 합으로 b>=1/4
    ok = abs(best - Rational(7, 6)) < 1e-30
    report('S02-07', ok, f'격자 탐색 최솟값={mp.nstr(best, 15)} (=7/6)')


# S02-08  a,b in [-3,3] 정수, g=(x-1)|x^2+ax+b| 비미분점 개수 합 = 57
def p08():
    total = 0
    for a, b in itertools.product(range(-3, 4), repeat=2):
        F = lambda t, a=a, b=b: (t - 1) * abs(t * t + a * t + b)
        D = a * a - 4 * b
        cands = []
        if D >= 0:
            cands = [(-a + mp.sqrt(D)) / 2, (-a - mp.sqrt(D)) / 2]
            if D == 0:
                cands = cands[:1]
        total += count_nondiff(F, cands)
    ok = total == 57
    report('S02-08', ok, f'49개 순서쌍 전수, 합={total}')


# S02-09  최고차항 1 사차, f(1)=0, f(|x|), |f(x)| 미분가능 -> f(2) 최대 9, 최소 4
def p09():
    p, q = symbols('p q', real=True)
    f = (x - 1)**2 * (x**2 + p * x + q)
    s = solve(diff(f, x).subs(x, 0), p)[0]  # p = 2q
    f2 = expand(f.subs(x, 2).subs(p, s))
    # 판별식 <= 0 인 q 범위
    disc = expand((s**2 - 4 * q))
    qs = solve(disc, q)
    qmin, qmax = min(qs), max(qs)
    M, m = f2.subs(q, qmax), f2.subs(q, qmin)
    # 수치: q 를 [0,1] 및 바깥에서 표본, 실제 미분가능성 검사
    def check(qv):
        P = lambda t: (t - 1)**2 * (t * t + 2 * qv * t + qv)
        F1 = lambda t: abs(P(t))
        F2 = lambda t: P(abs(t))
        roots = [mp.mpf(1)]
        D = 4 * qv * qv - 4 * qv
        if D >= 0:
            roots += [(-2 * qv + mp.sqrt(D)) / 2, (-2 * qv - mp.sqrt(D)) / 2]
        return count_nondiff(F1, roots) == 0 and is_diff(F2, 0)
    inside = all(check(mp.mpf(k) / 10) for k in range(0, 11))
    outside = not any(check(mp.mpf(v)) for v in ('-0.5', '1.5', '2'))
    # 다른 꼴 (x-1)^3(x-s) 은 s=1 이외에 단순근, (x-1)^4 은 f'(0)=-4
    other = diff((x - 1)**4, x).subs(x, 0) != 0
    ok = (M, m) == (9, 4) and inside and outside and other
    report('S02-09', ok, f'p=2q, q in [{qmin},{qmax}], f(2)=4+5q -> 최대 {M}, 최소 {m}')


# S02-10  g=max{f(x),f(-x)}, f=x^3+ax^2+bx+d; x=2 비미분, g'(1)=3, g(0)=1 -> int_{-2}^{2} g = 52/3
def p10():
    a, b, d = symbols('a b d', real=True)
    f = x**3 + a * x**2 + b * x + d
    g = Max(f, f.subs(x, -x))
    # 수치로 a,b,d 결정 확인: 후보 b 는 O(2)=0 에서
    bsol = solve(((f - f.subs(x, -x)) / 2).subs(x, 2), b)[0]
    O = expand(((f - f.subs(x, -x)) / 2).subs(b, bsol))
    E = expand((f + f.subs(x, -x)) / 2)
    gexpr = E + Abs(O)
    asol = solve(diff(E - O, x).subs(x, 1) - 3, a)[0]  # x=1 에서 O<0
    dsol = solve(gexpr.subs(x, 0) - 1, d)[0]
    vals = {a: asol, b: bsol, d: dsol}
    G = g.subs(vals)
    ans = integrate(E.subs(vals), (x, -2, 2)) + 2 * integrate(-O.subs(vals), (x, 0, 2))
    # 수치 교차검사: Max 그대로 적분, x=2 비미분, x=1 미분계수
    Gn = lambda t: mp.mpf(float(G.subs(x, t))) if False else max(
        t**3 + float(asol) * t**2 + float(bsol) * t + float(dsol),
        -t**3 + float(asol) * t**2 - float(bsol) * t + float(dsol))
    num = mp.quad(Gn, [-2, 0, 2])
    nd2 = not is_diff(Gn, 2)
    l, r = one_sided(Gn, 1, mp.mpf('1e-10'))
    ok = ans == Rational(52, 3) and abs(num - ans) < 1e-9 and nd2 and abs(r - 3) < 1e-6 and O.subs(x, 1) < 0
    report('S02-10', ok, f'(a,b,d)=({asol},{bsol},{dsol}), int={ans}, 수치 {mp.nstr(num, 12)}')


if __name__ == '__main__':
    for i, fn in enumerate([p01, p02, p03, p04, p05, p06, p07, p08, p09, p10], 1):
        try:
            fn()
        except Exception as e:  # noqa
            report(f'S02-{i:02d}', False, f'예외 {e!r}')
    k = sum(RESULTS)
    print(f'SUMMARY T02: {k}/{len(RESULTS)} PASS')
    sys.exit(0 if k == len(RESULTS) == 10 else 1)
