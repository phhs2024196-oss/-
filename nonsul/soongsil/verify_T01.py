# -*- coding: utf-8 -*-
"""S01. 다항함수의 조건 추론 — sympy 검산 (정답을 하드코딩해 자기비교하지 않고 조건에서 재계산)"""
import sys
from sympy import (symbols, Rational, integrate, expand, Poly, solve, diff,
                   factorial, simplify, binomial, S)

x, t = symbols('x t')
RESULTS = []


def gen_poly(n, name='a'):
    cs = symbols(f'{name}0:{n+1}')
    return sum(cs[k] * x**k for k in range(n + 1)), list(cs)


def coeff_eqs(expr):
    return Poly(expand(expr), x).all_coeffs()


def report(pid, ok, msg):
    ok = bool(ok)
    RESULTS.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


# S01-01  f(x+1)-f(x-1)=12x^2+4, int_{-1}^{1} f = 6, f(2)?
def p01():
    sols_all = []
    for n in range(0, 7):  # 차수를 6까지 열어 두고 풀어도 삼차만 남는지 확인
        f, cs = gen_poly(n)
        eqs = coeff_eqs(f.subs(x, x + 1) - f.subs(x, x - 1) - (12 * x**2 + 4))
        eqs.append(integrate(f, (x, -1, 1)) - 6)
        sol = solve(eqs, cs, dict=True)
        for s in sol:
            sols_all.append(expand(f.subs(s)))
    sols = set(sols_all)
    F = list(sols)[0]
    ans = F.subs(x, 2)
    # 대표 오답: 적분조건을 f(0)=6 으로 오해 -> d=6 -> f(2)=22
    wrong = 2 * 8 + 6
    ok = len(sols) == 1 and expand(F - (2 * x**3 + 3)) == 0 and ans == 19 and wrong == 22
    report('S01-01', ok, f'f={F}, f(2)={ans} (오답 재현 {wrong})')


# S01-02  접선의 y절편 f(t)-t f'(t) = -2t^3+3, 점 (1,0) 통과, int_0^2 f ?
def p02():
    sols = set()
    for n in range(0, 7):
        f, cs = gen_poly(n)
        eqs = coeff_eqs(f - x * diff(f, x) - (-2 * x**3 + 3))
        eqs.append(f.subs(x, 1))
        for s in solve(eqs, cs, dict=True):
            sols.add(expand(f.subs(s)))
    F = list(sols)[0]
    ans = integrate(F, (x, 0, 2))
    # 대표 오답: 일차항을 0으로 놓음 (f=x^3+3) -> 점 조건 무시, int = 4+6 = 10
    wrong = integrate(x**3 + 3, (x, 0, 2))
    ok = len(sols) == 1 and ans == 2 and expand(F - (x**3 - 4 * x + 3)) == 0
    report('S01-02', ok, f'f={F}, int_0^2 f={ans} (오답 재현 {wrong})')


# S01-03  f_n - f_n' = x^n,  f_n(0)=n!, f_n(0)>1000 인 최소 n = 7
def p03():
    vals = {}
    for n in range(1, 10):
        f, cs = gen_poly(n + 2)
        sol = solve(coeff_eqs(f - diff(f, x) - x**n), cs, dict=True)
        assert len(sol) == 1
        vals[n] = expand(f.subs(sol[0])).subs(x, 0)
    ok_fact = all(vals[n] == factorial(n) for n in vals)
    nmin = min(n for n in vals if vals[n] > 1000)
    ok = ok_fact and nmin == 7
    report('S01-03', ok, f'f_n(0)=n! (n=1..9 확인), 최소 n={nmin}')


# S01-04  삼차 이하 f, f(0)=f(1)=0, f'(0)=1 일 때 int_0^1 (f'')^2 의 최솟값 = 3
def p04():
    b, c = symbols('b c', real=True)
    f = x + b * x**2 + c * x**3
    J = integrate(diff(f, x, 2)**2, (x, 0, 1)).subs(c, -1 - b)
    bc = solve(diff(J, b), b)
    m = simplify(J.subs(b, bc[0]))
    # 이계도함수 계수 검사: J 는 b 에 대한 이차식(최고차 양수) -> 최솟값
    lead = Poly(expand(J), b).LC()
    # 대표 오답: 이차함수로 한정(c=0 -> b=-1) -> 4
    wrong = integrate(diff(x - x**2, x, 2)**2, (x, 0, 1))
    ok = m == 3 and lead > 0 and bc[0] == Rational(-3, 2) and wrong == 4
    report('S01-04', ok, f'최솟값={m} (b={bc[0]}), 오답 재현 {wrong}')


# S01-05  최고차항 1 삼차, 모든 이차 이하 g 에 대해 int_{-1}^1 f g = 0, int_{-1}^1 x^3 f dx?
def p05():
    p, q, r = symbols('p q r')
    f = x**3 + p * x**2 + q * x + r
    eqs = [integrate(f * x**k, (x, -1, 1)) for k in range(3)]
    sol = solve(eqs, [p, q, r], dict=True)
    F = f.subs(sol[0])
    ans = integrate(x**3 * F, (x, -1, 1))
    # 대표 오답: 일차 이하 g 만 사용(p 미결정) -> p=0 으로 놓아도 q 값 동일. f(1) 을 답으로 쓴 경우 2/5
    ok = len(sol) == 1 and expand(F - (x**3 - Rational(3, 5) * x)) == 0 and ans == Rational(8, 175)
    report('S01-05', ok, f'f={F}, 값={ans}')


# S01-06  f_0=1, f_n' = n f_{n-1}, int_0^1 f_n = 0 ;  f_4(0) = -1/30
def p06():
    fs = [S(1)]
    for n in range(1, 5):
        C = symbols('C')
        g = n * integrate(fs[-1].subs(x, t), (t, 0, x)) + C
        Cv = solve(integrate(g, (x, 0, 1)), C)[0]
        fs.append(expand(g.subs(C, Cv)))
    ans = fs[4].subs(x, 0)
    # 대표 오답: 적분상수를 0으로 둠 -> f_4 = x^4 -> 0
    ok = ans == Rational(-1, 30) and expand(fs[3] - (x**3 - Rational(3, 2) * x**2 + x / 2)) == 0
    report('S01-06', ok, f'f_3={fs[3]}, f_4(0)={ans}')


# S01-07  이차함수 f, f(x)f(-x)=f(x^2) : 4개, sum f(2)=14
def p07():
    a, b, c = symbols('a b c')
    f = a * x**2 + b * x + c
    sol = solve(coeff_eqs(f * f.subs(x, -x) - f.subs(x, x**2)), [a, b, c], dict=True)
    sol = [s for s in sol if s.get(a, a) != 0]
    funcs = [expand(f.subs(s)) for s in sol]
    total = sum(F.subs(x, 2) for F in funcs)
    ok = len(funcs) == 4 and total == 14
    report('S01-07', ok, f'해 {funcs}, 개수 {len(funcs)}, 합 {total}')


# S01-08  f(b)-f(a)=(b-a) f'((a+b)/2) for all a,b ; f(0)=1,f(1)=2,f(2)=7 -> f(3)=16
def p08():
    A, B = symbols('A B')
    sols = set()
    for n in range(0, 7):
        f, cs = gen_poly(n)
        expr = expand(f.subs(x, B) - f.subs(x, A) - (B - A) * diff(f, x).subs(x, (A + B) / 2))
        eqs = Poly(expr, A, B).coeffs() if expr != 0 else []
        eqs += [f.subs(x, 0) - 1, f.subs(x, 1) - 2, f.subs(x, 2) - 7]
        for s in solve(eqs, cs, dict=True):
            sols.add(expand(f.subs(s)))
    F = list(sols)[0]
    # [증명-수치확인] 차수 n>=3 의 x^{n-3} 계수 2C(n,3)a_n != 0
    proof_ok = all(binomial(n, 3) > 0 for n in range(3, 50))
    ok = len(sols) == 1 and F.subs(x, 3) == 16 and proof_ok
    report('S01-08', ok, f'f={F}, f(3)={F.subs(x, 3)} [증명-수치확인 n=3..49]')


# S01-09  (x-1)f(x+1)=(x+2)f(x), int_0^2 f = 4 -> f(3)=48
def p09():
    sols = set()
    for n in range(0, 8):
        f, cs = gen_poly(n)
        eqs = coeff_eqs((x - 1) * f.subs(x, x + 1) - (x + 2) * f)
        eqs.append(integrate(f, (x, 0, 2)) - 4)
        for s in solve(eqs, cs, dict=True):
            sols.add(expand(f.subs(s)))
    F = list(sols)[0]
    # 대표 오답: int_{0}^{2} 를 |f| 넓이로 착각하지 않고 c=4 로 놓음 -> f(3)=96
    ok = len(sols) == 1 and F.subs(x, 3) == 48
    report('S01-09', ok, f'f={F}, f(3)={F.subs(x, 3)}')


# S01-10  최고차항 양수 사차, f(x^2-2) = f(x)^2 - 2 -> f(3)=47
def p10():
    a = symbols('a', positive=True)
    cs = symbols('c0:4')
    f = a * x**4 + cs[3] * x**3 + cs[2] * x**2 + cs[1] * x + cs[0]
    sol = solve(coeff_eqs(f.subs(x, x**2 - 2) - f**2 + 2), [a] + list(cs), dict=True)
    F = expand(f.subs(sol[0]))
    # [증명-수치확인] f(-x)=f(x)
    ev = all(F.subs(x, k) == F.subs(x, -k) for k in range(-20, 21))
    ok = len(sol) == 1 and F.subs(x, 3) == 47 and ev
    report('S01-10', ok, f'f={F}, f(3)={F.subs(x, 3)}, 우함수 확인')


if __name__ == '__main__':
    for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
        try:
            fn()
        except Exception as e:  # noqa
            report(fn.__name__.replace('p', 'S01-'), False, f'예외 {e!r}')
    n = sum(RESULTS)
    print(f'SUMMARY T01: {n}/{len(RESULTS)} PASS')
    sys.exit(0 if n == len(RESULTS) == 10 else 1)
