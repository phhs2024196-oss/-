#!/usr/bin/env python3
"""S04. 정적분으로 정의된 함수 — 검산 스크립트 (숭실 T04)
각 문항의 설정에서 출발해 sympy/mpmath 로 정답을 독립적으로 다시 계산한다."""
import sys
import random
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
results = []
x, t = sp.symbols('x t', real=True)


def report(pid, ok, msg):
    results.append(bool(ok))
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


# ---------------------------------------------------------------- S04-01
def p01():
    a = sp.symbols('a', real=True)
    g = x**4 + a*x**3
    f = sp.diff(g, x, 2)                     # g'' = f
    # 확인: ∫_0^x (x-t) f(t) dt == g
    ok = sp.simplify(sp.integrate((x - t)*f.subs(x, t), (t, 0, x)) - g) == 0
    cond = sp.limit(sp.integrate(f.subs(x, t), (t, 1, x))/(x - 1), x, 1)
    asol = sp.solve(sp.Eq(cond, 0), a)
    ok &= asol == [-2]
    gg = g.subs(a, -2)
    crit = sp.solve(sp.diff(gg, x), x)
    mn = min(gg.subs(x, c) for c in crit)
    # 전역 최솟값(4차, 최고차계수 양수): 임계점에서의 최솟값
    ok &= mn == sp.Rational(-27, 16)
    wrong = sp.Rational(0)  # 극값 x=0 (변곡 겸 정류점)을 최솟값으로 착각
    report('S04-01', ok, f'a={asol}, g=x^4-2x^3, 최솟값 {mn}; 오답(x=0) {wrong}')


# ---------------------------------------------------------------- S04-02
def p02():
    F = sp.integrate((t**2 - 3*t + 2)*sp.exp(t), (t, 0, x))
    Fp = sp.diff(F, x)
    crit = sorted(sp.solve(sp.simplify(Fp*sp.exp(-x)), x))
    ok = crit == [1, 2]
    # 부호: 1 에서 +→-, 2 에서 -→+
    ok &= Fp.subs(x, sp.Rational(1, 2)) > 0 and Fp.subs(x, sp.Rational(3, 2)) < 0 and Fp.subs(x, 3) > 0
    diff = sp.simplify(F.subs(x, 1) - F.subs(x, 2))
    ok &= sp.simplify(diff - (3*sp.E - sp.E**2)) == 0
    report('S04-02', ok, f'F(1)={sp.simplify(F.subs(x,1))}, F(2)={sp.simplify(F.subs(x,2))}, 차 {diff}')


# ---------------------------------------------------------------- S04-03
def p03():
    def N(av):
        g = sp.integrate((t - 1)*(t - av), (t, 0, x))
        return len(set(sp.Poly(sp.expand(6*g), x).real_roots()))
    # 판별식 조건으로 얻은 경계 후보 + 격자 전수
    a2 = []
    grid = [sp.Rational(k, 12) for k in range(-60, 61)] + [sp.Rational(1, 3), sp.Integer(3), sp.Integer(0)]
    vals = {}
    for av in grid:
        vals[av] = N(av)
    a2 = sorted({av for av, n in vals.items() if n == 2})
    ok = a2 == [0, sp.Rational(1, 3), 3]
    # 구간별 확인
    ok &= all(vals[av] == 1 for av in grid if sp.Rational(1, 3) < av < 3)
    ok &= all(vals[av] == 3 for av in grid if (av < sp.Rational(1, 3) or av > 3) and av != 0)
    # 무작위 비격자 점 확인
    random.seed(1)
    for _ in range(20):
        av = sp.Rational(random.randint(-3000, 3000), 997)
        n = N(av)
        exp = 2 if av in (0, sp.Rational(1, 3), 3) else (1 if sp.Rational(1, 3) < av < 3 else 3)
        ok &= n == exp
    report('S04-03', ok, f'N(a)=2 인 a = {a2}, 합 {sum(a2)} (1/3<a<3 에서 1, 나머지 3)')


# ---------------------------------------------------------------- S04-04
def p04():
    k = sp.symbols('k', positive=True)
    # 1 <= k <= e 에서 e^t = k 인 t = ln k
    c = sp.log(k)
    G = sp.integrate(k - sp.exp(t), (t, 0, c)) + sp.integrate(sp.exp(t) - k, (t, c, 1))
    G = sp.simplify(G)
    crit = sp.solve(sp.diff(G, k), k)
    ok = crit == [sp.exp(sp.Rational(1, 2))]
    mn = sp.simplify(G.subs(k, crit[0]))
    ok &= sp.simplify(mn - (sp.sqrt(sp.E) - 1)**2) == 0
    # 수치: k 전역 탐색 (k<1, k>e 포함)
    f = lambda kv: mp.quad(lambda tt: abs(mp.e**tt - kv), [0, mp.log(kv) if 1 < kv < mp.e else 0.5, 1])
    best = min(f(mp.mpf(i)/1000) for i in range(0, 4000))
    ok &= abs(best - (mp.sqrt(mp.e) - 1)**2) < 1e-5
    wrong = sp.simplify(G.subs(k, sp.E - 1))  # 평균값 ∫e^t = e-1 을 넣는 오답
    report('S04-04', ok, f'최소점 x={crit[0]}, 최솟값 {mn}; 오답(x=e-1) {sp.N(wrong, 8)}')


# ---------------------------------------------------------------- S04-05
def p05():
    k = sp.symbols('k', integer=True, nonnegative=True)
    n = sp.symbols('n', integer=True, positive=True)
    # k 번째 구간 적분
    Ik = sp.integrate(t*sp.sin(t)*(-1)**k, (t, k*sp.pi, (k + 1)*sp.pi))
    ok = all(sp.simplify(Ik.subs(k, kv) - (2*kv + 1)*sp.pi) == 0 for kv in range(6))
    Sn = sp.summation((2*k + 1)*sp.pi, (k, 0, n - 1))
    ok &= sp.simplify(Sn - n**2*sp.pi) == 0
    # 샌드위치 하계·상계
    N_ = sp.symbols('N', positive=True)
    lo = sp.limit(N_**2*sp.pi/((N_ + 1)**2*sp.pi**2), N_, sp.oo)
    hi = sp.limit((N_ + 1)**2*sp.pi/(N_**2*sp.pi**2), N_, sp.oo)
    ok &= lo == hi == 1/sp.pi
    # 수치 확인 (비배수 x)
    X = mp.mpf(1000.7)
    val = mp.quad(lambda s: s*abs(mp.sin(s)), mp.linspace(0, X, 400))/X**2
    ok &= abs(val - 1/mp.pi) < 2e-3
    report('S04-05', ok, f'∫_(kπ)^((k+1)π) t|sin t| = (2k+1)π, 합 n^2π, 극한 {lo}; x=1000.7 수치 {mp.nstr(val, 8)}')


# ---------------------------------------------------------------- S04-06
def p06():
    fn = x
    a_list = []
    fs = [fn]
    for _ in range(60):
        an = sp.integrate(fn, (x, 0, 1))
        a_list.append(an)
        fn = sp.expand(x**2 + x*an)
        fs.append(fn)
    aL = sp.Rational(2, 3)
    nn = sp.symbols('nn', integer=True, positive=True)
    # 일반항 a_n = 2/3 - (1/6)(1/2)^(n-1) 확인
    ok = all(a_list[i] == aL - sp.Rational(1, 6)*sp.Rational(1, 2)**i for i in range(60))
    F = x**2 + aL*x
    ok &= abs(sp.N(fs[-1].subs(x, 2) - F.subs(x, 2))) < 1e-15
    # Σ ∫(F - f_n) = Σ (2/3 - a_n)
    total = sp.summation(sp.Rational(1, 6)*sp.Rational(1, 2)**(nn - 1), (nn, 1, sp.oo))
    part = sum(sp.integrate(F - fs[i], (x, 0, 1)) for i in range(60))
    ok &= total == sp.Rational(1, 3) and abs(sp.N(part - total)) < 1e-15
    report('S04-06', ok, f'a_n → 2/3, F(x)={F}, Σ∫(F-f_n) = {total}')


# ---------------------------------------------------------------- S04-07
def p07():
    f = x**3 + x
    y = sp.symbols('y', real=True)
    # 역함수 값 g(10): f(x)=10
    r = [s for s in sp.solve(f - 10, x) if s.is_real]
    ok = r == [2]
    # G(10) = ∫_0^10 g(t) dt — 수치로 직접(역함수 수치 계산) & 넓이 공식
    G10_formula = 2*10 - sp.integrate(f, (x, 0, 2))
    g = lambda tv: mp.findroot(lambda z: z**3 + z - tv, 1)
    G10_num = mp.quad(g, [0, 10])
    ok &= G10_formula == 14 and abs(G10_num - 14) < 1e-20
    # 극한: lim (G'(10+h)-G'(10))/h = g'(10) = 1/f'(2)
    h = mp.mpf(10)**-15
    num = (g(10 + h) - g(10 - h))/(2*h)
    gp = 1/sp.diff(f, x).subs(x, 2)
    ok &= gp == sp.Rational(1, 13) and abs(num - mp.mpf(1)/13) < 1e-12
    wrong = sp.diff(f, x).subs(x, 10)  # f'(10) 을 쓰는 오답
    report('S04-07', ok, f'G(10)={G10_formula} (수치 {mp.nstr(G10_num, 12)}), 극한 {gp}; 오답 f\'(10)={wrong}')


# ---------------------------------------------------------------- S04-08
def p08():
    a = sp.symbols('a', real=True)
    g = sp.integrate((t - a)*sp.exp(-t), (t, 0, x))
    ok = sp.simplify(g - ((1 - a) - (x - a + 1)*sp.exp(-x))) == 0

    def count_pos_roots(av):
        gf = sp.lambdify(x, g.subs(a, av), 'mpmath')
        xs = [mp.mpf(i)/50 for i in range(1, 3000)]
        cnt = 0
        for i in range(len(xs) - 1):
            if gf(xs[i]) == 0 or gf(xs[i])*gf(xs[i + 1]) < 0:
                cnt += 1
        return cnt
    res = {}
    for av in [sp.Rational(k, 10) for k in range(-10, 16)]:
        res[av] = count_pos_roots(av)
    good = sorted(av for av, c in res.items() if c >= 1)
    ok &= all((0 < av < 1) == (res[av] == 1) for av in res) and all(c <= 1 for c in res.values())
    report('S04-08', ok, f'양의 실근 존재하는 격자 a = {good[0]}..{good[-1]} (0<a<1 에서 정확히 1개), 그 외 0개')


# ---------------------------------------------------------------- S04-09
def p09():
    f0 = 3*x**2 + 2*x
    ok = sp.expand(sp.integrate(f0.subs(x, t), (t, x, 2*x)) - (7*x**3 + 3*x**2)) == 0
    # 미분한 관계식 2f(2x)-f(x) = 21x^2+6x 를 만족하는 다항식 해의 유일성 (차수 ≤ 6)
    cs = sp.symbols('c0:7')
    P = sum(c*x**i for i, c in enumerate(cs))
    sol = sp.solve(sp.Poly(sp.expand(2*P.subs(x, 2*x) - P - 21*x**2 - 6*x), x).coeffs(), cs, dict=True)
    ok &= len(sol) == 1 and sp.expand(P.subs(sol[0]) - f0) == 0
    # [증명-수치확인] 동차해 h: h(x)=2h(2x) 이면 h(x/2^n)=2^n h(x) — 연속 비자명 해 불가.
    # 예: h(x)=x^(-1) 는 동차식을 만족하나 x=0 에서 불연속 (반례 후보 확인)
    hcand = 1/x
    ok &= sp.simplify(2*hcand.subs(x, 2*x) - hcand) == 0
    val = f0.subs(x, 3)
    ok &= val == 33
    report('S04-09', ok, f'f=3x^2+2x 유일(다항 차수≤6), 동차해는 원점에서 불연속 [증명-수치확인], f(3)={val}')


# ---------------------------------------------------------------- S04-10
def p10():
    X = sp.symbols('X', positive=True)
    # ln2 - h(x) = ∫_x^{2x} (1-e^{-t})/t dt ; 부등식 1 - t/2 <= (1-e^{-t})/t <= 1 [증명-수치확인]
    ok = True
    for i in range(1, 1001):
        tv = mp.mpf(i)/100
        q = (1 - mp.e**(-tv))/tv
        ok &= (1 - tv/2 <= q <= 1)
    lo = sp.integrate(1 - t/2, (t, X, 2*X))
    hi = sp.integrate(sp.Integer(1), (t, X, 2*X))
    L1 = sp.limit(lo/X, X, 0, '+'); L2 = sp.limit(hi/X, X, 0, '+')
    ok &= L1 == L2 == 1
    # 수치: 직접 계산
    for xv in [mp.mpf('1e-4'), mp.mpf('1e-6')]:
        hval = mp.quad(lambda s: mp.e**(-s)/s, [xv, 2*xv])
        r = (mp.log(2) - hval)/xv
        ok &= abs(r - 1) < 3*xv
    # 단조감소: h'(x) = (e^{-2x}-e^{-x})/x < 0
    integrand = sp.exp(-t)/t
    hp = integrand.subs(t, 2*X)*sp.diff(2*X, X) - integrand.subs(t, X)   # 라이프니츠(적분의 미분)
    ok &= sp.simplify(hp - (sp.exp(-2*X) - sp.exp(-X))/X) == 0
    ok &= all(hp.subs(X, sp.Rational(k, 7)) < 0 for k in range(1, 30))
    report('S04-10', ok, f'샌드위치 하계 {L1}, 상계 {L2} → 극한 1; x=1e-6 수치 확인')


if __name__ == '__main__':
    for f in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
        try:
            f()
        except Exception as e:  # noqa
            report(f.__name__.replace('p', 'S04-'), False, f'예외 {e!r}')
    n_ok = sum(results)
    print(f'SUMMARY T04: {n_ok}/{len(results)} PASS')
    sys.exit(0 if n_ok == len(results) == 10 else 1)
