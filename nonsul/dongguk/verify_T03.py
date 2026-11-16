#!/usr/bin/env python3
"""D03 극대·극소와 매개변수에 따른 분류 — sympy 검산.
각 문항의 설정으로부터 독립적으로 다시 계산해 md의 정답과 비교한다."""
import sys
import numpy as np
import sympy as sp
import mpmath as mp

x, a, t, k = sp.symbols('x a t k', real=True)
E = sp.E
results = []


def report(pid, ok, msg):
    ok = bool(ok)
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def count_sign_changes(fp, lo, hi, n=20001):
    """수치 격자에서 f'의 부호 변화 횟수(=극값 개수)."""
    xs = np.linspace(lo, hi, n)
    v = np.array([float(fp(xx)) for xx in xs])
    s = np.sign(v)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


def n_extrema_exact(fexpr, var, domain=sp.S.Reals):
    """f'의 실근 중 부호가 바뀌는 근(홀수 중복도)의 개수."""
    fp = sp.factor(sp.diff(fexpr, var))
    num, _ = sp.fraction(sp.together(fp))
    num = sp.factor(num)
    cnt = 0
    for r in sp.solveset(num, var, domain):
        # 근 좌우 부호 비교
        eps = sp.Rational(1, 10**6)
        l = fp.subs(var, r - eps).evalf(30)
        rr = fp.subs(var, r + eps).evalf(30)
        if l * rr < 0:
            cnt += 1
    return cnt


# ---------------- D03-01 ----------------
def p01():
    f = sp.exp(x) / (x**2 + a)
    fp = sp.simplify(sp.diff(f, x))
    # (1) a=1: f'(1)=0, 부호변화 없음
    f1 = f.subs(a, 1)
    fp1 = sp.factor(sp.diff(f1, x))
    ok1 = sp.simplify(fp1.subs(x, 1)) == 0 and n_extrema_exact(f1, x) == 0
    # (2) 극값을 갖는 a 범위: 0<a<1 (a>0 가정)
    ok2 = all(n_extrema_exact(f.subs(a, av), x) == 2 for av in [sp.Rational(1, 10), sp.Rational(1, 2), sp.Rational(9, 10)]) \
        and all(n_extrema_exact(f.subs(a, av), x) == 0 for av in [1, sp.Rational(3, 2), 5])
    num = sp.fraction(sp.together(fp))[0]
    ok2 = ok2 and sp.factor(num / sp.exp(x)) == sp.factor(x**2 - 2*x + a)
    # (3) 극댓값·극솟값의 곱 = e^2 이 되는 a
    al = 1 - sp.sqrt(1 - a)
    be = 1 + sp.sqrt(1 - a)
    prod = sp.simplify(f.subs(x, al) * f.subs(x, be))
    ok3a = sp.simplify(prod - E**2 / (4*a)) == 0
    sol = sp.solve(sp.Eq(prod, E**2), a)
    sol = [s for s in sol if s.is_real and 0 < s < 1]
    ok3 = ok3a and sol == [sp.Rational(1, 4)]
    # 대표 오답: a<=1 로 등호 포함 → a=1 에서 극값 0개 확인
    report("D03-01", ok1 and ok2 and ok3,
           f"(1) f'(1)=0 & 극값 0개, (2) 0<a<1, (3) 곱=e^2/(4a) → a={sol}")


# ---------------- D03-02 ----------------
def p02():
    f = x**4 - 4*x**3 + 2*a*x**2
    # (1) a=-4 극값
    f1 = f.subs(a, -4)
    crit = sorted(sp.solve(sp.diff(f1, x), x))
    vals = [f1.subs(x, c) for c in crit]
    ok1 = crit == [-1, 0, 4] and vals == [-3, 0, -128]
    # (2) 극값 3개 ⇔ a<9/4, a≠0
    ok2 = True
    for av in [-5, -1, sp.Rational(1, 2), 2, sp.Rational(11, 5)]:
        ok2 &= n_extrema_exact(f.subs(a, av), x) == 3
    for av in [0, sp.Rational(9, 4), 3, 10]:
        ok2 &= n_extrema_exact(f.subs(a, av), x) == 1
    # (3) 두 극솟값이 같은 a: 수치 탐색 + 기호 확인
    found = []
    for av in [sp.Rational(n, 40) for n in range(-200, 90) if n != 0]:
        fa = f.subs(a, av)
        fpa = sp.diff(fa, x)
        cr = sorted(sp.solve(fpa, x), key=lambda r: float(r))
        if len(cr) != 3:
            continue
        mins = [c for c in cr if sp.diff(fa, x, 2).subs(x, c) > 0]
        if len(mins) == 2 and sp.simplify(fa.subs(x, mins[0]) - fa.subs(x, mins[1])) == 0:
            found.append(av)
    # 기호: 나머지 식 f ≡ (4a-9)x+3a-a^2 (mod x^2-3x+a)
    rem = sp.rem(sp.Poly(f, x), sp.Poly(x**2 - 3*x + a, x)).as_expr()
    ok3 = found == [2] and sp.expand(rem - ((4*a - 9)*x + 3*a - a**2)) == 0
    report("D03-02", ok1 and ok2 and ok3, f"(1) 극값 -3,0,-128, (2) a<9/4,a≠0, (3) a={found}, 나머지식 확인")


# ---------------- D03-03 ----------------
def p03():
    f = x**3 - 3*(a + 1)*x**2 + 12*a*x - 8*a
    fp = sp.factor(sp.diff(f, x))
    ok0 = sp.expand(fp - 3*(x - 2)*(x - 2*a)) == 0
    f3 = f.subs(a, 3)
    ok1 = f3.subs(x, 2) == 8 and f3.subs(x, 6) == -24
    ok2 = sp.expand(f.subs(x, 2) - 4*(a - 1)) == 0 and sp.expand(f.subs(x, 2*a) + 4*a*(a - 1)*(a - 2)) == 0

    def nroots(av):
        return len(sp.Poly(f.subs(a, av), x).real_roots()) and len(set(sp.Poly(f.subs(a, av), x).real_roots()))
    tests = {-3: 3, -1: 3, sp.Rational(-1, 10): 3, 0: 2, sp.Rational(1, 10): 1, 1: 1, sp.Rational(3, 2): 1,
             sp.Rational(19, 10): 1, 2: 2, sp.Rational(21, 10): 3, 5: 3}
    ok3 = all(nroots(av) == n for av, n in tests.items())
    report("D03-03", ok0 and ok1 and ok2 and ok3,
           "f'=3(x-2)(x-2a); a=3 극대 8 극소 -24; 실근 개수 a<0,a>2:3 / a=0,2:2 / 0<a<2:1")


# ---------------- D03-04 ----------------
def p04():
    f = (x**2 + a*x + 1)*sp.exp(x)
    fp = sp.factor(sp.diff(f, x))
    ok1 = sp.simplify(fp - sp.exp(x)*(x + 1)*(x + a + 1)) == 0
    ok1 &= n_extrema_exact(f.subs(a, 0), x) == 0 and n_extrema_exact(f.subs(a, 1), x) == 2

    def local_min(av):
        fa = f.subs(a, av)
        cr = sp.solve(sp.diff(fa, x), x)
        mins = [c for c in cr if sp.diff(fa, x, 2).subs(x, c).evalf() > 0]
        return fa.subs(x, mins[0])
    ok2 = sp.simplify(local_min(3) - (2 - 3)/E) == 0 and sp.simplify(local_min(-3) - (-3 + 2)*sp.exp(2)) == 0
    # (3) m(a) 최댓값: 수치 탐색
    grid = [i/200 for i in range(-1200, 1201) if i != 0]
    vals = [(float(local_min(sp.Rational(int(g*200), 200))), g) for g in grid[::7]]
    mval, marg = max(vals)
    # 정밀: a<0 구간 m(a)=(a+2)e^{-a-1}
    m_neg = (a + 2)*sp.exp(-a - 1)
    crit = sp.solve(sp.diff(m_neg, a), a)
    ok3 = crit == [-1] and m_neg.subs(a, -1) == 1 and abs(mval - 1) < 1e-3 and abs(marg + 1) < 0.05
    report("D03-04", ok1 and ok2 and ok3, f"f'=e^x(x+1)(x+a+1), 극값 없음 a=0, 극솟값의 최댓값 1 (a=-1); 수치 max={mval:.6f}@{marg}")


# ---------------- D03-05 ----------------
def p05():
    f = (x + a)/(x**2 + 1)
    num = sp.fraction(sp.together(sp.diff(f, x)))[0]
    ok1 = sp.expand(num + (x**2 + 2*a*x - 1)) == 0
    x1, x2 = -a + sp.sqrt(a**2 + 1), -a - sp.sqrt(a**2 + 1)
    ok1 &= sp.simplify(f.subs(x, x1) - 1/(2*x1)) == 0
    P = (x1, f.subs(x, x1)); Q = (x2, f.subs(x, x2))
    slope = sp.simplify((P[1] - Q[1])/(P[0] - Q[0]))
    icpt = sp.simplify(P[1] - slope*P[0])
    ok2 = slope == sp.Rational(1, 2) and sp.simplify(icpt - a) == 0
    area = sp.Abs(P[0]*Q[1] - Q[0]*P[1])/2
    sols = []
    for av in [sp.Rational(n, 4) for n in range(-40, 41)]:
        if abs(float(area.subs(a, av)) - float(2*sp.sqrt(5))) < 1e-12:
            sols.append(av)
    ok3 = sp.simplify((area.subs(a, 2))**2 - 20) == 0 and sols == [-2, 2]
    # 단조성 확인 (a>0에서 넓이 증가) → 해 유일
    ok3 &= all(float(area.subs(a, i/10)) < float(area.subs(a, (i+1)/10)) for i in range(1, 60))
    report("D03-05", ok1 and ok2 and ok3, f"극점 y=1/(2x), PQ 기울기 {slope}, y절편 a, 넓이=2√5 ⇔ a={sols}")


# ---------------- D03-06 ----------------
def p06():
    xp = sp.symbols('xp', positive=True)
    f = a*xp**2 - xp + sp.log(xp)
    cnt = {}
    for av in [-2, -sp.Rational(1, 3), 0, sp.Rational(1, 20), sp.Rational(1, 9), sp.Rational(1, 8), sp.Rational(1, 5), 1]:
        fp = sp.lambdify(xp, sp.diff(f.subs(a, av), xp))
        cnt[av] = count_sign_changes(fp, 1e-4, 200, 200001)
    ok1 = [cnt[v] for v in cnt] == [1, 1, 1, 2, 2, 0, 0, 0]
    # (2) S(a)
    ap = sp.symbols('ap', positive=True)
    al = (1 - sp.sqrt(1 - 8*ap))/(4*ap)
    be = (1 + sp.sqrt(1 - 8*ap))/(4*ap)
    fa = f.subs(a, ap)
    S = fa.subs(xp, al) + fa.subs(xp, be)
    target = -1/(4*ap) - 1 - sp.log(2*ap)
    ok2 = all(abs(float((S - target).subs(ap, v))) < 1e-12 for v in [0.01, 0.05, 0.1, 0.12])
    # (3) 치역 (-inf, -3+2ln2)
    ok3 = sp.simplify(sp.diff(target, ap) - (1 - 4*ap)/(4*ap**2)) == 0
    ok3 &= sp.simplify(target.subs(ap, sp.Rational(1, 8)) - (-3 + 2*sp.log(2))) == 0
    ok3 &= float(target.subs(ap, 1e-6)) < -1e5
    report("D03-06", ok1 and ok2 and ok3, "극값 개수 a≤0:1/0<a<1/8:2/a≥1/8:0; S=-1/(4a)-1-ln2a, 범위 (-∞,-3+2ln2)")


# ---------------- D03-07 ----------------
def p07():
    def M_num(av):
        mp.mp.dps = 30
        g = lambda xx: mp.log(xx) - av*xx
        xs = [1 + (mp.e - 1)*i/4000 for i in range(4001)]
        best = max(g(xx) for xx in xs)
        if av > 0 and 1 <= 1/av <= mp.e:
            best = max(best, g(1/mp.mpf(av)))
        return best
    ok1 = abs(M_num(0.5) - (mp.log(2) - 1)) < 1e-12
    M = sp.Piecewise((1 - a*E, a <= 1/E), (-sp.log(a) - 1, a < 1), (-a, True))
    ok2 = all(abs(M_num(av) - float(M.subs(a, av))) < 1e-7 for av in [-1, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2])
    I = sp.integrate(1 - a*E, (a, 0, 1/E)) + sp.integrate(-sp.log(a) - 1, (a, 1/E, 1)) + sp.integrate(-a, (a, 1, 2))
    ok3 = sp.simplify(I - (-sp.Rational(3, 2) - 1/(2*E))) == 0
    Inum = mp.quad(lambda s: M_num(float(s)), [0, 1/mp.e, 1, 2], maxdegree=4)
    ok3 &= abs(Inum - (-1.5 - 1/(2*mp.e))) < 1e-5
    # 대표 오답: 가운데 구간을 빠뜨리고 끝값 비교만 → ∫max(1-ae,-a)
    report("D03-07", ok1 and ok2 and ok3, f"M(1/2)=ln2-1, M(a) 3구간, ∫_0^2 M = {sp.nsimplify(I)} ≈ {float(I):.6f}")


# ---------------- D03-08 ----------------
def p08():
    f = a*x - sp.log(x**2 + 1)
    exp_cnt = {-2: 0, -1: 0, -sp.Rational(1, 2): 2, 0: 1, sp.Rational(1, 2): 2, sp.Rational(9, 10): 2, 1: 0, 3: 0}
    ok1 = all(n_extrema_exact(f.subs(a, av), x) == n for av, n in exp_cnt.items())
    # (2) αβ=1, 차 D(α)
    al = sp.symbols('al', positive=True)
    aa = 2*al/(al**2 + 1)
    D = (f.subs(a, aa).subs(x, al) - f.subs(a, aa).subs(x, 1/al))
    Dform = 2*(al**2 - 1)/(al**2 + 1) - 2*sp.log(al)
    ok2 = all(abs(float((D - Dform).subs(al, v))) < 1e-12 for v in [0.1, 0.3, 0.7, 0.95])
    ok2 &= sp.simplify(sp.diff(aa*x**2 - 2*x + aa, x).subs(x, al)) != 0  # α 단순근
    ok2 &= sp.simplify((aa*x**2 - 2*x + aa).subs(x, al)) == 0 and sp.simplify((aa*x**2 - 2*x + aa).subs(x, 1/al)) == 0
    # (3) D = ln3 -1 인 a 유일, a=√3/2
    dD = sp.simplify(sp.diff(Dform, al))
    ok3 = sp.simplify(dD + 2*(al**2 - 1)**2/(al*(al**2 + 1)**2)) == 0
    sol = sp.nsolve(Dform - (sp.log(3) - 1), al, 0.5)
    ok3 &= abs(sol - 1/sp.sqrt(3)) < 1e-20
    ok3 &= sp.simplify(aa.subs(al, 1/sp.sqrt(3)) - sp.sqrt(3)/2) == 0
    report("D03-08", ok1 and ok2 and ok3, "극값 개수 |a|≥1:0/0<|a|<1:2/a=0:1; D(α) 감소; D=ln3-1 ⇔ a=√3/2")


# ---------------- D03-09 ----------------
def p09():
    xp = sp.symbols('xp', positive=True)
    f = xp*sp.log(xp) - a*xp**2
    fp_half = sp.diff(f.subs(a, sp.Rational(1, 2)), xp)
    ok1 = fp_half.subs(xp, 1) == 0 and max(float(fp_half.subs(xp, v)) for v in np.linspace(0.01, 30, 3000)) <= 1e-12
    exp_cnt = {-1: 1, 0: 1, sp.Rational(1, 10): 2, sp.Rational(1, 4): 2, sp.Rational(49, 100): 2, sp.Rational(1, 2): 0, 1: 0}
    ok2 = True
    got = {}
    for av, n in exp_cnt.items():
        fp = sp.lambdify(xp, sp.diff(f.subs(a, av), xp))
        got[av] = count_sign_changes(fp, 1e-6, 400, 400001)
        ok2 &= got[av] == n
    # (3) 극댓값 >0 ⇔ 0<a<1/e
    def locmax(av):
        fp = lambda s: mp.log(s) + 1 - 2*av*s
        be = mp.findroot(fp, (1.0000001, 1e6), solver='bisect') if False else None
        # 이분법: (1, 큰수)에서 f' 부호 + → -
        lo, hi = mp.mpf(1), mp.mpf(10)
        while fp(hi) > 0:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi)/2
            if fp(mid) > 0:
                lo = mid
            else:
                hi = mid
        b = (lo + hi)/2
        return b*mp.log(b) - av*b*b
    ok3 = all(locmax(av) > 0 for av in [0.05, 0.2, 0.36]) and all(locmax(av) < 0 for av in [0.37, 0.4, 0.49])
    ok3 &= abs(locmax(1/mp.e)) < 1e-25
    report("D03-09", ok1 and ok2 and ok3, f"a=1/2 f'≤0; 극값 개수 {list(got.values())}; 극댓값>0 ⇔ 0<a<1/e")


# ---------------- D03-10 ----------------
def p10():
    f = sp.exp(x) + sp.exp(-x) + a*x**2
    fp = sp.diff(f, x)
    ok1 = sp.simplify(fp.subs(x, -x) + fp) == 0 and fp.subs(x, 0) == 0
    # a≥-1 : x>0 에서 f'>0
    xs = np.linspace(1e-3, 20, 4000)
    g = sp.lambdify((x, a), fp)
    ok1 &= all(g(xx, -1.0) > 0 for xx in xs)
    # (2) k(x)=(e^x-e^{-x})/x 증가, m(x)=x(e^x+e^{-x})-(e^x-e^{-x}) > 0
    kfun = (sp.exp(x) - sp.exp(-x))/x
    m = x*(sp.exp(x) + sp.exp(-x)) - (sp.exp(x) - sp.exp(-x))
    ok2 = sp.simplify(sp.diff(kfun, x) - m/x**2) == 0 and sp.simplify(sp.diff(m, x) - x*(sp.exp(x) - sp.exp(-x))) == 0
    ok2 &= sp.limit(kfun, x, 0, '+') == 2
    # (3) 극값 개수: a≥-1 → 1, a<-1 → 3, 극댓값 2
    cnt = {}
    for av in [-5, -2, -1.2, -1.01, -1, -0.5, 0, 3]:
        cnt[av] = count_sign_changes(lambda s: g(s, av), -12, 12, 240002)
    ok3 = [cnt[v] for v in cnt] == [3, 3, 3, 3, 1, 1, 1, 1]
    ok3 &= all(sp.diff(f, x, 2).subs({x: 0, a: av}) < 0 for av in [-5, -2, -sp.Rational(101, 100)])
    ok3 &= f.subs(x, 0) == 2
    report("D03-10", ok1 and ok2 and ok3, f"f' 홀함수, k 증가·k→2; 극값 개수 {list(cnt.values())} (1 또는 3), 극댓값 2")


for p in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        p()
    except Exception as e:  # noqa
        report(p.__name__, False, f"예외 {e!r}")

n = sum(results)
print(f"SUMMARY T03: {n}/{len(results)} PASS")
sys.exit(0 if n == len(results) == 10 else 1)
