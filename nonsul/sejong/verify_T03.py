# -*- coding: utf-8 -*-
"""세종 J03 역함수의 미분과 적분 — 검산 스크립트.
각 문항의 설정(f)으로부터 sympy 기호계산 또는 mpmath 고정밀 수치(역함수는 findroot)로
정답을 독립적으로 다시 계산하여 md의 정답과 비교한다."""
import sys, random
import sympy as sp
import mpmath as mp

mp.mp.dps = 30
x, y, t, s = sp.symbols('x y t s', real=True)
E, PI = sp.E, sp.pi
results = []


def close(a, b, tol=1e-9):
    a, b = mp.mpf(a), mp.mpf(b)
    return abs(a - b) <= tol * max(1, abs(b))


def inv(fn, yv, x0, lo=None, hi=None):
    """mpmath 역함수: fn(x)=yv 를 푸는 x (구간이 주어지면 이분법)."""
    if lo is not None:  # 증가함수 가정 이분법
        lo, hi = mp.mpf(lo), mp.mpf(hi)
        for _ in range(200):
            mid = (lo + hi) / 2
            if fn(mid) < yv:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return mp.findroot(lambda z: fn(z) - yv, x0)


def report(pid, ok, msg):
    ok = bool(ok)
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


# ---------------- J03-01 : f(x)=e^x+x ----------------
def p01():
    f = sp.exp(x) + x
    fp, fpp = sp.diff(f, x), sp.diff(f, x, 2)
    # (1) 증명-수치확인: f'>0 격자 1000점
    ok1 = all(float(fp.subs(x, v)) > 0 for v in [-50 + 0.1 * k for k in range(1000)])
    # (2) g(1)=0 (f(0)=1), g'(1)=1/f'(0), g''(1)=-f''(0)/f'(0)^3
    x0 = sp.solve(sp.Eq(f, 1), x)
    g1 = 1 / fp.subs(x, 0)
    g2 = -fpp.subs(x, 0) / fp.subs(x, 0) ** 3
    ok2 = x0 == [0] and g1 == sp.Rational(1, 2) and g2 == sp.Rational(-1, 8)
    # g''<0 : -f''/f'^3 at 1000 points
    ok2b = all(float((-fpp / fp ** 3).subs(x, v)) < 0 for v in [-20 + 0.04 * k for k in range(1000)])
    # (3) 넓이 = ∫_1^{e+1} [(x-1)/2 - g(x)] dx ; ∫ g = ∫_0^1 x f'(x) dx (y=f(x) 치환)
    intg = sp.integrate(x * fp, (x, 0, 1))
    area = sp.integrate((y - 1) / 2, (y, 1, E + 1)) - intg
    ans = E ** 2 / 4 - sp.Rational(3, 2)
    # 독립 수치: mpmath 역함수 직접 적분
    fn = lambda z: mp.e ** z + z
    num = mp.quad(lambda yy: (yy - 1) / 2 - inv(fn, yy, 0.5), [1, mp.e + 1])
    ok3 = sp.simplify(area - ans) == 0 and close(num, sp.N(ans, 30))
    # 대표오답: 접선 기울기를 f'(0)=2 로 쓴 경우 넓이
    wrong = sp.integrate(2 * (y - 1), (y, 1, E + 1)) - intg
    report('J03-01', ok1 and ok2 and ok2b and ok3,
           f"g'(1)=1/2, g''(1)=-1/8, 넓이={sp.simplify(area)} (수치 {mp.nstr(num, 12)}), [증명-수치확인] f'>0,g''<0; 오답(기울기2)={sp.simplify(wrong)}")


# ---------------- J03-02 : f(x)=x+ln x ----------------
def p02():
    f = x + sp.log(x)
    fn = lambda z: z + mp.log(z)
    # (1) 교점이 y=x 위에만 있음: f(a)=b, f(b)=a 의 수치해 탐색
    found = set()
    for a0 in [0.2 + 0.3 * k for k in range(40)]:
        try:
            a = mp.findroot(lambda z: fn(fn(z)) - z if fn(z) > 0 else 1e9, a0)
            if a > 0 and fn(a) > 0 and abs(fn(fn(a)) - a) < 1e-15:
                found.add(round(float(a), 8))
        except Exception:
            pass
    ok1 = found == {1.0}
    # (2) g'(x) = g/(g+1): 1/f'(t) 와 t/(t+1) 비교, 무작위 x 에서 수치 미분으로도 확인
    ok2 = sp.simplify(1 / sp.diff(f, x) - x / (x + 1)) == 0
    for _ in range(20):
        xv = mp.mpf(random.uniform(-3, 8))
        gv = inv(fn, xv, 1, lo=1e-12, hi=50)
        d = (inv(fn, xv + mp.mpf('1e-10'), 1, lo=1e-12, hi=50) - inv(fn, xv - mp.mpf('1e-10'), 1, lo=1e-12, hi=50)) / mp.mpf('2e-10')
        ok2 &= abs(d - gv / (gv + 1)) < 1e-6
    ok2 &= sp.simplify(f.subs(x, E) - (E + 1)) == 0
    intg = sp.integrate(x * sp.diff(f, x), (x, 1, E))  # ∫_1^{e+1} g = ∫_1^e t f'(t) dt
    ok2 &= sp.simplify(intg - (E ** 2 / 2 + E - sp.Rational(3, 2))) == 0
    # (3) ∫_1^{e+1} x g/(g+1) dx: 치환 x=f(t) 로 sympy, mpmath 수치로 독립 확인
    val = sp.simplify(sp.integrate(f * (x / (x + 1)) * sp.diff(f, x), (x, 1, E)))
    num = mp.quad(lambda z: z * inv(fn, z, 1, lo=1e-12, hi=50) / (inv(fn, z, 1, lo=1e-12, hi=50) + 1), [1, mp.e + 1])
    ok3 = sp.simplify(val - (E ** 2 + 1) / 2) == 0 and close(num, sp.N((E ** 2 + 1) / 2, 30))
    # 오답: 경계항에서 1*g(1) 누락
    wrong = sp.simplify((E + 1) * E - intg)
    ok3 &= sp.simplify(wrong - (E ** 2 + 3) / 2) == 0
    report('J03-02', ok1 and ok2 and ok3,
           f"[증명-수치확인] f(f(a))=a 해는 a=1뿐; g'=g/(g+1) 20점 확인, ∫g={sp.simplify(intg)}; (3)={val} (수치 {mp.nstr(num, 12)}); 오답={wrong}")


# ---------------- J03-03 : f(x)=x+sin x ----------------
def p03():
    f = x + sp.sin(x)
    fn = lambda z: z + mp.sin(z)
    # (1) f'=1+cos x>=0, 등호는 고립점 ; f(x)=x <=> x=kπ
    ok1 = all(1 + mp.cos(v) >= 0 for v in mp.linspace(-20, 20, 1000))
    # (2) 0<x<π 에서 g(x)<x, π<x<2π 에서 g(x)>x
    ok2 = all(inv(fn, v, v, 0, 2 * mp.pi) < v for v in mp.linspace(0.01, mp.pi - 0.01, 200)) and \
        all(inv(fn, v, v, 0, 2 * mp.pi) > v for v in mp.linspace(mp.pi + 0.01, 2 * mp.pi - 0.01, 200))
    I0pi = sp.simplify(PI * PI - sp.integrate(f, (x, 0, PI)))  # ∫_0^π g = π·g(π) - ∫_0^π f
    ok2b = sp.simplify(I0pi - (PI ** 2 / 2 - 2)) == 0
    # (3) ∫_0^{2π} |g-x|
    Ipi2pi = sp.simplify(2 * PI * 2 * PI - PI * PI - sp.integrate(f, (x, PI, 2 * PI)))
    area = sp.simplify((sp.integrate(y, (y, 0, PI)) - I0pi) + (Ipi2pi - sp.integrate(y, (y, PI, 2 * PI))))
    num = mp.quad(lambda v: abs(inv(fn, v, v, 0, 2 * mp.pi) - v), [0, mp.pi, 2 * mp.pi])
    ok3 = area == 4 and close(num, 4, 1e-8)
    report('J03-03', ok1 and ok2 and ok2b and ok3,
           f"[증명-수치확인] g<x on (0,π), g>x on (π,2π); ∫_0^π g={I0pi}; ∫|g-x|={area} (수치 {mp.nstr(num, 12)})")


# ---------------- J03-04 : f(x)=x e^x (x>=0) ----------------
def p04():
    fn = lambda z: z * mp.e ** z
    G = lambda v: inv(fn, v, None, 0, mp.log(v) + 2)
    # (1) x>e 에서 ln x - ln ln x < g(x) < ln x : 격자 1000점 (로그 간격)
    ok1 = True
    for k in range(1000):
        v = mp.e * mp.mpf(10) ** (k * 0.012) * 1.0001
        gv = G(v)
        if not (mp.log(v) - mp.log(mp.log(v)) < gv < mp.log(v)):
            ok1 = False
    # (2) g'(x)=g/(x(1+g)) 무작위 20점, g'(e)=1/(2e)
    ok2 = True
    random.seed(1)
    for _ in range(20):
        v = mp.mpf(random.uniform(0.1, 50))
        d = mp.diff(G, v)
        if not close(d, G(v) / (v * (1 + G(v))), 1e-8):
            ok2 = False
    f = x * sp.exp(x)
    gpe = 1 / sp.diff(f, x).subs(x, 1)
    ok2 = ok2 and sp.simplify(gpe - 1 / (2 * E)) == 0
    # (3) lim x g'(x) ln x / g(x) = lim ln x /(1+g) -> 1 : 큰 x에서 접근 확인 + 상하한 극한
    L = sp.limit(sp.log(t) / (1 + sp.log(t)), t, sp.oo)
    U = sp.limit(sp.log(t) / (1 + sp.log(t) - sp.log(sp.log(t))), t, sp.oo)
    vals = [mp.log(v) / (1 + G(v)) for v in [mp.mpf(10) ** 50, mp.mpf(10) ** 200]]
    ok3 = L == 1 and U == 1 and abs(vals[1] - 1) < abs(vals[0] - 1) < 0.1
    report('J03-04', ok1 and ok2 and ok3,
           f"[증명-수치확인] 부등식 1000점, g' 공식 20점; g'(e)=1/(2e); 하한·상한 극한 {L},{U} -> 답 1")


# ---------------- J03-05 : f(x)=sin x on [0,π/2], Young ----------------
def p05():
    f = sp.sin(x)
    G = lambda v: mp.asin(v)  # 수치 확인용 역함수
    # (1) 항등식: 무작위 20점
    ok1 = True
    random.seed(2)
    for _ in range(20):
        a = mp.mpf(random.uniform(0, float(mp.pi / 2)))
        lhs = mp.quad(mp.sin, [0, a]) + mp.quad(lambda u: mp.findroot(lambda z: mp.sin(z) - u, (0, mp.pi / 2), solver='anderson') if 0 < u < 1 else G(u), [0, mp.sin(a)])
        if not close(lhs, a * mp.sin(a), 1e-8):
            ok1 = False
    # (2) h(b)=max_a(ab-∫_0^a f) 를 격자 최대화로 직접 계산 vs ∫_0^b g
    ok2 = True
    for b in [0.1 * k for k in range(1, 11)]:
        best = max(a * b - (1 - mp.cos(a)) for a in mp.linspace(0, mp.pi / 2, 4001))
        exact = sp.integrate(sp.asin(y), (y, 0, b))
        if abs(best - float(exact)) > 1e-6:
            ok2 = False
    h1 = sp.simplify(PI / 2 * 1 - sp.integrate(f, (x, 0, PI / 2)))  # ∫_0^1 g = (π/2)·1 - ∫_0^{π/2} sin
    ok2 = ok2 and sp.simplify(h1 - (PI / 2 - 1)) == 0
    # (3) ∫_0^1 h(b) db = h(1) - ∫_0^1 b g(b) db, ∫ b g(b) db = ∫_0^{π/2} t sin t cos t dt
    bg = sp.integrate(t * sp.sin(t) * sp.cos(t), (t, 0, PI / 2))
    ans = sp.simplify(h1 - bg)
    hfun = lambda b: b * mp.asin(b) + mp.sqrt(1 - b * b) - 1
    num = mp.quad(lambda b: max(a * b - (1 - mp.cos(a)) for a in [mp.asin(b)]), [0, 1])
    num2 = mp.quad(hfun, [0, 1])
    ok3 = sp.simplify(ans - (3 * PI / 8 - 1)) == 0 and close(num, sp.N(ans, 30)) and close(num2, sp.N(ans, 30))
    report('J03-05', ok1 and ok2 and ok3,
           f"[증명-수치확인] 항등식 20점, h(b)=∫_0^b g (격자최대화 10점); h(1)={h1}; ∫h={ans} (수치 {mp.nstr(num, 12)})")


# ---------------- J03-06 : f(x)=tan x ----------------
def p06():
    G = lambda v: mp.findroot(lambda z: mp.tan(z) - v, mp.atan(v) + 0.01)
    random.seed(3)
    ok1 = all(close(mp.diff(G, v), 1 / (1 + v * v), 1e-8) for v in [mp.mpf(random.uniform(-5, 5)) for _ in range(20)])
    # (2) ∫_0^1 g = (π/4)·1 - ∫_0^{π/4} tan x dx
    I = sp.simplify(PI / 4 - sp.integrate(sp.tan(x), (x, 0, PI / 4)))
    ok2 = sp.simplify(I - (PI / 4 - sp.log(2) / 2)) == 0 and close(mp.quad(G, [0, 1]), sp.N(I, 30))
    # (3) Riemann 합 극한 = ∫_0^1 (1+x) g(x) dx ; ∫ x g = [x^2/2 g] - ∫ x^2/(2(1+x^2))
    Ixg = sp.simplify(PI / 8 - sp.integrate(x ** 2 / (2 * (1 + x ** 2)), (x, 0, 1)))
    ans = sp.simplify(I + Ixg)
    target = (PI - 1 - sp.log(2)) / 2
    n = 20000
    rs = sum((n + k) / mp.mpf(n) ** 2 * mp.atan(mp.mpf(k) / n) for k in range(1, n + 1))
    ok3 = sp.simplify(ans - target) == 0 and abs(rs - sp.N(target, 30)) < 1e-4 and close(mp.quad(lambda v: (1 + v) * G(v), [0, 1]), sp.N(target, 30))
    report('J03-06', ok1 and ok2 and ok3,
           f"[증명-수치확인] g'=1/(1+x^2) 20점; ∫g={I}; 극한={ans} (n=20000 합 {mp.nstr(rs, 8)})")


# ---------------- J03-07 : f(x)=2x+sin x, a_{n+1}=g(a_n) ----------------
def p07():
    f = 2 * x + sp.sin(x)
    fn = lambda z: 2 * z + mp.sin(z)
    ok1 = f.subs(x, 0) == 0
    gp0 = 1 / sp.diff(f, x).subs(x, 0)
    ok1 = ok1 and gp0 == sp.Rational(1, 3) and all(2 + mp.cos(v) > 0 for v in mp.linspace(-20, 20, 1000))
    # (2) 0<x<=2π 에서 x/3 <= g(x) <= x/2 : 1000점
    ok2 = True
    for v in mp.linspace(0.001, 2 * mp.pi, 1000):
        gv = inv(fn, v, v / 2.5, 0, 2 * mp.pi)
        if not (v / 3 - 1e-25 <= gv <= v / 2 + 1e-25):
            ok2 = False
    # 수열: 0<a_n<=π/2^{n-1}
    a = mp.pi
    seq = [a]
    for _ in range(60):
        a = inv(fn, a, a / 2.5, 0, 2 * mp.pi) if a > mp.mpf('1e-25') else mp.findroot(lambda z: fn(z) - a, a / 3)
        seq.append(a)
    ok2 = ok2 and all(0 < seq[k] <= mp.pi / 2 ** k * (1 + 1e-20) for k in range(len(seq)))
    # (3) a_{n+1}/a_n -> 1/3
    r = seq[-1] / seq[-2]
    ok3 = abs(r - mp.mpf(1) / 3) < 1e-12
    report('J03-07', ok1 and ok2 and ok3,
           f"g'(0)={gp0}; [증명-수치확인] x/3<=g(x)<=x/2 1000점, a_n<=π/2^(n-1); a_61/a_60={mp.nstr(r, 15)} -> 1/3")


# ---------------- J03-08 : ∫_0^{f(x)} g = (x-1)e^x+1 ----------------
def p08():
    # (1)(2) 미분방정식 g(f(x)) f'(x) = x f'(x) = d/dx[(x-1)e^x+1]
    F = sp.Function('F')
    rhs = sp.diff((x - 1) * sp.exp(x) + 1, x)
    sol = sp.dsolve(sp.Eq(x * F(x).diff(x), rhs), F(x), ics={F(0): 0})
    fsol = sp.simplify(sol.rhs)
    ok1 = sp.simplify(fsol - (sp.exp(x) - 1)) == 0
    # 조건 재확인: ∫_0^{f(x)} ln(1+t) dt == (x-1)e^x+1
    gsol = sp.log(1 + y)
    chk = sp.simplify(sp.integrate(gsol, (y, 0, sp.exp(x) - 1)) - ((x - 1) * sp.exp(x) + 1))
    ok2 = chk == 0 and all(mp.log(1 + v) < v < mp.e ** v - 1 for v in mp.linspace(0.001, 10, 1000))
    # (3) 넓이: f, g, x+y=e
    ans = E ** 2 / 2 - 3
    half = sp.integrate(sp.exp(x) - 1 - x, (x, 0, 1)) + (E - 2) * (E / 2 - 1) / 2
    num = mp.quad(lambda v: min(mp.e ** v - 1, mp.e - v) - mp.log(1 + v), [0, 1, mp.e - 1])
    ok3 = sp.simplify(2 * half - ans) == 0 and close(num, sp.N(ans, 30))
    report('J03-08', ok1 and ok2 and ok3,
           f"f={fsol}, 조건식 재확인 OK, [증명-수치확인] g<x<f; 넓이={sp.simplify(2*half)} (수치 {mp.nstr(num, 12)})")


# ---------------- J03-09 : f(x)=c-x^3, 교점 개수 ----------------
def count_intersections(c):
    """y=f(x) 와 y=g(x) 의 교점 수 = f(f(a))=a 의 서로 다른 실근 수."""
    mp.mp.dps = 60
    cc = mp.mpf(sp.N(c, 70))
    # f(f(x))-x = c-(c-x^3)^3-x 의 계수 (내림차순)
    coeffs = [1, 0, 0, -3 * cc, 0, 0, 3 * cc ** 2, 0, -1, -cc ** 3 + cc]
    rts = mp.polyroots(coeffs, maxsteps=500, extraprec=400)
    reals = sorted(mp.re(r) for r in rts if abs(mp.im(r)) < mp.mpf('1e-12'))
    distinct = []
    for r in reals:
        if not distinct or abs(r - distinct[-1]) > mp.mpf('1e-10'):
            distinct.append(r)
    mp.mp.dps = 30
    return len(distinct)


def p09():
    # (1) a≠b 교점이면 a^2+ab+b^2=1, s^3-2s+c=0 : 기호 확인
    a, b, c = sp.symbols('a b c', real=True)
    e1 = (c - a ** 3) - b
    e2 = (c - b ** 3) - a
    diff_ = sp.factor(e1 - e2)  # (b-a)(a^2+ab+b^2-1) 꼴
    ok1 = sp.simplify(diff_ - (b - a) * (a ** 2 + a * b + b ** 2 - 1)) == 0
    # 합: 2c - (a^3+b^3) = a+b, p=s^2-1 대입 -> s^3-2s+c=0
    sub = sp.expand((2 * c - (s ** 3 - 3 * (s ** 2 - 1) * s)) - s)
    ok1 = ok1 and sp.expand(sub / 2 - (s ** 3 - 2 * s + c)) == 0
    # (2) c=0
    sols = sp.solve([sp.Eq(b, -a ** 3), sp.Eq(a, -b ** 3)], [a, b], dict=True)
    real = sorted({(sp.nsimplify(d[a]), sp.nsimplify(d[b])) for d in sols if d[a].is_real and d[b].is_real})
    ok2 = real == sorted([(0, 0), (1, -1), (-1, 1)]) and count_intersections(0) == 3
    # (3) 정확히 5개인 c 의 범위: 4√3/9 < |c| < 4√6/9
    q = 4 * sp.sqrt(3) / 9
    m = 4 * sp.sqrt(6) / 9
    ok3 = True
    tests = []
    for cv in [sp.Rational(k, 100) for k in range(0, 151, 3)] + [q, m, -q, -m]:
        n = count_intersections(cv)
        pred = 5 if (q < abs(cv) < m) else (1 if abs(cv) > m else 3)
        tests.append((cv, n))
        if n != pred:
            ok3 = False
    report('J03-09', ok1 and ok2 and ok3,
           f"[증명-수치확인] 인수분해·합 조건 기호확인; c=0 교점 {real}; c 54개 표본(경계 ±4√3/9, ±4√6/9 포함)에서 교점수 1/3/5 예측 일치")


# ---------------- J03-10 : ∫f^2+∫g^2 >= 2/3 ----------------
def p10():
    # (1)(2) 무작위 증가함수 족 f(x)=x^p, (e^{kx}-1)/(e^k-1), sin(πx/2) 에서 확인
    fams = []
    for p in [0.3, 0.7, 1, 2, 5]:
        fams.append((lambda v, p=p: v ** p, lambda v, p=p: v ** (1 / p)))
    for k in [-3, 1, 4]:
        fams.append((lambda v, k=k: (mp.e ** (k * v) - 1) / (mp.e ** k - 1),
                     lambda v, k=k: mp.log(1 + v * (mp.e ** k - 1)) / k))
    ok1 = ok2 = True
    for F, G in fams:
        s1 = mp.quad(F, [0, 1]) + mp.quad(G, [0, 1])
        s2 = mp.quad(lambda v: F(v) ** 2, [0, 1]) + mp.quad(lambda v: G(v) ** 2, [0, 1])
        s2b = mp.mpf(2) / 3 + mp.quad(lambda v: (F(v) - v) ** 2, [0, 1])
        ok1 = ok1 and close(s1, 1, 1e-8)
        ok2 = ok2 and s2 >= mp.mpf(2) / 3 - 1e-12 and close(s2, s2b, 1e-8)
    # (3) f=sin(πx/2): ∫g = 1-∫f, ∫g^2 = 1 - 2∫ x f
    f = sp.sin(PI * x / 2)
    Ig = sp.simplify(1 - sp.integrate(f, (x, 0, 1)))
    Ig2 = sp.simplify(1 - 2 * sp.integrate(x * f, (x, 0, 1)))
    G = lambda v: 2 / mp.pi * mp.asin(v)
    ok3 = sp.simplify(Ig - (1 - 2 / PI)) == 0 and sp.simplify(Ig2 - (1 - 8 / PI ** 2)) == 0 and \
        close(mp.quad(G, [0, 1]), sp.N(Ig, 30)) and close(mp.quad(lambda v: G(v) ** 2, [0, 1]), sp.N(Ig2, 30))
    tot = sp.N(sp.Rational(1, 2) + Ig2)
    report('J03-10', ok1 and ok2 and ok3 and tot > sp.Rational(2, 3),
           f"[증명-수치확인] 8개 함수족에서 ∫f+∫g=1, ∫f²+∫g²=2/3+∫(f-x)²; ∫g={Ig}, ∫g²={Ig2}, 합 {tot:.6f}>2/3")


if __name__ == '__main__':
    for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
        try:
            fn()
        except Exception as ex:
            results.append(False)
            print(f"{fn.__name__} FAIL  exception {ex!r}")
    npass = sum(results)
    print(f"SUMMARY T03: {npass}/{len(results)} PASS")
    if npass != len(results) or len(results) != 10:
        sys.exit(1)
