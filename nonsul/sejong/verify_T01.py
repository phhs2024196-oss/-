#!/usr/bin/env python3
"""verify_T01.py -- J01. 접선·접원과 극한 (세종) 검산 스크립트.
각 문항의 설정으로부터 sympy/mpmath 로 독립 재계산하여 md 의 정답과 비교한다."""
import sys, random
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
random.seed(1)
results = []


def report(pid, ok, msg):
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def close(a, b, tol=1e-9):
    a, b = complex(a), complex(b)
    return abs(a - b) <= tol * max(1.0, abs(b))


x, y, t, s, u, a, m, r, b, th = sp.symbols('x y t s u a m r b theta', real=True)


# ---------------------------------------------------------------- J01-01
def p01():
    f = sp.exp(x)
    P = (t, sp.exp(t))
    k = sp.diff(f, x).subs(x, t)
    xA = sp.solve(sp.Eq(k * (x - t) + sp.exp(t), 0), x)[0]
    xB = sp.solve(sp.Eq(-(x - t) / k + sp.exp(t), 0), x)[0]
    ok1 = sp.simplify(xA - (t - 1)) == 0 and sp.simplify(xB - (t + sp.exp(2 * t))) == 0
    # (2) 직각 + 내접원 반지름 r = (PA+PB-AB)/2  == 2S/둘레
    PA = sp.sqrt((P[0] - xA) ** 2 + P[1] ** 2)
    PB = sp.sqrt((P[0] - xB) ** 2 + P[1] ** 2)
    AB = xB - xA
    dot = sp.simplify((xA - t) * (xB - t) + sp.exp(2 * t))
    ok2 = dot == 0
    r_md = ((1 + sp.exp(t)) * sp.sqrt(1 + sp.exp(2 * t)) - (1 + sp.exp(2 * t))) / 2
    for _ in range(20):
        tv = random.uniform(-6, 6)
        S = sp.exp(t) * AB / 2
        r_in = (2 * S / (PA + PB + AB)).subs(t, tv)
        ok2 &= close(sp.N(r_in, 30), sp.N(r_md.subs(t, tv), 30))
    # (3) 두 극한 (u=e^t)
    ru = (r_md / sp.exp(t)).subs(sp.exp(t), u)
    ru = ((1 + u) * sp.sqrt(1 + u ** 2) - (1 + u ** 2)) / (2 * u)
    L1 = sp.limit(ru, u, sp.oo)
    L2 = sp.limit(ru, u, 0, '+')
    ok3 = L1 == sp.Rational(1, 2) and L2 == sp.Rational(1, 2)
    # 대표 오답: 0 (분자→0 만 보고 판단) -- 실제 값과 다름을 확인
    report('J01-01', ok1 and ok2 and ok3,
           f"A(t-1,0),B(t+e^2t,0); 직각·r식 20점[증명-수치확인]; 극한 {L1},{L2}")


# ---------------------------------------------------------------- J01-02
def p02():
    tp = sp.symbols('tp', positive=True)
    f = 1 / x ** 2
    k = sp.diff(f, x).subs(x, tp)
    line = k * (x - tp) + 1 / tp ** 2
    xA = sp.solve(sp.Eq(line, 0), x)[0]
    yB = line.subs(x, 0)
    ok1 = sp.simplify(xA - 3 * tp / 2) == 0 and sp.simplify(yB - 3 / tp ** 2) == 0
    # (2) BP:PA = 2:1  <=> P = (2A + B)/3
    ok2 = sp.simplify(tp - (2 * xA + 0) / 3) == 0 and sp.simplify(1 / tp ** 2 - (0 + yB) / 3) == 0
    for _ in range(20):
        tv = random.uniform(0.1, 5)
        PA = mp.sqrt((tv - 1.5 * tv) ** 2 + (1 / tv ** 2) ** 2)
        PB = mp.sqrt(tv ** 2 + (1 / tv ** 2 - 3 / tv ** 2) ** 2)
        ok2 &= close(PB / PA, 2)
    # (3) PA*PB 최솟값
    prod = sp.sqrt((tp - xA) ** 2 + (1 / tp ** 2) ** 2) * sp.sqrt(tp ** 2 + (1 / tp ** 2 - yB) ** 2)
    prod = sp.simplify(prod)
    crit = [c for c in sp.solve(sp.diff(prod ** 2, tp), tp) if c.is_positive]
    vals = [sp.nsimplify(sp.simplify(prod.subs(tp, c))) for c in crit]
    grid = min(float(prod.subs(tp, 0.05 + i * 0.005)) for i in range(2000))
    ok3 = crit == [sp.sqrt(2)] and vals == [sp.Rational(3, 2)] and grid >= 1.5 - 1e-12
    report('J01-02', ok1 and ok2 and ok3,
           f"A(3t/2,0),B(0,3/t^2); BP:PA=2:1 [증명-수치확인]; min PA*PB={vals[0]} at t={crit[0]}")


# ---------------------------------------------------------------- J01-03
def p03():
    tt = sp.symbols('tt', nonzero=True)
    f = x ** 3
    line = 3 * tt ** 2 * (x - tt) + tt ** 3
    roots = sp.solve(sp.Eq(f, line), x)
    other = [rr for rr in roots if sp.simplify(rr - tt) != 0]
    ok1 = len(other) == 1 and sp.simplify(other[0] + 2 * tt) == 0
    # (2) S = 27 t^4 /4 (t>0, t<0 둘 다)
    ok2 = True
    for tv in [sp.Rational(1), sp.Rational(-2), sp.Rational(3, 7), sp.Rational(-5, 3)]:
        lo, hi = sorted([tv, -2 * tv])
        Sv = sp.integrate(sp.Abs(f - line.subs(tt, tv)), (x, lo, hi))
        ok2 &= sp.simplify(Sv - sp.Rational(27, 4) * tv ** 4) == 0
    # 수열 직접 생성 후 넓이 비 16
    xs = [sp.Integer(1)]
    for _ in range(8):
        xs.append(-2 * xs[-1])
    S = [sp.Rational(27, 4) * v ** 4 for v in xs]
    ok2 &= all(S[i + 1] / S[i] == 16 for i in range(8))
    n = sp.symbols('n', integer=True, positive=True)
    tot = sp.summation(1 / (sp.Rational(27, 4) * 16 ** (n - 1)), (n, 1, sp.oo))
    ok3 = tot == sp.Rational(64, 405)
    # 대표 오답: 첫 항 4/27 만 공비 1/2 로 잘못 → (4/27)*2 = 8/27
    report('J01-03', ok1 and ok2 and ok3, f"x_(n+1)=-2x_n; S_n=27x_n^4/4, 비16; 합={tot}")


# ---------------------------------------------------------------- J01-04
def p04():
    tp = sp.symbols('tp', positive=True)
    rr = sp.symbols('rr', positive=True)
    # 곡선 y=ln x 의 P(t, ln t) 에서 아래쪽 단위법선 (1,-t)/sqrt(1+t^2)
    nx, ny = 1 / sp.sqrt(1 + tp ** 2), -tp / sp.sqrt(1 + tp ** 2)
    cy = sp.log(tp) + rr * ny
    rsol = sp.solve(sp.Eq(cy, rr), rr)
    r_md = sp.sqrt(1 + tp ** 2) * (sp.sqrt(1 + tp ** 2) - tp) * sp.log(tp)
    a_md = tp + (sp.sqrt(1 + tp ** 2) - tp) * sp.log(tp)
    ok1 = True
    for tv in [1.3, 2, 5, 17.5]:
        rv = sp.N(rsol[0].subs(tp, tv), 30)
        ok1 &= close(rv, sp.N(r_md.subs(tp, tv), 30))
        ok1 &= close(sp.N((tp + rr * nx).subs({tp: tv, rr: rv}), 30), sp.N(a_md.subs(tp, tv), 30))
    # (2) (1/2) ln t < r < (2-sqrt2) ln t  (t>1) -- 격자 1000점
    ok2 = True
    for i in range(1, 1001):
        tv = 1 + i * 0.05
        g = mp.sqrt(1 + tv ** 2) * (mp.sqrt(1 + tv ** 2) - tv)
        ok2 &= (0.5 < g < 2 - mp.sqrt(2))
    # 원이 곡선 아래에 있는지(접원 설정의 타당성) 몇 t 에서 확인
    for tv in [1.5, 3.0, 10.0]:
        rv = float(r_md.subs(tp, tv)); av = float(a_md.subs(tp, tv))
        for j in range(400):
            ang = 2 * mp.pi * j / 400
            px, py = av + rv * mp.cos(ang), rv + rv * mp.sin(ang)
            if px > 0:
                ok2 &= py <= mp.log(px) + 1e-12
    L1 = sp.limit(r_md / (tp - 1), tp, 1, '+')
    L2 = sp.limit(r_md / sp.log(tp), tp, sp.oo)
    ok3 = sp.simplify(L1 - (2 - sp.sqrt(2))) == 0 and L2 == sp.Rational(1, 2)
    report('J01-04', ok1 and ok2 and ok3,
           f"r(t) 식 일치; 부등식 1000점[증명-수치확인]; 극한 {sp.simplify(L1)}, {L2}")


# ---------------------------------------------------------------- J01-05
def p05():
    f = x * sp.exp(-x)
    tang = sp.diff(f, x).subs(x, t) * (x - t) + f.subs(x, t)
    cond = sp.simplify(tang.subs(x, a) * sp.exp(t))
    ok1 = sp.expand(cond + (t ** 2 - a * t + a)) == 0 or sp.expand(cond - (t ** 2 - a * t + a)) == 0

    def count_lines(av):
        av = sp.nsimplify(av)
        sols = sp.solve(sp.Eq(t ** 2 - av * t + av, 0), t)
        sols = [sp.N(q, 30) for q in sols if q.is_real]
        lines = set()
        for q in sols:
            sl = sp.N(((1 - t) * sp.exp(-t)).subs(t, q), 20)
            ic = sp.N((t ** 2 * sp.exp(-t)).subs(t, q), 20)
            lines.add((round(float(sl), 12), round(float(ic), 12)))
        return len(lines)
    table = {-3: 2, -0.5: 2, 0: 1, 1: 0, 3.5: 0, 4: 1, 4.5: 2, 10: 2}
    ok2 = all(count_lines(k) == v for k, v in table.items())
    # 서로 다른 접점이 같은 접선을 주지 않음: a>4 무작위 50개
    for _ in range(50):
        av = random.uniform(4.01, 50)
        ok2 &= count_lines(av) == 2
    t1 = (a - sp.sqrt(a ** 2 - 4 * a)) / 2
    expr = a * (1 - t1) * sp.exp(-t1)
    L = sp.limit(expr, a, sp.oo)
    ok3 = sp.simplify(L + sp.exp(-1)) == 0
    report('J01-05', ok1 and ok2 and ok3, f"t^2-at+a=0; N(a) 표 일치; lim a*m = {L}")


# ---------------------------------------------------------------- J01-06
def p06():
    X, Y = sp.cos(th) ** 3, sp.sin(th) ** 3
    k = sp.simplify(sp.diff(Y, th) / sp.diff(X, th))
    line = k * (x - X) + Y
    xA = sp.simplify(sp.solve(sp.Eq(line, 0), x)[0])
    yB = sp.simplify(line.subs(x, 0))
    ok1 = sp.simplify(xA - sp.cos(th)) == 0 and sp.simplify(yB - sp.sin(th)) == 0
    AB = sp.sqrt(xA ** 2 + yB ** 2)
    AP = sp.sqrt((xA - X) ** 2 + Y ** 2)
    ok2 = True
    for _ in range(20):
        tv = random.uniform(0.01, float(sp.pi / 2) - 0.01)
        ok2 &= close(sp.N(AB.subs(th, tv)), 1) and close(sp.N(AP.subs(th, tv)), sp.N(sp.sin(tv) ** 2))
    # (3) 직선 AB 에 P 에서 접하고 x축에 접하며 중심이 O 쪽 반평면, y>0
    rho = sp.symbols('rho', positive=True)
    nvec = (sp.sin(th), sp.cos(th))  # O 반대쪽 법선(단위)
    cy = Y - rho * nvec[1]
    rs = sp.solve(sp.Eq(cy, rho), rho)
    rho_md = sp.sin(th) * (1 - sp.cos(th))
    ok3 = all(close(sp.N(rs[0].subs(th, v)), sp.N(rho_md.subs(th, v))) for v in [0.2, 0.7, 1.3])
    L = sp.limit(rho_md / th ** 3, th, 0, '+')
    ok3 &= L == sp.Rational(1, 2)
    report('J01-06', ok1 and ok2 and ok3, f"A(cos,0),B(0,sin); AB=1, AP=sin^2 [증명-수치확인]; rho=sin(1-cos), lim={L}")


# ---------------------------------------------------------------- J01-07
def p07():
    tp = sp.symbols('tp', positive=True)
    # y=1/x 의 P(t,1/t) 에서 접선: x + t^2 y = 2t
    aa, bb, cc = 1, tp ** 2, 2 * tp
    H = (aa * cc / (aa ** 2 + bb ** 2), bb * cc / (aa ** 2 + bb ** 2))
    ok1 = sp.simplify(H[0] - 2 * tp / (1 + tp ** 4)) == 0 and sp.simplify(H[1] - 2 * tp ** 3 / (1 + tp ** 4)) == 0
    # 접선 확인
    ok1 &= sp.simplify(sp.diff(1 / x, x).subs(x, tp) + aa / bb) == 0
    lem = sp.simplify((H[0] ** 2 + H[1] ** 2) ** 2 - 4 * H[0] * H[1])
    ok2 = lem == 0
    for _ in range(20):
        tv = random.uniform(0.05, 8)
        ok2 &= abs(float(lem.subs(tp, tv))) < 1e-12
    OH = sp.sqrt(H[0] ** 2 + H[1] ** 2)
    grid = max(float(OH.subs(tp, 0.01 + 0.005 * i)) for i in range(2000))
    ok3 = abs(grid - float(sp.sqrt(2))) < 1e-6 and sp.simplify(OH.subs(tp, 1) - sp.sqrt(2)) == 0
    HP = sp.sqrt((tp - H[0]) ** 2 + (1 / tp - H[1]) ** 2)
    S = OH * HP / 2
    S_md = sp.Abs(tp ** 4 - 1) / (tp ** 4 + 1)
    for tv in [0.3, 0.9, 1.7, 4.0]:
        ok3 &= close(sp.N(S.subs(tp, tv)), sp.N(S_md.subs(tp, tv)))
    L = sp.limit((tp ** 4 - 1) / (tp ** 4 + 1), tp, sp.oo)
    ok3 &= L == 1
    report('J01-07', ok1 and ok2 and ok3, f"H 좌표; 렘니스케이트 항등식 20점[증명-수치확인]; max OH=sqrt2, lim S={L}")


# ---------------------------------------------------------------- J01-08
def p08():
    Rr = sp.symbols('Rr', positive=True)
    P = (t, 1 - sp.cos(t))
    Rsol = sp.solve(sp.Eq(P[0] ** 2 + (P[1] - Rr) ** 2, Rr ** 2), Rr)[0]
    R_md = (t / 2) ** 2 / sp.sin(t / 2) ** 2 + sp.sin(t / 2) ** 2
    ok1 = all(close(sp.N(Rsol.subs(t, v)), sp.N(R_md.subs(t, v))) for v in [0.3, 1.1, 2.0, 3.0])
    g = x / sp.sin(x)
    gp = sp.diff(g, x)
    ok2 = all(float(gp.subs(x, 0.0015 + i * (1.5693 / 1000))) > 0 for i in range(1000))
    L0 = sp.limit(R_md, t, 0, '+')
    Lpi = sp.limit(R_md, t, sp.pi, '-')
    Rp = sp.diff(R_md, t)
    mono = all(float(Rp.subs(t, 0.003 + i * (3.1386 / 1000))) > 0 for i in range(1000))
    ok3 = L0 == 1 and sp.simplify(Lpi - (1 + sp.pi ** 2 / 4)) == 0 and mono
    report('J01-08', ok1 and ok2 and ok3, f"R(t) 일치; x/sin x 증가 1000점[증명-수치확인]; 범위 ({L0}, {sp.simplify(Lpi)})")


# ---------------------------------------------------------------- J01-09
def p09():
    rp, bp, Y = sp.symbols('rp bp Y', positive=True)
    quad = sp.expand(Y + (Y - bp) ** 2 - rp ** 2)  # y=x^2 대입 후 y 에 대한 이차식
    disc = sp.discriminant(quad, Y)
    bsol = sp.solve(sp.Eq(disc, 0), bp)
    ok1 = len(bsol) == 1 and sp.simplify(bsol[0] - (rp ** 2 + sp.Rational(1, 4))) == 0
    y0 = sp.solve(sp.diff(quad, Y), Y)[0].subs(bp, bsol[0])
    ok1 &= sp.simplify(y0 - (rp ** 2 - sp.Rational(1, 4))) == 0
    # (2) 반지름 수열을 직접 풀어서 생성 (n=1..60)
    rs = [sp.Rational(1, 2)]
    bs = [sp.Rational(1, 2)]
    for n in range(1, 60):
        rn = sp.symbols('rn', positive=True)
        sol = sp.solve(sp.Eq(rn ** 2 + sp.Rational(1, 4), bs[-1] + rs[-1] + rn), rn)
        rs.append(sol[0]); bs.append(sol[0] ** 2 + sp.Rational(1, 4))
    ok2 = all(rs[i] == sp.Rational(2 * (i + 1) - 1, 2) for i in range(60))
    # C1 이 포물선 안쪽에 있는지 (x^2 = y - y^2 <= y)
    n = sp.symbols('n', positive=True)
    xn = sp.sqrt((n - sp.Rational(1, 2)) ** 2 - sp.Rational(1, 4))
    ok3 = sp.simplify(xn ** 2 - (n ** 2 - n)) == 0
    L = sp.limit(n - xn, n, sp.oo)
    ok3 &= L == sp.Rational(1, 2)
    report('J01-09', ok1 and ok2 and ok3, f"b=r^2+1/4; r_n=n-1/2 (n<=60 직접풀이)[증명-수치확인]; lim(n-x_n)={L}")


# ---------------------------------------------------------------- J01-10
def p10():
    h = sp.exp(a) * (a - 1) - (a + 1)
    # (1) 조건 유도: e^x 접선과 ln x 접선 일치
    bb = sp.symbols('bb', positive=True)
    l1 = sp.exp(a) * (x - a) + sp.exp(a)
    l2 = (x - bb) / bb + sp.log(bb)
    bsol = sp.solve(sp.Eq(sp.diff(l1, x), sp.diff(l2, x)), bb)[0]
    eq = sp.simplify((l1 - l2).subs(x, 0).subs(bb, bsol))
    ok1 = sp.simplify(eq + h) == 0 or sp.simplify(eq - h) == 0
    # (2) 근이 정확히 2개, 서로 반대: 부호 변화 세기 + 대칭식
    hf = sp.lambdify(a, h, 'mpmath')
    grid = [-20 + 0.001 * i for i in range(40001)]
    changes = sum(1 for i in range(len(grid) - 1) if hf(grid[i]) * hf(grid[i + 1]) < 0)
    al = mp.findroot(hf, 1.5)
    ok2 = changes == 2 and abs(hf(-al)) < 1e-30 and sp.simplify(h.subs(a, -a) - sp.exp(-a) * h) == 0
    # (3) 기울기 곱, y절편 합
    sl = mp.e ** al * mp.e ** (-al)
    ic = mp.e ** al * (1 - al) + mp.e ** (-al) * (1 + al)
    ok3 = close(sl, 1) and close(ic, -2)
    # 실제로 ln x 에도 접하는지 수치 확인
    for av in [al, -al]:
        b0 = mp.e ** (-av)
        ok3 &= close(mp.e ** av * (1 - av), mp.log(b0) - 1)
    report('J01-10', ok1 and ok2 and ok3, f"e^a(a-1)=a+1; 근 2개 alpha={mp.nstr(al, 10)}, -alpha; 기울기곱={mp.nstr(sl, 5)}, 절편합={mp.nstr(ic, 5)}")


for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        fn()
    except Exception as e:  # noqa
        report(fn.__name__.replace('p', 'J01-'), False, f"예외 {e!r}")

npass = sum(results)
print(f"SUMMARY T01: {npass}/{len(results)} PASS")
sys.exit(0 if npass == len(results) == 10 else 1)
