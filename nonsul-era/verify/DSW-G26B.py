import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DSW-G26B", "items/DSW/PAST")
E = lambda v, **loc: sp.sympify(v, locals=loc)

def circles(N, K):
    """a=3N+3 개 원, K번 시행. 기하 조건으로 r_1..r_K: 원 중심은 반직선 위 거리 d_k, 이웃 원과 외접 -> r=d_k*sin(pi/a),
    k=1 은 단위원에 내접(d_1=1-r_1), k>=2 는 직전 시행 원과 외접(d_k=d_{k-1}-r_{k-1}-r_k)."""
    al = sp.sin(sp.pi/(3*N + 3))
    r = sp.symbols("r", positive=True)
    rs, d = [], None
    for i in range(K):
        dk = (1 - r) if i == 0 else (d - rs[-1] - r)
        sol = sp.solve(sp.Eq(r, dk*al), r)[0]
        rs.append(sp.simplify(sol)); d = dk.subs(r, sol)
    return rs

def p01():
    rs = circles(1, 10)
    V.check(1, "1-1 r1", rs[0], sp.Rational(1, 3))
    V.check(1, "1-1 r2", rs[1], sp.Rational(1, 9))
    S = lambda k: 6*sp.pi*rs[k-1]**2
    C = lambda k: 6*2*sp.pi*rs[k-1]
    V.check(1, "1-1 S2", S(2), V.ans(1, "1-1", "S_2"))
    V.check(1, "1-1 C2", C(2), V.ans(1, "1-1", "C_2"))
    V.check(1, "1-2 sumS", sum(S(k) for k in range(1, 11)), V.ans(1, "1-2", "sum_S"))
    V.check(1, "1-2 sumC", sum(C(k) for k in range(1, 11)), V.ans(1, "1-2", "sum_C"))
    N_, k_, al_ = sp.symbols("N k alpha", positive=True)
    f = E(V.ans(1, "1-3", "r_k"), alpha=al_, k=k_)
    adef = E(V.ans(1, "1-3", "alpha"), N=N_)
    for NN in (1, 2, 5):
        rr = circles(NN, 6)
        for kk in range(1, 7):
            V.check(1, f"1-3 N={NN} k={kk}", rr[kk-1], f.subs({al_: adef.subs(N_, NN), k_: kk}))
    V.check(1, "그림2 S3 = 6*pi*r3^2", S(3), 6*sp.pi/sp.Integer(27)**2)

def area(a, t):
    """a, t 구체값: f(x)=a x^2 (x<0), -a x^2 (x>=0). 곡선 c(x)=-(x-t)^2+f(t)와 y=f(x)의 두 교점 사이에서 |c| 를 x 축에 대해 적분."""
    ft = a*t**2 if t < 0 else -a*t**2
    c = -(x - t)**2 + ft
    xs = set()
    for br, ok in ((a*x**2, lambda r: r < 0), (-a*x**2, lambda r: r >= 0)):
        for r in sp.solve(sp.Eq(c, br), x):
            if ok(r): xs.add(sp.nsimplify(r))
    x1, x2 = sorted(xs, key=lambda r: float(r))
    assert len(xs) == 2
    mid = (x1 + x2)/2
    sg = 1 if c.subs(x, mid) > 0 else -1
    assert all(sg*c.subs(x, x1 + (x2 - x1)*j/20) >= 0 for j in range(21))   # 구간에서 부호 일정
    return sp.integrate(sg*c, (x, x1, x2))

def p02():
    h = sp.Rational(1, 2)
    V.check(2, "2-1 S(-2)", area(h, -2), V.ans(2, "2-1", "S(-2)"))
    V.check(2, "2-1 S(2)", area(h, 2), V.ans(2, "2-1", "S(2)"))
    a = sp.symbols("a", positive=True)
    # 2-2: t=-1 이면 S(t)/t^3 = -S(-1) = -g  -> g(a)=S(-1); 구체적 a 여러 값으로 p,q,r,s 를 연립해 구한다
    p_, q_, r_, s_ = sp.symbols("p q r s")
    eqs = [sp.Eq(area(av, -1)*3*(1 + av)**3, 6*av**4 + p_*av**3 + q_*av**2 + r_*av + s_) for av in [sp.Rational(1, 5), sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 4)]]
    so = sp.solve(eqs, [p_, q_, r_, s_], dict=True)[0]
    for nm, sy in (("p", p_), ("q", q_), ("r", r_), ("s", s_)):
        V.check(2, f"2-2 {nm}", so[sy], V.ans(2, "2-2", nm))
    # 2-3: t=1 이면 S(1)/1 = h(a); t 의존성(t^3 비례) 확인 후 극한
    for av in (sp.Rational(1, 3), sp.Rational(2, 3)):
        V.check(2, f"S(t)/t^3 t-불변 a={av}", area(av, 3)/27, area(av, 1))
        V.check(2, f"S(t)/t^3 (t<0) a={av}", area(av, -3)/(-27), -area(av, -1))
    gs = (6*a**4 + so[p_]*a**3 + so[q_]*a**2 + so[r_]*a + so[s_])/(3*(1 + a)**3)
    # h(a) 의 닫힌 식: 같은 방식으로 5개 값에서 6a^4 + P a^3 + Q a^2 + R a + S0 꼴을 적합하고 3(1-a)^3 으로 나눔
    P, Q, R, S0 = sp.symbols("P Q R S0")
    eqs2 = [sp.Eq(area(av, 1)*3*(1 - av)**3, 6*av**4 + P*av**3 + Q*av**2 + R*av + S0) for av in [sp.Rational(1, 5), sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 4)]]
    s2 = sp.solve(eqs2, [P, Q, R, S0], dict=True)[0]
    hs = (6*a**4 + s2[P]*a**3 + s2[Q]*a**2 + s2[R]*a + s2[S0])/(3*(1 - a)**3)
    V.check(2, "h 분자 계수(-4,6,0,0)", [s2[P], s2[Q], s2[R], s2[S0]], [-4, 6, 0, 0])
    V.check(2, "2-3 L1", sp.limit(gs/hs, a, 0, "+"), V.ans(2, "2-3", "L1"))
    V.check(2, "2-3 L2", sp.limit(gs/((1 - a)**3*hs), a, 1, "-"), V.ans(2, "2-3", "L2"))

def p03():
    # 3-1: l0 위의 점 (x,-x), P(-1,5). d 는 점-직선 거리, w 는 거리 제곱
    P = (-1, 5)
    p1, p2 = (-5, 5), (-1, 1)
    A_, B_ = p2[1] - p1[1], p1[0] - p2[0]                       # 직선 A x + B y + C = 0
    C_ = -(A_*p1[0] + B_*p1[1])
    d = sp.Abs(A_*P[0] + B_*P[1] + C_)/sp.sqrt(A_**2 + B_**2)
    ys = -x                                                       # 직선 위의 점
    w = (x - P[0])**2 + (ys - P[1])**2
    V.check(3, "3-1 integral", sp.integrate(sp.simplify(d/w), (x, -5, -1)), V.ans(3, "3-1"))
    # 3-2, 3-3: 직사각형 OABC, P(u,v). 정의대로 거리·거리제곱을 계산해 a_k, A_k 를 적분
    al, be, u, v = sp.symbols("alpha beta u v", positive=True)
    X = sp.symbols("X", real=True)
    segs = [  # (거리 d, 거리제곱 w(X), 적분구간 끝)  X = 적분변수(k=1,3: x / k=2,4: y)
        (v, (X - u)**2 + v**2, al),                 # l1: y=0
        (al - u, (al - u)**2 + (X - v)**2, be),     # l2: x=alpha
        (be - v, (X - u)**2 + (be - v)**2, al),     # l3: y=beta
        (u, u**2 + (X - v)**2, be),                 # l4: x=0
    ]
    vals = {al: 3, be: 2, u: sp.Rational(1, 2), v: sp.Rational(3, 4)}
    a = [sp.integrate((dd/ww).subs(vals), (X, 0, end.subs(vals) if hasattr(end, "subs") else end)) for dd, ww, end in segs]
    V.check(3, "3-2 sum a_k (수치 점)", sum(a), V.ans(3, "3-2"))
    vals2 = {al: 5, be: 4, u: 2, v: sp.Rational(5, 2)}
    a2 = [sp.integrate((dd/ww).subs(vals2), (X, 0, end.subs(vals2))) for dd, ww, end in segs]
    V.check(3, "3-2 sum a_k (다른 점)", sum(a2), V.ans(3, "3-2"))
    # 3-3: alpha=3, beta=2, v=1 고정, u 를 기호로 두고 극한
    fix = {al: 3, be: 2, v: 1}
    aa = sum(sp.integrate((dd/ww).subs(fix), (X, 0, end.subs(fix))) for dd, ww, end in segs)
    AA = sum(sp.integrate((1/ww).subs(fix), (X, 0, end.subs(fix))) for dd, ww, end in segs)
    Cu = sp.simplify(aa/AA)
    V.check(3, "3-3 lim C", sp.limit(Cu, u, 3, "-"), V.ans(3, "3-3"))
    # 수치 확인: u -> alpha- 일 때 C 가 0 으로 감소
    nums = [float(Cu.subs(u, 3 - sp.Rational(1, 10**j))) for j in (1, 2, 3, 4)]
    V.report(3, all(nums[i] > nums[i + 1] for i in range(3)) and nums[-1] < 1e-2, f"3-3 C(u) 수치 {nums}")

if __name__ == "__main__": V.run(globals())
