import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-G17A", "items/KW/PAST")
R_ = sp.Rational
tt = sp.symbols("t", real=True)
zz = sp.symbols("z", real=True)

def pt(s):
    return list(sp.sympify(s, locals={"x": x, "y": y}))

def p01():
    X, Y = sp.sqrt(2)*sp.cos(tt), 2 + sp.sin(tt)
    ell = sp.sympify(V.ans(1, "1", "ellipse"), locals={"x": x, "y": y})
    V.check(1, "1 타원식이 매개변수식에서 0", sp.simplify(ell.subs({x: X, y: Y})), 0)
    ys = sorted(sp.solve(ell.subs(x, 0), y))
    V.check(1, "1 A,B", ys, [pt(V.ans(1, "1", "A"))[1], pt(V.ans(1, "1", "B"))[1]])
    # [2] 극선: 타원 x^2/2+(y-2)^2=1 의 접점 (x0,y0) 접선 x x0/2+(y-2)(y0-2)=1 이 원점을 지날 조건
    x0, y0 = sp.symbols("x0 y0", real=True)
    sols = sp.solve([x0**2/2 + (y0-2)**2 - 1, -2*(y0-2) - 1], [x0, y0], dict=True)
    pts = sorted([(s[x0], s[y0]) for s in sols], key=lambda p: float(p[0]))
    V.check(1, "2 C", list(pts[0]), pt(V.ans(1, "2", "C")))
    V.check(1, "2 D", list(pts[1]), pt(V.ans(1, "2", "D")))
    # 매개변수 접선 조건과도 일치: t=7pi/6, 11pi/6
    for tv, p_ in zip((7*sp.pi/6, 11*sp.pi/6), pts):
        V.check(1, "2 접점 t", [sp.simplify(X.subs(tt, tv)), sp.simplify(Y.subs(tt, tv))], list(p_))
    c = sp.simplify((pts[0][0]*pts[1][0] + pts[0][1]*pts[1][1]) / (sp.sqrt(pts[0][0]**2+pts[0][1]**2) * sp.sqrt(pts[1][0]**2+pts[1][1]**2)))
    V.check(1, "2 cosθ", c, V.ans(1, "2", "costheta"))
    # [3] 영역 표본으로 8개 영역과 거울쌍 4개 확인
    A_, B_, C_, D_ = (0, 3), (0, 1), pts[0], pts[1]
    def side(P, Q, p):
        return sp.sign(float((Q[0]-P[0])*(p[1]-P[1]) - (Q[1]-P[1])*(p[0]-P[0])))
    lines = [(A_, C_), (A_, D_), (B_, C_), (B_, D_), (A_, B_), (C_, D_)]
    sig = lambda p: tuple(side(a, b, p) for a, b in lines)
    regions = {}
    N = 60
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            px, py = 1.4142*(i+0.37)/N*0.999, 2 + (j+0.31)/N*0.999
            if px**2/2 + (py-2)**2 < 1 and 0 not in sig((px, py)):
                regions.setdefault(sig((px, py)), (px, py))
    mirror = {s: sig((-p[0], p[1])) for s, p in regions.items()}
    pairs = {frozenset((s, m)) for s, m in mirror.items() if s != m}
    self_sym = [s for s, m in mirror.items() if s == m]
    nreg = len(regions)
    prob = 1 - R_(len(pairs), sp.binomial(nreg, 2))
    V.check(1, "3 영역 8개", nreg, 8)
    V.check(1, "3 자기대칭 영역 없음", len(self_sym), 0)
    V.check(1, "3 확률", prob, V.ans(1, "3"))
    # [4] 축에 평행한 내접 직사각형: 꼭짓점 (sqrt2 cos t, 2+sin t) → 가로 2*sqrt2 cos t, 세로 2 sin t
    S_t = 2*sp.sqrt(2)*sp.cos(tt) * 2*sp.sin(tt)
    V.check(1, "4 S(t)", sp.simplify(S_t - sp.sympify(V.ans(1, "4", "S"), locals={"t": tt})), 0)
    mx = max(sp.N(S_t.subs(tt, sp.pi/2*i/2000)) for i in range(1, 2000))
    V.check(1, "4 최댓값", sp.nsimplify(mx, [sp.sqrt(2)]), V.ans(1, "4", "max"), tol=1e-6)
    crit = [r for r in sp.solveset(sp.diff(S_t, tt), tt, sp.Interval.open(0, sp.pi/2))]
    V.check(1, "4 극점 t", crit, [sp.pi/4])
    # [5]
    P_ = sp.Matrix([-sp.sqrt(3)/2, 1, 0]); Q_ = sp.Matrix([sp.sqrt(3)/2, 1, 0])
    l, m, n_ = sp.symbols("l m n", real=True)
    sol = sp.solve([P_.dot(sp.Matrix([l, m, n_])), Q_.dot(sp.Matrix([l, m, n_])), l**2+m**2+n_**2-(Q_-P_).dot(Q_-P_)], [l, m, n_], dict=True)
    R_v = [sp.Matrix([s[l], s[m], s[n_]]) for s in sol if s[n_] > 0][0]
    V.check(1, "5 R", list(R_v), [0, 0, sp.sqrt(3)])
    nv = (Q_-P_).cross(R_v-P_)
    plane = sp.simplify(nv.dot(sp.Matrix([x, y, zz]) - P_))
    plane = sp.expand(plane / (nv[1]/3))
    V.check(1, "5 평면", sp.simplify(plane - sp.sympify(V.ans(1, "5", "plane").replace("z", "zz"), locals={"x": x, "y": y, "zz": zz})), 0)
    cth = sp.simplify(abs(nv[2]) / nv.norm())
    V.check(1, "5 cosθ", cth, sp.cos(sp.sympify(V.ans(1, "5", "theta"))))

def p02():
    f_ = sp.Function("f")
    # [1] ln f - ln g 의 도함수가 0 이면 f/g 가 상수
    g_ = sp.Function("g")
    d = sp.simplify(sp.diff(sp.log(f_(x)) - sp.log(g_(x)), x) - (sp.diff(f_(x), x)/f_(x) - sp.diff(g_(x), x)/g_(x)))
    V.check(2, "1 로그미분", d, 0)
    k_ = 3; gg = sp.exp(x) + 1
    V.check(2, "1 f=kg 이면 f'/f=g'/g", sp.simplify(sp.diff(k_*gg, x)/(k_*gg) - sp.diff(gg, x)/gg), 0)
    # [2](1) 여러 f 에 대해 0<P<x (수치)
    import mpmath as mp
    fs = [lambda s: 1 + s**2, lambda s: mp.e**(-s), lambda s: 1/(1+s), lambda s: 0.1 + s**3]
    bad = []
    for fn in fs:
        for xv in (0.05, 0.3, 0.5, 0.9, 0.99):
            num = mp.quad(lambda s: s*fn(s), [0, xv]); den = mp.quad(fn, [0, xv])
            if not (0 < num/den < xv): bad.append((xv,))
    V.proof(2, "2-1 0<P<x", bad)
    # [2](2)
    Pf = sp.integrate(t*(1+t**2), (t, 0, x)) / sp.integrate(1+t**2, (t, 0, x))
    V.check(2, "2-2 lim", sp.limit(Pf, x, 0, "+"), V.ans(2, "2-2"))
    # [3](1) P' 항등식: u'=x f, w'=f
    u, w = sp.Function("u")(x), sp.Function("w")(x)
    f0 = sp.Function("f")(x)
    Pexpr = u / w
    dP = sp.diff(Pexpr, x).subs({sp.Derivative(u, x): x*f0, sp.Derivative(w, x): f0})
    V.check(2, "3-1 P'", sp.simplify(dP - f0/w*(x - u/w)), 0)
    # [3](2),[4]: f=k g 이면 P=Q, f'∫g = g'∫f, f/∫f = g/∫g
    gx = 1 + t; kk = 2
    Gi = sp.integrate(gx, (t, 0, x)); Fi = sp.integrate(kk*gx, (t, 0, x))
    V.check(2, "4 (B) P=Q", sp.simplify(sp.integrate(t*kk*gx, (t, 0, x))/Fi - sp.integrate(t*gx, (t, 0, x))/Gi), 0)
    V.check(2, "3-2 f/∫f=g/∫g", sp.simplify(kk*gx.subs(t, x)/Fi - gx.subs(t, x)/Gi), 0)
    V.check(2, "4 f'∫g=g'∫f", sp.simplify(sp.diff(kk*gx.subs(t, x), x)*Gi - sp.diff(gx.subs(t, x), x)*Fi), 0)
    # 비례가 아니면 P≠Q: f=1, g=1+t
    Pn = sp.integrate(t, (t, 0, x))/sp.integrate(1, (t, 0, x)); Qn = sp.integrate(t*(1+t), (t, 0, x))/sp.integrate(1+t, (t, 0, x))
    V.check(2, "4 비례 아니면 P≠Q", bool(sp.simplify(Pn - Qn).subs(x, R_(1, 2)) != 0), True)

if __name__ == "__main__": V.run(globals())
