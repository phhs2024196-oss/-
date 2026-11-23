import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("HSN-G26N", "items/HSN/PAST")

def vals(no):
    return [S(v) for v in V.ans(no, "1").split(",")]

def p06():
    cnt = sum(1 for m in range(1, 200) if sp.Rational(1, 9)**m < sp.Integer(3)**(10-4*m))
    V.check(6, "자연수 개수", cnt, vals(6)[0])

def p07():
    c0, c1, c2, c3, k = sp.symbols("c0 c1 c2 c3 k")
    f = c0 + c1*x + c2*x**2 + c3*x**3
    t_ = sp.symbols("t")
    lhs = sp.integrate(f.subs(x, t_), (t_, 0, x))
    K = sp.integrate(f.subs(x, t_), (t_, 0, 1))
    eq = sp.Poly(sp.expand(lhs - (2*x**3 - 3*x**2*K)), x).all_coeffs()
    s = sp.solve(eq, [c0, c1, c2, c3], dict=True)[0]
    V.check(7, "f(1)", f.subs(s).subs(x, 1), vals(7)[0])

def p08():
    best = None
    for n_ in range(1, 60):
        for m_ in range(1, 3000):
            if m_**2 == (2*n_)**3:
                lhs = sp.log(sp.sqrt(m_), 2*n_) + 2*sp.log(m_, 2*n_)
                if sp.simplify(lhs - sp.Rational(15, 4)) == 0:
                    best = m_+n_ if best is None else min(best, m_+n_)
    V.check(8, "min m+n", best, vals(8)[0])

def p09():
    a1 = 25
    # a_n = a1 - n/2 (n>=2): 주어진 합 관계를 몇 개의 n에서 직접 확인
    a = lambda m: a1 if m == 1 else a1 - sp.Rational(m, 2)   # n>=2 에서 a_n = a_1 - n/2, a_1 은 주어진 값
    ok = all(sum(a(k_) - a(k_+1) for k_ in range(1, m)) == sum(sp.Rational(k_, m-1) for k_ in range(1, m)) for m in range(2, 12))
    V.report(9, ok, "n=2..11 에서 관계식 성립")
    V.check(9, "a40", a(40), vals(9)[0])

def p10():
    a = lambda m: 4*2**(m-1)
    S_ = lambda m: sum(a(i)**2 for i in range(1, m+1))
    sols = [m for m in range(3, 40) if S_(m) - S_(m-2) == 320]
    V.check(10, "n", sols[0], vals(10)[0])
    V.check(10, "해 유일", len(sols), 1)

def p11():
    a, b = sp.symbols("a b")
    f1 = x**3 + x**2 - x + 3
    f2 = a*x**2 + b
    s = sp.solve([f1.subs(x, 1) - f2.subs(x, 1), sp.diff(f1, x).subs(x, 1) - sp.diff(f2, x).subs(x, 1)], [a, b])
    V.check(11, "a-b", s[a] - s[b], vals(11)[0])

def p12():
    AC, AD = 6, 3*sp.sqrt(3)
    sinA = sp.sqrt(3)/3
    BC = AC*sinA
    AB = sp.sqrt(AC**2 - BC**2)
    cosD = BC/AC   # ∠ADB=∠ACB
    xx = sp.symbols("x")
    quad = sp.expand(AD**2 + xx**2 - 2*AD*xx*cosD - AB**2)   # AB^2 = AD^2+x^2-2 AD x cos
    c = sp.Poly(quad, xx).all_coeffs()
    exp = vals(12)
    V.check(12, "(가)BC", sp.simplify(BC), exp[0])
    V.check(12, "(나)AB", sp.simplify(AB), exp[1])
    V.check(12, "(다)cos", sp.simplify(cosD), exp[2])
    V.check(12, "(라) 계수", c[1], exp[3])
    V.check(12, "상수항 3", c[2], 3)

def p13():
    a, C = sp.symbols("a C")
    fp = a*x*(x-2)
    f = sp.integrate(fp, x) + C
    s = sp.solve([f.subs(x, 0) + 1, f.subs(x, 2) - 3], [a, C])
    g = f.subs(s)
    V.report(13, s[a] < 0, "a<0")
    V.check(13, "f(-1)", g.subs(x, -1), vals(13)[0])

def p14():
    roots = set()
    for r in sp.solve(sp.Eq(sp.sin(x), sp.cos(4*x)), x):
        pass
    import math
    h = lambda v: math.sin(v) - math.cos(4*v)
    def bis(u, w):
        for _ in range(100):
            m_ = (u+w)/2
            if h(u)*h(m_) <= 0: w = m_
            else: u = m_
        return (u+w)/2
    N = 20001
    grid = [2*math.pi*i/(N-1) for i in range(N)]
    rs = []
    for u, w in zip(grid[:-1], grid[1:]):
        if h(u)*h(w) < 0: rs.append(bis(u, w))
    # sin x - cos 4x 는 x=pi/2 에서 접하므로(부호 불변) 별도 확인
    if abs(h(math.pi/2)) < 1e-12 and h(math.pi/2-1e-3)*h(math.pi/2+1e-3) > 0: rs.append(math.pi/2)
    rs = sorted(set(round(r, 9) for r in rs))
    V.check(14, "근 개수", len(rs), 7)
    V.check(14, "근의 합", float(sum(rs)), float(vals(14)[0]), 1e-6)

def p15():
    # 후보: f = x^2 (x>=-1), f = 1 (x<-1)  -- (가) 인수분해와 (나)(다) 조건을 수치로 확인
    f = sp.Piecewise((1, x < -1), (x**2, True))
    F = sp.lambdify(x, f, "math")
    ok_a = all(abs((F(v)-v**2)*(F(v)-1)) < 1e-12 for v in [i/10 for i in range(-50, 51)])
    V.report(15, ok_a, "(가) 모든 x")
    V.report(15, min(F(i/100) for i in range(-500, 501)) == 0, "(다) 최솟값 0")
    # (나): x=-1 외의 점에서 좌우 미분계수 일치(수치), x=-1 에서는 불일치
    def dl(v, h=1e-6): return (F(v)-F(v-h))/h
    def dr(v, h=1e-6): return (F(v+h)-F(v))/h
    ok_b = all(abs(dl(v)-dr(v)) < 1e-4 for v in [-3, -2, -0.5, 0, 0.5, 1, 2, 3])
    V.report(15, ok_b and abs(dl(-1)-dr(-1)) > 1, "(나) x=-1 에서만 미분불가")
    V.check(15, "f(-2)+f'(2)", F(-2) + 2*2, vals(15)[0])

if __name__ == "__main__": V.run(globals())
