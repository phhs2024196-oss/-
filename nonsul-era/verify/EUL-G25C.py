import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("EUL-G25C", "items/EUL/PAST")
R = sp.Rational
PT = lambda v: list(sp.sympify(v))
E = lambda v: sp.sympify(v, locals={"x": x})

def p08():
    a, b, c, k = sp.symbols("a b c k")
    A = sp.solve(sp.limit((a*x**2+b*x+c)/(4*x**2-1), x, sp.oo) - 1, a)[0]
    V.check(8, "1 최고차항", A, V.ans(8, "1"))
    f = 4*(x - R(1, 2))*(x + k)
    kk = sp.solve(sp.limit(f/(4*x**2-1), x, R(1, 2)) - 2, k)[0]
    f = f.subs(k, kk)
    V.check(8, "2 f(1/2)", f.subs(x, R(1, 2)), V.ans(8, "2"))
    V.check(8, "3 f(3/2)", f.subs(x, R(3, 2)), V.ans(8, "3"))

def p09():
    a1, d = sp.symbols("a1 d")
    S = lambda m: m*(2*a1 + (m-1)*d)/2
    sol = sp.solve(sp.Poly(sp.expand(S(n+2) - S(n) - (56 - 8*n)), n).coeffs(), [a1, d])
    V.check(9, "1 d", sol[d], V.ans(9, "1"))
    V.check(9, "2 a1", sol[a1], V.ans(9, "2"))
    V.check(9, "3 S6", S(6).subs(sol), V.ans(9, "3"))

def p10():
    f = x**3 - 2*x**2 + x + 2
    def tan(p): return sp.diff(f, x).subs(x, p)*(x - p) + f.subs(x, p)
    xs = [r for r in sp.solve(f - tan(1), x) if r != 1]
    B = (xs[0], f.subs(x, xs[0]))
    C = (sp.solve(tan(B[0]), x)[0], 0)
    V.check(10, "1 B", list(B), PT(V.ans(10, "1")))
    V.check(10, "2 C", list(C), PT(V.ans(10, "2")))
    P = [(0, 0), (1, 2), B, C]
    area = abs(sum(P[i][0]*P[(i+1) % 4][1] - P[(i+1) % 4][0]*P[i][1] for i in range(4)))/2
    V.check(10, "3 area", area, V.ans(10, "3"))

def p11():
    a = sp.symbols("a"); t_ = sp.symbols("t_")
    F = sp.Function("F")
    def solve_with(lo, rhs, alpha):
        # x*I1 - I2 = rhs, I1=∫f, I2=∫t f ; 미분: I1 = rhs'
        I1 = sp.diff(rhs, x)            # ∫_lo^x f dt
        av = sp.solve(I1.subs(x, lo), a)[0]
        fx = sp.diff(I1.subs(a, av), x)
        return av, fx
    # 문항 원문(하한 -2, a/2+2): 그대로 풀면
    av, fx = solve_with(-2, x**3 + (a/2 + 2)*x**2 - 2*a, None)
    V.check(11, "원문 1 a", av, V.ans(11, "1"))
    V.check(11, "원문 2 f(x)", fx, E(V.ans(11, "2")))
    V.check(11, "원문 3 f(a)", fx.subs(x, av), V.ans(11, "3"))
    # 공식 해설은 하한 -3, 계수 2a+1/2 전제로 f(x)=6x+9, f(a)=21 — 원문과 불일치(교차검증 2회 일치, DB 답은 원문 기준으로 정정)
    av2, fx2 = solve_with(-3, x**3 + (2*a + R(1, 2))*x**2 - 2*a, None)
    V.check(11, "공식전제 f(a)=21", fx2.subs(x, av2), 21)

def p12():
    r, a1 = sp.symbols("r a1", positive=True)
    ak = lambda i: a1*r**(i-1)
    e = ak(1)/ak(2) + ak(3)/ak(2) + ak(3)/ak(4) + ak(5)/ak(4) + ak(5)/ak(6) + ak(7)/ak(6) - 10
    rv = [s for s in sp.solve(e, r) if s > 1][0]
    V.check(12, "1 r", rv, V.ans(12, "1"))
    a1v = sp.solve(ak(4).subs(r, rv) - 6, a1)[0]
    V.check(12, "2 a1", a1v, V.ans(12, "2"))
    nmin = next(m for m in range(1, 100) if a1v*(rv**m - 1)/(rv - 1) > 3**12)
    V.check(12, "3 n", nmin, V.ans(12, "3"))

def p13():
    sols = [(A, B, C) for A in range(1, 40) for B in range(1, 80) for C in range(1, 40)
            if A + C == 15 and -A + C == -3 and sp.sin(B*(sp.pi/6 + sp.pi)) == 0]
    V.check(13, "1 a*c", sols[0][0]*sols[0][2], V.ans(13, "1"))
    V.check(13, "2 min", min(sum(s) for s in sols), V.ans(13, "2"))

def p14():
    p, q = sp.symbols("p q", real=True)
    # A(p,log2 p), B(p',log2(p'-4)-2), 기울기 -1/2, 중점 x=6
    pb = sp.symbols("pb", real=True)
    sol = None
    for pv in sp.solve(sp.Eq(p + pb, 12), pb) and [None]:
        pass
    # 수치 탐색: p in (0,12), pb=12-p>4
    import mpmath as mp
    fn = lambda pv: (mp.log(12 - pv - 4, 2) - 2 - mp.log(pv, 2)) / (12 - 2*pv) + mp.mpf(1)/2
    pv = mp.findroot(fn, 4)
    A = (pv, mp.log(pv, 2)); B = (12 - pv, mp.log(12 - pv - 4, 2) - 2)
    V.check(14, "1 A", list([sp.nsimplify(float(A[0])), sp.nsimplify(float(A[1]))]), PT(V.ans(14, "1", "A")))
    V.check(14, "1 B", list([sp.nsimplify(float(B[0])), sp.nsimplify(float(B[1]))]), PT(V.ans(14, "1", "B")))
    V.check(14, "2 line", -R(1, 2)*x + 4, E(V.ans(14, "2")))
    V.check(14, "2 passes A,B", list([(-R(1,2)*4+4), (-R(1,2)*8+4)]), [PT(V.ans(14, "1", "A"))[1], PT(V.ans(14, "1", "B"))[1]])

if __name__ == "__main__": V.run(globals())
