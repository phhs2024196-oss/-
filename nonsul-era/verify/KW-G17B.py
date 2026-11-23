import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-G17B", "items/KW/PAST")
R_ = sp.Rational

def p01():
    import mpmath as mp
    # [1] |x/(1+x^2)| 의 최댓값
    f = x/(1 + x**2)
    cands = [f.subs(x, r) for r in sp.solveset(sp.diff(f, x), x, sp.Reals)] + [sp.limit(f, x, sp.oo), sp.limit(f, x, -sp.oo)]
    V.check(1, "1 p 최솟값", max(abs(c) for c in cands), V.ans(1, "1"))
    # [2] 타원 x^2/4+y^2=1, |x|<=sqrt3 의 넓이
    area = 2*sp.integrate(sp.sqrt(1 - x**2/4), (x, -sp.sqrt(3), sp.sqrt(3)))
    V.check(1, "2 넓이", sp.simplify(area), V.ans(1, "2", "S"))
    V.check(1, "2 초점거리", sp.sqrt(4 - 1), sp.sqrt(3))   # c^2=a^2-b^2, a=2 (거리의 합 4), b=1
    # [3] 수선의 발
    O = sp.Matrix([0, 0, 0]); A = sp.Matrix([2, 3, 0]); B = sp.Matrix([-1, 2, 1])
    s = sp.symbols("s", real=True)
    ts = sp.solve((A - s*B).dot(B), s)[0]
    H = s*B; H = H.subs(s, ts)
    ans_H = list(sp.sympify(V.ans(1, "3", "H")))
    V.check(1, "3 H", list(H), ans_H)
    V.check(1, "3 OH", sp.simplify(H.norm()), V.ans(1, "3", "OH"))
    # [4] 보조함수 g 의 양 끝값 부호와 한 예에서의 실제 근
    fm, f1 = sp.symbols("fm f1", real=True)
    g_ = lambda xv, fv: fv - xv - sp.cos(sp.pi/2*xv)
    V.check(1, "4 g(-1)=f(-1)+1", sp.simplify(g_(-1, fm) - (fm + 1)), 0)
    V.check(1, "4 g(1)=f(1)-1", sp.simplify(g_(1, f1) - (f1 - 1)), 0)
    ff = lambda xv: -3*xv   # f(-1)=3>0, f(1)=-3<0 인 연속함수 예
    root = mp.findroot(lambda xv: ff(xv) - xv - mp.cos(mp.pi/2*xv), 0.2)
    V.check(1, "4 예시 근이 (-1,1)", bool(-1 < root < 1), True)
    # [5] I_100 + 9900 I_98 (고정밀 수치적분)
    mp.mp.dps = 60
    I = lambda nn: mp.quad(lambda xv: xv**nn*mp.sin(xv), [0, mp.pi/2])
    lhs = I(100) + 9900*I(98)
    rhs = 100*(mp.pi/2)**99
    V.check(1, "5 값", sp.Float(str(lhs), 40), sp.N(sp.sympify(V.ans(1, "5")), 40), tol=1e-30)
    # 점화식도 작은 n 에서 정확히 확인
    for nn in (2, 3, 6):
        ex = sp.integrate(x**nn*sp.sin(x), (x, 0, sp.pi/2)) + nn*(nn-1)*sp.integrate(x**(nn-2)*sp.sin(x), (x, 0, sp.pi/2))
        V.check(1, f"5 점화식 n={nn}", sp.simplify(ex - nn*(sp.pi/2)**(nn-1)), 0)

def p02():
    a0, a1, a2, a3 = sp.symbols("a0:4", real=True)
    pp = sp.symbols("p", nonzero=True)
    f = lambda xv: a0 + a1*xv + a2*xv**2 + a3*xv**3
    # [1] f(x)=f(x+p), p!=0 이면 a1=a2=a3=0
    eqs = sp.Poly(sp.expand(f(x + pp) - f(x)), x).all_coeffs()
    sol = sp.solve(eqs, [a1, a2, a3], dict=True)
    V.check(2, "1 S=상수함수", [len(sol)] + [sol[0].get(v, v) for v in (a1, a2, a3)], [1, 0, 0, 0])
    V.report(2, "not all" in V.ans(2, "1") or "동시에" in V.ans(2, "1"), "1 T-S 서술 확인(a1,a2,a3 not all 0)")
    # [2] f(x^2)=x f(x-2)+12 를 차수 6 이하 다항식에서 풀기
    cs = sp.symbols("c0:7")
    F = lambda xv: sum(c*xv**i for i, c in enumerate(cs))
    eq = sp.Poly(sp.expand(F(x**2) - x*F(x - 2) - 12), x).all_coeffs()
    sol2 = sp.solve(eq, cs, dict=True)
    fx = sp.expand(F(x).subs(sol2[0])) if sol2 else None
    V.check(2, "2 f(x)", sp.expand(fx - sp.sympify(V.ans(2, "2", "f"), locals={"x": x})), 0)
    # [3] 몫과 나머지의 유일성: 작은 정수 계수 탐색
    A = x**3 + 2*x + 5; B = x**2 + 1
    cnt = 0
    rng = range(-4, 5)
    for q0 in rng:
        for q1 in rng:
            for r0 in range(-8, 9):
                for r1 in range(-8, 9):
                    if sp.expand(B*(q1*x + q0) + r1*x + r0 - A) == 0: cnt += 1
    V.check(2, "3 (Q,R) 유일", cnt, 1)
    # [4](1) 2^x 를 3차 다항식으로 보간한 뒤 다른 점에서 어긋남, 그리고 도함수 a1=ln2
    ai = sp.symbols("b0:4")
    interp = sp.solve([sum(ai[i]*xv**i for i in range(4)) - 2**xv for xv in range(4)], ai, dict=True)[0]
    P3 = sum(interp[ai[i]]*x**i for i in range(4))
    V.check(2, "4-1 보간은 x=4 에서 불일치", bool(P3.subs(x, 4) != 16), True)
    V.check(2, "4-1 a1=ln2", sp.diff(2**x, x).subs(x, 0), sp.log(2))
    V.check(2, "4-1 ln2 비정수", bool(sp.N(sp.log(2)) != sp.floor(sp.N(sp.log(2)))), True)
    # [4](2)
    fN = lambda nn: nn**2 - nn - 3
    V.proof(2, "4-2 f(n)<g(n) n<=80", [n_ for n_ in range(1, 81) if not fN(n_) < 2**n_])
    k_ = sp.symbols("k", positive=True)
    V.check(2, "4-2 f(k+1)=f(k)+2k", sp.expand(fN(k_ + 1) - fN(k_) - 2*k_), 0)
    V.proof(2, "4-2 2n<=2^n", [n_ for n_ in range(1, 81) if not 2*n_ <= 2**n_])

if __name__ == "__main__": V.run(globals())
