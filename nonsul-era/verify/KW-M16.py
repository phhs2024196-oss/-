import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
import mpmath as mp
V = Verifier("KW-M16", "items/KW/PAST")
M = sp.Matrix
SY = lambda s, **kw: sp.sympify(s, locals={"t": t, "x": x, "n": n, **kw})

def p01():
    # [1],[2],[3] 정수 성분(-6..6) 행렬 중 A^2015=O 인 것 전수탐색 (2x2 에서는 A^2015=O ⇔ A^2=O)
    nil = []
    for a, b, c, d in itertools.product(range(-6, 7), repeat=4):
        A = M([[a, b], [c, d]])
        if A**2 == sp.zeros(2): nil.append(A)
    V.report(1, all(A.det() == 0 for A in nil) and len(nil) > 1, f"1 A²=O 인 행렬 {len(nil)}개 모두 det=0 (역행렬 없음) → 교집합 공집합 (YAML {V.ans(1,'1')})")
    a11, a12, a21, a22 = sp.symbols("a11 a12 a21 a22")
    A = M([[a11, a12], [a21, a22]]); E = sp.eye(2)
    # 케일리·해밀턴
    V.report(1, sp.simplify(A*A - (a11 + a22)*A + A.det()*E) == sp.zeros(2), "2 케일리·해밀턴 A²-(a11+a22)A+(det A)E=O")
    V.check(1, "2 tr (A²=O 행렬 전수의 |tr| 최댓값)", max(abs(A[0, 0] + A[1, 1]) for A in nil), SY(V.ans(1, "2")))
    # det=0 이면 A^k = tr^(k-1) A
    k_ = 5
    ok = all(sp.simplify((B**k_ - (B[0, 0] + B[1, 1])**(k_ - 1)*B)) == sp.zeros(2) for B in [M([[2, 4], [1, 2]]), M([[3, 6], [1, 2]]), M([[1, 2], [2, 4]]), M([[0, 1], [0, 0]])])
    V.report(1, ok, "2 det A=0 이면 A^k=(tr A)^(k-1) A (표본)")
    # [3] G
    G = [B for B in nil if (B**2 + E)*(B + E) == E]
    V.report(1, G == [sp.zeros(2)], f"3 G={{O}} (전수탐색 {len(G)}개)")
    # [4] f(x)=-x, 극한
    X = sp.symbols("X", positive=True)
    V.check(1, "4 극한", sp.limit(1/X**2, X, sp.oo), SY(V.ans(1, "4")))
    V.report(1, all(B[1, 1] == -B[0, 0] for B in nil), "4 d=-a (A²=O 행렬 전수)")
    # [5](1) f(n)=r^n, r>1 → 증가
    bad = []
    for r in (sp.Rational(11, 10), sp.Integer(2), sp.Integer(3), sp.Rational(7, 2)):
        f = lambda m: r**m
        bad += [(r, m, nn) for m in range(-8, 9) for nn in range(m + 1, 9) if not f(m) < f(nn)]
        bad += [(r, m, nn) for m in range(-6, 7) for nn in range(-6, 7) if f(m + nn) != f(m)*f(nn)]
    V.proof(1, "5-1", bad)
    # [5](2) det g(A) = r^(a11a22+1) - r^(a12a21+1) ≠ 0 ⇔ det A ≠ 0 (정수 성분)
    bad = []
    for r in (sp.Rational(3, 2), sp.Integer(2), sp.Integer(5)):
        for a_, b_, c_, d_ in itertools.product(range(-2, 3), repeat=4):
            dg = r**(a_*d_)*r - r**(b_*c_)*r
            if a_*d_ - b_*c_ != 0 and dg == 0: bad.append((r, a_, b_, c_, d_))
    V.proof(1, "5-2", bad)

def p02():
    mp.mp.dps = 30
    # [1]
    V.check(2, "1 ∫e^x", sp.integrate(sp.exp(x), (x, 0, 1)), sp.E - 1)
    for nn in (50, 400, 3000):
        rs = sum(mp.e**(mp.mpf(k)/nn) for k in range(1, nn + 1))/nn
        closed = (mp.e**(1 + mp.mpf(1)/nn) - mp.e**(mp.mpf(1)/nn))/(mp.e**(mp.mpf(1)/nn) - 1)/nn
        V.check(2, f"1 등비합 폐쇄형 n={nn}", rs, closed, tol=1e-18)
    V.check(2, "1 극한 수치", mp.e**(mp.mpf(1)/10**6)*(mp.e - 1)*(mp.mpf(1)/10**6)/(mp.e**(mp.mpf(1)/10**6) - 1), mp.e - 1, tol=1e-5)
    # [2]
    hdef, lim_ = V.ans(2, "2", "h"), V.ans(2, "2", "lim")
    V.check(2, "2 정적분", sp.integrate(SY(hdef), (x, 0, 1)), SY(lim_, log=sp.log))
    nn = 200000
    V.check(2, "2 급수 수치", sum(1.0/(2*nn + k) for k in range(1, nn + 1)), sp.N(sp.log(sp.Rational(3, 2))), tol=1e-5)
    V.check(2, "2 Riemann 형", sum(1.0/(2 + k/nn) for k in range(1, nn + 1))/nn, sum(1.0/(2*nn + k) for k in range(1, nn + 1)), tol=1e-9)
    # [3] 수치 반례 탐색(부등식 ①)
    bad = []
    for f_ in (sp.sin(x), x**2 - 1, sp.exp(-x)*sp.cos(3*x), x**3 - 2*x):
        fv = sp.lambdify(x, f_, "mpmath")
        for a_, b_ in itertools.combinations([-2, -1, 0, 0.5, 1, 2.5, 4], 2):
            if abs(mp.quad(fv, [a_, b_])) > mp.quad(lambda u: abs(fv(u)), [a_, b_]) + 1e-20: bad.append((f_, a_, b_))
    V.proof(2, "3", bad)
    # [4] (1)(2) 수치
    bad = []
    for f_ in (sp.exp(x), x**2 + 1, 2 + sp.sin(x), 3 + sp.cos(2*x)):
        fv = sp.lambdify(x, f_, "mpmath"); dv = sp.lambdify(x, sp.diff(f_, x), "mpmath")
        for a_, b_ in ((0, 1), (-1, 2), (0.5, 3)):
            xs = [mp.mpf(a_) + (b_ - a_)*k/200 for k in range(201)]
            T = max(abs(dv(u)) for u in xs)
            bad += [("4-1", f_, u) for u in xs if abs(fv(u) - fv(a_)) > T*(u - a_) + 1e-12]
            err = abs(mp.quad(fv, [a_, b_]) - (b_ - a_)*fv(a_))
            if err > T/2*(b_ - a_)**2 + 1e-12: bad.append(("4-2", f_, a_, b_))
    V.proof(2, "4", bad)
    # [5](1) 오차 한계 ④
    bad = []
    for g_ in (1/(2 + x), 2 + sp.sin(x), sp.exp(x), x**2 + 1):
        gv = sp.lambdify(x, g_, "mpmath"); dv = sp.lambdify(x, sp.diff(g_, x), "mpmath")
        for c_, d_ in ((0, 1), (-1, 2), (1, 4)):
            M_ = max(abs(dv(mp.mpf(c_) + (d_ - c_)*k/2000)) for k in range(2001))
            exact = mp.quad(gv, [c_, d_])
            for N in (1, 2, 5, 25, 100):
                w = mp.mpf(d_ - c_)/N
                approx = w*sum(gv(c_ + (k - 1)*w) for k in range(1, N + 1))
                if abs(exact - approx) > M_/(2*N)*(d_ - c_)**2 + 1e-12: bad.append((g_, c_, d_, N))
    V.proof(2, "5-1", bad)
    # [5](2) N 최솟값
    hx = SY(hdef); M2 = max(abs(sp.diff(hx, x).subs(x, sp.Rational(k, 1000))) for k in range(0, 1001))
    V.check(2, "5-2 M", M2, sp.Rational(1, 4))
    Nmin = min(N for N in range(1, 500) if M2/(2*N)*(1 - 0)**2 <= sp.Rational(1, 200))
    V.check(2, "5-2 N 최솟값", Nmin, SY(V.ans(2, "5-2")))
    w = mp.mpf(1)/Nmin
    approx = w*sum(1/(2 + (k - 1)*w) for k in range(1, Nmin + 1))
    V.report(2, abs(mp.log(mp.mpf(3)/2) - approx) <= mp.mpf(1)/200, f"5-2 실제 오차 {float(abs(mp.log(mp.mpf(3)/2) - approx)):.5f} ≤ 1/200")

if __name__ == "__main__": V.run(globals())
