import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
from math import comb, lgamma, log
from fractions import Fraction
V = Verifier("UOS-M26", "items/UOS/PAST")


def p01():
    a = sp.symbols("a", positive=True)
    p, q, r = sp.symbols("p q r")
    f = x**3 + p*x**2 + q*x + r
    # (2) 에서 f 결정(a 를 매개변수로), (3) 에서 a 결정
    sol = sp.solve([f.subs(x, 0), f.subs(x, a) - a**2, f.subs(x, a**2) - a**4], [p, q, r], dict=True)[0]
    fa = f.subs(sol)
    V.check(1, "f-x^2 인수분해", sp.factor(sp.simplify(fa - x**2)), x*(x - a)*(x - a**2))
    A1 = sp.integrate(fa - x**2, (x, 0, a)); A2 = sp.integrate(x**2 - fa, (x, a, a**2))
    roots = [s_ for s_ in sp.solve(sp.Eq(A1, A2), a) if s_.is_real and s_ > 1]
    V.report(1, len(roots) == 1, f"a>1 해 {roots}")
    fx = sp.expand(fa.subs(a, roots[0]))
    V.check(1, "f(x)", fx, sp.sympify(V.ans(1, "1-1"), locals={"x": x}))
    # 조건 확인(a=2): 부호 변화(두 영역) 및 (2)
    A = roots[0]
    V.proof(1, "조건 (2)", [] if (fx.subs(x, 0) == 0 and fx.subs(x, A) == A**2 and fx.subs(x, A**2) == A**4) else ["위반"])


def p02():
    n_ = sp.Symbol("n")
    for N in range(2, 6):
        M = 2**N; hit = 0
        for a_ in range(1, M + 1):
            for k_ in range(1, N):
                c = a_ * 2**(2*k_)
                if c <= M: hit += 1
        # 직접 삼중항 열거로 교차확인
        cnt = 0
        for a_ in range(1, M + 1):
            for b_ in range(a_ + 1, M + 1):
                if b_ % a_: continue
                rr = b_ // a_
                c = b_ * rr
                if c > M: continue
                # 공비 rr 이 2^k (k>=1)
                if rr >= 2 and (rr & (rr - 1)) == 0: cnt += 1
        assert hit == cnt
        pr = sp.Rational(cnt, comb(M, 3))
        par = "even" if N % 2 == 0 else "odd"
        V.check(2, f"n={N}", pr, S(V.ans(2, "2-1", par)).subs(n_, N))


def p03():
    exact = S(V.ans(3, "3-1"))
    V.check(3, "정적분 값", sp.integrate(sp.log(1 + x)**2, (x, 0, 1)), exact)
    def val(N):
        lnN = log(N)
        a = (N*lnN - 2*(lgamma(2*N + 1) - lgamma(N + 1))) * lnN
        ks = np.arange(N + 1, 2*N + 1, dtype=float)
        b = float(np.sum(np.log(ks)**2))
        return (a + b)/N
    v1, v2 = val(10**6), val(2*10**6)
    # 오차 ~ c ln n / n 형태이므로 두 값의 차이가 줄어드는지와 정확값과의 근접을 확인
    V.check(3, "수치 n=2e6", v2, float(sp.N(exact)), 1e-4)
    V.report(3, abs(v2 - float(sp.N(exact))) <= abs(v1 - float(sp.N(exact))) + 1e-12, "수렴 방향")


def p04():
    b = sp.symbols("b", positive=True)
    a2 = sp.Rational(1, 2)          # a=√2/2 (장축 √2)
    m = -sp.sqrt(3); c = sp.sqrt(3) - b
    bs = sp.solve(sp.Eq(c**2, a2*m**2 + b**2), b)   # 직선 y=mx+c 가 타원에 접할 조건
    bs = [s_ for s_ in bs if s_ > 0]
    V.report(4, len(bs) == 1, f"b 해 {bs}")
    bv = bs[0]
    # 접점 Q: 타원과 직선 AC 의 교점(중근)
    Qs = sp.solve([x**2/a2 + y**2/bv**2 - 1, y - (m*x + c.subs(b, bv))], [x, y], dict=True)
    V.report(4, len(Qs) == 1, f"접점 개수 {len(Qs)} (중근)")
    Q = (sp.simplify(Qs[0][x]), sp.simplify(Qs[0][y])); P = (-Q[0], Q[1])
    kk = sp.simplify(Q[1] / Q[0]**2)          # 포물선 y = k x^2 (P, O, Q 를 지남)
    top = sp.sqrt(3) - bv - sp.sqrt(3)*sp.Abs(x)
    # 포물선과 변 AC 의 교점 중 삼각형 안쪽(|x|<=1) 은 Q 뿐인지 확인
    xs = [s_ for s_ in sp.solve(sp.Eq(kk*x**2, m*x + c.subs(b, bv)), x)]
    V.proof(4, "포물선-AC 교점 중 |x|<=1", [s_ for s_ in xs if abs(s_) <= 1 and s_ != Q[0]])
    S_above = 2*sp.integrate(top.subs(sp.Abs(x), x) - kk*x**2, (x, 0, Q[0]))
    tri = sp.sqrt(3)       # 한 변 2 인 정삼각형
    V.check(4, "작은 넓이", sp.simplify(min(S_above, tri - S_above, key=lambda v: float(v))), V.ans(4, "4-1"))
    V.check(4, "두 넓이의 합=삼각형", S_above + (tri - S_above), tri)


if __name__ == "__main__": V.run(globals())
