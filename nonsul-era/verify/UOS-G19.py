import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
V = Verifier("UOS-G19", "items/UOS/PAST")


def fa(a):
    return [1, 6*a + 2, 11*a**2 + 10*a + 1, 6*a**3 + 14*a**2 + 4*a, 3*a**3 + 5*a**2 + a]


def p01():
    a = sp.symbols("a")
    A, B = sp.symbols("A B")
    f = x**4 + (6*a + 2)*x**3 + (11*a**2 + 10*a + 1)*x**2 + (6*a**3 + 14*a**2 + 4*a)*x + 3*a**3 + 5*a**2 + a
    diff = sp.expand(f - (x**2 + A*x + 3*a**2 + 5*a + 1)*(x**2 + B*x + a))
    sol = sp.solve(sp.Poly(diff, x).all_coeffs(), [A, B], dict=True)
    ok = len(sol) == 1
    V.report(1, ok, f"(a) 계수비교 해 {sol}")
    V.check(1, "1-1 A", sol[0][A], sp.sympify(V.ans(1, "1-1", "A"), locals={"a": a}))
    V.check(1, "1-1 B", sol[0][B], sp.sympify(V.ans(1, "1-1", "B"), locals={"a": a}))
    # (b) a 격자에서 m_a>=0 ⇔ a=-1 또는 0<=a<=1 (최솟값은 도함수의 실근 + 정확 유리수 격자)
    bad = []
    for num in range(-120, 121):
        av = sp.Rational(num, 60)
        c = [sp.Rational(v) for v in fa(av)]
        poly = sp.Poly(sum(ci*x**(4 - i) for i, ci in enumerate(c)), x)
        crit = [r for r in sp.real_roots(sp.diff(poly, x))]
        m = min(poly.eval(r) for r in crit)
        pred = (av == -1) or (0 <= av <= 1)
        if (m >= -sp.Rational(1, 10**12)) != pred and not (abs(m) < 1e-12):
            bad.append((av, float(m)))
        elif abs(m) < 1e-12 and not pred:
            bad.append((av, float(m)))
    V.proof(1, "1-2 집합 {a | a=-1 또는 0<=a<=1} (격자 241점)", bad)
    # a=-1 에서 f=(x^2-2x-1)^2
    V.check(1, "a=-1 완전제곱", sp.expand(f.subs(a, -1) - (x**2 - 2*x - 1)**2), 0)


def p02():
    expr = S(V.ans(2, "2-1"))
    n_ = sp.symbols("n")
    for N in range(2, 9):
        tot = 2*N
        # 당첨 3개: 첫 당첨의 위치(1..tot-2)의 분포를 정확히 계산, A 는 홀수 번째
        p = sp.Rational(0)
        # 직접 계산: 확률의 곱
        for pos in range(1, tot - 1):
            prob = sp.Rational(1)
            for j in range(pos - 1):
                prob *= sp.Rational(tot - 3 - j, tot - j)
            prob *= sp.Rational(3, tot - (pos - 1))
            if pos % 2 == 1:
                p += prob
        V.check(2, f"2-1 n={N} 정확 확률 {p}", p, expr.subs(n_, N))
    # 전수 열거(작은 n): 제비 순열
    N = 4
    tot = 2*N
    # 위치 3개의 모든 조합에서 가장 작은 위치가 짝/홀
    odd = sum(1 for win in itertools.combinations(range(tot), 3) if min(win) % 2 == 0)
    tot_c = math.comb(tot, 3)
    V.check(2, "n=4 조합 전수", sp.Rational(odd, tot_c), expr.subs(n_, 4))


def p03():
    a, b = sp.symbols("a b", positive=True)
    ex = sp.sympify(V.ans(3, "3-1"), locals={"a": a, "b": b})
    # 수치 검증: 임의의 (a,b)에서 사면체를 만들어 cos^2θ 직접 계산
    import random
    random.seed(3)
    ok_all = []
    for _ in range(40):
        av, bv = random.uniform(1, 8), random.uniform(1, 8)
        A_ = np.array([bv, 0, 0]); B_ = np.array([0, av, 0])
        xx = (bv**2 - 9)/(2*bv); yy = (av**2 - 9)/(2*av)
        h2 = 16 - xx**2 - yy**2
        if h2 <= 0:
            continue
        D = np.array([xx, yy, math.sqrt(h2)])
        ok = abs(np.linalg.norm(D - A_) - 5) < 1e-9 and abs(np.linalg.norm(D - B_) - 5) < 1e-9 and abs(np.linalg.norm(D) - 4) < 1e-9
        cos2 = (D[2]/np.linalg.norm(D))**2     # 법선 (0,0,1)과 CD 의 각
        ok_all.append(ok and abs(cos2 - float(ex.subs({a: av, b: bv}))) < 1e-9)
    V.report(3, len(ok_all) > 5 and all(ok_all), f"3-1 임의 (a,b) {len(ok_all)}쌍 사면체 구성 후 cos²θ 일치")
    # 기호 풀이: x,y,h 연립으로 직접
    x_, y_, h_ = sp.symbols("x_ y_ h_", real=True)
    sol = sp.solve([x_**2 + y_**2 + h_**2 - 16, (x_ - b)**2 + y_**2 + h_**2 - 25, x_**2 + (y_ - a)**2 + h_**2 - 25], [x_, y_, h_**2], dict=True)
    V.check(3, "cos²θ=h²/16", sp.simplify((16 - sol[0][x_]**2 - sol[0][y_]**2)/16), ex)


def p04():
    # 두 직사각형: R1=[t1,a] 높이 e^t1, R2=[a2,r2] 높이 e^a2, 0<=t1<=a<=a2<=r2<=2 의 전수 최적화
    best = 0
    N = 200
    xs = np.linspace(0, 2, N + 1)
    for i in range(N + 1):
        for j in range(i, N + 1):
            for k in range(j, N + 1):
                # t1=xs[i], a=xs[j], a2=xs[k], r2=2 (오른쪽 끝까지가 항상 유리)
                s = (xs[j] - xs[i])*math.exp(xs[i]) + (2 - xs[k])*math.exp(xs[k])
                if s > best:
                    best = s
    exact = float(sp.N(S(V.ans(4, "4-1"))))
    V.report(4, 0 <= exact - best < 2e-3, f"4-1 격자 최댓값 {best:.5f} / 정답 {exact:.5f}")
    g = sp.exp(x)*(2 + sp.exp(-1) - x)
    V.check(4, "g 최대", sp.simplify(g.subs(x, 1 + sp.exp(-1))), V.ans(4, "4-1"))
    t = sp.symbols("t")
    a = sp.symbols("a")
    V.check(4, "S1 최대 e^(a-1)", sp.simplify(((a - t)*sp.exp(t)).subs(t, a - 1)), sp.exp(a - 1))


if __name__ == "__main__":
    V.run(globals())
